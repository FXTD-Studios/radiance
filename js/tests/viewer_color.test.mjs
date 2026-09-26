/**
 * The Viewer shows what the node says the pixels are (3.5.0).
 *
 * Drives a real RadianceViewer in Chromium (WebGL under SwiftShader) with
 * tagged results and reads the visible canvas back:
 *
 *   - an sRGB-encoded ComfyUI IMAGE sent as floats is shown untouched
 *     (it used to be decoded as linear and encoded again: washed out)
 *   - a linear source is shown through OpenColorIO ACES 2.0, the real one:
 *     0.18 lands at 0.349 sRGB, not the Narkowicz fit's 0.556
 *   - the node's PNG preview, already a display image, is not transformed again
 *   - compare_image reaches the wipe on WebGL (the base-class stub threw)
 *   - going back to a cached frame does not show black
 *   - the source fps drives playback
 *
 * Skips when Playwright is unavailable.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { deflateSync } from 'node:zlib';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = join(JS, '..');
const W = 64, H = 48;

const STUBS = {
    '/scripts/app.js': `export const app = {
        registerExtension(e) { (globalThis.__exts ||= []).push(e); },
        graph: { _nodes: [], setDirtyCanvas() {} },
        canvas: { setDirty() {} },
        extensionManager: { registerSidebarTab() {} },
        ui: { settings: { addSetting() {}, getSettingValue() {} } },
    }; globalThis.app = app;`,
    '/scripts/api.js': `export const api = {
        addEventListener() {}, removeEventListener() {},
        apiURL(p) { return p; },
        fetchApi() { return Promise.resolve({ ok: false, json: () => ({}) }); },
    };`,
};
const MIME = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html',
    '.wasm': 'application/wasm', '.png': 'image/png' };

function png(v8) {
    const raw = Buffer.alloc((W * 3 + 1) * H);
    for (let y = 0; y < H; y++) {
        raw[y * (W * 3 + 1)] = 0;
        raw.fill(v8, y * (W * 3 + 1) + 1, (y + 1) * (W * 3 + 1));
    }
    const crc32 = (buf) => {
        let c, crc = 0xffffffff;
        for (let n = 0; n < buf.length; n++) {
            c = (crc ^ buf[n]) & 0xff;
            for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
            crc = (crc >>> 8) ^ c;
        }
        return (crc ^ 0xffffffff) >>> 0;
    };
    const chunk = (type, data) => {
        const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
        const td = Buffer.concat([Buffer.from(type), data]);
        const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
        return Buffer.concat([len, td, crc]);
    };
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
    ihdr[8] = 8; ihdr[9] = 2;
    return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

// Reference triplets evaluated with OCIO 2.5 ACES studio CPU processors.
const GAMUT_CASES = [
    {
        "id": "0",
        "space": "ACEScg",
        "source": [
            0.3,
            0.2,
            0.1
        ],
        "reference": [
            0.37883102893829346,
            0.1880291998386383,
            0.08230243623256683
        ]
    },
    {
        "id": "1",
        "space": "ACES2065-1",
        "source": [
            0.3,
            0.2,
            0.1
        ],
        "reference": [
            0.4909241199493408,
            0.18197593092918396,
            0.08162686228752136
        ]
    },
    {
        "id": "2",
        "space": "Linear Rec.2020",
        "source": [
            0.3,
            0.2,
            0.1
        ],
        "reference": [
            0.3733340799808502,
            0.18837988376617432,
            0.08631196618080139
        ]
    },
    {
        "id": "3",
        "space": "Linear P3-D65",
        "source": [
            0.3,
            0.2,
            0.1
        ],
        "reference": [
            0.3224940299987793,
            0.19579429924488068,
            0.0882088840007782
        ]
    }
];

function rhdr(v) {
    const f32 = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) { if (Array.isArray(v)) f32.set(v, i * 4); else f32.fill(v, i * 4, i * 4 + 3); f32[i * 4 + 3] = 1; }
    const header = Buffer.alloc(12);
    header.write('RHDR', 0);
    header.writeUInt16LE(W, 4); header.writeUInt16LE(H, 6);
    header.writeUInt16LE(4, 8); header.writeUInt16LE(1, 10);
    return Buffer.concat([header, deflateSync(Buffer.from(f32.buffer))]);
}

function stripes() {
    const f32 = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) { const v = (i % W) % 2 ? 1 : 0; f32.fill(v, i * 4, i * 4 + 3); f32[i * 4 + 3] = 1; }
    const header = Buffer.alloc(12);
    header.write('RHDR', 0); header.writeUInt16LE(W, 4); header.writeUInt16LE(H, 6);
    header.writeUInt16LE(4, 8); header.writeUInt16LE(1, 10);
    return Buffer.concat([header, deflateSync(Buffer.from(f32.buffer))]);
}

function file(name) {
    if (name === 'stripes.rhdr') return stripes();
    const gamut = /^gamut_(\d+)_(source|reference)\.rhdr$/.exec(name);
    if (gamut) return rhdr(GAMUT_CASES[Number(gamut[1])][gamut[2]]);
    let m = /^png_(\d+)\.png$/.exec(name);
    if (m) return png(Number(m[1]));
    m = /^f_(-?[\d.]+)\.rhdr$/.exec(name);
    if (m) return rhdr(Number(m[1]));
    return null;
}

function entry(value, { encoding, colorspace, pngValue = 128, withHdr = true, compare = false } = {}) {
    return {
        filename: `png_${pngValue}.png`, subfolder: '', type: 'temp',
        ...(withHdr ? { hdr_sidecar: `f_${value}.rhdr`, hdr_primary: true, hdr_fp32: true } : {}),
        has_hdr: false, data_range: [value, value], source_width: W, source_height: H,
        channel_names: ['R', 'G', 'B', 'A'],
        ...(encoding ? { source_encoding: encoding, source_colorspace: colorspace } : {}),
        ...(compare ? { is_compare: true } : {}),
    };
}

async function findChromium() {
    for (const p of [process.env.RADIANCE_TEST_CHROMIUM, '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].filter(Boolean)) {
        try { await access(p); return p; } catch { /* next */ }
    }
    return null;
}
async function loadPlaywright() {
    const require = createRequire(import.meta.url);
    for (const spec of ['playwright', '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) {
        try { return require(spec); } catch { /* next */ }
    }
    return null;
}
async function serve() {
    const server = createServer(async (req, res) => {
        const u = new URL(req.url, 'http://x');
        if (STUBS[u.pathname]) { res.writeHead(200, { 'Content-Type': 'text/javascript' }); res.end(STUBS[u.pathname]); return; }
        if (u.pathname === '/view') {
            const name = u.searchParams.get('filename');
            const f = file(name);
            if (!f) { res.writeHead(404); res.end(); return; }
            res.writeHead(200, { 'Content-Type': name.endsWith('.png') ? 'image/png' : 'application/octet-stream' });
            res.end(f); return;
        }
        try {
            const p = join(ROOT, normalize(decodeURIComponent(u.pathname)));
            if (!p.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
            const data = await readFile(p);
            res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream' });
            res.end(data);
        } catch { res.writeHead(404); res.end('not found'); }
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    return server;
}

const playwright = await loadPlaywright();
const skip = playwright ? false : 'Playwright is not installed';
const R = {};
const pageErrors = [];
const hardwareGpu = process.env.RADIANCE_TEST_HARDWARE_GPU === '1';
const gpuArgs = hardwareGpu ? (process.platform === 'win32'
    ? ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist'] : ['--ignore-gpu-blocklist'])
    : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
if (!skip) {
    const server = await serve();
    const exe = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(exe ? { executablePath: exe } : {}),
        args: ['--no-sandbox', ...gpuArgs],
    });
    try {
        const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
        page.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/colorharness.html`, { waitUntil: 'load' });
        await page.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
        const show = (msg, opts) => page.evaluate(([m, o]) => window.__show(m, o), [msg, opts || {}]);
        const SRGB = { encoding: 'srgb', colorspace: 'sRGB Encoded Rec.709 (sRGB)' };
        const LIN = { encoding: 'linear', colorspace: 'Linear Rec.709 (sRGB)' };
        R.gamutViews = [];
        for (const item of GAMUT_CASES) {
            const pair = {};
            for (const kind of ['source', 'reference']) {
                await show({ radiance_images: [{ ...entry(0.3, {
                    encoding: 'linear', colorspace: kind === 'source' ? item.space : LIN.colorspace,
                }), hdr_sidecar: `gamut_${item.id}_${kind}.rhdr` }] }, { waitOcio: true });
                pair[kind] = {};
                for (const mode of ['srgb', 'rec709', 'filmic', 'aces2']) {
                    pair[kind][mode] = await page.evaluate((v) => window.__viewMode(v), mode);
                }
            }
            R.gamutViews.push({ space: item.space, ...pair });
        }
        await page.evaluate(() => localStorage.removeItem('radiance_view_mode'));
        R.srgb = await show({ radiance_images: [entry(0.5, SRGB)], fps: [25] });
        R.linear = await show({ radiance_images: [entry(0.18, LIN)] }, { waitOcio: true });
        R.preview = await show({ radiance_images: [entry(0, { ...LIN, withHdr: false, pngValue: 128 })] });
        R.compare = await show({ radiance_images: [entry(0.5, SRGB), entry(0, { ...SRGB, withHdr: false, pngValue: 200, compare: true })] });
        R.falseColor = await show({ radiance_images: [entry(0.18, LIN)] }, { flags: { falseColor: true } });
        R.gamut = await show({ radiance_images: [entry(-0.2, LIN)] }, { flags: { gamutWarning: true } });
        R.scope = await show({ radiance_images: [entry(0.5, SRGB)] }, { flags: { falseColor: true }, scopes: true });
        R.viewEv = await show({ radiance_images: [entry(0.5, SRGB)] }, { flags: { viewExposure: 1 }, scopes: true });
        R.keys = await page.evaluate((m) => window.__keys(m),
            { radiance_images: [0, 1, 2, 3].map((i) => ({ ...entry(0.1 * (i + 1), SRGB), frame: i })) });
        R.cache = await show({ radiance_images: [{ ...entry(0.25, SRGB), frame: 0 }, { ...entry(0.75, SRGB), frame: 1 }] },
            { frames: [1, 0, 1, 0] });
        // 2x device pixel ratio: the canvas is device-sized and 1:1+ is crisp.
        const page2 = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
        page2.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page2.goto(`http://127.0.0.1:${server.address().port}/js/tests/colorharness.html`, { waitUntil: 'load' });
        await page2.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
        R.dpr = await page2.evaluate(async (msg) => {
            const out = await window.__show(msg, {});
            return out;
        }, { radiance_images: [{ ...entry(0, SRGB), hdr_sidecar: 'stripes.rhdr' }] });
        R.dprPixels = await page2.evaluate(async () => {
            const v = window.__lastViewer;
            v.setZoom(4);
            await new Promise((r) => setTimeout(r, 200));
            v.render();
            const c = v.canvas, ctx = c.getContext('2d');
            const cssW = c.getBoundingClientRect().width;
            const y = Math.floor(c.height / 2);
            const row = ctx.getImageData(Math.floor(c.width / 2) - 40, y, 80, 1).data;
            const vals = []; for (let i = 0; i < row.length; i += 4) vals.push(row[i]);
            return { canvasW: c.width, cssW, vals };
        });
    } finally {
        await browser.close();
        server.close();
    }
}

// Display P3: a browser with a forced P3 output profile.
if (!skip) {
    const server = await serve();
    const exe = await findChromium();
    const b3 = await playwright.chromium.launch({
        ...(exe ? { executablePath: exe } : {}),
        args: ['--no-sandbox', ...gpuArgs,
               '--force-color-profile=display-p3-d65'],
    });
    try {
        const pg = await b3.newPage({ viewport: { width: 1280, height: 900 } });
        pg.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await pg.addInitScript(() => { try { localStorage.setItem('radiance_display_target', 'Display P3'); } catch {} });
        await pg.goto(`http://127.0.0.1:${server.address().port}/js/tests/colorharness.html`, { waitUntil: 'load' });
        await pg.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
        R.p3 = await pg.evaluate((m) => window.__show(m, { waitOcio: true }),
            { radiance_images: [entry(0.18, { encoding: 'linear', colorspace: 'Linear Rec.709 (sRGB)' })] });
        R.p3.extra = await pg.evaluate(() => ({
            capable: window.__lastViewer.displayP3Capable,
            buffer: window.__lastViewer.renderer.displayColorSpace || null,
            display: window.__lastViewer.ocioDisplay, view: window.__lastViewer.ocioView,
        }));
    } finally { await b3.close(); server.close(); }
}

const ok = (r) => r && r.ok ? false : `harness failed: ${r?.error}`;

test('the viewer loads without page errors', { skip }, () => {
    assert.deepEqual(pageErrors, [], pageErrors.join('\n'));
});

test('an sRGB-encoded IMAGE is shown untouched (no second encode)', { skip: skip || ok(R.srgb) }, () => {
    const [r, g, b] = R.srgb.centre;
    assert.equal(R.srgb.state.effective, 'srgb');
    assert.equal(R.srgb.state.isLinear, false, 'float sRGB data must be decoded, not read as linear');
    for (const c of [r, g, b]) assert.ok(Math.abs(c - 127.5) <= 3, `0.5 sRGB shows as ${c.toFixed(1)}, expected 127.5`);
});

test('the source fps drives playback', { skip: skip || ok(R.srgb) }, () => {
    assert.equal(R.srgb.state.fps, 25);
});

test('a linear source is shown through OCIO ACES 2.0, not the approximation', { skip: skip || ok(R.linear) }, () => {
    assert.ok(R.linear.ocioReady, `OCIO auto view never became active: ${R.linear.state.status}`);
    assert.equal(R.linear.state.ocioView, 'ACES 2.0 - SDR 100 nits (Rec.709)');
    assert.equal(R.linear.state.ocioSource, 'Linear Rec.709 (sRGB)');
    // OCIO CPU reference: 0.18 -> 0.3492 sRGB = 89.0 / 255. The Narkowicz fit
    // plus sRGB puts it at 0.556 = 142.
    const [r] = R.linear.centre;
    assert.ok(Math.abs(r - 89.0) <= 4, `18% grey shows as ${r.toFixed(1)}, ACES 2.0 gives 89`);
});

test('the node PNG preview is a display image and is not transformed again', { skip: skip || ok(R.preview) }, () => {
    const [r] = R.preview.centre;
    assert.ok(Math.abs(r - 128) <= 3, `display preview 128 shows as ${r.toFixed(1)}`);
});

test('compare_image reaches the wipe on WebGL', { skip: skip || ok(R.compare) }, () => {
    assert.equal(R.compare.state.compareMode, 'wipe');
    assert.ok(R.compare.state.hasReference, 'no compare texture was uploaded');
    const [bSide] = R.compare.left;
    assert.ok(Math.abs(bSide - 200) <= 4, `B side shows ${bSide.toFixed(1)}, compare preview is 200`);
});

test('returning to a cached frame shows it, not black', { skip: skip || ok(R.cache) }, () => {
    const vals = R.cache.frames.map((x) => x.centre[0]);
    const want = [0.75, 0.25, 0.75, 0.25].map((v) => v * 255);
    vals.forEach((v, i) => assert.ok(Math.abs(v - want[i]) <= 4,
        `step ${i} (frame ${R.cache.frames[i].f}) shows ${v.toFixed(1)}, expected ${want[i]}`));
});

test('the WebGL shader compiles and is the backend under test', { skip: skip || ok(R.srgb) }, () => {
    assert.equal(R.srgb.state.backend, 'RadianceWebGLRenderer');
    const shaderErrors = R.cache.state.errors.filter((e) => /shader|compil/i.test(e));
    assert.deepEqual(shaderErrors, [], shaderErrors.join('\n'));
});

test('false colour puts 18 % grey in the ARRI green band, under any view', { skip: skip || ok(R.falseColor) }, () => {
    const [r, g, b] = R.falseColor.centre;
    assert.ok(r < 30 && g > 190 && b < 80, `18 % grey reads (${R.falseColor.centre}) — ARRI green is 38-42 %`);
});

test('the gamut warning fires on a negative scene value', { skip: skip || ok(R.gamut) }, () => {
    const [r, g, b] = R.gamut.centre;
    assert.ok(r > 230 && g < 30 && b > 230, `negative value shows (${R.gamut.centre}), expected magenta`);
});

test('scopes read the displayed picture, not the overlays on it', { skip: skip || ok(R.scope) }, () => {
    // False colour is on, so the screen is a band colour; the scope signal is
    // still the picture: 0.5 sRGB -> 128.
    assert.ok(R.scope.signal, 'no display signal was rendered');
    R.scope.signal.forEach((c) => assert.ok(Math.abs(c - 128) <= 2, `scope signal ${R.scope.signal}`));
    assert.deepEqual(R.scope.centreAfterSignal, R.scope.centre, 'rendering the scope signal changed the screen');
});

test('viewer f-stop changes the screen only, not the scopes or the export', { skip: skip || ok(R.viewEv) }, () => {
    // 0.5 sRGB = 0.214 linear; +1 stop = 0.428 -> 0.687 sRGB = 175.
    assert.ok(Math.abs(R.viewEv.centre[0] - 175) <= 3, `screen shows ${R.viewEv.centre}`);
    R.viewEv.signal.forEach((c) => assert.ok(Math.abs(c - 128) <= 2, `scope signal moved: ${R.viewEv.signal}`));
    assert.ok(Math.abs(R.viewEv.exportValue - 0.214) < 0.003, `graded EXR moved: ${R.viewEv.exportValue}`);
});

test('the canvas is in device pixels and magnified pixels stay crisp', { skip: skip || ok(R.dpr) }, () => {
    const { canvasW, cssW, vals } = R.dprPixels;
    assert.ok(Math.abs(canvasW - cssW * 2) <= 2, `canvas ${canvasW}px for ${cssW} CSS px at DPR 2`);
    const mid = vals.filter((v) => v > 12 && v < 243);
    assert.deepEqual(mid, [], `smoothed values at 4x zoom: ${vals.join(',')}`);
    assert.ok(vals.some((v) => v > 243) && vals.some((v) => v < 12), 'the stripes are not visible');
});

test('keys reach only the viewer in use, and never take Ctrl combinations', { skip: skip || ok(R.keys) }, () => {
    assert.deepEqual(R.keys.ev, [0, 0.5], `f-stop per viewer: ${R.keys.ev}`);
    assert.equal(R.keys.afterLeave, 0.5, 'leaving an unselected Viewer releases keyboard ownership');
});

test('in/out points bound playback and ping-pong reverses at them', { skip: skip || ok(R.keys) }, () => {
    assert.deepEqual(R.keys.seq, [2, 1, 2, 1, 2, 1]);
    assert.equal(R.keys.home, 1, 'Home goes to the in point');
    assert.equal(R.keys.channel, 'a', 'A shows alpha, as in RV');
});

test('on a P3 monitor the ACES view targets Display P3 and says so to the browser', { skip: skip || ok(R.p3) }, () => {
    const x = R.p3.extra;
    if (!x.capable) return;   // this Chromium build ignores the forced profile
    assert.equal(x.display, 'Display P3 - Display');
    assert.equal(x.view, 'ACES 2.0 - SDR 100 nits (P3 D65)');
    assert.equal(x.buffer, 'display-p3');
});


test('wide-gamut sources match OCIO-converted Rec.709 in every built-in view', { skip }, () => {
    for (const item of R.gamutViews) {
        for (const mode of ['srgb', 'rec709', 'filmic', 'aces2']) {
            const a = item.source[mode], b = item.reference[mode];
            for (let c = 0; c < 3; c++) {
                assert.ok(Math.abs(a.centre[c] - b.centre[c]) <= 2,
                    `${item.space} ${mode} channel ${c}: ${a.centre} vs ${b.centre}`);
            }
            assert.ok(Math.abs(a.exportRed - 0.3) < 1e-5,
                `${item.space} ${mode}: display conversion leaked into EXR export`);
        }
    }
});


test('hardware acceptance uses the actual GPU', { skip: skip || !hardwareGpu }, () => {
    assert.match(R.srgb.state.gpuRenderer || '', /NVIDIA|AMD|Intel/);
    assert.doesNotMatch(R.srgb.state.gpuRenderer, /SwiftShader|llvmpipe|Microsoft Basic/i);
    console.log('Hardware renderer:', R.srgb.state.gpuRenderer);
});
