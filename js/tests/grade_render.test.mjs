/**
 * The grade, rendered by the real viewer.
 *
 * Builds a RadianceViewer through the node registration in Chromium, loads
 * constant-colour float frames through onExecuted, sets grade fields and reads
 * the graded scene-linear signal back (readPixelsFloat32). Each test is one of
 * the review's findings, measured where the user sees it:
 *
 *   H2  contrast pivots on 18% grey and is a power curve about it
 *   H3  Temperature and Tint are gains: black stays black, negatives survive,
 *       the eyedropper neutralises the picked pixel at any level
 *   H4  ACEScct mode at neutral is the identity, for Rec.709 and ACEScg sources
 *   M3  Luma Mix 0 keeps the exposure change
 *   C4  the composite shader computes what gradePixelFull() computes, which is
 *       what the CDL and LUT exporters are built on
 *   M5  the bar's LUT select and the Output Transform are one control
 *   H6  the grade is saved with the workflow and restored on load
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/grade_render.test.mjs
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

const STUBS = {
    '/scripts/app.js': `export const app = {
        registerExtension(e) { (globalThis.__exts ||= []).push(e); },
        graph: { _nodes: [], setDirtyCanvas() {} },
        canvas: { setDirty() {}, selected_nodes: {} },
        extensionManager: { registerSidebarTab() {} },
        ui: { settings: { addSetting() {}, getSettingValue() {} } },
    }; globalThis.app = app;`,
    '/scripts/api.js': `export const api = {
        addEventListener() {}, removeEventListener() {},
        apiURL(p) { return p; },
        fetchApi() { return Promise.resolve({ ok: false, json: () => ({}) }); },
    };`,
};

const MIME = {
    '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html',
    '.wasm': 'application/wasm', '.css': 'text/css', '.png': 'image/png',
};

/** An 8x8 RHDR float32 sidecar of one colour (header + deflated RGBA). */
function rhdr(rgb) {
    const W = 8, H = 8;
    const f32 = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) f32.set([rgb[0], rgb[1], rgb[2], 1], i * 4);
    const header = Buffer.alloc(12);
    header.write('RHDR', 0);
    header.writeUInt16LE(W, 4); header.writeUInt16LE(H, 6);
    header.writeUInt16LE(4, 8); header.writeUInt16LE(1, 10);
    return Buffer.concat([header, deflateSync(Buffer.from(f32.buffer))]);
}

async function findChromium() {
    for (const p of [process.env.RADIANCE_TEST_CHROMIUM,
        '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].filter(Boolean)) {
        try { await access(p); return p; } catch { /* keep looking */ }
    }
    return null;
}

async function loadPlaywright() {
    const require = createRequire(import.meta.url);
    for (const spec of ['playwright', '/home/claude/.npm-global/lib/node_modules/playwright/index.js',
        '/opt/node22/lib/node_modules/playwright']) {
        try { return require(spec); } catch { /* keep looking */ }
    }
    return null;
}

async function serveRepo() {
    const server = createServer(async (req, res) => {
        const u = new URL(req.url, 'http://x');
        if (STUBS[u.pathname]) {
            res.writeHead(200, { 'Content-Type': 'text/javascript' });
            res.end(STUBS[u.pathname]);
            return;
        }
        if (u.pathname === '/view') {
            const m = /^c_(-?[\d.e-]+)_(-?[\d.e-]+)_(-?[\d.e-]+)\.rhdr$/.exec(u.searchParams.get('filename') || '');
            if (!m) { res.writeHead(404); res.end(); return; }
            res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
            res.end(rhdr([Number(m[1]), Number(m[2]), Number(m[3])]));
            return;
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
const skip = playwright ? false
    : 'Playwright is not installed — the viewer cannot be rendered. Install it, or set RADIANCE_TEST_CHROMIUM.';

let report = null;
let pageErrors = [];
if (!skip) {
    const server = await serveRepo();
    const chromiumPath = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
    });
    try {
        const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
        page.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/graderenderharness.html`);
        await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
        report = await page.evaluate(() => window.__run());
    } finally {
        await browser.close();
        server.close();
    }
}

const close = (a, b, eps) => Math.abs(a - b) <= eps;
const luma = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

test('the viewer renders a graded frame', { skip }, () => {
    assert.ok(report, 'no report');
    assert.ok(report.ok, report.error);
    assert.deepEqual(pageErrors, []);
});

test('H2: contrast pivots on 18% grey and spreads stops evenly', { skip }, () => {
    const c = report.cases;
    assert.equal(c.defaultPivot, 0.18, 'a new viewer must pivot on 0.18');
    // Contrast 1.2 used to drop grey 0.63 stop, 1.5 by 3.2 stops, 2.0 sent it negative.
    for (const [k, out] of Object.entries(c.contrast)) {
        assert.ok(close(out[1], 0.18, 1e-5), `contrast ${k} moved grey to ${out[1]}`);
    }
    // One stop above the pivot becomes two at contrast 2.
    assert.ok(close(Math.log2(c.contrastStopAbove[1] / 0.18), 2, 1e-4), `${c.contrastStopAbove}`);
});

test('H3: Temperature and Tint balance colour instead of adding it', { skip }, () => {
    const c = report.cases;
    assert.deepEqual(c.wbBlack, [0, 0, 0], 'Temperature 0.1 must leave black black');
    assert.ok(c.wbNegative[0] < -0.04, `a tiny Tint clamped a negative: ${c.wbNegative}`);
    for (const key of ['pick', 'pickTenth']) {
        const o = c[key].out;
        assert.ok(close(o[0], o[1], 1e-5 * Math.max(1, o[1] * 10)) && close(o[1], o[2], 1e-5 * Math.max(1, o[1] * 10)),
            `${key}: the picked pixel did not come out neutral: ${o}`);
    }
    assert.ok(close(c.pick.state[0], c.pickTenth.state[0], 1e-9) && close(c.pick.state[1], c.pickTenth.state[1], 1e-9),
        'the pick at a tenth of the level must solve to the same Temperature and Tint');
});

test('H4: ACEScct mode with nothing graded leaves the picture alone', { skip }, () => {
    for (const key of ['cct709', 'cctAP1']) {
        const o = report.cases[key];
        [0.5, 0.2, 0.1].forEach((v, i) => assert.ok(close(o[i], v, 1e-5), `${key}: ${o}`));
    }
});

test('M3: Luma Mix 0 keeps the exposure change and holds saturation to it', { skip }, () => {
    const c = report.cases;
    const y0 = luma([0.2, 0.18, 0.1]);
    assert.ok(close(luma(c.lumaMixExposure), 2 * y0, 1e-5), `exposure +1 at Luma Mix 0 gave ${luma(c.lumaMixExposure)}`);
    assert.ok(close(luma(c.lumaMixSat), y0, 1e-5), 'saturation changed brightness at Luma Mix 0');
});

test('C4/M10: the composite shader computes what gradePixelFull computes', { skip }, () => {
    // The exporters are built on gradePixelFull, so this is what makes an
    // exported grade match the screen. Curves included, read at texel centres.
    assert.equal(report.cases.full.length, 10);
    for (const c of report.cases.full) {
        const d = Math.max(...c.gpu.map((v, i) => Math.abs(v - c.cpu[i]) / Math.max(1, Math.abs(c.cpu[i]))));
        assert.ok(d <= 2e-4, `${c.cct ? 'ACEScct ' : ''}${JSON.stringify(c.in)}: GPU ${JSON.stringify(c.gpu)} vs JS ${JSON.stringify(c.cpu)} (Δ ${d})`);
    }
});

test('M5: the LUT selects route through one manual pick and stay in step', { skip }, () => {
    const m = report.cases.m5;
    assert.ok(m.hasBar);
    // The bar set displayLut directly and left the view on Auto, so the next
    // image replaced the pick.
    assert.deepEqual(m.afterBar, { viewMode: 'manual', displayLut: 'Rec.709 (Broadcast)', gradeSelect: 'Rec.709 (Broadcast)' });
    assert.deepEqual(m.afterGrade, { viewMode: 'manual', displayLut: 'Reinhard Tonemap', bar: 'Reinhard Tonemap' });
    assert.deepEqual(m.afterNextImage, { viewMode: 'manual', displayLut: 'Reinhard Tonemap', bar: 'Reinhard Tonemap', gradeSelect: 'Reinhard Tonemap' });
    assert.equal(m.deadKeyWritten, null, 'the bar still writes a localStorage key nothing reads');
});

test('H6: the grade is saved into the workflow and restored by onConfigure', { skip }, () => {
    const h = report.cases.h6;
    assert.deepEqual(h.saved, { saturation: 0, contrast: 1.4, offset: [0.02, 0, 0] });
    assert.deepEqual(h.restored, { saturation: 0, contrast: 1.4, offset: [0.02, 0, 0], logMidtone: [0, 0.05, 0], temperature: 0.5 });
    // And the restored grade is what the shader draws, the fields render()
    // does not push every frame (offset, log wheels) included.
    h.gpu.forEach((v, i) => assert.ok(Math.abs(v - h.cpu[i]) < 1e-5, `restored grade renders ${h.gpu}, expected ${h.cpu}`));
});
