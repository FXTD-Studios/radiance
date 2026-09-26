/**
 * A frame that arrives through the paging window reaches the screen.
 *
 * The bounded frame window (radiance_frame_window.js) replaced the per-frame
 * onload/then handlers in the Viewer's onExecuted. frame_window.test.mjs pins
 * the window's memory bound; nothing pinned that a paged-in frame is actually
 * displayed. This builds a real RadianceViewer in Chromium (constructor,
 * canvas, WebGL renderer under SwiftShader), serves it a synthetic result --
 * a PNG proxy plus an fp32 RHDR sidecar, the same two files the node writes --
 * drives onExecuted, and reads the pixels back off the visible canvas.
 *
 * A black canvas after a successful load is the failure this file is for.
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/sequence_load.test.mjs
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

const MIME = {
    '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html',
    '.wasm': 'application/wasm', '.css': 'text/css', '.png': 'image/png',
};

// ── Synthetic frames ────────────────────────────────────────────────────────
const W = 64, H = 48;

/** A mid-grey PNG with a bright bar: the proxy the node writes. */
function makePNG() {
    // Minimal uncompressed-ish PNG encoder (RGB, 8-bit) via zlib.
    const raw = Buffer.alloc((W * 3 + 1) * H);
    for (let y = 0; y < H; y++) {
        raw[y * (W * 3 + 1)] = 0;
        for (let x = 0; x < W; x++) {
            const o = y * (W * 3 + 1) + 1 + x * 3;
            const v = y < H / 4 ? 240 : 128;
            raw[o] = v; raw[o + 1] = v; raw[o + 2] = v;
        }
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
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    return Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
    ]);
}

/** An fp32 RHDR sidecar: scene-linear, mid-grey 0.18 with a 4.0 highlight bar. */
function makeRHDR(channels = 4) {
    const f32 = new Float32Array(W * H * channels);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const v = y < H / 4 ? 4.0 : 0.18;
        const o = (y * W + x) * channels;
        f32[o] = v; f32[o + 1] = v; f32[o + 2] = v;
        if (channels === 4) f32[o + 3] = 1.0;
    }
    const header = Buffer.alloc(12);
    header.write('RHDR', 0);
    header.writeUInt16LE(W, 4); header.writeUInt16LE(H, 6);
    header.writeUInt16LE(channels, 8); header.writeUInt16LE(1, 10);   // flags=1: fp32
    return Buffer.concat([header, deflateSync(Buffer.from(f32.buffer))]);
}

const FILES = { 'frame_0.png': makePNG(), 'frame_0.rhdr': makeRHDR(),
                'frame_1.png': makePNG(), 'frame_1.rhdr': makeRHDR() };

function frameEntry(i) {
    return {
        filename: `frame_${i}.png`, subfolder: '', type: 'temp',
        hdr_sidecar: `frame_${i}.rhdr`, hdr_primary: true, hdr_fp32: true,
        has_hdr: true, data_range: [0.18, 4.0],
        source_width: W, source_height: H, preview_tonemapped: true,
        channel_names: ['R', 'G', 'B', 'A'],
    };
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
    for (const spec of ['playwright',
        '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) {
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
            const f = FILES[u.searchParams.get('filename')];
            if (!f) { res.writeHead(404); res.end(); return; }
            res.writeHead(200, { 'Content-Type': u.searchParams.get('filename').endsWith('.png') ? 'image/png' : 'application/octet-stream' });
            res.end(f);
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
    : 'Playwright is not installed — the viewer cannot be driven. '
      + 'Install it, or set RADIANCE_TEST_CHROMIUM.';

let report = null;
let pageErrors = [];
if (!skip) {
    const server = await serveRepo();
    const chromiumPath = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
               '--ignore-gpu-blocklist'],
    });
    try {
        const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
        page.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/sequenceharness.html`,
            { waitUntil: 'load' });
        try {
            await page.waitForFunction(() => typeof window.__run === 'function',
                null, { timeout: 20000 });
        } catch {
            pageErrors.push('window.__run never appeared — the harness module did not run');
        }
        report = await page.evaluate((opts) => window.__run(opts), {
            timeoutMs: 8000,
            message: { radiance_images: [frameEntry(0), frameEntry(1)] },
        });
    } finally {
        await browser.close();
        server.close();
    }
}

test('the viewer module loads and the harness ran', { skip }, () => {
    assert.deepEqual(pageErrors, [], 'the module threw while loading:\n' + pageErrors.join('\n'));
    assert.ok(report, 'no report');
    assert.ok(report.ok, `harness failed: ${report?.error}\n${(report?.log || []).join('\n')}`);
});

const noReport = report?.ok ? false : 'the harness did not finish';

test('a frame paged in through the window is on screen, not black', { skip: skip || noReport }, () => {
    assert.equal(report.totalFrames, 2);
    assert.equal(report.currentFrame, 0);
    assert.ok(report.hasHDR, `frame 0 has no HDR data after load (fallback: ${report.fallbackReason}); log:\n${report.log.join('\n')}`);
    assert.deepEqual(report.imageSize, [W, H], 'the viewer did not take the frame size');
    assert.ok(report.textureBound, 'the renderer has no image texture bound');
    const px = report.pixels;
    assert.ok(!px.error, `canvas unreadable: ${px.error}`);
    assert.ok(px.max > 40,
        `the visible canvas is black after the frame loaded (max ${px.max}, mean ${px.mean}); `
        + `backend ${report.backend}; log:\n${report.log.join('\n')}`);
});

test('scrubbing to a second frame displays it', { skip: skip || noReport }, () => {
    assert.ok(report.frame1, 'no second frame was sampled');
    assert.equal(report.frame1.currentFrame, 1);
    assert.ok(report.frame1.hasHDR, 'frame 1 has no HDR data after scrub');
    assert.ok(report.frame1.pixels.max > 40, `frame 1 is black (max ${report.frame1.pixels.max})`);
});

test('the status badge reports the float source, not the proxy', { skip: skip || noReport }, () => {
    assert.match(String(report.badge), /FP32/, `badge reads ${report.badge}`);
    assert.doesNotMatch(String(report.badge), /PROXY/);
});

test('the viewer does not grow its host to its panel content', { skip: skip || noReport }, () => {
    // Live on ComfyUI 0.32 / frontend 1.48 the sidebar and inspector content
    // (about 2000 px) set the Vue node's height, the canvas column stretched
    // with it, and fitToView centred the frame below the visible area: the
    // second half of the black viewer. contain: size on the container holds
    // the host at the viewer's own minimum.
    assert.ok(report.autoContentHeight > 600,
        `panel content is only ${report.autoContentHeight}px; the check proves nothing`);
    assert.ok(report.autoHostHeight <= 400,
        `an auto-height host grew to ${report.autoHostHeight}px to fit the panels`);
});
