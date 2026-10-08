/**
 * Shared support for the browser tests that drive whole viewers.
 *
 * panel_dom.test.mjs builds panels on an Object.create'd instance. The
 * failures these tests are about live in the parts that harness skips: two
 * real viewers on one page, frames arriving from a server, a video element,
 * a lost GPU context, a deleted node. So this serves the repository, stubs
 * ComfyUI's two modules, and answers /view with synthetic frames the tests
 * can recognise, and sessionharness.html constructs viewers the way ComfyUI
 * does, through the extension's onNodeCreated.
 *
 * Not a test file itself (no .test. in the name), so `node --test` skips it.
 */
import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { deflateSync } from 'node:zlib';

const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = join(HERE, '..', '..');

const STUBS = {
    '/scripts/app.js': `export const app = {
        registerExtension(e) { (globalThis.__exts ||= []).push(e); },
        graph: { _nodes: [], setDirtyCanvas() {} },
        canvas: { setDirty() {}, selected_nodes: {} },
        extensionManager: { registerSidebarTab() {} },
        ui: { settings: { addSetting() {}, getSettingValue() {} } },
    }; globalThis.app = app;`,
    '/scripts/api.js': `const L = (globalThis.__apiListeners = new Map());
    export const api = {
        addEventListener(t, h) { if (!L.has(t)) L.set(t, new Set()); L.get(t).add(h); },
        removeEventListener(t, h) { L.get(t)?.delete(h); },
        apiURL(p) { return p; },
        fetchApi(p, o) { return fetch(p, o); },
    };`,
};

const MIME = {
    '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html',
    '.wasm': 'application/wasm', '.css': 'text/css', '.png': 'image/png',
    '.webm': 'video/webm', '.json': 'application/json',
};

/** IEEE 754 half from a number, round to nearest. Enough for test frames. */
export function toHalf(v) {
    const f = new Float32Array([v]);
    const x = new Uint32Array(f.buffer)[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 0xff) - 127 + 15;
    let mant = x & 0x7fffff;
    if (exp <= 0) return sign;                       // flush tiny values to zero
    if (exp >= 31) return sign | 0x7c00;
    mant += 0x1000;                                  // round
    if (mant & 0x800000) { mant = 0; exp++; if (exp >= 31) return sign | 0x7c00; }
    return sign | (exp << 10) | (mant >>> 13);
}

function crc32(buf) {
    let c, crc = 0xffffffff;
    for (let n = 0; n < buf.length; n++) {
        c = (crc ^ buf[n]) & 0xff;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        crc = (crc >>> 8) ^ c;
    }
    return (crc ^ 0xffffffff) >>> 0;
}

/** An RGB PNG; the top quarter is light, so B placement can be measured. */
export function makePNG(W, H, value = 128) {
    const raw = Buffer.alloc((W * 3 + 1) * H);
    for (let y = 0; y < H; y++) {
        raw[y * (W * 3 + 1)] = 0;
        for (let x = 0; x < W; x++) {
            const o = y * (W * 3 + 1) + 1 + x * 3;
            raw[o] = raw[o + 1] = raw[o + 2] = value;
        }
    }
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
        chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 1 })), chunk('IEND', Buffer.alloc(0))]);
}

/**
 * An fp16 RGBA RHDR frame. Every pixel is `base`, except pixel 0, whose red
 * channel carries `mark` so a test can tell which frame is on screen.
 * `black` makes the first `black` fraction of pixels pure 0, and `peak` puts
 * one bright value in the last pixel.
 */
export function makeRHDR(W, H, { base = 0.18, mark = 0, black = 0, peak = 0, level = 1 } = {}) {
    const n = W * H;
    const arr = new Uint16Array(n * 4);
    const hb = toHalf(base), one = toHalf(1);
    const blackPixels = Math.floor(n * black);
    for (let i = 0; i < n; i++) {
        const v = i < blackPixels ? 0 : hb;
        arr[i * 4] = v; arr[i * 4 + 1] = v; arr[i * 4 + 2] = v; arr[i * 4 + 3] = one;
    }
    arr[0] = toHalf(mark);
    if (peak) { const p = toHalf(peak); arr[(n - 1) * 4] = p; arr[(n - 1) * 4 + 1] = p; arr[(n - 1) * 4 + 2] = p; }
    const header = Buffer.alloc(12);
    header.write('RHDR', 0);
    header.writeUInt16LE(W, 4); header.writeUInt16LE(H, 6);
    header.writeUInt16LE(4, 8); header.writeUInt16LE(0, 10);
    return Buffer.concat([header, deflateSync(Buffer.from(arr.buffer), { level })]);
}

/**
 * Synthetic /view files, by name:
 *   f_<W>x<H>_<i>.png | .rhdr     frame i (the RHDR's pixel 0 red is i)
 *   b_<W>x<H>_<v>.png | .rhdr     a flat compare frame at value v (PNG 0-255, RHDR v/100)
 *   black_<W>x<H>_<i>.rhdr        half the frame pure black, one 41.6 peak
 *   bad.rhdr, trunc.rhdr          corrupt payloads
 *   clip24.webm                   js/tests/fixtures/clip24.webm
 * Anything else is a 404, as a deleted ComfyUI temp file is.
 */
async function viewFile(name) {
    let m = name.match(/^f_(\d+)x(\d+)_(\d+)\.(png|rhdr)$/);
    if (m) {
        const [W, H, i] = [+m[1], +m[2], +m[3]];
        return m[4] === 'png' ? makePNG(W, H, 100 + (i % 100)) : makeRHDR(W, H, { mark: i });
    }
    m = name.match(/^b_(\d+)x(\d+)_(\d+)\.(png|rhdr)$/);
    if (m) {
        const [W, H, v] = [+m[1], +m[2], +m[3]];
        return m[4] === 'png' ? makePNG(W, H, v) : makeRHDR(W, H, { base: v / 100, mark: v / 100 });
    }
    m = name.match(/^black_(\d+)x(\d+)_(\d+)\.rhdr$/);
    if (m) return makeRHDR(+m[1], +m[2], { base: 0.18, black: 0.5, peak: 41.6 });
    if (name === 'bad.rhdr') {
        const h = Buffer.alloc(12); h.write('RHDR', 0); h.writeUInt16LE(64, 4); h.writeUInt16LE(48, 6); h.writeUInt16LE(4, 8);
        return Buffer.concat([h, Buffer.from('this is not zlib data at all, just garbage bytes'.repeat(20))]);
    }
    if (name === 'trunc.rhdr') return makeRHDR(64, 48).subarray(0, 200);
    if (name === 'clip24.webm') return readFile(join(HERE, 'fixtures', 'clip24.webm'));
    return null;
}

export async function serve() {
    const log = [];
    const server = createServer(async (req, res) => {
        const u = new URL(req.url, 'http://x');
        log.push(u.pathname + u.search);
        if (STUBS[u.pathname]) {
            res.writeHead(200, { 'Content-Type': 'text/javascript' });
            res.end(STUBS[u.pathname]);
            return;
        }
        if (u.pathname === '/view') {
            const name = u.searchParams.get('filename') || '';
            const f = await viewFile(name).catch(() => null);
            if (!f) { res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('404: Not Found'); return; }
            const type = name.endsWith('.png') ? 'image/png' : name.endsWith('.webm') ? 'video/webm' : 'application/octet-stream';
            // Range support, which a <video> needs to seek.
            const range = req.headers.range && /bytes=(\d+)-(\d*)/.exec(req.headers.range);
            if (range) {
                const start = +range[1], end = range[2] ? +range[2] : f.length - 1;
                res.writeHead(206, { 'Content-Type': type, 'Accept-Ranges': 'bytes',
                    'Content-Range': `bytes ${start}-${end}/${f.length}`, 'Content-Length': end - start + 1 });
                res.end(f.subarray(start, end + 1));
                return;
            }
            res.writeHead(200, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': f.length });
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
    return { server, log, port: server.address().port };
}

async function findChromium() {
    for (const p of [process.env.RADIANCE_TEST_CHROMIUM,
        '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].filter(Boolean)) {
        try { await access(p); return p; } catch { /* keep looking */ }
    }
    return null;
}

export function loadPlaywright() {
    const require = createRequire(import.meta.url);
    for (const spec of ['playwright',
        '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) {
        try { return require(spec); } catch { /* keep looking */ }
    }
    return null;
}

/**
 * One browser for a test file. `open()` gives a fresh page on the harness with
 * the viewer module loaded; errors thrown on the page are collected on it.
 */
export async function startSession() {
    const playwright = loadPlaywright();
    if (!playwright) return null;
    const { server, log, port } = await serve();
    const chromiumPath = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
            '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required', '--js-flags=--expose-gc'],
    });
    return {
        log,
        async open() {
            const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
            page.errors = [];
            page.on('pageerror', (e) => page.errors.push(String(e.message)));
            await page.goto(`http://127.0.0.1:${port}/js/tests/sessionharness.html`, { waitUntil: 'load' });
            await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
            return page;
        },
        async close() { await browser.close(); server.close(); },
    };
}
