/**
 * The Viewer's picture and numbers, measured in a real browser.
 *
 * Drives a real RadianceViewer in Chromium (WebGL under SwiftShader) with
 * synthetic float frames delivered the way the node delivers them, and reads
 * back what a user would read: the scopes, the probe, the status bar, the
 * labels and the saved PNG.
 *
 *   - scopes plot code values on the shared scale, from every pixel, in float
 *   - the probe decodes an sRGB-encoded float source before it measures light
 *   - the built-in filmic view sits near ACES 2.0, and Fit averages in light
 *   - the colour-space labels follow the source tag
 *   - Save PNG (Result) leaves out the viewer-only look and the overlays
 *   - gamut, NaN and Inf are flagged rather than drawn black
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/viewer_truth.test.mjs
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
const MIME = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html',
    '.wasm': 'application/wasm', '.png': 'image/png' };

// ── Fixtures ────────────────────────────────────────────────────────────────

function half(f) {
    const fv = new Float32Array(1), iv = new Uint32Array(fv.buffer);
    fv[0] = f;
    const x = iv[0];
    if (Number.isNaN(f)) return 0x7e00;
    const s = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 127 + 15;
    const m = (x >>> 13) & 0x3ff;
    if (e >= 31) return s | 0x7c00;
    if (e <= 0) return s;
    return s | (e << 10) | m;
}

/** An RHDR sidecar, fp32 or fp16, from a per-pixel fill function. */
function rhdrFrom(W, H, fill, fp16 = false) {
    const f32 = new Float32Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const v = fill(x, y), i = (y * W + x) * 4;
        f32[i] = v[0]; f32[i + 1] = v[1]; f32[i + 2] = v[2]; f32[i + 3] = 1;
    }
    const header = Buffer.alloc(12);
    header.write('RHDR', 0);
    header.writeUInt16LE(W, 4); header.writeUInt16LE(H, 6);
    header.writeUInt16LE(4, 8); header.writeUInt16LE(fp16 ? 0 : 1, 10);
    let payload;
    if (fp16) {
        const u16 = new Uint16Array(f32.length);
        for (let i = 0; i < f32.length; i++) u16[i] = half(f32[i]);
        payload = Buffer.from(u16.buffer);
    } else payload = Buffer.from(f32.buffer);
    return Buffer.concat([header, deflateSync(payload)]);
}

function png(W, H, v8) {
    const raw = Buffer.alloc((W * 3 + 1) * H);
    for (let y = 0; y < H; y++) { raw[y * (W * 3 + 1)] = 0; raw.fill(v8, y * (W * 3 + 1) + 1, (y + 1) * (W * 3 + 1)); }
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
    ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
    return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

const FIXTURES = {
    // 2000x1000 grey 0.5 with one white column (x = 1001) and 105 isolated
    // white pixels: 1105 pixels at 1.0 in all.
    'spike.rhdr': () => rhdrFrom(2000, 1000, (x, y) =>
        (x === 1001 || (x % 97 === 13 && y % 211 === 7)) ? [1, 1, 1] : [0.5, 0.5, 0.5]),
    // 2000x1000 grey 0.4 with a one-pixel pure red column.
    'redline.rhdr': () => rhdrFrom(2000, 1000, (x) => (x === 777 ? [1, 0, 0] : [0.4, 0.4, 0.4])),
    // One-pixel black and white stripes.
    'stripes.rhdr': () => rhdrFrom(512, 256, (x) => (x % 2 ? [1, 1, 1] : [0, 0, 0])),
    // 90 % of the frame at 0.25, 10 % at 0.75 (a histogram with two peaks).
    'split.rhdr': () => rhdrFrom(64, 48, (x) => (x < 58 ? [0.25, 0.25, 0.25] : [0.75, 0.75, 0.75])),
    // An 8x8 NaN block and an 8x8 +Inf block in 18 % grey, half float.
    'naninf.rhdr': () => rhdrFrom(64, 48, (x, y) => {
        if (y >= 20 && y < 28 && x >= 8 && x < 16) return [NaN, 0.18, 0.18];
        if (y >= 20 && y < 28 && x >= 40 && x < 48) return [Infinity, 0.18, 0.18];
        return [0.18, 0.18, 0.18];
    }, true),
};

function fixture(name) {
    if (FIXTURES[name]) return FIXTURES[name]();
    let m = /^c_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)\.rhdr$/.exec(name);
    if (m) { const v = [Number(m[1]), Number(m[2]), Number(m[3])]; return rhdrFrom(64, 48, () => v); }
    m = /^png_(\d+)x(\d+)_(\d+)\.png$/.exec(name);
    if (m) return png(Number(m[1]), Number(m[2]), Number(m[3]));
    return null;
}

const SRGB = { encoding: 'srgb', colorspace: 'sRGB Encoded Rec.709 (sRGB)' };
const LIN = { encoding: 'linear', colorspace: 'Linear Rec.709 (sRGB)' };

/** One result entry, shaped like the node's. */
function entry(hdr, { encoding, colorspace, w = 64, h = 48 } = LIN) {
    return {
        filename: `png_${w}x${h}_128.png`, subfolder: '', type: 'temp',
        hdr_sidecar: hdr, hdr_filename: hdr, hdr_primary: true, hdr_fp32: !/naninf/.test(hdr),
        exr_filename: hdr.replace('.rhdr', '.exr'),
        has_hdr: true, data_range: [0, 1], source_width: w, source_height: h,
        channel_names: ['R', 'G', 'B', 'A'], source_encoding: encoding, source_colorspace: colorspace,
        frame: 0, total_frames: 1,
    };
}
const msg = (e) => ({ radiance_images: [e], source_encoding: [e.source_encoding], source_colorspace: [e.source_colorspace] });

// ── Browser ─────────────────────────────────────────────────────────────────

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
            const f = fixture(name);
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

// ── Measurements (each runs in the page) ────────────────────────────────────

/**
 * The sidebar scopes (Scopes tab of the right panel), drawn by the viewer's
 * own update, and the same draw on a black frame for a baseline. The trace is
 * what differs between the two, so the graticule never reads as signal.
 */
const sidebarScopes = async () => {
    const v = window.__lastViewer;
    const host = document.createElement('div');
    document.body.appendChild(host);
    v._renderReferenceScopes(host);
    const grab = () => {
        v._updateReferenceScopes();
        const out = {};
        for (const [k, c] of Object.entries(v._referenceScopeCanvases)) {
            out[k] = { w: c.width, h: c.height, px: Array.from(window.__pixels(c)), text: (c.__text || []).slice(), rects: c.__strokeRects || 0 };
            c.__text = []; c.__strokeRects = 0;
        }
        return out;
    };
    const t0 = performance.now();
    const live = grab();
    const ms = performance.now() - t0;
    const exp = v.exposure;
    v.exposure = -12; v.render();
    const black = grab();
    v.exposure = exp; v.render();
    host.remove();
    return { live, black, ms, plot: [0, 0.5, 1].map((x) => window.__units.plotPos?.(x) ?? null) };
};

/** Rows (top-down) where the trace differs from the baseline, per column band. */
function traceRows(live, black, x0, x1, ch = 1, thresh = 20) {
    const rows = [];
    for (let y = 0; y < live.h; y++) {
        let best = 0;
        for (let x = Math.floor(x0 * live.w); x < Math.ceil(x1 * live.w); x++) {
            const i = (y * live.w + x) * 4 + ch;
            best = Math.max(best, live.px[i] - black.px[i]);
        }
        if (best > thresh) rows.push(y);
    }
    return rows;
}
const rowOf = (pos, h) => (1 - pos) * h;

if (!skip) {
    const server = await serve();
    const exe = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(exe ? { executablePath: exe } : {}),
        args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
    });
    try {
        const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
        page.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/truthharness.html`, { waitUntil: 'load' });
        await page.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
        await page.evaluate(() => localStorage.clear());
        const load = (m, o) => page.evaluate(([mm, oo]) => window.__load(mm, oo), [m, o || {}]);
        const ev = (fn, arg) => page.evaluate(fn, arg);
        const guard = async (key, fn) => {
            try { R[key] = await fn(); } catch (e) { R[key] = { error: String(e && e.stack || e) }; }
        };

        // C1: sRGB white and mid grey on the sidebar waveform and parade.
        for (const val of ['1', '0.5']) {
            await guard(`c1_${val}`, async () => {
                await load(msg(entry(`c_${val}_${val}_${val}.rhdr`, SRGB)), { view: 'srgb' });
                return ev(sidebarScopes);
            });
        }

        // C2: one-pixel detail and super-white on a 2000x1000 frame.
        await guard('c2_spike', async () => {
            await load(msg(entry('spike.rhdr', { ...SRGB, w: 2000, h: 1000 })), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                const src = v._scopeSource();
                const sig = v.renderer.readDisplaySignal(src.width || 0, src.height || 0, 1.0, true);
                const f = sig?.float || null;
                let max = 0, n = 0;
                if (f) for (let i = 0; i < f.length; i += 4) { if (f[i] > max) max = f[i]; if (f[i] >= 0.999) n++; }
                return { size: [src.width, src.height], floatMax: max, n1: n, isFloat: !!f };
            });
        });
        await guard('c2_spike_scope', () => ev(sidebarScopes));
        await guard('c2_tab', () => ev(() => {
            // The Scopes tab, waveform and vectorscope, as a user opens them.
            const v = window.__lastViewer;
            const out = {};
            for (const mode of ['waveform', 'parade', 'histogram']) {
                v.scopeMode = mode; v.scopeTransformed = true; v.scopeLogView = false;
                const host = document.createElement('div'); document.body.appendChild(host);
                v.renderScopesTab(host);
                const c = host.querySelector('canvas');
                out[mode] = { w: c.width, h: c.height, px: Array.from(window.__pixels(c)),
                    note: host.textContent.replace(/\s+/g, ' ') };
                host.remove();
            }
            out.plot1 = window.__units.plotPos(1);
            return out;
        }));
        await guard('c2_red', async () => {
            await load(msg(entry('redline.rhdr', { ...SRGB, w: 2000, h: 1000 })), { view: 'srgb' });
            return ev(sidebarScopes);
        });
        await guard('c2_over', async () => {
            await load(msg(entry('c_1.5_1.5_1.5.rhdr', LIN)), { view: 'srgb' });
            const sc = await ev(sidebarScopes);
            const sig = await ev(() => {
                const v = window.__lastViewer;
                const s = v.renderer.readDisplaySignal(64, 48, 1.0, true);
                return s?.float ? s.float[(24 * 64 + 32) * 4] : null;
            });
            return { ...sc, sig };
        });
        await guard('c2_under', async () => {
            await load(msg(entry('c_-0.1_0.5_0.5.rhdr', LIN)), { view: 'srgb' });
            return ev(() => {
                const v = window.__lastViewer;
                const s = v.renderer.readDisplaySignal(64, 48, 1.0, true);
                return { sig: s?.float ? s.float[(24 * 64 + 32) * 4] : null };
            });
        });

        // C2: throttled, and a slow machine degrades rather than stalls.
        await guard('c2_cost', async () => {
            await load(msg(entry('c_0.5_0.5_0.5.rhdr', SRGB)), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer, RV = window.RadianceViewer;
                const host = document.createElement('div'); document.body.appendChild(host);
                v._renderReferenceScopes(host);
                v._referenceRightTab = 'scopes';
                // Let the panel's own first draw (queued on build) run first.
                await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
                await window.__sleep(300);
                let n = 0;
                const orig = v._updateReferenceScopes;
                v._updateReferenceScopes = function () { n++; return orig.call(this); };
                v._referenceScopeAt = performance.now();
                for (let i = 0; i < 6; i++) v._scheduleReferenceScopeUpdate();
                await window.__waitFor(() => n > 0, 5000);
                await window.__sleep(300);
                const throttled = n;
                v._updateReferenceScopes = orig;
                // One update slower than the guard halves the signal size.
                RV.SCOPE_SLOW_UPDATE_MS = 0;
                v.scopeSignalMax = 4096;
                v._updateReferenceScopes();
                const afterSlow = v.scopeSignalMax;
                RV.SCOPE_SLOW_UPDATE_MS = Infinity;
                // A software rasteriser starts in the degraded mode.
                delete v.scopeSignalMax;
                const software = v._scopeSignalMaxNow();
                host.remove();
                return { throttled, afterSlow, software, budget: v.renderer.scopePointBudget };
            });
        });

        // M9: histogram bins code values per channel; log is a real axis.
        await guard('m9_hist', async () => {
            await load(msg(entry('c_0.5_0.5_0.5.rhdr', SRGB)), { view: 'srgb' });
            return ev(sidebarScopes);
        });
        await guard('m9_log', async () => {
            await load(msg(entry('split.rhdr', SRGB)), { view: 'srgb' });
            return ev(() => {
                const v = window.__lastViewer;
                const src = v._scopeSource();
                const draw = (log) => {
                    const c = document.createElement('canvas'); c.width = 512; c.height = 224;
                    c.__text = [];
                    v.renderer.renderHistogram(c, log, src.tex, src.isLinear, { width: src.width, height: src.height });
                    const d = window.__pixels(c);
                    // The two tallest curve columns (red channel, lit pixels per column).
                    const col = [];
                    for (let x = 0; x < 512; x++) { let n = 0; for (let y = 0; y < 224; y++) if (d[(y * 512 + x) * 4] > 150) n++; col.push(n); }
                    const order = col.map((n, x) => [n, x]).sort((p, q) => q[0] - p[0]);
                    const peaks = [order[0][1]];
                    for (const [, x] of order) if (Math.abs(x - peaks[0]) > 20) { peaks.push(x); break; }
                    const label = c.__text.find((t) => t.text === '512');
                    return { peaks: peaks.sort((p, q) => p - q).map((x) => x / 511), label512: label ? label.x / 512 : null };
                };
                const u = window.__units;
                return { lin: draw(false), log: draw(true),
                    want: { lin: [0.25, 0.75].map((x) => u.plotPos(x)), log: [0.25, 0.75].map((x) => u.plotPos(u.logAssistPos(x))),
                        label: u.plotPos(u.logAssistPos(512 / 1023)) } };
            });
        });

        // C3: probe and status bar on an sRGB-encoded float 0.5.
        await guard('c3', async () => {
            await load(msg(entry('c_0.5_0.5_0.5.rhdr', SRGB)), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                const host = document.createElement('div');
                document.body.appendChild(host);
                v.renderProbeTab(host);
                const rect = v.canvas.getBoundingClientRect();
                const cx = v.panX + v.imageWidth * v.zoom * 0.5, cy = v.panY + v.imageHeight * v.zoom * 0.5;
                v._lastCanvasRect = null;
                v.updateProbe({ clientX: rect.left + cx / (v._canvasScaleX || 1), clientY: rect.top + cy / (v._canvasScaleY || 1) });
                v._probeRenderCurrent();
                const p = v._probeCurrent;
                const out = {
                    hdrEncoding: v.hdrData?.sourceEncoding ?? null, hdrIsLinear: v.hdrData?.isLinear,
                    probe: p ? [p.r, p.g, p.b] : null,
                    caption: v._probeDescribe().label,
                    panel: host.textContent.replace(/\s+/g, ' '),
                    status: v.infoLeft.textContent.replace(/\s+/g, ' ').trim(),
                    screen: window.__sample(v),
                };
                host.remove();
                return out;
            });
        });

        // H9: the built-in filmic view and ACES 2.0 on 18 % grey.
        await guard('h9', async () => {
            await load(msg(entry('c_0.18_0.18_0.18.rhdr', LIN)), { view: 'filmic' });
            return ev(() => ({ filmic: window.__sample(window.__lastViewer)[0] }));
        });

        // H10: one-pixel stripes at 50 % and 25 % zoom.
        await guard('h10', async () => {
            await load(msg(entry('stripes.rhdr', { ...SRGB, w: 512, h: 256 })), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                const out = {};
                for (const z of [1, 0.5, 0.25]) {
                    v.setZoom(z); await window.__sleep(150); v.render();
                    const ctx = v.canvas.getContext('2d');
                    const cx = Math.round(v.panX + v.imageWidth * v.zoom / 2), cy = Math.round(v.panY + v.imageHeight * v.zoom / 2);
                    const d = ctx.getImageData(cx - 10, cy, 20, 1).data;
                    let s = 0; for (let i = 0; i < d.length; i += 4) s += d[i];
                    out['z' + z] = s / (d.length / 4);
                }
                // The full-size paths must stay full size.
                const sig = v.renderer.readDisplaySignal(512, 256, 1.0, true);
                out.signalStripes = [sig.data[(128 * 512 + 100) * 4], sig.data[(128 * 512 + 101) * 4]];
                return out;
            });
        });

        // H11 and M16: labels for a Linear Rec.709 source, and the File Info.
        await guard('h11', async () => {
            await load(msg(entry('c_0.18_0.18_0.18.rhdr', LIN)), {});
            return ev(() => {
                const v = window.__lastViewer;
                v.updateBottomBar?.();
                v._updateProMetadata?.();
                const insp = document.createElement('div'); document.body.appendChild(insp);
                v._renderReferenceInspector(insp);
                const kv = {};
                insp.querySelectorAll('.radiance-ref-kv').forEach((box) => {
                    const ks = box.querySelectorAll('.k');
                    ks.forEach((k) => { kv[k.textContent] = k.nextElementSibling?.textContent; });
                });
                const grade = document.createElement('div'); document.body.appendChild(grade);
                v._renderReferenceGrade?.(grade);
                const opts = [...grade.querySelectorAll('select')].flatMap((s) => [...s.options].map((o) => o.textContent));
                const out = { header: v._proColor?.textContent, inspector: kv, gradeOptions: opts,
                    html: insp.textContent + ' ' + grade.textContent };
                insp.remove(); grade.remove();
                return out;
            });
        });
        await guard('h11_acescg', async () => {
            await load(msg(entry('c_0.18_0.18_0.18.rhdr', { encoding: 'linear', colorspace: 'ACEScg' })), {});
            return ev(() => {
                const v = window.__lastViewer;
                v._updateProMetadata?.();
                return { header: v._proColor?.textContent };
            });
        });

        // H17: Save PNG (Result) with the viewer f-stop, false colour and a P3 tag.
        await guard('h17', async () => {
            await load(msg(entry('c_0.5_0.5_0.5.rhdr', SRGB)), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                let href = null, name = null;
                const orig = HTMLAnchorElement.prototype.click;
                HTMLAnchorElement.prototype.click = function () { href = this.href; name = this.download; };
                const grab = async () => {
                    href = null; v.exportSnapshot('png');
                    const img = new Image(); img.src = href; await img.decode();
                    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
                    const x = c.getContext('2d'); x.drawImage(img, 0, 0);
                    return { px: Array.from(x.getImageData(img.width >> 1, img.height >> 1, 1, 1).data.slice(0, 3)), name, size: [img.width, img.height] };
                };
                const base = await grab();
                v.viewExposure = 2; v.viewGamma = 1.5; v.render();
                const look = await grab();
                v.viewExposure = 0; v.viewGamma = 1; v.falseColor = true; v.zebra = true; v.render();
                const overlays = await grab();
                v.falseColor = false; v.zebra = false; v.render();
                HTMLAnchorElement.prototype.click = orig;
                return { base, look, overlays, logs: (v._termLines || []).slice(-3) };
            });
        });

        // M6: ACEScg pure green is out of the display gamut.
        await guard('m6', async () => {
            await load(msg(entry('c_0_1_0.rhdr', { encoding: 'linear', colorspace: 'ACEScg' })), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                const plain = window.__sample(v);
                v.gamutWarning = true; v.render(); await window.__sleep(50);
                const warn = window.__sample(v);
                v.gamutWarning = false; v.render();
                return { plain, warn };
            });
        });
        // M6: false colour reads luminance with the source gamut's Y row.
        // ACEScg (0, 0, 4): AP1 Y = 4 x 0.0537 = 0.215, 45 % on the ARRI
        // signal, which is no band (grey). Rec.709 weights read 0.289, 53 %,
        // the pink one-stop-over band.
        await guard('m6_luma', async () => {
            await load(msg(entry('c_0_0_4.rhdr', { encoding: 'linear', colorspace: 'ACEScg' })), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                v.falseColor = true; v.render(); await window.__sleep(50);
                const fc = window.__sample(v);
                v.falseColor = false; v.render();
                return { fc };
            });
        });

        // M7: NaN and Inf are flagged and counted.
        await guard('m7', async () => {
            await load(msg(entry('naninf.rhdr', LIN)), { view: 'srgb' });
            return ev(async () => {
                const v = window.__lastViewer;
                await window.__waitFor(() => v.hdrData?.data, 5000);
                v.render(); await window.__sleep(100);
                const nan = window.__sampleImg(v, 12 / 64, 24 / 48);
                const inf = window.__sampleImg(v, 44 / 64, 24 / 48);
                const grey = window.__sampleImg(v, 30 / 64, 10 / 48);
                v.updateBottomBar?.();
                const hud = (v.container.textContent || '').replace(/\s+/g, ' ');
                const m = /(\d+)\s*NaN/.exec(hud), n = /(\d+)\s*Inf/.exec(hud);
                // Below 100 % the footprint path must flag them too.
                v.setZoom(0.5); await window.__sleep(100); v.render();
                const px1 = (u, w) => Array.from(v.canvas.getContext('2d').getImageData(
                    Math.floor(v.panX + u * v.imageWidth * v.zoom), Math.floor(v.panY + w * v.imageHeight * v.zoom), 1, 1).data.slice(0, 3));
                const fitNan = px1(12 / 64, 24 / 48);
                const fitInf = px1(44 / 64, 24 / 48);
                return { nan, inf, grey, fitNan, fitInf, nanCount: m ? Number(m[1]) : null, infCount: n ? Number(n[1]) : null,
                    rawInf: v.hdrData?.data ? v.hdrData.data[(24 * 64 + 44) * 4] : null };
            });
        });

        // M8: the chromaticity scope plots linear source xy.
        await guard('m8', async () => {
            await load(msg(entry('c_1_0.5_0.rhdr', SRGB)), { view: 'srgb' });
            return ev(() => {
                const v = window.__lastViewer;
                // The Scopes tab, as a user opens it, with the plotted points
                // recorded off its canvas.
                const proto = CanvasRenderingContext2D.prototype;
                const orig = proto.fillRect;
                proto.fillRect = function (x, y, w, h) {
                    if (w <= 2 && h <= 2) (this.canvas.__pts ||= []).push([x, y]);
                    return orig.call(this, x, y, w, h);
                };
                v.scopeMode = 'chromaticity';
                const host = document.createElement('div'); document.body.appendChild(host);
                try { v.renderScopesTab(host); } finally { proto.fillRect = orig; }
                const c = host.querySelector('canvas');
                const pts = c?.__pts || [];
                const W = c?.width || 1000, H = c?.height || 1000;
                host.remove();
                // Back from canvas to xy with the panel's own mapping.
                const pad = 60, sW = W - pad * 2, sH = H - pad * 2;
                const xy = pts.map(([x, y]) => [((x - pad) / sW) * 0.8, ((H - pad - y) / sH) * 0.9]);
                const mean = xy.reduce((a, p) => [a[0] + p[0] / xy.length, a[1] + p[1] / xy.length], [0, 0]);
                return { n: pts.length, xy: mean };
            });
        });

        // M19: the canvas tag follows what the shader writes.
        await guard('m19', async () => {
            await load(msg(entry('c_0.18_0.18_0.18.rhdr', LIN)), {});
            return ev(async () => {
                const v = window.__lastViewer, r = v.renderer, gl = r.gl;
                if (!('drawingBufferColorSpace' in gl)) return { unsupported: true };
                v.setViewMode('aces2');
                await window.__waitFor(() => v.ocioActive && v._ocioAutoActive, 20000);
                r.setDisplayColorSpace('display-p3'); v.render();
                const p3View = gl.drawingBufferColorSpace;
                // The same view, but the frame fell back to the 8-bit sRGB preview.
                const png = await new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = '/view?filename=png_64x48_128.png'; });
                r.loadImageTexture(png); r.setSourceEncoding('linear'); v.render();
                const preview = gl.drawingBufferColorSpace;
                return { ocio: !!r.ocioEnabled, p3View, preview, reported: r.displayColorSpace };
            });
        });

    } finally {
        await browser.close();
        server.close();
    }
}

const ok = (r) => (r && !r.error ? false : `harness failed: ${r?.error}`);
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

test('the viewer loads without page errors', { skip }, () => {
    assert.deepEqual(pageErrors, [], pageErrors.join('\n'));
});

// ── C1 ──────────────────────────────────────────────────────────────────────

test('C1: sRGB white plots at the 100% line of the sidebar waveform and parade', { skip: skip || ok(R.c1_1) }, () => {
    const { live, black, plot } = R.c1_1;
    assert.ok(plot[2] !== null, 'radiance_scope_units has no plotPos');
    for (const k of ['waveform', 'parade']) {
        const rows = traceRows(live[k], black[k], 0.1, 0.9, 1);
        assert.ok(rows.length, `${k}: no trace`);
        const want = rowOf(plot[2], live[k].h);
        assert.ok(rows.some((y) => Math.abs(y - want) <= 3), `${k}: white at rows ${rows}, the 100% line is row ${want.toFixed(1)}`);
        assert.ok(rows.every((y) => Math.abs(y - want) <= 3), `${k}: trace off the 100% line: ${rows}`);
    }
});

test('C1: mid grey (code 0.5) plots at 50%', { skip: skip || ok(R['c1_0.5']) }, () => {
    const { live, black, plot } = R['c1_0.5'];
    assert.equal(plot[1], 0.5, 'the plot must put code 0.5 at half height');
    for (const k of ['waveform', 'parade']) {
        const rows = traceRows(live[k], black[k], 0.1, 0.9, 1);
        const want = rowOf(0.5, live[k].h);
        assert.ok(rows.length && rows.every((y) => Math.abs(y - want) <= 3), `${k}: grey at rows ${rows}, want ${want}`);
    }
});

test('C1: the sidebar graticule is in code values, with no nit labels on an SDR view', { skip: skip || ok(R.c1_1) }, () => {
    for (const k of ['waveform', 'parade']) {
        const text = R.c1_1.live[k].text.map((t) => t.text);
        assert.ok(text.includes('1023') && text.includes('512'), `${k} graticule says ${JSON.stringify(text)}`);
        assert.ok(!text.some((t) => /nit/i.test(t)), `${k} has nit labels on an sRGB view: ${text}`);
    }
});

// ── C2 ──────────────────────────────────────────────────────────────────────

test('C2: the scope signal is full resolution float and keeps every white pixel', { skip: skip || ok(R.c2_spike) }, () => {
    const s = R.c2_spike;
    assert.ok(s.isFloat, 'the scope signal is not a float target');
    assert.deepEqual(s.size, [2000, 1000], 'the scope signal is resampled');
    assert.ok(s.floatMax >= 0.999, `signal max ${s.floatMax}`);
    assert.ok(s.n1 >= 1100, `${s.n1} white pixels in the signal, the frame has 1105`);
});

test('C2: the sidebar waveform shows the white column and the isolated pixels', { skip: skip || ok(R.c2_spike_scope) }, () => {
    const { live, black, plot } = R.c2_spike_scope;
    const wf = live.waveform, bk = black.waveform;
    const y1 = Math.round(rowOf(plot[2], wf.h));
    const lit = new Set();
    for (let y = y1 - 2; y <= y1 + 2; y++) for (let x = 0; x < wf.w; x++) {
        const i = (y * wf.w + x) * 4 + 1;
        if (wf.px[i] - bk.px[i] > 12) lit.add(x);
    }
    const col = Math.round(1001 / 2000 * wf.w);
    assert.ok([...lit].some((x) => Math.abs(x - col) <= 2), `the white column (scope x ${col}) is missing: lit ${[...lit]}`);
    const isolated = [...lit].filter((x) => Math.abs(x - col) > 3);
    assert.ok(isolated.length >= 15, `isolated white pixels: ${isolated.length} columns lit, 21 expected`);
});

test('C2: the Scopes tab plots the white column from every pixel, in float', { skip: skip || ok(R.c2_tab) }, () => {
    const wf = R.c2_tab.waveform;
    const y1 = Math.round((1 - R.c2_tab.plot1) * wf.h);
    const col = Math.round(1001 / 2000 * wf.w);
    let best = 0;
    for (let y = y1 - 3; y <= y1 + 3; y++) for (let x = col - 2; x <= col + 2; x++) best = Math.max(best, wf.px[(y * wf.w + x) * 4 + 1]);
    assert.ok(best > 60, `no white column at the 100% line on the Scopes tab waveform: ${best}`);
    assert.match(wf.note, /Every pixel, in float/, 'the caption still claims 8-bit sampling');
    // Parade: the column is white in all three channels.
    const pd = R.c2_tab.parade;
    for (let ch = 0; ch < 3; ch++) {
        const cx = Math.round((ch + 1001 / 2000) / 3 * pd.w);
        let b = 0;
        for (let y = y1 - 3; y <= y1 + 3; y++) for (let x = cx - 2; x <= cx + 2; x++) b = Math.max(b, pd.px[(y * pd.w + x) * 4 + ch]);
        assert.ok(b > 60, `parade channel ${ch}: no white column (${b})`);
    }
});

test('C2: a one-pixel red column shows on the vectorscope', { skip: skip || ok(R.c2_red) }, () => {
    const { live, black } = R.c2_red;
    const vs = live.vectorscope, bk = black.vectorscope;
    // 100% red: Cb = -0.1146, Cr = 0.5 on BT.709.
    const half = Math.min(vs.w, vs.h) / 2;
    const cx = vs.w / 2 + (-0.114572 / 0.5) * half * 0.9, cy = vs.h / 2 - (0.5 / 0.5) * half * 0.9;
    let best = 0;
    for (let y = Math.round(cy) - 4; y <= Math.round(cy) + 4; y++) for (let x = Math.round(cx) - 4; x <= Math.round(cx) + 4; x++) {
        const i = (y * vs.w + x) * 4;
        best = Math.max(best, vs.px[i] - bk.px[i]);
    }
    assert.ok(best > 30, `no red trace at the 100% red target (${cx.toFixed(0)}, ${cy.toFixed(0)}): ${best}`);
});

test('C2: super-white and below-black survive into the scope signal and plot in the headroom', { skip: skip || ok(R.c2_over) || ok(R.c2_under) }, () => {
    // Linear 1.5 through the plain sRGB view encodes to 1.19.
    near(R.c2_over.sig, 1.194, 0.01, 'super-white in the scope signal');
    assert.ok(R.c2_under.sig < -0.05, `below-black clamped in the scope signal: ${R.c2_under.sig}`);
    const { live, black, plot } = R.c2_over;
    assert.ok(plot[2] < 0.97, `no headroom above the 100% line (plotPos(1) = ${plot[2]})`);
    const rows = traceRows(live.waveform, black.waveform, 0.1, 0.9, 1);
    const y1 = rowOf(plot[2], live.waveform.h);
    assert.ok(rows.length && rows.every((y) => y < y1 - 1), `super-white plotted at rows ${rows}, the 100% line is ${y1}`);
});

test('C2: scope updates are throttled, and a slow machine degrades instead of stalling', { skip: skip || ok(R.c2_cost) }, () => {
    const c = R.c2_cost;
    assert.equal(c.throttled, 1, `six requests inside the interval gave ${c.throttled} updates`);
    assert.equal(c.afterSlow, 2048, 'a slow update did not halve the scope signal');
    // SwiftShader is a software rasteriser: the scopes start at 1024 px and
    // 128K points (a GPU starts at 4096 px and every pixel of a 4K frame).
    assert.equal(c.software, 1024);
    assert.ok(c.budget <= 131072, `software point budget ${c.budget}`);
});

// ── M9 ──────────────────────────────────────────────────────────────────────

test('M9: the histogram bins code values, so mid grey peaks at half width', { skip: skip || ok(R.m9_hist) }, () => {
    const h = R.m9_hist.live.histogram;
    let bestX = -1, best = -1;
    for (let x = 0; x < h.w; x++) {
        let s = 0;
        for (let y = 0; y < h.h; y++) s += h.px[(y * h.w + x) * 4];
        if (s > best) { best = s; bestX = x; }
    }
    near(bestX / (h.w - 1), 0.5, 0.03, 'histogram peak position');
});

test('M9: the log histogram is a log axis, not moved grid lines', { skip: skip || ok(R.m9_log) }, () => {
    // Log is the LogC assist curve, as in the Scopes tab: the bins move with
    // the graticule, so the labels stay on their values.
    const { lin, log, want } = R.m9_log;
    lin.peaks.forEach((p, i) => near(p, want.lin[i], 0.02, `linear peak ${i}`));
    log.peaks.forEach((p, i) => near(p, want.log[i], 0.02, `log peak ${i}`));
    assert.ok(log.label512 !== null, 'the log histogram has no 512 label');
    near(log.label512, want.label, 0.03, 'log graticule 512 label');
});

test('M9: the sidebar vectorscope draws its graticule', { skip: skip || ok(R.m9_hist) }, () => {
    assert.ok(R.m9_hist.live.vectorscope.rects >= 6, 'no colour-bar targets on the sidebar vectorscope');
});

// ── C3 ──────────────────────────────────────────────────────────────────────

test('C3: hdrData carries the node source encoding', { skip: skip || ok(R.c3) }, () => {
    assert.equal(R.c3.hdrEncoding, 'srgb');
    assert.equal(R.c3.hdrIsLinear, false);
});

test('C3: the probe decodes an sRGB-encoded float before measuring light', { skip: skip || ok(R.c3) }, () => {
    const { probe, caption, panel } = R.c3;
    for (const c of probe) near(c, 0.214, 0.001, 'probe linear');
    assert.doesNotMatch(caption, /scene-linear/, `caption: ${caption}`);
    assert.match(panel, /EV \+0\.2[45]/, `panel EV: ${panel}`);
    assert.match(panel, /nits 43\.[45]/, `panel nits: ${panel}`);
});

test('C3: the status bar Disp is the rendered display value', { skip: skip || ok(R.c3) }, () => {
    const m = /Disp: ([\d.]+) ([\d.]+) ([\d.]+)/.exec(R.c3.status);
    assert.ok(m, `no Disp in "${R.c3.status}"`);
    for (const v of m.slice(1)) near(Number(v) * 255, 127.5, 1.5, `Disp (screen ${R.c3.screen})`);
    assert.doesNotMatch(R.c3.status, /#000000/, R.c3.status);
    assert.match(R.c3.status, /0\.2140/, `status linear: ${R.c3.status}`);
});

// ── H9 / H10 ────────────────────────────────────────────────────────────────

test('H9: the built-in filmic view puts 18% grey near ACES 2.0 (89)', { skip: skip || ok(R.h9) }, () => {
    near(R.h9.filmic, 89, 4, 'filmic 18% grey');
});

test('H10: Fit averages fine detail in light, not in code values', { skip: skip || ok(R.h10) }, () => {
    near(R.h10['z0.5'], 188, 6, 'stripes at 50%');
    near(R.h10['z0.25'], 188, 6, 'stripes at 25%');
    assert.deepEqual(R.h10.signalStripes.map((v) => v > 128), [false, true], 'the full-size signal lost its stripes');
});

// ── H11 / M16 ───────────────────────────────────────────────────────────────

test('H11: a Linear Rec.709 source is not labelled ACEScg anywhere', { skip: skip || ok(R.h11) }, () => {
    const { header, inspector, gradeOptions, html } = R.h11;
    assert.match(header, /Rec\.709/, `header: ${header}`);
    assert.match(inspector['Input Transform'], /Rec\.709/, `inspector: ${inspector['Input Transform']}`);
    assert.ok(gradeOptions.some((o) => /Rec\.709 \(sRGB\)/.test(o)), `grade options: ${gradeOptions}`);
    assert.doesNotMatch(`${header} ${html} ${gradeOptions.join(' ')}`, /ACEScg/);
});

test('H11: an ACEScg source still says ACEScg', { skip: skip || ok(R.h11_acescg) }, () => {
    assert.match(R.h11_acescg.header, /ACEScg/);
});

test('M16: File Info names the frame, its resolution and its real format', { skip: skip || ok(R.h11) }, () => {
    const kv = R.h11.inspector;
    assert.doesNotMatch(kv['File Name'], /\.png$/, `file name: ${kv['File Name']}`);
    assert.equal(kv.Resolution, '64 x 48');
    assert.match(kv.Format, /RHDR/, `format: ${kv.Format}`);
});

// ── H17 ─────────────────────────────────────────────────────────────────────

test('H17: Save PNG (Result) leaves out the viewer look, overlays and dither', { skip: skip || ok(R.h17) }, () => {
    const { base, look, overlays } = R.h17;
    base.px.forEach((c) => near(c, 128, 1, 'base PNG'));
    assert.deepEqual(look.px, base.px, 'viewer f-stop / gamma baked into the PNG');
    assert.deepEqual(overlays.px, base.px, 'false colour / zebra baked into the PNG');
    assert.deepEqual(base.size, [64, 48]);
});

test('H17: the saved PNG states its colour space', { skip: skip || ok(R.h17) }, () => {
    assert.match(R.h17.base.name, /sRGB|DisplayP3/i, `file name: ${R.h17.base.name}`);
});

// ── M6 ──────────────────────────────────────────────────────────────────────

test('M6: the gamut warning tests the display gamut, not the source gamut', { skip: skip || ok(R.m6) }, () => {
    const [r, g, b] = R.m6.warn;
    assert.ok(r > 230 && g < 30 && b > 230, `ACEScg pure green with the warning on: ${R.m6.warn} (plain ${R.m6.plain})`);
});

test('M6: false colour uses the source gamut luminance row', { skip: skip || ok(R.m6_luma) }, () => {
    // ACEScg (0, 0, 4): AP1 Y = 0.215 -> ARRI signal 45 %, no band, grey 115.
    // Rec.709 weights read Y = 0.289 -> 53 %, the pink band (255, 128, 191).
    const [r, g, b] = R.m6_luma.fc;
    assert.ok(Math.abs(r - g) < 8 && Math.abs(g - b) < 8, `false colour is banded: ${R.m6_luma.fc}`);
    near(r, 115, 8, 'false colour grey level');
});

// ── M7 ──────────────────────────────────────────────────────────────────────

test('M7: NaN and Inf pixels have their own colour, not black', { skip: skip || ok(R.m7) }, () => {
    const { nan, inf, grey } = R.m7;
    const differs = (a, b) => a.some((c, i) => Math.abs(c - b[i]) > 40);
    assert.ok(differs(nan, [0, 0, 0]) && differs(nan, grey), `NaN shows as ${nan}`);
    assert.ok(differs(inf, grey), `Inf shows as ${inf}`);
    assert.ok(differs(inf, nan), `Inf and NaN share a colour: ${inf}`);
    assert.ok(!differs(R.m7.fitNan, nan), `NaN at 50 %: ${R.m7.fitNan}, at 100 % ${nan}`);
    assert.ok(!differs(R.m7.fitInf, inf), `Inf at 50 %: ${R.m7.fitInf}, at 100 % ${inf}`);
});

test('M7: the HUD counts NaN and Inf for the frame', { skip: skip || ok(R.m7) }, () => {
    assert.equal(R.m7.nanCount, 64);
    assert.equal(R.m7.infCount, 64);
});

// ── M8 ──────────────────────────────────────────────────────────────────────

test('M8: the chromaticity scope plots sRGB orange (1, 0.5, 0) at its true xy, from light', { skip: skip || ok(R.m8) }, () => {
    assert.ok(R.m8.n > 0, 'nothing plotted');
    near(R.m8.xy[0], 0.544, 0.01, 'x');
    near(R.m8.xy[1], 0.407, 0.01, 'y');
});

// ── M19 ─────────────────────────────────────────────────────────────────────

test('M19: the Display P3 tag does not outlive the P3 output', { skip: skip || ok(R.m19) || (R.m19?.unsupported && 'no drawingBufferColorSpace') }, () => {
    assert.ok(R.m19.ocio, 'OCIO never became active');
    assert.equal(R.m19.p3View, 'display-p3', 'the P3 view did not tag the canvas');
    assert.equal(R.m19.preview, 'srgb', 'an 8-bit sRGB preview frame was left tagged Display P3');
    assert.equal(R.m19.reported, 'srgb');
});

// ── L5 ──────────────────────────────────────────────────────────────────────

test('measured values, for the record', { skip }, (t) => {
    const pick = {
        h9_filmic_18pct: R.h9?.filmic,
        h10: R.h10 && { z1: R.h10.z1, z05: R.h10['z0.5'], z025: R.h10['z0.25'] },
        c3_status: R.c3?.status,
        c3_probe: R.c3?.probe,
        sidebar_ms_2000x1000: R.c2_spike_scope?.ms,
        m6_luma_fc: R.m6_luma?.fc,
        m7: R.m7 && { nan: R.m7.nan, inf: R.m7.inf, counts: [R.m7.nanCount, R.m7.infCount] },
        m8: R.m8?.xy,
    };
    t.diagnostic(JSON.stringify(pick));
});
