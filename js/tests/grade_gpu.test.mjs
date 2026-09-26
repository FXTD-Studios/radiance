/**
 * The grade maths, on a GPU.
 *
 * `radiance_grade.js` emits GLSL and WGSL from the same file as its JS
 * functions. That removes the *opportunity* for the backends to disagree — it
 * does not prove the transliteration is right. This compiles the emitted GLSL
 * in a real WebGL2 context and compares it to the JS across the grade settings
 * a user can actually reach, including the ones the old unguarded paths turned
 * into Infinity or NaN.
 *
 * The WGSL half lives in `grade_wgsl_gpu.test.mjs`. It is a separate file
 * because it needs a separate runtime: headless Chromium has no
 * `navigator.gpu` at all — absent, not disabled — so the WebGPU shader is
 * compiled under Deno against Mesa's lavapipe software Vulkan driver instead.
 *
 * Skips when Playwright is unavailable — a skipped GPU test is visible in the
 * output, a missing one is not.
 *
 * Run: node --test js/tests/grade_gpu.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = join(JS, '..');

async function findChromium() {
    for (const p of [process.env.RADIANCE_TEST_CHROMIUM,
        '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].filter(Boolean)) {
        try { await access(p); return p; } catch { /* keep looking */ }
    }
    return null;
}

async function loadPlaywright() {
    const require = createRequire(import.meta.url);
    for (const spec of ['playwright', '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) {
        try { return require(spec); } catch { /* keep looking */ }
    }
    return null;
}

const MIME = {
    '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.wasm': 'application/wasm', '.html': 'text/html',
};

async function serveRepo() {
    const server = createServer(async (req, res) => {
        try {
            const p = join(ROOT, normalize(decodeURIComponent(req.url.split('?')[0])));
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
    : 'Playwright is not installed — GPU verification skipped. Install it, or set RADIANCE_TEST_CHROMIUM.';

let report = null;
if (!skip) {
    const server = await serveRepo();
    const chromiumPath = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
    });
    try {
        const page = await browser.newPage();
        const errors = [];
        page.on('pageerror', (e) => errors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/gradeharness.html`);
        report = await page.evaluate(() => window.__run());
        report.pageErrors = errors;
    } finally {
        await browser.close();
        server.close();
    }
}

// fp32 in the shader against fp64 in JS. Anything looser than this would let a
// real formula difference through; anything tighter is float noise.
const TOLERANCE = 2e-6;

test('the emitted GLSL compiles', { skip }, () => {
    assert.ok(report, 'no report came back from the browser');
    assert.ok(report.ok, report.error);
    assert.deepEqual(report.pageErrors, [], 'the harness page threw');
    assert.equal(report.compiled, true);
});

test('the emitted GLSL computes what the JS computes', { skip }, () => {
    // The claim the single-definition refactor rests on. If a transliteration
    // is wrong, the backends are back to disagreeing — just with better
    // comments.
    assert.ok(report.cases.length >= 50, `only ${report.cases.length} comparisons`);
    const comparable = report.cases.filter((c) => c.representable);
    assert.ok(comparable.length >= 50, `only ${comparable.length} comparable cases`);
    for (const c of comparable) {
        assert.ok(c.gpu.every(Number.isFinite),
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)} → ${JSON.stringify(c.gpu)}`);
        assert.ok(c.delta <= TOLERANCE,
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)}: `
            + `GPU ${JSON.stringify(c.gpu)} vs JS ${JSON.stringify(c.cpu)} (Δ ${c.delta})`);
    }
});

test('the only samples the GPU cannot match are ones fp32 cannot hold', { skip }, () => {
    // Gamma 0 floors to 0.01, so the exponent is 100 and a scene-linear 4.0
    // becomes 1.6e60 — past fp32's 3.4e38 ceiling. That is the format, not the
    // formula, and it is worth stating rather than hiding behind a loose
    // tolerance. Visually it makes no difference: anything that large clips to
    // white through any display transform.
    //
    // The assertion is that this is the *only* reason a sample is excluded. If
    // an ordinary grade ever lands here, the exclusion is covering a real bug.
    for (const c of report.cases.filter((x) => !x.representable)) {
        assert.ok(c.grade.gamma && c.grade.gamma[0] <= 0.01,
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)} exceeded fp32 without an extreme gamma`);
        assert.ok(Math.max(...c.in.map(Math.abs)) > 1,
            'an in-range pixel should never exceed fp32');
    }
});

test('no grade setting produces a non-finite pixel on the GPU', { skip }, () => {
    // gamma 0, contrast 99 and pivot 0 are all reachable on the sliders, and
    // all three produced Infinity or NaN on at least one of the old paths.
    const risky = report.cases.filter((c) => c.representable
        && (c.grade.gamma?.[0] === 0 || c.grade.contrast === 99 || c.grade.pivot === 0));
    assert.ok(risky.length > 0, 'the harness stopped covering the extreme settings');
    for (const c of risky) {
        assert.ok(c.gpu.every(Number.isFinite),
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)} → ${JSON.stringify(c.gpu)}`);
    }
});

// The WGSL half used to sit here as a standing `todo`, because headless
// Chromium has no navigator.gpu under any flag combination. It is now a real
// test in grade_wgsl_gpu.test.mjs, run under Deno on a software Vulkan
// adapter, over this same case matrix.
