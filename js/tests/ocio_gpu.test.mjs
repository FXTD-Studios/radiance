/**
 * The OpenColorIO display transform, on a GPU.
 *
 * `ocio.test.mjs` proves the config is read and the maths is OpenColorIO's.
 * That is not the same as proving the viewer *shows* it: between the two sits a
 * shader splice, a LUT upload and a texture-unit assignment, none of which Node
 * can exercise. This drives `glharness.html` in headless Chromium, compiles the
 * real spliced shader, uploads the real LUT textures, renders known pixels and
 * compares the GPU's answer to OpenColorIO's own CPU answer for the same
 * transform.
 *
 * It earned its place immediately. On the first run it found two defects that
 * every other form of checking had passed over, because both produced a
 * *plausible* result rather than an error:
 *
 *   1. Scene-linear black came back as 1.3e16, 7.7e14 or NaN depending on the
 *      display. GLSL leaves `pow(x, y)` undefined for x == 0, y <= 0, and
 *      OCIO's inverse-EOTF chains reach exactly that. Black is the most common
 *      pixel in a frame.
 *   2. Every HDR view rendered solid black. 32-bit float textures are not
 *      filterable in WebGL2 unless `OES_texture_float_linear` is *enabled*, and
 *      listing it in `getSupportedExtensions()` does not enable it. LINEAR
 *      sampling of an unfilterable R32F LUT silently returns zero — no error,
 *      no warning, clean compile.
 *
 * Both are pinned below.
 *
 * Playwright is not a dependency of this package, so these skip rather than
 * fail when it is absent. A skipped GPU test is visible in the output; a
 * missing one is not.
 *
 * Run: node --test js/tests/ocio_gpu.test.mjs
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

const CHROMIUM_CANDIDATES = [
    process.env.RADIANCE_TEST_CHROMIUM,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
].filter(Boolean);

async function findChromium() {
    for (const p of CHROMIUM_CANDIDATES) {
        try { await access(p); return p; } catch { /* keep looking */ }
    }
    // Let Playwright resolve its own default install if it has one.
    return null;
}

async function loadPlaywright() {
    // Playwright is CommonJS and is not a dependency of this package; resolve it
    // from wherever the environment has it, and give up quietly if it does not.
    const require = createRequire(import.meta.url);
    for (const spec of ['playwright', '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) {
        try { return require(spec); } catch { /* keep looking */ }
    }
    return null;
}

const MIME = {
    '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.wasm': 'application/wasm', '.html': 'text/html', '.json': 'application/json',
};

async function serveRepo() {
    const server = createServer(async (req, res) => {
        try {
            const p = join(ROOT, normalize(decodeURIComponent(req.url.split('?')[0])));
            if (!p.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
            const data = await readFile(p);
            res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream' });
            res.end(data);
        } catch {
            res.writeHead(404); res.end('not found');
        }
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    return server;
}

const playwright = await loadPlaywright();
const chromiumPath = await findChromium();
const skip = playwright
    ? false
    : 'Playwright is not installed — GPU verification skipped. Install it, or set RADIANCE_TEST_CHROMIUM.';

let report = null;
if (!skip) {
    const server = await serveRepo();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        // SwiftShader, so this runs on a CI box with no GPU. It is a software
        // rasteriser but it is a conformant one: the transfer-function
        // behaviour under test is arithmetic, not hardware.
        args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
    });
    try {
        const page = await browser.newPage();
        const errors = [];
        page.on('pageerror', (e) => errors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/glharness.html`);
        report = await page.evaluate(() => window.__run());
        report.pageErrors = errors;
    } finally {
        await browser.close();
        server.close();
    }
}

// Everything OCIO does is arithmetic on floats, so the tolerance is a float
// tolerance, not a "looks about right" tolerance. The one exception is the
// exact-zero nudge, which by construction lands slightly off the CPU's answer
// for black — still under a hundredth of an 8-bit code value.
const TOLERANCE = 1e-4;

test('the harness ran and produced a report', { skip }, () => {
    assert.ok(report, 'no report came back from the browser');
    assert.equal(report.ok, true, report.error || 'harness reported failure');
    assert.deepEqual(report.pageErrors, [], 'the harness page threw');
    assert.match(report.ocioVersion, /^2\./);
});

test('the stub entry point compiles when no config is loaded', { skip }, () => {
    // The composite shader always contains the OCIODisplay() call site, so the
    // no-config build has to declare a stub or nothing compiles at all — which
    // would break the viewer for every user who never touches OCIO.
    assert.equal(report.stubCompiles, true, report.stubError);
});

test('every display and view in the config compiles', { skip }, () => {
    assert.ok(report.cases.length >= 5, `only ${report.cases.length} displays tested`);
    for (const c of report.cases) {
        assert.ok(!c.error, `${c.display} / ${c.view}: ${c.error}`);
        assert.equal(c.compiled, true, `${c.display} / ${c.view} did not compile`);
    }
});

test('the GPU renders what OpenColorIO computes', { skip }, () => {
    // The claim the whole feature rests on. If this drifts, the viewer is
    // showing its own approximation under the config's labels.
    for (const c of report.cases) {
        for (const p of c.probes) {
            assert.ok(p.gpu.every((v) => Number.isFinite(v)),
                `${c.display} / ${c.view} at ${JSON.stringify(p.in)} returned ${JSON.stringify(p.gpu)}`);
            assert.ok(p.maxDelta <= TOLERANCE,
                `${c.display} / ${c.view} at ${JSON.stringify(p.in)}: `
                + `GPU ${JSON.stringify(p.gpu)} vs OCIO CPU ${JSON.stringify(p.cpu)} (Δ ${p.maxDelta})`);
        }
    }
});

test('scene-linear black does not blow up', { skip }, () => {
    // Regression, and the reason this file exists. Without the exact-zero nudge
    // black rendered as 1.3e16 on Rec.1886, 7.7e14 on P3-D65 and NaN on sRGB
    // and Display P3 — all of it compiling cleanly and reporting nothing.
    for (const c of report.cases) {
        const black = c.probes.find((p) => p.in.every((v) => v === 0));
        assert.ok(black, `${c.display} was not probed at black`);
        assert.ok(black.gpu.every(Number.isFinite),
            `${c.display} / ${c.view} renders black as ${JSON.stringify(black.gpu)}`);
        assert.ok(black.gpu.every((v) => Math.abs(v) < 0.01),
            `${c.display} / ${c.view} renders black as ${JSON.stringify(black.gpu)} — not black`);
    }
});

test('views that need a LUT texture are not silently black', { skip }, () => {
    // Regression for OES_texture_float_linear. Sampling an unfilterable R32F
    // texture with LINEAR returns zero: no error, no warning, clean compile,
    // and every HDR view in the config renders solid black.
    const withTextures = report.cases.filter((c) => c.textures > 0);
    assert.ok(withTextures.length > 0,
        'no view in this config needed a LUT texture — the regression would go untested');
    for (const c of withTextures) {
        const lit = c.probes.find((p) => p.in.some((v) => v > 0.1));
        assert.ok(lit.gpu.some((v) => Math.abs(v) > 1e-3),
            `${c.display} / ${c.view} has ${c.textures} LUT texture(s) and renders black — `
            + 'OES_texture_float_linear is probably not enabled');
    }
});

test('nothing is stripped from the generated shader beyond the declarations', { skip }, () => {
    for (const c of report.cases) {
        for (const line of c.stripped || []) {
            assert.match(line, /^(#version|precision)/,
                `${c.display}: the splice removed a line it should not have: ${line}`);
        }
    }
});

test('the GPU has enough texture units for the config\'s LUTs', { skip }, () => {
    const most = Math.max(...report.cases.map((c) => c.textures));
    assert.ok(7 + most <= report.maxTextureUnits,
        `a view needs ${7 + most} units and this context has ${report.maxTextureUnits}`);
});
