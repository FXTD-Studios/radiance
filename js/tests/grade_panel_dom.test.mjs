/**
 * The Grade and Effects tabs, operated in a real browser.
 *
 * Builds the live tabs from radiance_viewer.js in Chromium (ComfyUI's modules
 * stubbed) and drives them with pointer events, typed values, double-clicks
 * and the Reset buttons. Each test is one finding from the viewer review:
 *
 *   H1   holding a wheel puck still keeps the level
 *   H5   one undo step per drag, a complete grade state, 0 kept, Reset then Undo
 *   H6   presets in the live tab; the grade written into the node
 *   M1   readouts show the value applied, take typed values, one exposure range
 *   M2   Reset and Reset All differ; section resets; double-click; Fringe
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/grade_panel_dom.test.mjs
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

const STUBS = {
    '/scripts/app.js': `export const app = {
        registerExtension(e) { (globalThis.__exts ||= []).push(e); },
        graph: { _nodes: [], setDirtyCanvas() {} },
        canvas: { setDirty() {} },
        extensionManager: { registerSidebarTab() {} },
        ui: { settings: { addSetting() {}, getSettingValue() {} } },
    };`,
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
        const url = req.url.split('?')[0];
        if (STUBS[url]) {
            res.writeHead(200, { 'Content-Type': 'text/javascript' });
            res.end(STUBS[url]);
            return;
        }
        try {
            const p = join(ROOT, normalize(decodeURIComponent(url)));
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
    : 'Playwright is not installed — the panels cannot be driven. Install it, or set RADIANCE_TEST_CHROMIUM.';

let report = null;
const pageErrors = [];
if (!skip) {
    const server = await serveRepo();
    const chromiumPath = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    });
    try {
        const page = await browser.newPage({ viewport: { width: 1200, height: 1600 } });
        page.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/gradepanelharness.html`);
        await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
        report = await page.evaluate(() => window.__run());
    } finally {
        await browser.close();
        server.close();
    }
}

test('the Grade and Effects tabs build and run without errors', { skip }, () => {
    assert.ok(report, 'no report');
    assert.deepEqual(report.errors, []);
    assert.deepEqual(pageErrors, []);
});

test('H1: holding a wheel puck still keeps the master level', { skip }, () => {
    // Holding the Gain puck on red took the mean from 1.00 to 0.82, 0.64 and
    // then 0.27.
    const w = report.wheels;
    for (const [prop, level] of [['lift', 0], ['gamma', 1], ['gain', 1]]) {
        for (const m of w[prop].means) assert.ok(Math.abs(m - level) < 1e-3, `${prop} mean drifted: ${w[prop].means}`);
        assert.ok(w[prop].redWins, `${prop}: the puck on the right should push red: ${w[prop].final}`);
        assert.equal(w[prop].undoSteps, 1, `${prop}: one undo step per drag`);
    }
});

test('M1: readouts show the value the grade holds, at the step precision', { skip }, () => {
    const r = report.readouts;
    // Saturation 0 read 1.00 and pivot 0 read 0.50 (|| defaults).
    assert.deepEqual(r.saturation, ['0', '0.00']);
    assert.deepEqual(r.pivot, ['0', '0.00']);
    // 0.25 showed as 0.3; 0.05 steps rounded to 0.1.
    assert.deepEqual(r.exposure, ['0.25', '0.25']);
    assert.equal(r.temperature, '0.05');
    assert.equal(r.tint, '-0.15');
    assert.equal(r.grainSize, '1.15');
    // One exposure range: the slider covers what adjustEV and the shader allow.
    assert.deepEqual(r.exposureRange, ['-12', '12']);
    assert.deepEqual(r.exposure12, ['12', '12.00']);
});

test('M1: adjustEV moves the Exposure slider, and clamps to its range', { skip }, () => {
    const r = report.readouts;
    assert.deepEqual(r.adjustEV, { state: 0.25, slider: '0.25', readout: '0.25' });
    assert.deepEqual(r.adjustEVClamp, { state: 12, slider: '12' });
});

test('M1: a readout takes a typed value', { skip }, () => {
    const r = report.readouts;
    assert.equal(r.typed.state, 1.37);
    assert.equal(r.typed.slider, '1.37');
    assert.equal(r.typed.readout, '1.37');
    assert.ok(r.typed.undo >= 1, 'typing a value is an undo step');
    assert.equal(r.typedClamped, 3, 'a typed value is held to the slider range');
});

test('H5: a slider drag is one undo step, and undo / redo move it', { skip }, () => {
    const u = report.undo;
    assert.deepEqual(u.afterDrag, { highlights: 0.3, steps: 1 });
    assert.equal(u.afterUndo, 0);
    assert.equal(u.afterRedo, 0.3);
});

test('H5: the grade state is complete', { skip }, () => {
    for (const k of ['offset', 'highlights', 'shadows', 'midDetail', 'colorBoost', 'lumaMix', 'curves',
        'logShadow', 'logMidtone', 'logHighlight', 'printerR', 'printerG', 'printerB', 'inputSpace', 'manualLut',
        'saturation', 'pivot', 'grain', 'lensFringe', 'maskState', 'qualifierState']) {
        assert.ok(report.undo.keys.includes(k), `${k} is missing from the grade state`);
    }
});

test('H5/M2: Reset All then Undo restores everything, saturation 0 included', { skip }, () => {
    const u = report.undo;
    assert.equal(u.resetSat, 1);
    assert.equal(u.resetCurves, null);
    assert.equal(u.resetInput, 'None');
    assert.equal(u.resetLut, null);
    assert.ok(u.restoredEqual, 'undo after Reset All did not bring the whole grade back');
    assert.equal(u.restoredSat, 0, 'saturation 0 came back as something else');
    assert.equal(u.restoredCurve, 3, 'the curve point did not come back');
    assert.deepEqual(u.restoredLut, ['manual', 'Rec.709 (Broadcast)']);
});

test('M2: Reset is the Grade tab, Reset All is every grade field', { skip }, () => {
    const r = report.reset;
    assert.deepEqual(r.buttons, ['Reset', 'Reset All']);
    assert.equal(r.afterReset.exposure, 0);
    assert.deepEqual(r.afterReset.gain, [1, 1, 1]);
    assert.equal(r.afterReset.temperature, 0);
    // Reset leaves what is not on the tab.
    assert.equal(r.afterReset.printerR, 9);
    assert.equal(r.afterReset.grain, 0.4);
    assert.equal(r.afterReset.inputSpace, 'IDT: S-Log3 → Linear');
    // Reset All takes everything.
    assert.deepEqual(r.afterResetAll, { exposure: 0, gain: [1, 1, 1], printerR: 0, grain: 0, bloom: 0, lensFringe: 0, inputSpace: 'None' });
});

test('M2: each section resets on its own, and double-click resets one slider', { skip }, () => {
    const r = report.reset;
    for (const s of ['color_transform', 'exposure', 'tone', 'color', 'color_wheels']) {
        assert.ok(r.sectionResets.includes(s), `no reset on ${s}: ${r.sectionResets}`);
    }
    assert.deepEqual(r.afterSection, { exposure: 0, contrast: 1, highlights: 0.3, temperature: 0.4 });
    assert.deepEqual(r.dbl, { temperature: 0, exposure: 1 });
});

test('M2: Reset Effects includes Fringe', { skip }, () => {
    const r = report.reset;
    assert.deepEqual(r.fxAfter, { lensFringe: 0, grain: 0, bloom: 0, exposure: 1 });
    assert.equal(r.fringeDbl, 0, 'double-click on Fringe');
});

test('H6: grade presets save and load from the live Grade tab', { skip }, () => {
    const p = report.presets;
    assert.deepEqual(p.actionsBefore, ['load', 'save', 'delete']);
    assert.deepEqual(p.options, ['bw']);
    assert.deepEqual(p.loaded, { saturation: 0, contrast: 1.3, offset: [0.02, 0, 0] });
    assert.equal(p.persisted, 0, 'the grade written to the node lost saturation 0');
    assert.ok(p.persistedKeys > 40);
});
