/**
 * The viewer panels, built and operated in a real browser.
 *
 * Everything else that covers `radiance_viewer.js` reads it as text --
 * probe_wiring, scope_wiring and ocio_wiring all grep the source, because the
 * module imports ComfyUI's `app.js` and `api.js` and cannot be imported in
 * Node. Those tests are worth keeping: they pin the connections that make the
 * instruments unambiguous. But a source assertion cannot tell you that a panel
 * builds, and it did not stop the module going unparseable for seven commits
 * while every check reported success.
 *
 * This loads the real module in Chromium against stubbed ComfyUI modules,
 * builds each panel, and then *operates* it -- changes the selects, clicks the
 * toggles -- and checks the instance and the DOM afterwards.
 *
 * The first thing it found: picking the Legacy 480-line safe-area preset left
 * the caption underneath still reading "SMPTE ST 2046-1 and EBU R 95 specify
 * the same two boxes". The boxes redrew correctly at 90/80; only their stated
 * provenance was wrong, which in a QC guide is the worse half to get wrong. No
 * amount of reading the source would have shown it, because every line involved
 * was correct on its own.
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/panel_dom.test.mjs
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

/**
 * ComfyUI's two modules, stubbed.
 *
 * The viewer imports these at the top of the file, so without them nothing
 * loads at all. They are deliberately inert: the point is to exercise
 * Radiance's panel code, not to simulate a graph editor. `registerExtension`
 * records rather than acts, which is also how the test knows registration ran.
 */
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
    for (const spec of ['playwright',
        '/home/claude/.npm-global/lib/node_modules/playwright/index.js']) {
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
    : 'Playwright is not installed — the panels cannot be driven. '
      + 'Install it, or set RADIANCE_TEST_CHROMIUM.';

let report = null;
let loaded = null;
let pageErrors = [];
if (!skip) {
    const server = await serveRepo();
    const chromiumPath = await findChromium();
    const browser = await playwright.chromium.launch({
        ...(chromiumPath ? { executablePath: chromiumPath } : {}),
        args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    });
    try {
        const page = await browser.newPage();
        page.on('pageerror', (e) => pageErrors.push(String(e.message)));
        await page.goto(`http://127.0.0.1:${server.address().port}/js/tests/panelharness.html`,
            { waitUntil: 'load' });
        // A module that fails to parse never runs, so __run never appears and
        // this times out. Swallowing the timeout here turns that into a named
        // assertion below instead of an unattributed file-level crash — which
        // matters, because "the module did not load" is the failure this file
        // was written for and it should say so.
        try {
            await page.waitForFunction(() => typeof window.__run === 'function',
                null, { timeout: 20000 });
        } catch {
            pageErrors.push('window.__run never appeared — the harness module did not run');
        }
        loaded = await page.evaluate(() => ({
            hasClass: typeof window.RadianceViewer,
            extensions: (globalThis.__exts || []).map((e) => e.name),
        }));
        report = await page.evaluate(() => (window.__run ? window.__run() : null));
    } finally {
        await browser.close();
        server.close();
    }
}

test('the viewer module loads in a browser and registers its extension', { skip }, () => {
    // The regression this whole file exists for. When a backtick in a comment
    // inside a shader template literal broke the module, the node rendered with
    // three inputs, one output and no UI — and `node --check` passed on every
    // commit for a week. A module that does not parse cannot register, so this
    // is the assertion that would have gone red on the first one.
    assert.deepEqual(pageErrors, [], 'the module threw while loading:\n' + pageErrors.join('\n'));
    assert.equal(loaded.hasClass, 'function', 'window.RadianceViewer is missing');
    assert.ok(loaded.extensions.includes('FXTD.RadianceViewer'),
        `the viewer extension did not register — got ${JSON.stringify(loaded.extensions)}`);
});

// Every assertion below reads `report`. If the harness never ran, the first
// test says so and this keeps the rest from failing with a TypeError that
// names a property instead of the problem.
const noReport = report?.ok ? false
    : 'the harness did not finish — see the module-load test above';

test('every panel the viewer defines is built, not a chosen few', { skip: skip || noReport }, () => {
    // The panel list is discovered from the prototype, not written here. This
    // harness drove five of the viewer's fourteen panels while reading as
    // though it covered the viewer — the same failure as a source assertion:
    // it passes, and it is not looking at the thing. The floor makes losing a
    // panel a failure too, so the list cannot quietly shrink back.
    assert.ok(report.panels.length >= 14,
        `only ${report.panels.length} panels discovered: ${report.panels.join(', ')}`);
    assert.deepEqual(Object.keys(report.built).sort(), report.panels,
        'a discovered panel was not built');
    for (const [name, r] of Object.entries(report.built)) {
        assert.ok(r.ok, `${name} threw: ${r.error}`);
        assert.ok(r.nodes > 3, `${name} produced ${r.nodes} elements — that is an empty panel`);
    }
});

test('the framing selects offer exactly the presets that are defined', { skip: skip || noReport }, () => {
    // Against the definitions, not against a list retyped here. Adding a matte
    // ratio and forgetting to surface it is the failure this catches.
    const { safeOptions, matteOptions } = report.driven.framing;
    assert.deepEqual(safeOptions.map((o) => o.value), report.presets.safeArea.map((p) => p.id));
    assert.deepEqual(safeOptions.map((o) => o.text), report.presets.safeArea.map((p) => p.label));
    assert.deepEqual(matteOptions.map((o) => o.value), report.presets.matte.map((p) => p.id));
    assert.deepEqual(matteOptions.map((o) => o.text), report.presets.matte.map((p) => p.label));
});

test('changing a framing select updates the state, the store and the caption', { skip: skip || noReport }, () => {
    const d = report.driven.framing;
    assert.equal(d.safeState, 'legacy', 'the instance did not take the new preset');
    assert.equal(d.safeStored, 'legacy', 'the choice was not persisted');
    assert.equal(d.matteState, '2.39');
    assert.equal(d.matteStored, '2.39');

    // The caption is the provenance of the guide — "not a current delivery
    // spec" against "SMPTE ST 2046-1 and EBU R 95 specify the same two boxes".
    // It used to be written once at build time, so switching preset left the
    // wrong standard named under the right boxes.
    assert.equal(d.noteWasFirst, true, 'the caption did not start on the first preset');
    assert.equal(d.noteNowMatchesSelection, true,
        'the caption still describes the previous preset after the selection changed');
});

test('the framing toggles move their label with their state', { skip: skip || noReport }, () => {
    // A toggle whose label lags its state reports the mode it is about to
    // leave, which on a magnification filter means the user cannot tell whether
    // they are looking at real pixels.
    const d = report.driven.framing;
    assert.equal(d.magnifyBefore.state, 'linear');
    assert.match(d.magnifyBefore.label, /LINEAR/);
    assert.equal(d.magnifyAfter.state, 'nearest');
    assert.match(d.magnifyAfter.label, /NEAREST/);

    assert.equal(d.positionBefore.state, 'frames');
    assert.match(d.positionBefore.label, /FRAMES/);
    assert.equal(d.positionAfter.state, 'seconds');
    assert.match(d.positionAfter.label, /SECONDS/);
});

test('the scope scale selector offers the shared scales and takes a change', { skip: skip || noReport }, () => {
    // Against SCOPE_SCALES from radiance_scope_units.js. A scope that quietly
    // drops a scale is back to being an unlabelled instrument.
    const d = report.driven.scopes;
    assert.ok(d.found, 'no scale selector in the scopes tab');
    assert.deepEqual(d.options, report.presets.scopeScales);
    assert.equal(d.state, 'nits-pq');
    assert.equal(d.stored, 'nits-pq');
});

test('the mask type selector writes an int, not the string the DOM hands back',
    { skip: skip || noReport }, () => {
        // `<option value="1">` hands back "1". The mask type is compared
        // numerically and passed to the renderer, so a string is a mask that
        // silently stops matching. A source assertion cannot catch this: the
        // parseInt is right there in the line it would read.
        const d = report.driven.masks;
        assert.ok(d.found, 'no mask type selector in the masks tab');
        assert.equal(d.before, 0);
        assert.equal(d.after, 1);
        assert.equal(d.type, 'number', `mask type came back as ${d.type}`);
    });

test('drawing the colour panel does not fetch the OCIO WebAssembly', { skip: skip || noReport }, () => {
    // 4.7 MB on the chance someone opens a tab is the reason the façade
    // dynamic-imports it. This is the assertion that keeps it lazy.
    assert.equal(report.driven.ocioWasmRequested, false,
        'ocio-wasm.wasm was fetched just from building the panel');
});
