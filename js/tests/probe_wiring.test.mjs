/**
 * The probe panel's wiring.
 *
 * `radiance_viewer.js` cannot be imported in Node — it pulls in ComfyUI's
 * `app.js` — so this reads the source and pins the connections that a probe
 * silently loses when someone refactors around it. Every assertion here
 * corresponds to a way the panel can go from *wrong* to *plausibly wrong*,
 * which is the dangerous failure mode for an instrument: it keeps printing
 * numbers.
 *
 * These are source assertions, not behaviour tests. They are cheap and they
 * hold the seams; a real behavioural harness for the viewer needs a browser and
 * is tracked separately.
 *
 * Run: node --test js/tests/probe_wiring.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');

/**
 * Code only. A comment that *names* the thing a test forbids would otherwise
 * fail the test — and the comment explaining why the call is forbidden is
 * exactly the comment that gets written.
 */
function code(src) {
    return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
}

/** The body of a method, from its declaration to the next same-indent method. */
function methodBody(src, name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    assert.notEqual(end, -1, `could not find the end of ${name}`);
    return rest.slice(0, end);
}

test('the probe panel is reachable — tab registered and dispatched', () => {
    // A tab in the list but missing from the dispatch renders an empty pane and
    // reads as a broken feature rather than a missing branch.
    assert.match(viewer, /\{ id: 'probe', label: '[^']*PROBE' \}/,
        'no probe entry in the HUD tab list');
    assert.match(viewer, /activeTab === 'probe'\s*\)\s*\{\s*\n\s*active\.renderProbeTab\(/,
        'the probe tab is listed but never dispatched to renderProbeTab');
});

test('the probe measures through the shared module, not its own arithmetic', () => {
    // The reason radiance_probe.js exists. Four copies of the grade maths
    // already diverged across the two backends; the statistics must not
    // repeat that.
    assert.match(viewer, /from "\.\/radiance_probe\.js"/,
        'the viewer must import the probe module');
    for (const fn of ['_probeSampleStats', '_probeRgbToHsv', '_probeNits', '_probeEV']) {
        assert.ok(viewer.includes(fn), `${fn} imported but never used — the panel is computing its own`);
    }
});

test('the cursor readout never triggers a GPU readback', () => {
    // readPixelsFloat32 pulls the entire frame off the GPU. Calling it from the
    // per-pointer-move path stalls the render loop at 4K, and it is an easy
    // "simplification" for a later reader to make. The rendered pixels are a
    // capture; only _probeCaptureRendered may read them.
    assert.doesNotMatch(code(methodBody(viewer, '_probeSampleOne')), /readPixelsFloat32/,
        '_probeSampleOne must read the capture, not the GPU');
    assert.doesNotMatch(code(methodBody(viewer, '_probeBuffer')), /readPixelsFloat32/,
        '_probeBuffer must read the capture, not the GPU');
    assert.doesNotMatch(code(methodBody(viewer, '_probeDescribe')), /readPixelsFloat32/,
        '_probeDescribe must not read back either — it runs on every pointer move');
    assert.match(methodBody(viewer, '_probeCaptureRendered'), /readPixelsFloat32/,
        '_probeCaptureRendered is the only place allowed to read back');
});

test('a capture states whether it is graded, and when it was taken', () => {
    // WebGPU returns ungraded source. Labelling that "Rendered" without
    // qualification attributes every number to a grade that was never applied —
    // the same contract failure that made the EXR export write source pixels
    // into a file called "graded".
    const buf = methodBody(viewer, '_probeDescribe');
    assert.match(buf, /UNGRADED/, 'the caption must say when the pixels are not graded');
    assert.match(buf, /captured \$\{cap\.at\}/, 'the caption must say when the capture was taken');
    assert.match(methodBody(viewer, '_probeCaptureRendered'), /graded: r\.graded !== false/,
        'the capture must record the backend\'s graded flag rather than assuming');
});

test('every path that replaces the frame invalidates the probe', () => {
    // The failure this prevents is the worst one available: the panel keeps
    // showing last frame's statistics, correctly formatted, next to the new
    // image. Each `this.imageData = <new data>` must be followed by an
    // invalidation.
    const lines = viewer.split('\n');
    const assigns = [];
    lines.forEach((line, i) => {
        if (/this\.imageData = /.test(line) && !/this\.imageData = null/.test(line)) {
            assigns.push({ n: i + 1, next: lines[i + 1] || '' });
        }
    });
    assert.ok(assigns.length >= 4, `expected several imageData assignments, found ${assigns.length}`);
    for (const a of assigns) {
        assert.match(a.next, /_probeInvalidate\(\)/,
            `line ${a.n} replaces imageData without invalidating the probe`);
    }
});

test('the invalidation actually drops every cache', () => {
    const body = methodBody(viewer, '_probeInvalidate');
    for (const field of ['_probeLinearCache', '_probeRendered', '_probeStats', '_probeCurrent']) {
        assert.ok(body.includes(field), `_probeInvalidate leaves ${field} stale`);
    }
});

test('the region selection is drawn, and only while it is live', () => {
    // A rectangle left on screen after the mode changed is a lie about what is
    // being measured.
    assert.match(viewer, /if \(this\.probeRect && this\._probeRegionActive\(\)\) this\._probeDrawRegion\(ctx\)/,
        'renderOverlay must draw the region, gated on the mode');
    assert.match(methodBody(viewer, '_probeRegionActive'),
        /activeTab === 'probe' && this\.probeMode === 'region'/,
        'the region is only live on the probe tab in region mode');
});

test('a region drag does not steal the wipe handle or a mask handle', () => {
    // Direct manipulation of something already on screen must keep priority
    // over starting a new selection.
    const wipe = viewer.indexOf('this.isDraggingWipe = true;');
    const mask = viewer.indexOf('this.maskDragMode = handle;');
    const probe = viewer.indexOf('this._probeDragging = true;');
    assert.ok(wipe > 0 && mask > 0 && probe > 0, 'expected all three drag starts to exist');
    assert.ok(probe > wipe, 'the probe drag must be checked after the wipe handle');
    assert.ok(probe > mask, 'the probe drag must be checked after the mask handles');
});

test('shift-drag still pans while a region can be drawn', () => {
    // Losing pan on the probe tab would be a bad trade for a selection.
    assert.match(viewer, /_probeRegionActive\(\) && e\.button === 0 && !e\.shiftKey/,
        'the region drag must yield to shift, which is the pan modifier');
});

test('a region drag ends by measuring, and clears its own drag state', () => {
    const up = viewer.slice(viewer.indexOf('this._winMouseUpHandler = '));
    assert.match(up, /this\._probeDragging = false/, 'the drag flag must be cleared on mouseup');
    assert.match(up, /this\._probeComputeStats\(\)/, 'finishing a drag should measure it');
});

test('the panel repaint is coalesced to one frame', () => {
    // A pointer emits events faster than the DOM can be rewritten. Without the
    // guard the panel becomes the reason the viewer feels slow.
    assert.match(viewer, /_probeRepaintQueued/, 'no repaint coalescing on the probe readout');
    assert.match(viewer, /requestAnimationFrame\(\(\) => \{\s*\n\s*this\._probeRepaintQueued = false;/,
        'the coalescing flag must be cleared inside the frame callback, or it latches');
});

test('the readout writes into a live node or does nothing', () => {
    // The panel is rebuilt whenever a tab switches, so held DOM references go
    // stale. Writing into a detached node is invisible and looks like a frozen
    // probe.
    for (const m of ['_probeRenderCurrent', '_probeRenderStats']) {
        assert.match(methodBody(viewer, m), /isConnected/,
            `${m} must check its node is still in the document`);
    }
});

test('non-finite and negative counts are surfaced, not just computed', () => {
    const body = methodBody(viewer, '_probeRenderStats');
    for (const word of ['NaN', 'Inf', 'negative']) {
        assert.ok(body.includes(word), `the stats block never shows the ${word} count`);
    }
    assert.match(body, /Excluded from every statistic/,
        'the panel must say that flagged samples are excluded from the statistics');
});

test('an estimated median is labelled as estimated', () => {
    assert.match(methodBody(viewer, '_probeRenderStats'), /medianExact === false/,
        'the panel must distinguish an exact median from a histogram estimate');
});

test('8-bit sources are declared as quantised', () => {
    // Five decimal places on a value that only has 256 possible states implies
    // a precision the source does not carry.
    assert.match(methodBody(viewer, '_probeDescribe'), /quantised: true/);
    assert.match(methodBody(viewer, '_probeRenderCurrent'), /8-bit quantised/);
});

test('describing the source does not materialise it', () => {
    // The caption repaints on every pointer move. If it went through the
    // buffer, the first move over an 8-bit 4K frame would decode it into a
    // 132 MB float array to produce a string.
    const desc = code(methodBody(viewer, '_probeDescribe'));
    assert.doesNotMatch(desc, /_probeLinearFromImageData/,
        '_probeDescribe must not decode the frame');
    assert.doesNotMatch(desc, /\bdata:/,
        '_probeDescribe must not carry pixel data');
    assert.match(code(methodBody(viewer, '_probeRenderCurrent')), /this\._probeDescribe\(\)/,
        'the per-move repaint must use the description, not the buffer');
});

test('the panel never shows an unlabelled number', () => {
    // Every state of the caption resolves to either an error or a label; there
    // is no branch that leaves it blank.
    const body = methodBody(viewer, '_probeRenderCurrent');
    assert.match(body, /nodes\.caption\.innerHTML = buf\.error/,
        'the caption must be written on every repaint, error or not');
});
