/**
 * One answer to "what am I looking at?".
 *
 * Four controls set the display transform (the bar's View and LUT, the Grade
 * tab's Output Transform and the OCIO View) and on screen they disagreed: with
 * ACES 2.0 applied through OpenColorIO, LUT and Output Transform read
 * "ACES Filmic", the curve held in reserve while OCIO loads, and the status bar
 * said "Scene Linear / ACES Filmic". The bar now has the View select only, the
 * Grade tab reads "From View" unless a custom curve is picked, and the status
 * bar names the whole chain the renderer applies.
 *
 * Whole viewers, built the way ComfyUI builds them (sessionharness.html).
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/viewer_view_control.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

async function viewerPage() {
    const page = await session.open();
    await page.evaluate(async () => {
        const node = __make('advanced');
        window.v = node.radianceViewer;
        node.onExecuted(__frames(1, 64, 48));
        await __until(() => v.hdrData, 30000);
        await __until(() => v.ocioActive, 20000);
        await __sleep(100);
    });
    return page;
}

const state = () => ({
    status: v.colorspaceInfo.textContent,
    view: v._viewSelect.selectedOptions[0].textContent,
    output: v._gradeOutputSelect.selectedOptions[0].textContent,
    ocio: !!v.ocioActive,
});

test('the viewer bar has one display control, the View select', { skip }, async () => {
    const page = await viewerPage();
    try {
        const r = await page.evaluate(() => ({
            selects: [...v.viewerBar.querySelectorAll('select')].map(s => s.title.split('.')[0]),
            lutLabel: [...v.viewerBar.querySelectorAll('span')].some(s => s.textContent === 'LUT'),
        }));
        assert.equal(r.lutLabel, false, 'the LUT select is gone from the bar');
        assert.equal(r.selects.filter(t => /View transform/.test(t)).length, 1);
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('with ACES 2.0 through OCIO, every control and the status bar say ACES 2.0', { skip }, async () => {
    const page = await viewerPage();
    try {
        const s = await page.evaluate(`(${state})()`);
        assert.equal(s.ocio, true);
        assert.equal(s.view, 'Auto: ACES 2.0 SDR');
        assert.match(s.output, /^From View \(ACES 2\.0 - SDR 100 nits/);
        assert.equal(s.status, 'Linear Rec.709 (sRGB) → sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709)');
        assert.doesNotMatch(s.status + s.output, /Filmic/, 'the reserve curve is not named as the view');
    } finally { await page.close(); }
});

test('a custom curve shows in the View select and the status bar, and "From View" returns to the last view', { skip }, async () => {
    const page = await viewerPage();
    try {
        const r = await page.evaluate(async (fn) => {
            const st = eval(fn);
            v.setViewMode('srgb'); await __sleep(50);
            const srgb = st();
            v._gradeOutputSelect.value = 'Reinhard Tonemap';
            v._gradeOutputSelect.dispatchEvent(new Event('change')); await __sleep(50);
            const custom = st();
            v._gradeOutputSelect.value = '__from_view__';
            v._gradeOutputSelect.dispatchEvent(new Event('change')); await __sleep(50);
            return { srgb, custom, back: st(), mode: v.viewMode };
        }, `(${state})`);
        assert.equal(r.srgb.status, 'Linear Rec.709 (sRGB) → sRGB (Display)');
        assert.equal(r.srgb.ocio, false);
        assert.equal(r.custom.view, 'Custom: Reinhard Tonemap');
        assert.equal(r.custom.output, 'Reinhard Tonemap');
        assert.equal(r.custom.status, 'Linear Rec.709 (sRGB) → Reinhard Tonemap');
        assert.equal(r.mode, 'srgb');
        assert.equal(r.back.status, r.srgb.status);
    } finally { await page.close(); }
});

test('the input space is part of the chain', { skip }, async () => {
    const page = await viewerPage();
    try {
        const status = await page.evaluate(async () => {
            v.inputSpace = 'IDT: LogC3 → Linear';
            v._gradeChanged();
            return v.colorspaceInfo.textContent;
        });
        assert.match(status, /^LogC3 → /);
    } finally { await page.close(); }
});

test('picking a view takes the display back from a loaded config, which the picture used to keep', { skip }, async () => {
    const page = await viewerPage();
    try {
        const r = await page.evaluate(async () => {
            await v._ocioLoad({ builtin: 'ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5' }, 'Studio');
            await __until(() => v._ocioUserConfig && v.ocioActive, 20000);
            const loaded = { view: v._viewSelect.selectedOptions[0].textContent, value: v._viewSelect.value };
            v.setViewMode('srgb'); await __sleep(50);
            return { loaded, after: { ocio: !!v.ocioActive, renderer: !!v.renderer?.ocioEnabled,
                user: v._ocioUserConfig, view: v._viewSelect.value, options: [...v._viewSelect.options].map(o => o.value) } };
        });
        assert.equal(r.loaded.value, 'ocio-config');
        assert.match(r.loaded.view, /^OCIO: /);
        assert.equal(r.after.user, false);
        assert.equal(r.after.ocio, false, 'the config no longer owns the display');
        assert.equal(r.after.renderer, false, 'nor does its transform stay on the renderer');
        assert.equal(r.after.view, 'srgb');
        assert.ok(!r.after.options.includes('ocio-config'));
    } finally { await page.close(); }
});

// ── The picture first: nothing floats over it ──────────────────────────────

test('the viewer bar and the sequence dock sit under the picture, not on it', { skip }, async () => {
    const page = await session.open();
    try {
        const r = await page.evaluate(async () => {
            const node = __make('advanced');
            window.v = node.radianceViewer;
            node.onExecuted(__frames(6, 64, 48));
            await __until(() => v.hdrData, 30000);
            await __sleep(200);
            const c = v.canvas.getBoundingClientRect();
            const bar = v.viewerBar.getBoundingClientRect(), dock = v.sequenceDock.getBoundingClientRect();
            return { barTop: bar.top, dockTop: dock.top, canvasBottom: c.bottom,
                inWrapper: v.canvasWrapper.contains(v.viewerBar) || v.canvasWrapper.contains(v.sequenceDock),
                frame: !!v.container.querySelector('.radiance-viewer-frame'),
                cross: getComputedStyle(v.viewerCross).display };
        });
        assert.equal(r.inWrapper, false);
        assert.ok(r.barTop >= r.canvasBottom - 0.5, `the bar overlaps the picture (${r.barTop} < ${r.canvasBottom})`);
        assert.ok(r.dockTop > r.barTop);
        assert.equal(r.frame, false, 'the decorative green frame is gone');
        assert.equal(r.cross, 'none', 'the centre cross starts off');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('a still shows the transport only; the Ref Wipe tools and V2 lane need a sequence', { skip }, async () => {
    const page = await session.open();
    try {
        const r = await page.evaluate(async () => {
            const shown = (v, sel) => { const el = v.sequenceDock.querySelector(sel); return !!el && getComputedStyle(el).display !== 'none'; };
            const one = __make('advanced'); one.onExecuted(__frames(1, 64, 48));
            const many = __make('advanced'); many.onExecuted(__frames(6, 64, 48));
            const a = one.radianceViewer, b = many.radianceViewer;
            await __until(() => a.hdrData && b.hdrData, 30000);
            await __sleep(200);
            const out = {
                still: { tools: shown(a, '.radiance-pro-timeline-tools'), v2: shown(a, '.radiance-pro-lane-v2'), play: shown(a, '.radiance-pro-sequence-controls') },
                seq: { tools: shown(b, '.radiance-pro-timeline-tools'), v2: shown(b, '.radiance-pro-lane-v2'), guide: shown(b, '.radiance-pro-v2-guide') },
            };
            b._setTimelineTool('reference');
            out.seq.guideWithTool = shown(b, '.radiance-pro-v2-guide');
            return out;
        });
        assert.deepEqual(r.still, { tools: false, v2: false, play: true });
        assert.deepEqual(r.seq, { tools: true, v2: true, guide: false, guideWithTool: true });
    } finally { await page.close(); }
});
