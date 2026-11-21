/**
 * Inspect reads the frame; Review compares against a version you choose.
 *
 *  - Inspect opens on the Inspector (the 4.1 shell: "Channels and pixel
 *    probe"); its Scopes tab (More tools > Scopes) shows waveform, parade,
 *    vectorscope and the QC result in a 2 x 2, a 10-bit | % | Nits switch (Nits
 *    only for a PQ or HLG view), and a probe row that follows the pointer.
 *    QC Details opens the QC workspace, which has its own tab since 4.1.
 *  - Review: Grab still keeps the frame as shown and makes it B; the B menu
 *    lists the stills, the compare input, Release and Clear. Releasing B keeps
 *    the stills; Clear forgets them.
 *
 * Skips when Playwright is unavailable.
 * Run: node --test js/tests/viewer_inspect_review.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

test('Inspect scopes are a 2 x 2 of scopes and QC, with the scale switch and a probe row', { skip }, async () => {
    const page = await session.open();
    try {
        const r = await page.evaluate(async () => {
            const n = __make('advanced'); const v = n.radianceViewer;
            n.onExecuted(__frames(6, 160, 90));
            await __until(() => v.hdrData); await __sleep(300);
            v.setWorkspace('inspect', { persist: false }); await __sleep(300);
            const out = { opensOn: v._referenceRightTab };
            v._setReferenceTab('scopes'); await __sleep(300);
            out.tab = v._referenceRightTab;
            const grid = v.controlsPanel.querySelector('.radiance-ref-scopes-grid');
            out.quad = grid.classList.contains('is-quad');
            out.cells = [...grid.children].map((e) => e.dataset.scope);
            out.firstSection = v.controlsPanel.querySelector('.radiance-ref-section .radiance-ref-title')?.textContent;
            const sw = [...v.controlsPanel.querySelectorAll('.radiance-scope-scale button')];
            out.scale = sw.map((b) => [b.textContent, b.disabled, b.getAttribute('aria-pressed')]);
            sw.find((b) => b.textContent === '%').click(); await __sleep(100);
            out.picked = v.scopeScale;
            out.pressed = v.controlsPanel.querySelector('.radiance-scope-scale button[aria-pressed="true"]')?.textContent;
            out.qcBefore = v.controlsPanel.querySelector('.radiance-scope-qc-run')?.textContent;
            await v.runQC(); await __sleep(200);
            out.qcLines = v.controlsPanel.querySelectorAll('.radiance-scope-qc-line').length;
            out.qcAfter = v.controlsPanel.querySelector('.radiance-scope-qc-run')?.textContent;
            v.controlsPanel.querySelector('.radiance-scope-qc-more').click(); await __sleep(100);
            out.details = [v.workspace, v._referenceRightTab];
            v.setWorkspace('inspect', { persist: false, tab: 'scopes' }); await __sleep(200);
            window.__v = v; window.__n = n;
            return out;
        });
        // The pointer over the middle of the frame fills the probe.
        const box = await page.evaluate(() => { const b = __v.canvas.getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; });
        await page.mouse.move(box.x + box.w / 2, box.y + box.h / 2);
        await page.mouse.move(box.x + box.w / 2 + 2, box.y + box.h / 2 + 1);
        await page.waitForTimeout(150);
        Object.assign(r, await page.evaluate(async () => {
            const v = __v, n = __n, out = {};
            out.probe = Object.fromEntries([...v.controlsPanel.querySelectorAll('.radiance-scope-probe .cell')]
                .map((c) => [c.dataset.probe, c.querySelector('.value').textContent]));
            // Color keeps the stack, histogram first.
            v.setWorkspace('color', { persist: false }); v._setReferenceTab('scopes'); await __sleep(200);
            const g2 = v.controlsPanel.querySelector('.radiance-ref-scopes-grid');
            out.color = { quad: g2.classList.contains('is-quad'), cells: [...g2.children].map((e) => e.dataset.scope),
                probe: !!v.controlsPanel.querySelector('.radiance-scope-probe') };
            __remove(n);
            return out;
        }));
        assert.deepEqual(page.errors, []);
        assert.equal(r.opensOn, 'inspector', 'Inspect opens on the Inspector (4.1 shell)');
        assert.equal(r.tab, 'scopes');
        assert.equal(r.quad, true);
        assert.deepEqual(r.cells, ['waveform', 'parade', 'vectorscope', 'qc']);
        assert.equal(r.firstSection, 'SCOPES', 'the picture reading comes before its options');
        assert.deepEqual(r.scale.map(([l]) => l), ['10-bit', '%', 'Nits']);
        assert.equal(r.scale[2][1], true, 'Nits is not offered on an SDR view');
        assert.equal(r.picked, 'percent');
        assert.equal(r.pressed, '%');
        assert.equal(r.qcBefore, 'Run QC');
        assert.ok(r.qcLines > 0, 'the QC cell lists the checks after a run');
        assert.equal(r.qcAfter, 'Run again');
        assert.deepEqual(r.details, ['qc', 'qc'], 'QC Details opens the QC workspace');
        assert.match(r.probe.light, /^\d+ nits {2}· {2}[+-]\d+\.\d st$/);
        assert.match(r.probe.where, /^\d+, \d+ {2}· {2}f\d+$/);
        assert.notEqual(r.probe.linear, '—');
        assert.deepEqual(r.color, { quad: false, cells: ['histogram', 'waveform', 'parade', 'vectorscope'], probe: false });
    } finally { await page.close(); }
});

test('Grab still keeps versions for B; the B menu picks one, Release keeps them, Clear forgets them', { skip }, async () => {
    const page = await session.open();
    try {
        const r = await page.evaluate(async () => {
            const n = __make('advanced'); const v = n.radianceViewer;
            n.onExecuted(__frames(12, 96, 54));
            await __until(() => v.hdrData); await __sleep(300);
            v.setWorkspace('review', { persist: false }); await __sleep(200);
            const out = { canKeep: v._canKeepStills() };
            v.setFrame(1); await __sleep(80); v._sbPin.click(); await __sleep(80);
            v.setFrame(5); await __sleep(80); v._sbPin.click(); await __sleep(80);
            out.grabbed = { stills: v._stills.map((x) => [x.slot, x.frame]), mode: v.compareMode, label: v._sbBLabel.textContent };
            v._sbBLabel.click(); await __sleep(60);
            const menu = document.querySelector('.radiance-b-menu');
            out.items = [...menu.querySelectorAll('button')].map((b) => b.dataset.b);
            out.current = menu.querySelector('button.is-current')?.dataset.b;
            menu.querySelector('[data-b="still-1"]').click(); await __sleep(60);
            out.menuClosed = !document.querySelector('.radiance-b-menu');
            out.picked = { label: v._sbBLabel.textContent, slot: v._pinnedSlot, src: v.compareSource };
            // Escape closes the menu.
            v._sbBLabel.click(); await __sleep(30);
            document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
            out.escClosed = !document.querySelector('.radiance-b-menu');
            v.releaseReference(); await __sleep(60);
            out.released = { src: v.compareSource, stills: v._stills.length, kept: !!v.renderer.referenceShelf[2] };
            out.again = v.useStillAsB(2) && v._sbBLabel.textContent;
            // The plain pin (File > Pin) uses its own slot and leaves the stills.
            v.pinReference(); out.pin = { slot: v._pinnedSlot, stills: v._stills.length, still1: !!v.renderer.referenceShelf[1] };
            v.clearStills(); await __sleep(60);
            out.cleared = { stills: v._stills.length, src: v.compareSource };
            v.releaseReference();
            out.afterAll = v.compareSource;
            __remove(n);
            return out;
        });
        assert.deepEqual(page.errors, []);
        assert.equal(r.canKeep, true);
        assert.deepEqual(r.grabbed.stills, [[1, 1], [2, 5]]);
        assert.equal(r.grabbed.mode, 'wipe', 'grabbing with compare off opens a wipe');
        assert.equal(r.grabbed.label, 'B: still 2');
        assert.deepEqual(r.items, ['input', 'still-2', 'still-1', 'grab', 'release', 'clear']);
        assert.equal(r.current, 'still-2');
        assert.equal(r.menuClosed, true);
        assert.deepEqual(r.picked, { label: 'B: still 1', slot: 1, src: 'pinned' });
        assert.equal(r.escClosed, true);
        assert.deepEqual(r.released, { src: null, stills: 2, kept: true });
        assert.equal(r.again, 'B: still 2');
        assert.deepEqual(r.pin, { slot: 0, stills: 2, still1: true });
        assert.deepEqual(r.cleared, { stills: 0, src: 'pinned' }, 'clearing stills keeps a plain pin');
        assert.equal(r.afterAll, null);
    } finally { await page.close(); }
});

test('a loaded graph opens in View, unless the V5 shell saved it in Pro tools', { skip }, async () => {
    const page = await session.open();
    try {
        const r = await page.evaluate(async () => {
            const tabs = (v) => [...v.controlsPanel.querySelectorAll('.radiance-ref-tab')].map((b) => b.dataset.tabId);
            const out = {};
            // Saved by the V5 shell in Pro tools > Inspect: it opens there again.
            const a = new FakeNode(); a.onNodeCreated(); await __sleep(100);
            a.onConfigure({ properties: { radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'inspect', radiance_viewer_shell: 'v5' } });
            await __sleep(100);
            out.inspect = { ui: a.radianceViewer.uiMode, ws: a.radianceViewer.workspace, tabs: tabs(a.radianceViewer) };
            // Saved before V5 in the Full layout (or before workspaces existed): View.
            const b = new FakeNode(); b.onNodeCreated(); await __sleep(100);
            b.onConfigure({ properties: { radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'color' } });
            await __sleep(100);
            out.old = { ui: b.radianceViewer.uiMode, ws: b.radianceViewer.workspace, task: b.radianceViewer._shell.task,
                marked: b.properties.radiance_viewer_shell };
            const d = new FakeNode(); d.onNodeCreated(); await __sleep(100);
            d.onConfigure({ properties: {} });
            out.none = { ui: d.radianceViewer.uiMode, ws: d.radianceViewer.workspace };
            // A shipped workflow: View.
            const c = new FakeNode(); c.onNodeCreated(); await __sleep(100);
            c.onConfigure({ properties: { radiance_viewer_mode: 'simple', radiance_viewer_workspace: 'review' } });
            out.review = { ui: c.radianceViewer.uiMode, ws: c.radianceViewer.workspace };
            [a, b, c, d].forEach((n) => __remove(n));
            return out;
        });
        assert.deepEqual(page.errors, []);
        assert.deepEqual(r.inspect, { ui: 'advanced', ws: 'inspect', tabs: ['inspector', 'scopes', 'analysis'] });
        assert.deepEqual(r.old, { ui: 'simple', ws: 'review', task: 'view', marked: 'v5' });
        assert.deepEqual(r.none, { ui: 'simple', ws: 'review' });
        assert.deepEqual(r.review, { ui: 'simple', ws: 'review' });
    } finally { await page.close(); }
});
