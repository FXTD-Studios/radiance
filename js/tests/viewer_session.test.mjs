/**
 * Whole viewers in a browser, used the way a session uses them.
 *
 * Each of these was reproduced in the October 2026 review with a real viewer
 * in headless Chromium: two viewers on one graph, a video loaded straight
 * into the viewer, frames that 404 after a restart, a lost GPU context, a
 * deleted node. None of it shows up on an Object.create'd instance, so the
 * viewers here are built through the extension exactly as ComfyUI builds
 * them (see sessionharness.html and session_support.mjs).
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/viewer_session.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

/** Run `fn` in a fresh page; page errors fail the test. */
async function inPage(fn, arg) {
    const page = await session.open();
    try {
        const out = await page.evaluate(fn, arg);
        return { out, errors: page.errors, page };
    } finally {
        await page.close();
    }
}

// ── C5: one panel per viewer ────────────────────────────────────────────────

test('two viewers each keep a full control panel of their own', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const count = (v) => v.rightControlPanel?.querySelectorAll('*').length || 0;
        const a = __make(); await __sleep(300);
        const A = a.radianceViewer;
        const alone = count(A);
        const b = __make(); await __sleep(300);
        const B = b.radianceViewer;
        const r = { alone, aWithB: count(A), b: count(B),
            aPanelInA: A.rightControlPanel.contains(A.controlsPanel),
            bPanelInB: B.rightControlPanel.contains(B.controlsPanel),
            distinct: A.controlsPanel !== B.controlsPanel };

        // Exposure on B's GRADE tab changes B, and only B.
        const gradeTab = B.controlsPanel.querySelector('[data-tab-id="grade"]');
        gradeTab.click();
        const row = [...B.controlsPanel.querySelectorAll('.radiance-ref-slider')]
            .find((el) => el.querySelector('label')?.textContent.trim() === 'Exposure');
        const input = row.querySelector('input[type="range"]');
        input.value = '2';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        r.exposure = { A: A.exposure || 0, B: B.exposure || 0 };

        // Deleting B leaves A whole.
        __remove(b); await __sleep(200);
        r.aAfterRemoveB = count(A);
        r.aPanelStillInA = A.rightControlPanel.contains(A.controlsPanel);
        __remove(a);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.ok(out.alone > 50, `a lone viewer's panel has ${out.alone} elements`);
    assert.ok(out.aWithB > 50, `viewer A's panel dropped to ${out.aWithB} elements when B was added`);
    assert.ok(out.b > 50);
    assert.ok(out.distinct && out.aPanelInA && out.bPanelInB, JSON.stringify(out));
    assert.deepEqual(out.exposure, { A: 0, B: 2 }, 'moving Exposure in B did not change B alone');
    assert.ok(out.aAfterRemoveB > 50, `deleting B left A with ${out.aAfterRemoveB} panel elements`);
    assert.ok(out.aPanelStillInA);
});

// ── M11, M12: the renderer starts once, the GPU budget is the page's ───────

test('the renderer compiles its shaders once per viewer', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const P = window.RadianceWebGLRenderer.prototype;
        let builds = 0, lostHandlers = 0;
        const createPrograms = P.createPrograms;
        P.createPrograms = function () { builds++; return createPrograms.call(this); };
        const add = HTMLCanvasElement.prototype.addEventListener;
        HTMLCanvasElement.prototype.addEventListener = function (type, ...rest) {
            if (type === 'webglcontextlost') lostHandlers++;
            return add.call(this, type, ...rest);
        };
        const n = __make(); await __sleep(200);
        const r = { builds, lostHandlers, backend: n.radianceViewer._gpuBackend };
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.backend, 'webgl');
    assert.equal(out.builds, 1, `shaders were compiled ${out.builds} times for one viewer`);
    assert.equal(out.lostHandlers, 1, `${out.lostHandlers} context-loss handlers for one renderer`);
});

test('the GPU frame cache budget is shared by the viewers on the page', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const a = __make(); await __sleep(100);
        const one = a.radianceViewer.renderer._frameCacheByteBudget;
        const b = __make(); await __sleep(100);
        const two = [a.radianceViewer.renderer._frameCacheByteBudget, b.radianceViewer.renderer._frameCacheByteBudget];
        __remove(b); await __sleep(50);
        const back = a.radianceViewer.renderer._frameCacheByteBudget;
        __remove(a);
        return { one, two, back };
    });
    assert.deepEqual(errors, []);
    assert.ok(out.one > 0);
    assert.deepEqual(out.two, [Math.floor(out.one / 2), Math.floor(out.one / 2)],
        'each viewer kept the whole budget');
    assert.equal(out.back, out.one, 'a deleted viewer kept its share');
});
