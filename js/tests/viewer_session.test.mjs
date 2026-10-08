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
