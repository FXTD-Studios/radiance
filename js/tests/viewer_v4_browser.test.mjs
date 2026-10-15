/**
 * V4 viewer features, driven in a real browser (Chromium through Playwright):
 * the Compare hint, review strokes, the 3D Normal channel and retimed compare.
 * Uses the session harness (sessionharness.html, session_support.mjs); skips
 * when Playwright is not installed.
 *
 * Run: node --test js/tests/viewer_v4_browser.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

async function inPage(fn, arg) {
    const page = await session.open();
    try {
        const out = await page.evaluate(fn, arg);
        return { out, errors: page.errors };
    } finally {
        await page.close();
    }
}

test('a connected B puts a hint on Compare and switches nothing', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        const msg = __frames(3, 64, 48);
        for (let i = 0; i < 3; i++) msg.radiance_images.push({ filename: 'b_64x48_46.png', subfolder: '', type: 'temp', is_compare: true, frame: i });
        n.onExecuted(msg);
        await __until(() => v.image, 10000);
        const btn = v._modeBtns.get('compare');
        const modeBefore = v.compareMode;
        v._syncWorkspaceHint();
        const r = { hint: btn.classList.contains('has-hint'), title: btn.title, ws: v.workspace, modeKept: v.compareMode === modeBefore };
        v.setWorkspace('compare'); r.modeAfterCompare = v.compareMode === modeBefore; v.setWorkspace('review');
        r.afterSeen = btn.classList.contains('has-hint');
        n.onExecuted(__frames(2, 64, 48));
        await __sleep(50);
        r.noB = btn.classList.contains('has-hint');
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.hint, true, 'no hint with a B connected');
    assert.match(out.title, /A reference \(B\) is connected/);
    assert.equal(out.ws, 'review', 'the hint switched the workspace');
    assert.equal(out.modeKept, true, 'the hint changed the compare mode');
    assert.equal(out.modeAfterCompare, true, 'opening Compare changed the compare mode');
    assert.equal(out.afterSeen, false, 'the hint stayed after Compare was opened');
    assert.equal(out.noB, false, 'a run without B still shows the hint');
});

test('an Alt+drag stroke is stored per frame in UV, follows zoom, persists and reloads', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(3, 64, 48));
        await __until(() => v.image, 10000);
        await __sleep(100);
        const rect = v.canvas.getBoundingClientRect();
        const sx = rect.width / v.canvas.width, sy = rect.height / v.canvas.height;
        // canvas px of image pixels (16, 12) and (48, 36)
        const at = (ix, iy) => ({ clientX: rect.left + (v.panX + ix * v.zoom) * sx, clientY: rect.top + (v.panY + iy * v.zoom) * sy });
        const p0 = at(16, 12), p1 = at(48, 36);
        v.canvas.dispatchEvent(new MouseEvent('mousedown', { ...p0, altKey: true, button: 0, bubbles: true }));
        document.dispatchEvent(new MouseEvent('mousemove', { ...p1, altKey: true, bubbles: true }));
        document.dispatchEvent(new MouseEvent('mouseup', { ...p1, bubbles: true }));
        const saved = JSON.parse(n.properties.radiance_viewer_review || 'null');
        const stroke = saved?.marks?.[0];
        const r = { kind: stroke?.kind, frame: stroke?.frame, pts: stroke?.pts };
        const before = v._reviewStrokesForView()[0]?.pts[0];
        v.zoom *= 2; v.panX += 7;
        const afterZoom = v._reviewStrokesForView()[0]?.pts[0];
        r.follows = !!before && !!afterZoom && Math.abs(afterZoom.x - (v.panX + stroke.pts[0].u * v.imageWidth * v.zoom)) < 0.01
            && Math.abs(afterZoom.x - before.x) > 1;
        v.currentFrame = 1;
        r.otherFrame = v._reviewStrokesForView().length;
        // Reload: a new node built from the saved properties has the stroke.
        const n2 = new FakeNode(); n2.properties = { radiance_viewer_review: n.properties.radiance_viewer_review }; n2.onNodeCreated();
        r.reloaded = n2.radianceViewer._review.forFrame(0, 'stroke').length;
        // Shift+Alt+click clears only the current frame (frame 1 has nothing; frame 0 keeps its stroke).
        v.canvas.dispatchEvent(new MouseEvent('click', { ...p0, altKey: true, shiftKey: true, bubbles: true }));
        r.keptOnFrame0 = v._review.forFrame(0, 'stroke').length;
        v.currentFrame = 0;
        v.canvas.dispatchEvent(new MouseEvent('click', { ...p0, altKey: true, shiftKey: true, bubbles: true }));
        r.clearedFrame0 = v._review.forFrame(0, 'stroke').length;
        __remove(n); __remove(n2);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.kind, 'stroke', 'no stroke saved on the node');
    assert.equal(out.frame, 0);
    assert.ok(Math.abs(out.pts[0].u - 0.25) < 0.02 && Math.abs(out.pts[0].v - 0.25) < 0.02, `stroke starts at ${JSON.stringify(out.pts[0])}, expected UV 0.25`);
    assert.equal(out.follows, true, 'the stroke did not follow zoom and pan');
    assert.equal(out.otherFrame, 0, 'the stroke shows on another frame');
    assert.equal(out.reloaded, 1, 'the saved stroke did not reload');
    assert.equal(out.keptOnFrame0, 1, 'clearing frame 1 removed frame 0\'s stroke');
    assert.equal(out.clearedFrame0, 0);
});

test('the Normal channel draws the normal pass, is A only, and leaves the timeline alone', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        const msg = __frames(2, 64, 48);
        for (let i = 0; i < 2; i++) msg.radiance_images.push({ filename: 'b_64x48_200.png', subfolder: '', type: 'temp', is_normal: true, frame: i, normal_input: 'signed' });
        n.onExecuted(msg);
        await __until(() => v.image && v.normalImage, 10000);
        const r = { total: v.totalFrames };
        const row = v._getReferenceChannelRows().find((x) => x.name === 'Normal');
        r.available = !!row?.available;
        v._selectReferenceChannel(row);
        v.renderOverlay();
        const ox = Math.round(v.panX + 32 * v.zoom), oy = Math.round(v.panY + 24 * v.zoom);
        r.overlay = Array.from(v.overlayCtx.getImageData(ox, oy, 1, 1).data);
        r.probe = v._sampleNormalAt(32, 24);
        v.compareMode = 'b'; v.renderOverlay();
        r.hiddenUnderB = Array.from(v.overlayCtx.getImageData(ox, oy, 1, 1).data)[3];
        v.compareMode = 'none';
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.total, 2, 'normal frames became timeline frames');
    assert.equal(out.available, true);
    assert.ok(Math.abs(out.overlay[0] - 200) <= 2 && out.overlay[3] === 255, `overlay pixel ${out.overlay}`);
    assert.ok(Math.abs(out.probe[0] - (200 / 255 * 2 - 1)) < 0.01, `probe ${out.probe}`);
    assert.equal(out.hiddenUnderB, 0, 'the normal pass was drawn over B');
});

test('a paged B at another frame rate is loaded and shown by time', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        const msg = __frames(10, 64, 48);
        msg.compare_fps = [30];                       // A plays at 24 (fps), B is a 30 fps VIDEO
        for (let i = 0; i < 13; i++) msg.radiance_images.push({ filename: `b_64x48_${40 + i}.png`, subfolder: '', type: 'temp', is_compare: true, frame: i });
        n.onExecuted(msg);
        await __until(() => v.image, 10000);
        v.setFrame(8);
        await __until(() => v.frameCompareImages?.[8], 10000);
        await __sleep(100);
        v._updateCompareForFrame(8, true);
        return { slot8: v.frameCompareImages[8]?.src || '', shown: v.compareImage?.src || '', paged: v._compareEntries.length };
    });
    assert.deepEqual(errors, []);
    assert.equal(out.paged, 13);
    // A frame 8 at 24 fps is 0.333 s: B frame floor(0.333 * 30) = 10, file b_64x48_50.png
    assert.match(out.slot8, /b_64x48_50\.png/, `A's slot 8 holds ${out.slot8}`);
    assert.match(out.shown, /b_64x48_50\.png/, `B shows ${out.shown}`);
});
