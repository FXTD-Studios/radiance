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

test('each workspace shows only its panel tabs, and a hidden tab moves to the workspace that has it', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('advanced'); const v = n.radianceViewer;
        n.onExecuted(__frames(2, 64, 48));
        await __until(() => v.image, 10000);
        const labels = () => [...v.controlsPanel.querySelectorAll('.radiance-ref-tab')].map((b) => b.dataset.tabId);
        const active = () => v.controlsPanel.querySelector('.radiance-ref-tab.is-active')?.dataset.tabId;
        const r = {};
        v.setWorkspace('inspect'); r.inspect = labels(); r.inspectActive = active();
        v.setWorkspace('color'); r.color = labels(); r.colorActive = active();
        v.setWorkspace('compare'); r.compare = labels(); r.compareActive = active();
        v.setWorkspace('inspect');
        v._setReferenceTab('grade');
        r.afterShortcut = { ws: v.workspace, tabs: labels(), active: active() };
        r.compareMode = v.compareMode;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(out.inspect, ['inspector', 'scopes', 'analysis']);
    assert.equal(out.inspectActive, 'inspector');
    assert.deepEqual(out.color, ['grade', 'masks', 'effects', 'scopes']);
    assert.equal(out.colorActive, 'grade');
    assert.deepEqual(out.compare, ['inspector', 'scopes', 'analysis']);
    assert.equal(out.compareActive, 'scopes');
    assert.deepEqual(out.afterShortcut, { ws: 'color', tabs: ['grade', 'masks', 'effects', 'scopes'], active: 'grade' });
    assert.equal(out.compareMode, 'none');
});

test('a deleted viewer whose Grade and Inspector panels were open can be collected', { skip }, async () => {
    const page = await session.open();
    try {
        await page.evaluate(async () => {
            const n = __make(); const v = n.radianceViewer;
            n.onExecuted(__frames(3, 64, 48));
            await __until(() => v.hdrData);
            v._setReferenceTab('grade'); await __sleep(100);
            v._setReferenceTab('inspector'); await __sleep(100);   // rebuilds the panel: the old preset <select> goes
            window.__ref = new WeakRef(v);
            __remove(n);
        });
        const cdp = await page.context().newCDPSession(page);
        let collected = false;
        for (let i = 0; i < 40 && !collected; i++) {
            await page.evaluate(() => new Promise((r) => setTimeout(r, 250)));
            await cdp.send('HeapProfiler.collectGarbage');
            collected = await page.evaluate(() => window.__ref.deref() === undefined);
        }
        assert.deepEqual(page.errors, []);
        assert.ok(collected, 'the deleted viewer is still reachable (a detached <select> keeps the panel alive)');
    } finally {
        await page.close();
    }
});

test('Difference gain and signed mode change what the renderer draws', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        const msg = __frames(2, 64, 48);
        for (let i = 0; i < 2; i++) msg.radiance_images.push({ filename: 'b_64x48_46.png', subfolder: '', type: 'temp', is_compare: true, frame: i });
        n.onExecuted(msg);
        await __until(() => v.image && v._hasCompareB(), 15000);
        await __sleep(200);
        const gl = v.renderer.gl;
        const read = () => { v.render(); const px = new Uint8Array(4); gl.readPixels(32, 24, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); return px[0]; };
        const r = {};
        v.setCompareMode('none'); r.a = read();
        v.setCompareMode('b'); r.b = read();
        v.setCompareMode('difference');
        v.setDifference({ gain: 1, mode: 'abs' }); r.abs1 = read();
        v.setDifference({ gain: 2 }); r.abs2 = read();
        v.setDifference({ gain: 1, mode: 'signed' }); r.signed1 = read();
        r.gainLabel = v._sbDiffGain.textContent; r.signLabel = v._sbDiffSigned.textContent;
        r.shown = v._sbDiffGain.style.display !== 'none';
        v.setDifference({ gain: 3 }); r.badGain = v.diffSpec.gain;      // not a step: falls back to 4
        // equal A and B: pin A as B
        v.setCompareMode('none'); v.pinReference(); v.setCompareMode('difference');
        v.setDifference({ gain: 4, mode: 'abs' }); r.equalAbs = read();
        v.setDifference({ mode: 'signed' }); r.equalSigned = read();
        v.setCompareMode('none'); r.hiddenOff = v._sbDiffGain.style.display === 'none';
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    const d = Math.abs(out.a - out.b);
    assert.ok(d > 10, `test premise: A ${out.a} and B ${out.b} differ`);
    assert.ok(Math.abs(out.abs1 - d) <= 2, `|A-B| x1 is ${out.abs1}, expected ${d}`);
    assert.ok(Math.abs(out.abs2 - Math.min(255, 2 * d)) <= 3, `|A-B| x2 is ${out.abs2}, expected ${2 * d}`);
    const expectSigned = Math.round(127.5 + 0.5 * (out.a - out.b));
    assert.ok(Math.abs(out.signed1 - expectSigned) <= 3, `signed x1 is ${out.signed1}, expected ${expectSigned}`);
    assert.equal(out.gainLabel, 'x1'); assert.equal(out.signLabel, 'A-B'); assert.equal(out.shown, true);
    assert.equal(out.badGain, 4);
    assert.ok(out.equalAbs <= 2, `A against itself, |A-B| is ${out.equalAbs}`);
    assert.ok(Math.abs(out.equalSigned - 128) <= 2, `A against itself, signed is ${out.equalSigned}, expected mid grey`);
    assert.equal(out.hiddenOff, true, 'the difference controls show outside Difference');
});

test('onion skin draws the neighbouring frames, at their opacity, and not while comparing', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(5, 64, 48));
        await __until(() => v.frameImages?.[0] && v.frameImages?.[1] && v.frameImages?.[2], 15000);
        v.setFrame(1); await __sleep(100);
        const ox = Math.round(v.panX + 32 * v.zoom), oy = Math.round(v.panY + 24 * v.zoom);
        const px = () => Array.from(v.overlayCtx.getImageData(ox, oy, 1, 1).data);
        const r = {};
        v.setOnion({ enabled: true, before: 1, after: 0, tint: false, opacity: 0.4 }); v.renderOverlay();
        r.one = px(); r.count = { ...v._onionDrawn };
        v.setOnion({ before: 1, after: 1, tint: true }); v.renderOverlay();
        r.two = { ...v._onionDrawn }; r.tinted = px();
        v.setCompareMode('b'); v.renderOverlay(); r.whileB = { ...v._onionDrawn };
        v.setCompareMode('none');
        v.setOnion({ enabled: false }); v.renderOverlay(); r.off = px()[3];
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    // frame 0's preview is grey 100; at opacity 0.4 the overlay holds it with alpha ~102
    assert.ok(Math.abs(out.one[0] - 100) <= 3 && Math.abs(out.one[3] - 102) <= 3, `onion pixel ${out.one}`);
    assert.deepEqual(out.count, { drawn: 1, wanted: 1 });
    assert.deepEqual(out.two, { drawn: 2, wanted: 2 });
    assert.ok(out.tinted[0] > out.tinted[2], `tinted layers ${out.tinted}: before is red, after green, so blue is lowest`);
    assert.deepEqual(out.whileB, { drawn: 0, wanted: 0 });
    assert.equal(out.off, 0);
});

test('bookmarks, notes, QC and the review report work from the viewer', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(6, 64, 48));
        await __until(() => v.image && v.hdrData, 15000);
        const r = {};
        // Bookmarks on frames 1 and 4, then step between them (wrapping).
        v.setFrame(1); v._runKeyAction('review.bookmark');
        v.setFrame(4); v._runKeyAction('review.bookmark');
        r.markActive = v._sbMark.classList.contains('is-active');
        v.setFrame(0); v._runKeyAction('review.nextBookmark'); r.next1 = v.currentFrame;
        v._runKeyAction('review.nextBookmark'); r.next2 = v.currentFrame;
        v._runKeyAction('review.nextBookmark'); r.wrap = v.currentFrame;
        v._runKeyAction('review.prevBookmark'); r.prev = v.currentFrame;
        // Toggle off on frame 4.
        v.setFrame(4); v._runKeyAction('review.bookmark'); r.afterOff = v._review.bookmarks().map((b) => b.frame);
        // A note on frame 2: typing in the editor never reaches the shortcuts.
        v.setFrame(2);
        let leaked = 0; const spy = () => leaked++;
        window.addEventListener('keydown', spy);
        v._runKeyAction('review.note');
        const input = v._noteEditor;
        r.editorOpen = !!input && document.activeElement === input;
        input.value = 'Fix flicker here';
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'f', code: 'KeyF', bubbles: true }));
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
        window.removeEventListener('keydown', spy);
        r.leaked = leaked;
        r.editorClosed = !v._noteEditor && !input.isConnected;
        r.note = v._review.forFrame(2, 'note')[0]?.text;
        r.saved = JSON.parse(n.properties.radiance_viewer_review).marks.length;
        // The note is drawn at the centre of the picture on frame 2 only.
        v.renderOverlay();
        const cx = Math.round(v.panX + 32 * v.zoom), cy = Math.round(v.panY + 24 * v.zoom);
        r.notePixel = Array.from(v.overlayCtx.getImageData(cx, cy, 1, 1).data);
        v.setFrame(3); v.renderOverlay();
        r.otherFramePixel = v.overlayCtx.getImageData(cx, cy, 1, 1).data[3];
        // QC over the six float frames.
        await v.runQC();
        r.qc = v._qcReport && { overall: v._qcReport.overall, complete: v._qcReport.complete, analysed: v._qcReport.frames.analysed };
        r.qcLabel = v._sbQC.textContent;
        // The report: capture the downloads instead of saving them.
        const files = [];
        v._downloadText = (name, text) => files.push({ name, text });
        const rep = v._runKeyAction('review.exportReport');
        r.files = files.map((f) => f.name.replace(/_\d.*\./, '_*.'));
        r.md = files[0]?.text || '';
        r.json = files[1] ? JSON.parse(files[1].text) : null;
        // A new run clears the QC result.
        n.onExecuted(__frames(2, 64, 48));
        await __sleep(50);
        r.qcAfterNewRun = v._qcReport; r.qcLabelAfter = v._sbQC.textContent;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.markActive, true);
    assert.deepEqual([out.next1, out.next2, out.wrap, out.prev], [1, 4, 1, 4]);
    assert.deepEqual(out.afterOff, [1]);
    assert.equal(out.editorOpen, true);
    assert.equal(out.leaked, 0, 'a key typed in the note editor reached the page');
    assert.equal(out.editorClosed, true);
    assert.equal(out.note, 'Fix flicker here');
    assert.equal(out.saved, 2);       // one bookmark, one note
    assert.ok(out.notePixel[3] > 200 && out.notePixel[0] > 200 && out.notePixel[2] < 120, `note marker pixel ${out.notePixel}`);
    assert.equal(out.otherFramePixel, 0, 'the note shows on another frame');
    assert.ok(out.qc && out.qc.complete && out.qc.analysed === 6, `QC ${JSON.stringify(out.qc)}`);
    assert.ok(['PASS', 'WARN', 'FAIL', 'INDETERMINATE'].includes(out.qc.overall));
    assert.equal(out.qcLabel, `QC: ${out.qc.overall}`);
    assert.deepEqual(out.files, ['radiance_review_*.md', 'radiance_review_*.json']);
    assert.match(out.md, /Fix flicker here/);
    assert.match(out.md, new RegExp(`Overall: ${out.qc.overall}`));
    assert.equal(out.json.schema, 'radiance.review.report');
    assert.equal(out.json.qc.overall, out.qc.overall);
    assert.equal(out.qcAfterNewRun, null);
    assert.equal(out.qcLabelAfter, 'QC');
});

test('a QC run can be stopped, and keeps what it measured as incomplete', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(40, 64, 48));
        await __until(() => v.image && v.hdrData, 15000);
        const run = v.runQC();
        await __until(() => /QC [1-9]/.test(v._sbQC.textContent), 10000);
        v._sbQC.click();                         // the same button stops it
        await run;
        const r = { overall: v._qcReport?.overall, complete: v._qcReport?.complete, analysed: v._qcReport?.frames.analysed, label: v._sbQC.textContent };
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.complete, false);
    assert.ok(out.analysed > 0 && out.analysed < 40, `analysed ${out.analysed}`);
    assert.notEqual(out.overall, 'PASS', 'a stopped run must not read as a pass');
    assert.match(out.label, /\(stopped\)$/);
});
