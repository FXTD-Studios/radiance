import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { ReviewStore, canvasToUV, uvToCanvas, thinPoints, LIMITS, REVIEW_VERSION } from '../radiance_review.js';

const view = (o = {}) => ({ zoom: 1, panX: 0, panY: 0, width: 1920, height: 1080, ...o });

test('canvas and UV are inverses at any zoom and pan', () => {
    for (const v of [view(), view({ zoom: 0.37, panX: 120, panY: -40 }), view({ zoom: 8, panX: -5000, panY: 900 })]) {
        const uv = canvasToUV(640, 360, v);
        const back = uvToCanvas(uv.u, uv.v, v);
        assert.ok(Math.abs(back.x - 640) < 1e-9 && Math.abs(back.y - 360) < 1e-9);
    }
});

test('a mark stays on the same feature when the view changes', () => {
    // pointing at the exact centre of the picture
    const a = view({ zoom: 0.5, panX: 100, panY: 50 });
    const uv = canvasToUV(100 + 960 * 0.5, 50 + 540 * 0.5, a);
    assert.ok(Math.abs(uv.u - 0.5) < 1e-12 && Math.abs(uv.v - 0.5) < 1e-12);
    const zoomed = view({ zoom: 2, panX: -300, panY: -100 });
    const at = uvToCanvas(uv.u, uv.v, zoomed);
    assert.ok(Math.abs(at.x - (-300 + 960 * 2)) < 1e-9);
    assert.ok(Math.abs(at.y - (-100 + 540 * 2)) < 1e-9);
});

test('a mark survives a change of resolution (UV is resolution independent)', () => {
    const hd = view({ width: 1920, height: 1080 });
    const uhd = view({ width: 3840, height: 2160 });
    const uv = canvasToUV(480, 270, hd);                     // a quarter of the way in
    const at = uvToCanvas(uv.u, uv.v, uhd);
    assert.ok(Math.abs(at.x - 960) < 1e-9 && Math.abs(at.y - 540) < 1e-9);
});

test('an unusable view gives null, not NaN', () => {
    assert.equal(canvasToUV(1, 1, view({ zoom: 0 })), null);
    assert.equal(canvasToUV(1, 1, null), null);
    assert.equal(uvToCanvas(0.5, 0.5, view({ panX: NaN })), null);
});

test('strokes belong to one frame and show only on it', () => {
    const s = new ReviewStore();
    s.addStroke(3, [{ u: 0.1, v: 0.1 }, { u: 0.2, v: 0.2 }]);
    s.addStroke(7, [{ u: 0.5, v: 0.5 }, { u: 0.6, v: 0.6 }]);
    assert.equal(s.canvasStrokes(3, view()).length, 1);
    assert.equal(s.canvasStrokes(4, view()).length, 0);
    assert.equal(s.canvasStrokes(7, view()).length, 1);
});

test('canvas strokes follow the view', () => {
    const s = new ReviewStore();
    s.addStroke(0, [{ u: 0, v: 0 }, { u: 1, v: 1 }]);
    const [line] = s.canvasStrokes(0, view({ zoom: 0.5, panX: 10, panY: 20 }));
    assert.deepEqual(line.pts.map((p) => [p.x, p.y]), [[10, 20], [10 + 960, 20 + 540]]);
});

test('an in-progress stroke can be drawn without being stored', () => {
    const s = new ReviewStore();
    const live = { pts: [{ u: 0.1, v: 0.1 }, { u: 0.3, v: 0.3 }], color: '#f00' };
    assert.equal(s.canvasStrokes(0, view(), [live]).length, 1);
    assert.equal(s.marks.length, 0);
});

test('bad strokes are refused', () => {
    const s = new ReviewStore();
    assert.equal(s.addStroke(0, [{ u: 0.1, v: 0.1 }]), null);              // one point
    assert.equal(s.addStroke(-1, [{ u: 0, v: 0 }, { u: 1, v: 1 }]), null);
    assert.equal(s.addStroke(1.5, [{ u: 0, v: 0 }, { u: 1, v: 1 }]), null);
    assert.equal(s.addStroke(0, [{ u: NaN, v: 0 }, { u: 1, v: 1 }]), null);
    assert.equal(s.addStroke(0, 'x'), null);
});

test('long strokes are thinned but keep both ends', () => {
    const pts = Array.from({ length: 5000 }, (_, i) => ({ u: i / 4999, v: 0 }));
    const t = thinPoints(pts);
    assert.equal(t.length, LIMITS.maxPointsPerStroke);
    assert.equal(t[0].u, 0);
    assert.equal(t[t.length - 1].u, 1);
});

test('notes are clamped into the picture and need text', () => {
    const s = new ReviewStore();
    const n = s.addNote(2, 1.5, -0.2, '  fix the edge  ');
    assert.deepEqual([n.u, n.v, n.text], [1, 0, 'fix the edge']);
    assert.equal(s.addNote(2, 0.5, 0.5, '   '), null);
    assert.equal(s.addNote(2, 0.5, 0.5, 'x'.repeat(5000)).text.length, LIMITS.maxNoteChars);
});

test('bookmarks: one per frame, ordered, and stepping wraps', () => {
    const s = new ReviewStore();
    s.addBookmark(40, 'hero'); s.addBookmark(10, 'start'); s.addBookmark(25);
    s.addBookmark(10, 'start v2');                                    // updates, does not duplicate
    assert.deepEqual(s.bookmarks().map((b) => b.frame), [10, 25, 40]);
    assert.equal(s.bookmarks()[0].label, 'start v2');
    assert.equal(s.stepBookmark(10, 1), 25);
    assert.equal(s.stepBookmark(40, 1), 10);                           // wraps forward
    assert.equal(s.stepBookmark(10, -1), 40);                          // wraps back
    assert.equal(s.stepBookmark(30, -1), 25);
    assert.equal(new ReviewStore().stepBookmark(3), null);
});

test('removing and clearing', () => {
    const s = new ReviewStore();
    const a = s.addStroke(1, [{ u: 0, v: 0 }, { u: 1, v: 1 }]);
    s.addNote(1, 0.5, 0.5, 'n'); s.addBookmark(1); s.addStroke(2, [{ u: 0, v: 0 }, { u: 1, v: 1 }]);
    assert.equal(s.remove(a.id), true);
    assert.equal(s.remove('nope'), false);
    assert.equal(s.clearFrame(1, 'note'), 1);
    assert.equal(s.clear('stroke'), 1);
    assert.equal(s.marks.length, 1);                                   // the bookmark
    assert.equal(s.clear(), 1);
});

test('the store has a hard cap and refuses rather than dropping older marks', () => {
    const s = new ReviewStore();
    for (let i = 0; i < LIMITS.maxMarks; i++) assert.ok(s.addNote(i, 0.5, 0.5, 'n'));
    assert.equal(s.addNote(9999, 0.5, 0.5, 'one too many'), null);
    assert.equal(s.marks.length, LIMITS.maxMarks);
    assert.equal(s.marks[0].frame, 0);
});

test('save and load round-trips every kind of mark', () => {
    const s = new ReviewStore();
    s.addStroke(3, [{ u: 0.1, v: 0.2 }, { u: 0.3, v: 0.4 }], '#f55');
    s.addNote(3, 0.5, 0.6, 'check the matte');
    s.addBookmark(12, 'cut');
    const r = ReviewStore.parse(s.serialize());
    assert.deepEqual(r.marks.map(({ id, ...m }) => m), s.marks.map(({ id, ...m }) => m));
    assert.equal(JSON.parse(s.serialize()).version, REVIEW_VERSION);
});

test('loading never throws: junk, wrong version, oversized and hostile values give an empty store', () => {
    for (const bad of [null, undefined, '', 'not json', '{"version":99,"marks":[]}', '{"version":1}', '[]', 42,
        { version: 1, marks: 'x' }, JSON.stringify({ version: 1, marks: [null, 5, { kind: 'evil' }] }),
        'x'.repeat(LIMITS.maxJsonChars + 1)]) {
        const r = ReviewStore.parse(bad);
        assert.ok(r instanceof ReviewStore);
        assert.equal(r.marks.length, 0, String(bad).slice(0, 30));
    }
});

test('loading re-validates each mark and caps the count', () => {
    const marks = [
        { kind: 'stroke', frame: 1, pts: [{ u: 0, v: 0 }, { u: 1, v: 1 }], color: '#fff' },
        { kind: 'stroke', frame: -3, pts: [{ u: 0, v: 0 }, { u: 1, v: 1 }] },       // bad frame
        { kind: 'note', frame: 1, u: 9, v: 9, text: 'ok' },                          // clamped
        { kind: 'note', frame: 1, u: 0.5, v: 0.5, text: '' },                        // no text
    ];
    const r = ReviewStore.parse({ version: 1, marks });
    assert.equal(r.marks.length, 2);
    assert.equal(r.marks[1].u, 1);
    const many = Array.from({ length: 2000 }, (_, i) => ({ kind: 'bookmark', frame: i }));
    assert.equal(ReviewStore.parse({ version: 1, marks: many }).marks.length, LIMITS.maxMarks);
});

test('serialize refuses output over the size cap instead of writing it', () => {
    const s = new ReviewStore();
    for (let i = 0; i < 200; i++) s.addNote(i, 0.5, 0.5, 'y'.repeat(LIMITS.maxNoteChars));
    // 200 full notes is ~110 kB: under the cap, so it is written
    assert.ok(s.serialize());
    for (let i = 200; i < LIMITS.maxMarks; i++) s.addNote(i, 0.5, 0.5, 'y'.repeat(LIMITS.maxNoteChars));
    assert.equal(s.serialize(), null);   // 500 full notes is over: refused, not truncated
    const big = new ReviewStore();
    for (let i = 0; i < 150; i++) big.addStroke(i, Array.from({ length: LIMITS.maxPointsPerStroke }, (_, k) => ({ u: k / 400, v: 0.123456789 })));
    assert.equal(big.serialize(), null);                               // too big: caller keeps the old saved copy
});

test('the marks are drawn, not baked: no pixel buffer is touched', () => {
    const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_review.js'), 'utf8');
    assert.doesNotMatch(src, /getImageData|putImageData|\.data\[/);
});

test('stroke points are stored to 1e-4 of the frame, so saved marks stay small', () => {
    const s = new ReviewStore();
    const m = s.addStroke(0, [{ u: 0.123456789, v: 0.987654321 }, { u: 0.5, v: 0.5 }]);
    assert.deepEqual(m.pts[0], { u: 0.1235, v: 0.9877 });
    assert.ok(LIMITS.maxJsonChars <= 150000);
});
