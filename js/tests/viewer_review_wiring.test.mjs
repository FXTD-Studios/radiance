/**
 * Viewer review-mark wiring. Strokes are stored per frame in image UV and
 * persisted in node.properties.radiance_viewer_review. Methods are lifted from
 * radiance_viewer.js (it cannot be imported in Node). The drawing itself and
 * the Alt+drag gesture need a browser and are not covered here.
 *
 * Run: node --test js/tests/viewer_review_wiring.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { ReviewStore } from '../radiance_review.js';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    return rest.slice(0, rest.search(/\n    \}\n/)) + '\n    }';
}
const Proto = new Function('_ReviewStore',
    `return class { ${['_reviewView', '_reviewStrokesForView', '_persistReview'].map(methodSource).join('\n')} }`)(ReviewStore);

function viewer(props = {}) {
    const v = new Proto();
    Object.assign(v, { zoom: 2, panX: 10, panY: 20, imageWidth: 100, imageHeight: 50, currentFrame: 0,
        node: { properties: props }, logs: [], _termLog(l, m) { this.logs.push([l, m]); } });
    v._review = ReviewStore.parse(props.radiance_viewer_review);
    return v;
}

test('strokes belong to a frame and are drawn only there', () => {
    const v = viewer();
    v._review.addStroke(3, [{ u: 0, v: 0 }, { u: 1, v: 1 }]);
    v.currentFrame = 2;
    assert.equal(v._reviewStrokesForView().length, 0);
    v.currentFrame = 3;
    assert.equal(v._reviewStrokesForView().length, 1);
});

test('strokes follow zoom and pan', () => {
    const v = viewer();
    v._review.addStroke(0, [{ u: 0, v: 0 }, { u: 1, v: 1 }]);
    const a = v._reviewStrokesForView()[0].pts;
    assert.deepEqual(a[0], { x: 10, y: 20 });
    assert.deepEqual(a[1], { x: 210, y: 120 });
    v.zoom = 1; v.panX = 0; v.panY = 0;
    assert.deepEqual(v._reviewStrokesForView()[0].pts[1], { x: 100, y: 50 });
});

test('a live stroke is drawn from UV points too', () => {
    const v = viewer();
    const out = v._reviewStrokesForView([{ color: '#fff', uv: [{ u: 0, v: 0 }, { u: 0.5, v: 0.5 }] }]);
    assert.equal(out.length, 1);
});

test('persist writes versioned JSON and a reload restores it', () => {
    const v = viewer();
    v._review.addStroke(1, [{ u: 0.1, v: 0.1 }, { u: 0.2, v: 0.2 }]);
    v._persistReview();
    const saved = v.node.properties.radiance_viewer_review;
    assert.equal(typeof saved, 'string');
    const v2 = viewer({ radiance_viewer_review: saved });
    assert.equal(v2._review.forFrame(1, 'stroke').length, 1);
});

test('an oversized set keeps the previous saved copy and warns', () => {
    const v = viewer({ radiance_viewer_review: 'previous' });
    v._review.serialize = () => null;
    v._persistReview();
    assert.equal(v.node.properties.radiance_viewer_review, 'previous');
    assert.equal(v.logs[0][0], 'warn');
});

test('corrupt saved value loads as empty', () => {
    assert.equal(viewer({ radiance_viewer_review: '{nope' })._review.marks.length, 0);
});

test('the old canvas-pixel annotation state is gone', () => {
    assert.ok(!/_annotationLines/.test(src));
});

test('a live stroke shows only on the frame it was started on', () => {
    const v = viewer();
    const line = { frame: 2, color: '#fff', uv: [{ u: 0, v: 0 }, { u: 0.5, v: 0.5 }] };
    v.currentFrame = 3;
    assert.equal(v._reviewStrokesForView([line]).length, 0);
    v.currentFrame = 2;
    assert.equal(v._reviewStrokesForView([line]).length, 1);
});

test('a stroke is stored on the frame where it started, and Shift+Alt+click clears one frame', () => {
    assert.match(src, /addStroke\(this\._annotationCurrentLine\.frame,/);
    assert.match(src, /this\._review\.clearFrame\(f, 'stroke'\)/);
    assert.doesNotMatch(src, /this\._review\.clear\('stroke'\)/);
});
