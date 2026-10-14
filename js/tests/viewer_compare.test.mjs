import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mapCompareFrame, compareGeometry } from '../radiance_compare.js';

test('frames pair by index when rates are unknown or equal, as before', () => {
    assert.deepEqual(mapCompareFrame(5, 10), { index: 5, held: false });
    assert.deepEqual(mapCompareFrame(5, 10, { aFps: 24, bFps: 24 }), { index: 5, held: false });
    assert.deepEqual(mapCompareFrame(5, 10, { aFps: 23.976, bFps: 23.98 }), { index: 5, held: false });  // within tolerance
    assert.deepEqual(mapCompareFrame(5, 10, { aFps: 24 }), { index: 5, held: false });                    // B rate unknown
});

test('past the end of B the last frame is held and flagged', () => {
    assert.deepEqual(mapCompareFrame(9, 10), { index: 9, held: false });
    assert.deepEqual(mapCompareFrame(10, 10), { index: 9, held: true });
    assert.deepEqual(mapCompareFrame(500, 10), { index: 9, held: true });
});

test('different rates pair by time, not by frame number', () => {
    // A at 30 fps, B at 24: A frame 30 is one second in, which is B frame 24
    assert.deepEqual(mapCompareFrame(30, 100, { aFps: 30, bFps: 24 }), { index: 24, held: false });
    assert.deepEqual(mapCompareFrame(0, 100, { aFps: 30, bFps: 24 }), { index: 0, held: false });
    // A at 24, B at 48: B runs twice as fast through its frames
    assert.deepEqual(mapCompareFrame(12, 100, { aFps: 24, bFps: 48 }), { index: 24, held: false });
    // and the end is still detected on the mapped index
    assert.deepEqual(mapCompareFrame(30, 20, { aFps: 30, bFps: 24 }), { index: 19, held: true });
});

test('empty, negative and junk input', () => {
    assert.deepEqual(mapCompareFrame(3, 0), { index: -1, held: false });
    assert.deepEqual(mapCompareFrame(3, -2), { index: -1, held: false });
    assert.deepEqual(mapCompareFrame(-4, 10), { index: 0, held: false });
    assert.deepEqual(mapCompareFrame(NaN, 10), { index: 0, held: false });
    assert.deepEqual(mapCompareFrame(5, 10, { aFps: 0, bFps: 24 }), { index: 5, held: false });
    assert.deepEqual(mapCompareFrame(5, 10, { aFps: -1, bFps: NaN }), { index: 5, held: false });
});

test('equal sizes and unknown sizes say nothing', () => {
    assert.equal(compareGeometry({ width: 1920, height: 1080 }, { width: 1920, height: 1080 }).note, null);
    assert.equal(compareGeometry({ width: 1920, height: 1080 }, null).note, null);
    assert.equal(compareGeometry({ width: 0, height: 0 }, { width: 10, height: 10 }).note, null);
    assert.equal(compareGeometry({ width: 1920, height: 1080 }, { width: 1920, height: 1080 }).match, true);
});

test('a size mismatch is named, with both sizes and how B is placed', () => {
    const a = { width: 3840, height: 2160 }, b = { width: 1920, height: 1080 };
    const px = compareGeometry(a, b, 'pixel');
    assert.equal(px.match, false);
    assert.match(px.note, /B is 1920×1080, A is 3840×2160/);
    assert.match(px.note, /pixel for pixel, centred/);
    assert.match(compareGeometry(a, b, 'fit').note, /scaled to fit A/);
});

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

test('the viewer uses the mapping and shows a held B and a size mismatch', () => {
    assert.match(src, /_mapCompareFrame\(idx, n, \{ aFps: this\.frameRate, bFps: this\.compareFrameRate \}\)/);
    assert.match(src, /\(ended at \$\{this\._compareCount\}, held\)/);
    assert.match(src, /_compareGeometryNote\(\) \{/);
    assert.doesNotMatch(src, /const i = Math\.min\(idx, n - 1\);\n\s+const img = list\[i\] \|\| null, hdr = floats\[i\]/,
        'the silent clamp must be gone');
});
