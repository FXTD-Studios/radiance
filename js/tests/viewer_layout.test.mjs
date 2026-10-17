// FLUX 3 layout boxes: UV boxes to the table FLUX 3 Image reads.
// Run: node --test js/tests/viewer_layout.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    boxFromCorners, parseStoredBoxes, serializeBoxes, layoutRows, layoutJSON, boxesFromLayout, undescribed, LAYOUT_LIMITS,
} from '../radiance_layout.js';

test('a box from two corners, either way round, clamped to the picture', () => {
    assert.deepEqual(boxFromCorners({ u: 0.8, v: 0.9 }, { u: 0.2, v: 0.1 }, 'sky'), { x: 0.2, y: 0.1, w: 0.6, h: 0.8, desc: 'sky' });
    assert.deepEqual(boxFromCorners({ u: -0.5, v: 0.5 }, { u: 0.5, v: 1.7 }), { x: 0, y: 0.5, w: 0.5, h: 0.5, desc: '' });
    assert.equal(boxFromCorners({ u: 0.5, v: 0.5 }, { u: 0.505, v: 0.9 }), null, 'a click is not a box');
    assert.equal(boxFromCorners({ u: NaN, v: 0 }, { u: 1, v: 1 }), null);
});

test('rows are what ComfyUI\'s FLUX 3 Image Layout Prompt writes: [ymin, xmin, ymax, xmax] on 0-1000', () => {
    const boxes = [{ x: 0, y: 0, w: 1, h: 0.4, desc: ' a stormy sky ' }, { x: 0.3, y: 0.45, w: 0.4, h: 0.53, desc: 'a lighthouse' }];
    assert.deepEqual(layoutRows(boxes), [
        { id: 'obj_1', bbox: [0, 0, 400, 1000], desc: 'a stormy sky' },
        { id: 'obj_2', bbox: [450, 300, 980, 700], desc: 'a lighthouse' },
    ]);
    assert.equal(layoutJSON(boxes), JSON.stringify(layoutRows(boxes)));
    assert.equal(layoutJSON([]), '', 'no boxes clears the Prompt instead of sending []');
});

test('a table reads back into the same boxes', () => {
    const boxes = [{ x: 0.1, y: 0.2, w: 0.3, h: 0.4, desc: 'a cat' }];
    assert.deepEqual(boxesFromLayout(layoutJSON(boxes)), boxes);
    assert.deepEqual(boxesFromLayout('not json'), []);
    assert.deepEqual(boxesFromLayout('[{"bbox":[1,2,3]}]'), []);
});

test('stored boxes survive a round trip and bad entries are dropped', () => {
    const boxes = [{ x: 0.1, y: 0.1, w: 0.2, h: 0.2, desc: 'a' }, { x: 'x' }, null, { x: 0.5, y: 0.5, w: 0.001, h: 0.4 }];
    assert.deepEqual(parseStoredBoxes(serializeBoxes(boxes)), [{ x: 0.1, y: 0.1, w: 0.2, h: 0.2, desc: 'a' }]);
    assert.deepEqual(parseStoredBoxes('{oops'), []);
    assert.deepEqual(parseStoredBoxes(undefined), []);
    const many = Array.from({ length: 50 }, () => ({ x: 0, y: 0, w: 0.5, h: 0.5 }));
    assert.equal(parseStoredBoxes(JSON.stringify(many)).length, LAYOUT_LIMITS.maxBoxes);
});

test('boxes with no description are named, so the Viewer can warn', () => {
    assert.deepEqual(undescribed([{ x: 0, y: 0, w: 0.5, h: 0.5, desc: 'x' }, { x: 0, y: 0, w: 0.5, h: 0.5, desc: '  ' }]), ['obj_2']);
});
