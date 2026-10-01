/**
 * Viewer timeline: a small mark for each frame the paging window holds, paler
 * while it loads, above the simple bar's slider and on the sequence dock's V1
 * track (the dev asked for Nuke-like cache marks).
 *
 * Run: node --test js/tests/cache_marks.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.search(new RegExp(`\\n    ${name}\\(`));
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n\n/);
    assert.notEqual(end, -1, `could not find the end of ${name}`);
    return rest.slice(0, end) + '\n    }';
}

globalThis.window = { devicePixelRatio: 1 };
const Viewer = new Function(`return class { ${methodSource('_drawCacheMarks')} }`)();

function fakeCanvas() {
    const rects = [];
    const ctx = {
        fillStyle: '', shadowColor: '', shadowBlur: 0,
        clearRect() {},
        fillRect(x, y, w, h) { rects.push({ x, y, w, h, style: this.fillStyle }); },
    };
    return { width: 0, height: 0, rects, getContext: () => ctx };
}

test('one mark per frame held, at the thumb position of that frame, paler while loading', () => {
    const canvas = fakeCanvas();
    const v = Object.assign(new Viewer(), {
        totalFrames: 101,
        _frameWindow: { cacheState: () => ({ held: [0, 50, 100], loading: [51] }) },
        // An 816 px slider whose 16 px thumb puts frame 0 at 8 px and frame 100 at 808 px.
        _cacheMarks: [{ canvas, geo: { w: 816, h: 20, x0: 8, span: 800, top: 1 } }],
    });
    v._drawCacheMarks();
    const held = canvas.rects.filter((r) => r.style.endsWith(',1)')).map((r) => r.x);
    const loading = canvas.rects.filter((r) => r.style.endsWith(',0.35)')).map((r) => r.x);
    assert.deepEqual(held, [8, 408, 808]);
    assert.deepEqual(loading, [416]);
    assert.ok(canvas.rects.every((r) => r.y === 1 && r.h === 4), 'the marks are 4 px from their top');
    assert.equal(canvas.width, 816);
});

test('nothing is drawn for a slider not laid out, or a single image', () => {
    const hidden = fakeCanvas();
    const shown = fakeCanvas();
    const v = Object.assign(new Viewer(), {
        totalFrames: 1,
        _frameWindow: { cacheState: () => ({ held: [0], loading: [] }) },
        _cacheMarks: [
            { canvas: hidden, geo: { w: 0, h: 0, x0: 0, span: 0, top: 0 } },
            { canvas: shown, geo: { w: 320, h: 60, x0: 0, span: 320, top: 30 } },
        ],
    });
    v._drawCacheMarks();
    assert.equal(hidden.rects.length, 0);
    assert.equal(shown.rects.length, 0, 'a single image has no timeline to mark');
});
