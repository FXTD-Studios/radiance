/**
 * A video's frame rate, read from the times of its frames.
 *
 * The viewer assumed 25 frames/s for every video loaded straight into it and
 * stepped by float seconds, so the frame counter, the scrubbers and the
 * picture drifted apart. It now reads the rate from requestVideoFrameCallback
 * media times, which containers keep to the millisecond: 24 fps comes back
 * as 0, 42, 83, 125 ms. RadianceViewer._frameRateFromTimes turns those into
 * a rate; viewer_session.test.mjs checks the whole path on a real webm.
 *
 * Run: node --test js/tests/video_frame_rate.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.search(new RegExp(`\\n    (static )?${name}\\(`));
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    return rest.slice(0, end) + '\n    }';
}

const RadianceViewer = new Function(`return class RadianceViewer { ${methodSource('_frameRateFromTimes')} }`)();
const rateOf = (times) => RadianceViewer._frameRateFromTimes(times);

/** Frame start times at `rate`, rounded to the millisecond as a webm stores them. */
const frames = (rate, indices) => indices.map((k) => Math.round((k * 1000) / rate) / 1000);
const run = (from, n) => Array.from({ length: n }, (_, i) => from + i);

test('whole-number rates from a few frames', () => {
    for (const rate of [24, 25, 30, 48, 50, 60]) {
        assert.equal(rateOf(frames(rate, run(0, 9))), rate, `${rate} fps`);
    }
});

test('a skipped frame does not change the rate', () => {
    assert.equal(rateOf(frames(24, [0, 1, 2, 4, 5, 6, 8, 9])), 24);
    assert.equal(rateOf(frames(30, [3, 4, 6, 7, 8, 9, 10])), 30);
});

test('a frame far from the start tells NTSC rates from whole ones', () => {
    // A third of a second of millisecond stamps fits both; ten seconds do not.
    assert.equal(rateOf(frames(23.976, [...run(0, 9), 240])), 23.976);
    assert.equal(rateOf(frames(24, [...run(0, 9), 240])), 24);
    assert.equal(rateOf(frames(29.97, [...run(0, 9), 300])), 29.97);
    assert.equal(rateOf(frames(30, [...run(0, 9), 300])), 30);
});

test('a short clip with no far frame reads as the whole-number rate', () => {
    assert.equal(rateOf(frames(24, run(0, 9))), 24);
});

test('a rate off the standard list is kept as measured', () => {
    assert.equal(rateOf(frames(12.5, run(0, 9))), 12.5);
});

test('too few frames give no rate', () => {
    assert.equal(rateOf([0, 0.042]), null);
    assert.equal(rateOf([]), null);
});
