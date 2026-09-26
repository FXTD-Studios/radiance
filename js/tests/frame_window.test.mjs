/**
 * The viewer must survive a shot it cannot hold.
 *
 * The sequence loader used to be `mainImages.forEach(...)` in
 * radiance_viewer.js: one `new Image()` and one `fetch(hdrUrl)` per frame,
 * fired the instant a result arrived, with no concurrency limit, storing every
 * decoded buffer at `frameHDRData[idx]`. Those arrays were reset only when a
 * new generation started and were never evicted from. The GPU texture LRU in
 * radiance_webgl.js is bounded, which is exactly what hid this: the textures
 * were capped, the source pixel arrays behind them were not.
 *
 * Measured on the old code, a 300-frame 1080p RGBA shot meant 300 simultaneous
 * fetches and 300 x 33 MB, roughly 10 GB of Float32Array in one tab. 10,000
 * frames is roughly 330 GB: the tab died during load and the bounded texture
 * cache never got a chance to help.
 *
 * These tests measure the bound rather than inspect it. The stub loader
 * allocates a real Float32Array per frame and registers it; the window's own
 * eviction hook deregisters it. What is asserted is how many of those
 * allocations are still referenced after paging through 10,000 frames. An
 * unbounded implementation cannot pass: at the sizes used here it would have
 * to hold 10 GB.
 *
 * Run: node --test js/tests/frame_window.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
    RadianceFrameWindow,
    DEFAULT_FRAME_WINDOW,
    DEFAULT_FETCH_CONCURRENCY,
    measureFramePayload,
} from '../radiance_frame_window.js';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');

/** Bytes per simulated frame. Real allocation, deliberately. */
const FRAME_BYTES = 1024 * 1024;

/**
 * A stubbed fetch+decode that allocates a real buffer per frame and reports
 * every buffer still referenced by the window.
 */
function stubLoader() {
    const live = new Map();          // idx -> buffer
    let peakInFlight = 0;
    let inFlight = 0;
    let loads = 0;

    return {
        live,
        get liveBytes() {
            let n = 0;
            for (const buf of live.values()) n += buf.byteLength;
            return n;
        },
        get peakInFlight() { return peakInFlight; },
        get loads() { return loads; },
        load(entry, idx) {
            loads++;
            inFlight++;
            if (inFlight > peakInFlight) peakInFlight = inFlight;
            return Promise.resolve().then(() => {
                inFlight--;
                const data = new Float32Array(FRAME_BYTES / 4);
                data[0] = idx;                     // so the buffer is really written
                return { idx, hdr: { data, width: 512, height: 512, channels: 4 } };
            });
        },
        track(idx, payload) { live.set(idx, payload.hdr.data); },
        release(idx) { live.delete(idx); },
    };
}

function sequence(n) {
    return Array.from({ length: n }, (_, i) => ({
        filename: `frame_${i}.png`,
        hdr_sidecar: `frame_${i}.rhdr`,
        frame: i,
    }));
}

/** Let every queued load settle. The stub resolves on the microtask queue. */
async function drain(win, maxTurns = 500000) {
    let turns = 0;
    while ((win.inFlight > 0 || win.queuedCount > 0) && turns++ < maxTurns) {
        await Promise.resolve();
    }
    assert.ok(turns < maxTurns, 'window never drained — a load was lost');
}

function makeWindow(stub, opts = {}) {
    return new RadianceFrameWindow({
        windowSize: 16,
        concurrency: 4,
        maxBytes: 1024 * 1024 * 1024,
        load: (entry, idx) => stub.load(entry, idx),
        onReady: (idx, payload) => stub.track(idx, payload),
        onEvict: (idx) => stub.release(idx),
        ...opts,
    });
}

// ── the measurement ─────────────────────────────────────────────────────────

test('retained buffers stay at the window across a 10,000 frame scrub', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub);
    const TOTAL = 10000;

    win.setSequence(sequence(TOTAL), 0);
    await drain(win);

    const afterLoad = stub.live.size;
    assert.ok(afterLoad <= win.windowSize,
        `arrival alone retained ${afterLoad} frames; the old loader retained all ${TOTAL}`);

    // Scrub the whole shot, one frame at a time, as playback does.
    const samples = [];
    for (let f = 0; f < TOTAL; f++) {
        win.setPlayhead(f);
        if (f % 250 === 0) {
            await drain(win);
            samples.push(stub.live.size);
        }
    }
    await drain(win);

    // The number that matters: what is still allocated at the far end of a
    // 10,000-frame shot, having touched every frame in it.
    assert.ok(stub.live.size <= win.windowSize,
        `retained ${stub.live.size} buffers after ${TOTAL} frames, window is ${win.windowSize}`);
    assert.ok(stub.liveBytes <= win.windowSize * FRAME_BYTES,
        `retained ${stub.liveBytes} bytes, ceiling is ${win.windowSize * FRAME_BYTES}`);

    // Flat, not growing: the reading at frame 250 and the reading at frame
    // 9750 are the same. On the old code this series was the frame index.
    const first = samples[1];
    for (const s of samples.slice(1)) {
        assert.equal(s, first, `retention drifted with sequence position: ${samples.join(',')}`);
    }

    // And the window's own accounting agrees with the census.
    assert.equal(win.residentCount, stub.live.size);
    assert.ok(win.peakResident <= win.windowSize,
        `peak residency ${win.peakResident} exceeded the window ${win.windowSize}`);
});

test('a longer sequence does not retain more than a shorter one', async () => {
    const readings = [];
    for (const total of [100, 1000, 10000]) {
        const stub = stubLoader();
        const win = makeWindow(stub);
        win.setSequence(sequence(total), 0);
        await drain(win);
        for (let f = 0; f < total; f += 7) { win.setPlayhead(f); }
        await drain(win);
        readings.push(stub.live.size);
    }
    assert.equal(readings[0], readings[1], `retention grew from 100 to 1000 frames: ${readings}`);
    assert.equal(readings[1], readings[2], `retention grew from 1000 to 10000 frames: ${readings}`);
});

// ── the concurrency bound ───────────────────────────────────────────────────

test('at most `concurrency` loads are ever in flight', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub, { concurrency: 4 });
    win.setSequence(sequence(3000), 0);
    await drain(win);
    for (let f = 0; f < 3000; f += 3) win.setPlayhead(f);
    await drain(win);

    assert.ok(stub.peakInFlight <= 4,
        `${stub.peakInFlight} loads were in flight at once; the bound is 4`);
    assert.ok(win.peakInFlight <= 4);
});

test('arrival of a 300 frame result does not open 300 sockets', async () => {
    // The measured shape of the original defect: a 300-frame 1080p shot meant
    // 300 simultaneous fetches the moment the result landed.
    const stub = stubLoader();
    const win = makeWindow(stub);
    win.setSequence(sequence(300), 0);
    // Before anything settles, only the concurrency slots are occupied.
    assert.ok(win.inFlight <= win.concurrency,
        `${win.inFlight} fetches opened immediately on arrival`);
    await drain(win);
    assert.ok(stub.loads <= win.windowSize,
        `arrival fetched ${stub.loads} of 300 frames; only the window should be read`);
});

// ── the byte ceiling ────────────────────────────────────────────────────────

test('the byte ceiling binds before the frame count on large frames', async () => {
    // 4K RGBA fp32 is 141 MB a frame, so 16 frames is 2.2 GB. The frame count
    // alone is not a memory bound; maxBytes is.
    const stub = stubLoader();
    const win = makeWindow(stub, { windowSize: 16, maxBytes: 4 * FRAME_BYTES });
    win.setSequence(sequence(500), 0);
    await drain(win);
    for (let f = 0; f < 500; f++) win.setPlayhead(f);
    await drain(win);

    assert.ok(win.retainedBytes <= 4 * FRAME_BYTES,
        `retained ${win.retainedBytes} bytes over a ${4 * FRAME_BYTES} byte ceiling`);
    assert.ok(stub.live.size <= 4,
        `${stub.live.size} buffers held under a 4-frame byte ceiling`);
});

test('the frame on screen is never evicted', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub, { windowSize: 4, maxBytes: FRAME_BYTES });  // ceiling below one frame
    win.setSequence(sequence(50), 0);
    await drain(win);
    for (const f of [10, 40, 3, 25]) {
        win.setPlayhead(f);
        await win.ensure(f);
        await drain(win);
        assert.ok(win.has(f), `playhead frame ${f} was evicted out from under the display`);
    }
});

// ── scrubbing outside the window ────────────────────────────────────────────

test('a jump outside the window pages the frame in on demand', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub);
    win.setSequence(sequence(5000), 0);
    await drain(win);
    assert.ok(!win.has(4200), 'frame 4200 should not be resident after loading frame 0');

    win.setPlayhead(4200);
    const payload = await win.ensure(4200);
    assert.ok(payload, 'ensure() did not deliver the scrubbed-to frame');
    assert.equal(payload.idx, 4200);
    await drain(win);
    assert.ok(win.has(4200));
    assert.ok(stub.live.size <= win.windowSize);
});

test('a new sequence releases the previous one', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub);
    win.setSequence(sequence(200), 0);
    await drain(win);
    assert.ok(stub.live.size > 0);
    win.setSequence(sequence(200), 0);
    // clear() ran the evict hook for every held frame before refilling.
    await drain(win);
    assert.ok(stub.live.size <= win.windowSize);
    win.clear();
    assert.equal(stub.live.size, 0, 'clear() left buffers referenced');
    assert.equal(win.retainedBytes, 0);
});

test('isWindowReady does not walk the sequence', async () => {
    // The old _allFramesReady() was O(N) and ran from every frame's onload, so
    // a load cost O(N^2) on the main thread. This one is bounded by the window.
    const stub = stubLoader();
    const win = makeWindow(stub);
    win.setSequence(sequence(10000), 0);
    await drain(win);
    assert.equal(win.isWindowReady(), true);
    win.setPlayhead(9000);
    assert.equal(win.isWindowReady(), false, 'a window it has not loaded yet reads as ready');
    await drain(win);
    assert.equal(win.isWindowReady(), true);
});

test('measureFramePayload charges the float data, the proxy and the brackets', () => {
    assert.equal(measureFramePayload(null), 0);
    const hdr = { data: new Float32Array(100), fp16data: new Uint16Array(100) };
    assert.equal(measureFramePayload({ hdr }), 400 + 200);
    assert.equal(measureFramePayload({ hdr, img: { width: 10, height: 10 } }), 400 + 200 + 400);
    // Brackets are two more decoded bitmaps a frame, and they used to be a
    // second unbounded pass over the sequence.
    assert.equal(
        measureFramePayload({
            hdr,
            img: { width: 10, height: 10 },
            bracketLow: { width: 10, height: 10 },
            bracketHigh: { width: 10, height: 10 },
        }),
        400 + 200 + 400 * 3,
    );
});

test('exposure brackets are paged with their frame, not loaded all at once', () => {
    const src = read('radiance_viewer.js');
    assert.doesNotMatch(src, /bracketImages\.forEach\(\(imgData, idx\) => \{\s*const label/,
        'the unbounded bracket loader is back');
    assert.match(src, /payload\.bracketLow/,
        'brackets are not carried on the paged frame payload');
    assert.match(src, /this\.frameBracketImages\.low\[idx\] = null/,
        'brackets are not released when their frame is evicted');
});

test('z-depth is paged with its frame too', () => {
    // Depth is written at FULL resolution by the node, with no thumbnail cap,
    // and used to be a third unbounded forEach alongside colour and brackets.
    const src = read('radiance_viewer.js');
    assert.doesNotMatch(src, /zdepthImages\.forEach/,
        'the unbounded z-depth loader is back');
    assert.match(src, /payload\.zdepth/,
        'depth is not carried on the paged frame payload');
    assert.match(src, /this\.frameZdepthImages\[idx\] = null/,
        'depth is not released when its frame is evicted');
});

// ── the viewer is wired to it ───────────────────────────────────────────────

test('the viewer pages the sequence instead of loading all of it', () => {
    const src = read('radiance_viewer.js');

    assert.match(src, /from "\.\/radiance_frame_window\.js"/,
        'radiance_viewer.js does not import the bounded window');
    assert.match(src, /_installFrameWindow\(mainImages, currentGen/,
        'the result handler does not route the sequence through the window');
    assert.doesNotMatch(src, /mainImages\.forEach/,
        'the unbounded per-frame fetch+Image loop is back');
});

test('the sequence loader is the only place frames are retained', () => {
    const src = read('radiance_viewer.js');
    // The window nulls these on eviction. Nothing else may write a decoded
    // frame into them, or the bound is only advisory.
    const writes = [...src.matchAll(/(?:viewer|this)\.frameHDRData\[\w+\]\s*=/g)];
    assert.ok(writes.length <= 3,
        `frameHDRData is written from ${writes.length} places; it is owned by the frame window`);
});

test('_allFramesReady no longer walks every frame', () => {
    const src = read('radiance_viewer.js');
    const at = src.indexOf('    _allFramesReady() {');
    assert.ok(at > 0, '_allFramesReady is gone');
    const body = src.slice(at, src.indexOf('\n    }', at));
    assert.doesNotMatch(body, /for \(let i = 0; i < this\.totalFrames; i\+\+\)/,
        '_allFramesReady is O(N) again, and it is called from every frame arrival');
    assert.match(body, /_frameWindow/, '_allFramesReady does not consult the paging window');
});

test('the window bounds are named constants, not magic numbers', () => {
    const src = read('radiance_frame_window.js');
    assert.match(src, /export const DEFAULT_FRAME_WINDOW\b/);
    assert.match(src, /export const DEFAULT_FRAME_WINDOW_BYTES\b/);
    assert.match(src, /export const DEFAULT_FETCH_CONCURRENCY\b/);
    assert.ok(DEFAULT_FRAME_WINDOW > 0 && DEFAULT_FRAME_WINDOW < 128);
    assert.ok(DEFAULT_FETCH_CONCURRENCY > 0 && DEFAULT_FETCH_CONCURRENCY <= 8);
});
