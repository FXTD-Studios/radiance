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
import test, { afterEach } from 'node:test';
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

// The byte bound is shared by every window with a sequence (one per viewer),
// so a window a test leaves holding frames would shrink the next test's share.
const made = [];
afterEach(() => { while (made.length) made.pop().clear(); });

function makeWindow(stub, opts = {}) {
    return track(new RadianceFrameWindow({
        windowSize: 16,
        concurrency: 4,
        maxBytes: 1024 * 1024 * 1024,
        load: (entry, idx) => stub.load(entry, idx),
        onReady: (idx, payload) => stub.track(idx, payload),
        onEvict: (idx) => stub.release(idx),
        ...opts,
    }));
}

function track(win) { made.push(win); return win; }

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

test('a looping range reads its in point ahead of the wrap', async () => {
    // Playback stopped at the loop to load the in point: it was outside the
    // window until the playhead got there.
    const stub = stubLoader();
    const win = makeWindow(stub);
    win.setSequence(sequence(200), 0);
    win.loop = { start: 0, end: 199 };
    win.setPlayhead(195);
    await drain(win);
    for (let f = 0; f < 6; f++) assert.ok(win.has(f), `frame ${f}, past the out point, is not loaded`);
    assert.ok(win.inSpan(0) && win.isWindowReady());
});

test('under the byte ceiling the loop keeps its in point and lets played frames go', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub, { maxBytes: 13 * FRAME_BYTES });   // what a 1080p clip leaves room for
    win.setSequence(sequence(200), 0);
    win.loop = { start: 0, end: 199 };
    for (let f = 180; f <= 197; f++) { win.setPlayhead(f); await drain(win); }
    assert.ok(win.has(0) && win.has(1), 'the in point was evicted before frames already played');
    assert.ok(!win.has(185), 'a frame already played is still held');
});

test('cacheState reports the frames held and the frames loading', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub, { concurrency: 2 });
    win.setSequence(sequence(100), 0);
    await Promise.resolve();
    await Promise.resolve();
    const loading = win.cacheState().loading;
    assert.equal(loading.length, 2, 'the loads in flight are the frames loading');
    await drain(win);
    const { held, loading: after } = win.cacheState();
    assert.deepEqual(after, []);
    assert.deepEqual(held.sort((a, b) => a - b), [...Array(16).keys()]);
});

test('without a loop the window stops at the last frame', async () => {
    const stub = stubLoader();
    const win = makeWindow(stub);
    win.setSequence(sequence(200), 0);
    win.setPlayhead(195);
    await drain(win);
    assert.ok(!win.has(0) && !win.inSpan(0));
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
    // A paged compare frame is one more full-size bitmap.
    assert.equal(measureFramePayload({ hdr, compare: { width: 10, height: 10 } }), 400 + 200 + 400);
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

test('a compare sequence is paged with its frame too', () => {
    // It was the last unbounded loader: every compare frame requested at once,
    // full size, and held for the whole run. A still is still loaded whole.
    const src = read('radiance_viewer.js');
    assert.match(src, /if \(!pageCompare\) compareImages\.forEach/,
        'a compare sequence as long as the clip is loaded all at once again');
    assert.match(src, /payload\.compare = compare/,
        'compare is not carried on the paged frame payload');
    assert.match(src, /this\.frameCompareImages\[idx\] = null/,
        'compare is not released when its frame is evicted');
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

test('the simple bar is enabled as soon as a sequence is installed', () => {
    // It waited for a full window, which a 1080p clip never reaches (13 of 16
    // frames fit the byte budget): Play and the slider stayed off in Simple
    // mode until the user switched to Advanced and back.
    const src = read('radiance_viewer.js');
    const start = src.indexOf('    _installFrameWindow(');
    const body = src.slice(start, src.indexOf('\n    }\n', start));
    assert.match(body, /setSequence\([^)]*\);[\s\S]*this\._syncSimpleTransport/,
        'installing a sequence does not update the simple bar');
});

// ── sized by bytes, and no thrash (4K and 8K) ───────────────────────────────

/** A loader whose frames weigh `bytes` each, without allocating them. */
function sizedLoader(bytesOf) {
    const loads = [];
    return {
        loads,
        load(entry, idx) {
            loads.push(idx);
            return Promise.resolve({ idx, hdr: { data: { byteLength: bytesOf(idx) } } });
        },
    };
}

test('once a frame lands the window holds what fits, and loads about one frame per step', async () => {
    // Measured on the old window (768 MB, 16 frames): 1.8, 3.8, 13 and 15
    // loads per playhead step at 50, 58, 199 and 796 MB a frame. It queued 16
    // frames whatever they weighed, evicted the ones over the budget as they
    // landed, and asked for them again on the next step.
    const MB = 1024 * 1024;
    for (const size of [50, 58, 199, 796]) {
        const stub = sizedLoader(() => size * MB);
        const win = track(new RadianceFrameWindow({
            windowSize: 16, maxBytes: 768 * MB, concurrency: 4, load: (e, i) => stub.load(e, i),
        }));
        win.setSequence(sequence(200), 0);
        await drain(win);
        const fits = Math.min(16, Math.max(1, Math.floor(768 / size)));
        assert.equal(win.span().end - win.span().start + 1, fits,
            `${size} MB frames: the span is not what fits the budget`);
        assert.ok(win.retainedBytes <= Math.max(768, size) * MB, "over the budget (one frame is always kept)");
        const before = stub.loads.length;
        for (let f = 1; f <= 24; f++) { win.setPlayhead(f); await drain(win); }
        const perStep = (stub.loads.length - before) / 24;
        assert.ok(perStep <= 1.0, `${size} MB frames: ${perStep.toFixed(2)} loads per playhead step`);
        assert.ok(win.has(24), 'the playhead frame is not held');
        win.clear();
    }
});

test('a frame the byte bound just evicted is not queued again until the playhead moves', async () => {
    // Frames of uneven size: the span is sized from the average, so a heavy
    // frame can still push one out. It must stay out at this playhead.
    const MB = 1024 * 1024;
    const stub = sizedLoader((i) => (i % 2 ? 300 : 20) * MB);
    const win = track(new RadianceFrameWindow({
        windowSize: 16, maxBytes: 700 * MB, concurrency: 4, load: (e, i) => stub.load(e, i),
    }));
    win.setSequence(sequence(100), 10);
    await drain(win);
    const before = stub.loads.length;
    for (let i = 0; i < 20; i++) { win.setPlayhead(10); await drain(win); }
    assert.equal(stub.loads.length, before, 'frames were fetched again at an unchanged playhead');
    assert.ok(win.retainedBytes <= 700 * MB);
    win.setPlayhead(11);
    await drain(win);
    assert.ok(win.has(11));
});

test('the byte bound is shared by every window on the page, not given to each', async () => {
    // One window per viewer. Three viewers each kept the full 768 MB.
    const stubs = [stubLoader(), stubLoader(), stubLoader()];
    const wins = stubs.map((stub) => makeWindow(stub, { windowSize: 64, maxBytes: 12 * FRAME_BYTES }));
    wins[0].setSequence(sequence(100), 0);
    await drain(wins[0]);
    assert.equal(wins[0].residentCount, 12, 'alone, a window has the whole budget');

    wins[1].setSequence(sequence(100), 0);
    wins[2].setSequence(sequence(100), 0);
    for (const w of wins) await drain(w);
    const total = wins.reduce((n, w) => n + w.retainedBytes, 0);
    assert.ok(total <= 12 * FRAME_BYTES, `three windows hold ${total / FRAME_BYTES} frames over a 12 frame bound`);
    for (const w of wins) assert.equal(w.residentCount, 4, 'the budget is not split evenly');

    // A viewer closing hands its share back.
    wins[2].clear();
    wins[1].clear();
    await drain(wins[0]);
    assert.equal(wins[0].residentCount, 12, 'the remaining window did not grow back');
});

test('measureFramePayload does not decode an fp16 frame to weigh it', () => {
    let decoded = 0;
    const hdr = { fp16data: new Uint16Array(100) };
    Object.defineProperty(hdr, 'data', { enumerable: false, get() { decoded++; return new Float32Array(100); } });
    assert.equal(measureFramePayload({ hdr }), 200, 'an fp16 frame is its half floats');
    assert.equal(decoded, 0, 'weighing the frame decoded it');
});

test('a paged float B is charged to its frame', () => {
    const hdr = { fp16data: new Uint16Array(100) };
    assert.equal(measureFramePayload({ hdr, compareHdr: { fp16data: new Uint16Array(50) } }), 200 + 100);
});
