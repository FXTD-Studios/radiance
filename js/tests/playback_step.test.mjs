/**
 * Viewer playback: what the next step is, and that playback never freezes.
 *
 * Found by driving the Viewer in a browser (3.5.0). The readiness check in the
 * playback loop looked at `currentFrame + 1` wrapped to the in point whatever
 * the loop mode, while the step itself honoured ping-pong and play-once. At
 * the end of the range ping-pong and play-once waited for a frame they would
 * never show, and a whole-clip loop longer than the 16-frame paging window
 * waited for a frame 0 that had been paged out: all three froze. J did nothing
 * on a video loaded straight into the Viewer, and the node grew without limit
 * (760 to 5688 px in 3 s) because its widget sized itself from the node.
 *
 * `radiance_viewer.js` cannot be imported in Node (it pulls in ComfyUI's
 * app.js), so the methods under test are lifted from the source and run on a
 * small stand-in viewer with a fake clock and a fake paging window.
 *
 * Run: node --test js/tests/playback_step.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    assert.notEqual(end, -1, `could not find the end of ${name}`);
    return rest.slice(0, end) + '\n    }';
}

const METHODS = ['_nextPlayFrame', '_advance', '_seqPlaybackLoop', 'shuttle', '_videoReverse', '_frameReady', '_syncSequenceAudio'];

// Fake clock and RAF, shared by every stand-in.
let now = 0;
const rafQueue = [];
globalThis.performance = { now: () => now };
globalThis.requestAnimationFrame = (fn) => { rafQueue.push(fn); return rafQueue.length; };
globalThis.cancelAnimationFrame = () => { rafQueue.length = 0; };

const Proto = new Function(`return class { ${METHODS.map(methodSource).join('\n')} }`)();

test('sequence audio follows play, seek, rate, stalls and reverse without starting duplicate plays', async () => {
    const v = viewer();
    let plays = 0;
    const audio = v._sequenceAudio = {
        currentTime: 0, paused: true, playbackRate: 1,
        play() { plays++; this.paused = false; return Promise.resolve(); },
        pause() { this.paused = true; },
    };
    v._sequenceAudioFps = 24;
    v.currentFrame = 24; v.isPlaying = true;
    v._syncSequenceAudio(); v._syncSequenceAudio();
    assert.equal(plays, 1); assert.equal(audio.currentTime, 1);
    v.playbackFps = 48; v._syncSequenceAudio();
    assert.equal(audio.playbackRate, 2);
    v.frameImages[25] = undefined; v._syncSequenceAudio();
    assert.equal(audio.paused, true, 'buffering pauses audio');
    v.frameImages[25] = true; v.playDirection = -1; v._syncSequenceAudio();
    assert.equal(audio.paused, true, 'reverse playback is silent');
    v.currentFrame = 0; v.isPlaying = false; v._syncSequenceAudio();
    assert.equal(audio.currentTime, 0, 'loop/seek returns audio to the playhead');
});

/** Paging window like RadianceFrameWindow: holds `size` frames ahead of the playhead. */
class FakeWindow {
    constructor(viewer, size = 16) { this.v = viewer; this.size = size; }
    span() { const c = this.v.currentFrame; return [c - 2, c + this.size - 3]; }
    inSpan(i) { const [a, b] = this.span(); return i >= a && i <= b; }
    has(i) { return !!this.v.frameImages[i]; }
    ensure(i) { this.v.frameImages[i] = true; return Promise.resolve(); }
    /** Loads the whole span, drops the rest (what the real window does after a seek). */
    settle() {
        for (let i = 0; i < this.v.frameImages.length; i++) this.v.frameImages[i] = this.inSpan(i) || undefined;
    }
}

function viewer({ total = 96, fps = 24, mode = 'loop', inPt = null, outPt = null, paged = false } = {}) {
    const v = new Proto();
    Object.assign(v, {
        totalFrames: total, currentFrame: 0, playbackFps: fps, loopMode: mode, playDirection: 1,
        inPoint: inPt, outPoint: outPt, isPlaying: false, videoMode: false, lastFrameTime: 0,
        frameImages: new Array(total).fill(true), frameHDRData: [], shown: [],
        _range() { return [this.inPoint ?? 0, this.outPoint ?? this.totalFrames - 1]; },
        setFrame(i) { this.currentFrame = i; this.shown.push(i); if (this._frameWindow) this._frameWindow.settle(); },
        _updatePlayBtn() {},
        togglePlayback() {
            this.isPlaying = !this.isPlaying;
            if (this.isPlaying) { this.lastFrameTime = now; this._seqPlaybackLoop(); }
        },
    });
    if (paged) { v._frameWindow = new FakeWindow(v); v._frameWindow.settle(); }
    return v;
}

/** Run the RAF loop for `ms` of fake time in 1 ms ticks (window reloads between ticks). */
function run(v, ms) {
    const end = now + ms;
    while (now < end && rafQueue.length) {
        now += 1;
        const fn = rafQueue.shift();
        fn(now);
    }
    now = end;
}

test('the next step honours in/out, direction and every loop mode', () => {
    const v = viewer({ total: 10 });
    v.currentFrame = 4;
    assert.deepEqual(v._nextPlayFrame(), { frame: 5, dir: 1, stop: false });
    v.currentFrame = 9;
    assert.deepEqual(v._nextPlayFrame(), { frame: 0, dir: 1, stop: false }, 'loop wraps to the in point');
    v.loopMode = 'pingpong';
    assert.deepEqual(v._nextPlayFrame(), { frame: 8, dir: -1, stop: false }, 'ping-pong turns round');
    v.loopMode = 'once';
    assert.equal(v._nextPlayFrame().stop, true, 'play-once stops at the out point');

    const r = viewer({ total: 10, inPt: 3, outPt: 6 });
    r.currentFrame = 6;
    assert.equal(r._nextPlayFrame().frame, 3, 'loop stays inside in/out');
    r.playDirection = -1; r.currentFrame = 3;
    assert.equal(r._nextPlayFrame().frame, 6, 'reverse loop wraps to the out point');
});

test('play-once stops playback instead of waiting', () => {
    rafQueue.length = 0;
    const v = viewer({ total: 10, mode: 'once' });
    v.currentFrame = 7;
    v.togglePlayback();
    run(v, 1000);
    assert.equal(v.currentFrame, 9);
    assert.equal(v.isPlaying, false, 'still "playing" at the last frame');
});

test('ping-pong turns at both ends', () => {
    rafQueue.length = 0;
    const v = viewer({ total: 6, mode: 'pingpong' });
    v.togglePlayback();
    run(v, 1000 / 24 * 12 + 5);
    v.isPlaying = false; rafQueue.length = 0;
    assert.deepEqual(v.shown.slice(0, 12), [1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2]);
});

test('a whole-clip loop longer than the paging window wraps instead of freezing', () => {
    rafQueue.length = 0;
    const v = viewer({ total: 96, paged: true });
    v.setFrame(93); v.shown.length = 0;
    assert.equal(v._frameReady(0), false, 'test premise: frame 0 is paged out near the end');
    v.togglePlayback();
    run(v, 1000);
    v.isPlaying = false; rafQueue.length = 0;
    assert.ok(v.shown.includes(0), `never wrapped: ${v.shown.join(',')}`);
    assert.ok(v.currentFrame > 5, `stuck after the wrap at ${v.currentFrame}`);
});

test('every frame is shown in order through the wrap', () => {
    rafQueue.length = 0;
    const v = viewer({ total: 40, paged: true });
    v.setFrame(30); v.shown.length = 0;
    v.togglePlayback();
    run(v, 1000);
    v.isPlaying = false; rafQueue.length = 0;
    const seq = v.shown.slice(0, 20);
    for (let i = 1; i < seq.length; i++) assert.equal(seq[i], (seq[i - 1] + 1) % 40, `skipped: ${seq.join(',')}`);
});

test('J plays a video backwards by seeking, K stops it, the start stops it', () => {
    rafQueue.length = 0;
    const events = [];
    const vid = { currentTime: 2, duration: 4, paused: false, seeking: false,
        pause() { this.paused = true; events.push('pause'); }, play() { this.paused = false; events.push('play'); } };
    const v = viewer();
    Object.assign(v, { videoMode: true, videoEl: vid, _videoNativeFps: 24, loop: false });
    v.shuttle(-1);
    assert.equal(vid.paused, true, 'J must pause the element (browsers cannot play backwards)');
    assert.equal(v.isPlaying, true);
    run(v, 500);
    assert.ok(vid.currentTime < 1.6 && vid.currentTime > 1.4, `after 0.5 s of J: ${vid.currentTime}`);
    v.shuttle(0);
    const t = vid.currentTime;
    run(v, 500);
    assert.equal(vid.currentTime, t, 'K did not stop reverse');
    assert.equal(v.isPlaying, false);

    vid.currentTime = 0.1;
    v.shuttle(-1);
    run(v, 500);
    assert.equal(vid.currentTime, 0);
    assert.equal(v._videoReversing, false, 'reverse ran past the start');

    v.loop = true; vid.currentTime = 0.05;
    v.shuttle(-1);
    run(v, 200);
    assert.ok(vid.currentTime > 3, `looping reverse did not wrap: ${vid.currentTime}`);
    v.shuttle(1);
    assert.equal(v._videoReversing, false);
    assert.equal(events.at(-1), 'play');
});

test('the viewer widget does not size itself from the node', () => {
    // computeSize returning [node width, node height - 110] made LiteGraph grow
    // the node by the widget, every frame, without limit.
    const setup = src.slice(src.indexOf('addDOMWidget("viewer"'), src.indexOf('addDOMWidget("viewer"') + 400);
    assert.match(setup, /getMinHeight/, 'the viewer widget needs a fixed minimum height');
    assert.doesNotMatch(src, /viewerWidget\.computeSize\s*=/, 'computeSize override is back');
});

test('the main 2D canvas stays GPU-backed', () => {
    // willReadFrequently forces a software canvas, so every frame blitted
    // from WebGL went through the CPU.
    const line = src.match(/this\.ctx = this\.canvas\.getContext\('2d'[^\n]*/g) || [];
    assert.ok(line.length > 0);
    for (const l of line) assert.doesNotMatch(l, /willReadFrequently:\s*true/, l);
});
