/**
 * Viewer 3D Normal channel. Normal frames arrive as `is_normal` entries (never
 * timeline frames) and are drawn as data on the overlay canvas when
 * channel === 'normal'. Methods are lifted from radiance_viewer.js. How it looks
 * needs a browser and is not covered here.
 *
 * Run: node --test js/tests/viewer_normal_channel.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');
const exp = readFileSync(join(JS, 'radiance_viewer_export.js'), 'utf8');

function methodSource(name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    return rest.slice(0, rest.search(/\n    \}\n/)) + '\n    }';
}
const Proto = new Function(`return class { ${['_drawNormalChannel', '_normalHiddenByCompare', '_getReferenceChannelRows', '_selectReferenceChannel'].map(methodSource).join('\n')} }`)();

function ctxRecorder() {
    const calls = [];
    return { calls, save() {}, restore() {}, set imageSmoothingEnabled(v) { calls.push(['smooth', v]); },
        drawImage(...a) { calls.push(['draw', ...a.slice(1)]); } };
}
function viewer(over = {}) {
    const v = new Proto();
    Object.assign(v, { zoom: 2, panX: 10, panY: 20, imageWidth: 100, imageHeight: 50, channel: 'normal',
        normalImage: { naturalWidth: 100, naturalHeight: 50 }, frameNormalImages: [], hdrData: { channels: 4 }, showZdepth: false,
        render() {}, updateBottomBar() {}, _renderReferenceRightHUD() {} }, over);
    return v;
}

test('the normal pass is drawn at the image rectangle for the current zoom and pan', () => {
    const v = viewer(), c = ctxRecorder();
    v._drawNormalChannel(c);
    // source rect (whole image), then destination rect
    assert.deepEqual(c.calls.find((x) => x[0] === 'draw').slice(1), [0, 0, 100, 50, 10, 20, 200, 100]);
});

test('nothing is drawn without a normal frame or an image size', () => {
    const c = ctxRecorder();
    viewer({ normalImage: null })._drawNormalChannel(c);
    viewer({ imageWidth: 0 })._drawNormalChannel(c);
    assert.equal(c.calls.length, 0);
});

test('Normal appears in the channel list, available only when a normal frame exists', () => {
    const row = (v) => v._getReferenceChannelRows().find((r) => r.name === 'Normal');
    assert.equal(row(viewer({ normalImage: null })).available, false);
    assert.equal(row(viewer()).available, true);
    assert.equal(row(viewer()).active, true);
});

test('selecting Normal sets the channel; selecting another leaves it', () => {
    const v = viewer({ channel: 'rgb' });
    v._selectReferenceChannel({ available: true, kind: 'normal', channel: 'normal' });
    assert.equal(v.channel, 'normal');
    v._selectReferenceChannel({ available: true, kind: 'channel', channel: 'r' });
    assert.equal(v.channel, 'r');
});

test('normal entries are kept out of the timeline and out of the export source frame', () => {
    assert.match(src, /mainImages = message\.radiance_images\.filter\(img => !img\.is_compare && !img\.is_zdepth && !img\.is_normal/);
    assert.match(exp, /!d\.is_zdepth && !d\.is_normal/);
    assert.ok(!/!d\.is_compare && !d\.is_zdepth\)/.test(src), 'a frame lookup still lets normal entries through');
});

test('the normal pass is A\'s: hidden while B is on screen', () => {
    for (const over of [{ compareMode: 'b' }, { compareMode: 'difference' }, { compareMode: 'sidebyside' },
        { compareMode: 'blink', _blinkB: true }]) {
        const c = ctxRecorder();
        viewer(over)._drawNormalChannel(c);
        assert.equal(c.calls.length, 0, JSON.stringify(over));
    }
    const c = ctxRecorder();
    viewer({ compareMode: 'blink', _blinkB: false })._drawNormalChannel(c);
    assert.equal(c.calls.filter((x) => x[0] === 'draw').length, 1);
});

test('in a wipe the normal pass is drawn only on A\'s side of the line', () => {
    const c = ctxRecorder();
    viewer({ compareMode: 'wipe', wipePosition: 0.25 })._drawNormalChannel(c);
    // x0 = 25 px of a 100 px image: source from 25 wide 75, destination from 10 + 25*2
    assert.deepEqual(c.calls.find((x) => x[0] === 'draw').slice(1), [25, 0, 75, 50, 60, 20, 150, 100]);
});

test('Save PNG on the Normal channel saves the normal pass, and the probe reads it', () => {
    assert.match(src, /_normal_encoded_8bit\.png/);
    assert.match(src, /const n = this\._sampleNormalAt\(imgX, imgY\);/);
    assert.match(src, /'NORMAL \(A\) HIDDEN WHILE B SHOWN'/);
});
