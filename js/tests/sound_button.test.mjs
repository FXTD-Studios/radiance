/**
 * Viewer: the sound on/off button next to Play, in the simple bar and the
 * sequence dock. Muting does not make playback lighter (measured in Edge: the
 * browser plays the sound on its own thread); it is there for comfort.
 *
 * Run: node --test js/tests/sound_button.test.mjs
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

globalThis.document = {
    createElement: () => ({ attrs: {}, setAttribute(k, v) { this.attrs[k] = v; } }),
};
const Viewer = new Function(
    `return class { ${['_muteButton', '_setAudioMuted', '_syncMuteButtons'].map(methodSource).join('\n')} }`)();

test('the button mutes and unmutes the sequence sound, in every bar', () => {
    const v = new Viewer();
    v._sequenceAudio = { muted: false };
    const simple = v._muteButton(), dock = v._muteButton();
    assert.equal(simple.disabled, false);
    simple.onclick();
    assert.equal(v._sequenceAudio.muted, true);
    for (const b of [simple, dock]) {
        assert.equal(b.attrs['aria-pressed'], 'true');
        assert.match(b.innerHTML, /M11 6l4 4/, 'the muted icon shows a cross');
    }
    dock.onclick();
    assert.equal(v._sequenceAudio.muted, false);
    assert.equal(simple.attrs['aria-pressed'], 'false');
});

test('without a sound the button is disabled and says why', () => {
    const v = new Viewer();
    const b = v._muteButton();
    assert.equal(b.disabled, true);
    assert.equal(b.title, 'No sound in this clip');
});

test('a new run starts its sound muted when the button is off', () => {
    assert.match(src, /viewer\._sequenceAudio\.muted = !!viewer\.audioMuted;/,
        'a new sequence sound ignores the mute button');
});
