import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeOnion, onionLayers, differenceSpec, differenceValue, ONION_LIMITS } from '../radiance_onion.js';

test('defaults show one frame either side', () => {
    const l = onionLayers(5, 10);
    assert.deepEqual(l.map((x) => [x.frame, x.side]), [[4, 'before'], [6, 'after']]);
});

test('no wrapping or duplicates at the ends', () => {
    assert.deepEqual(onionLayers(0, 10, { before: 2, after: 2 }).map((x) => x.frame), [1, 2]);
    assert.deepEqual(onionLayers(9, 10, { before: 2, after: 2 }).map((x) => x.frame), [8, 7]);
    assert.deepEqual(onionLayers(0, 1), []);
});

test('invalid frame or count gives no layers', () => {
    for (const [f, c] of [[-1, 5], [5, 5], [1.5, 5], [0, 0], ['2', 5], [NaN, 5]]) assert.deepEqual(onionLayers(f, c), []);
});

test('opacity falls off per step and faint layers are dropped', () => {
    const l = onionLayers(10, 30, { before: 4, after: 0, opacity: 0.4, falloff: 0.5 });
    assert.deepEqual(l.map((x) => x.opacity), [0.4, 0.2, 0.1, 0.05]);
    const dropped = onionLayers(10, 30, { before: 4, after: 0, opacity: 0.4, falloff: 0.1 });
    assert.equal(dropped.length, 2);  // 0.4, 0.04; 0.004 is below the floor
});

test('step spaces the layers; tint is optional', () => {
    const l = onionLayers(10, 30, { before: 2, after: 0, step: 3, tint: false });
    assert.deepEqual(l.map((x) => x.frame), [7, 4]);
    assert.ok(l.every((x) => x.tint === null));
    assert.ok(onionLayers(10, 30)[0].tint);
});

test('settings are clamped and bad values fall back', () => {
    const n = normalizeOnion({ before: 99, after: -3, opacity: 7, falloff: 'x', step: 0, tint: 'yes' });
    assert.equal(n.before, ONION_LIMITS.maxSide);
    assert.equal(n.after, 0);
    assert.equal(n.opacity, 1);
    assert.equal(n.falloff, 0.5);
    assert.equal(n.step, 1);
    assert.equal(n.tint, true);
    assert.doesNotThrow(() => normalizeOnion(null ?? undefined));
});

test('difference spec states its domain and gain, and rejects unknown values', () => {
    const s = differenceSpec({ domain: 'weird', gain: 3 });
    assert.equal(s.domain, 'display');
    assert.equal(s.gain, 4);
    assert.match(s.label, /\|A - B\| x4, display-referred/);
    assert.match(differenceSpec({ domain: 'scene-linear', gain: 8, mode: 'signed' }).label, /A - B \(grey = equal\) x8, scene-linear/);
});

test('difference value', () => {
    const abs = differenceSpec({ gain: 2 }), signed = differenceSpec({ gain: 2, mode: 'signed' });
    assert.equal(differenceValue(0.2, 0.5, abs), 0.6);
    assert.equal(differenceValue(0.2, 0.5, signed), -0.6);
    assert.equal(differenceValue(3, 3, abs), 0);
});
