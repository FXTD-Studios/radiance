/**
 * C9: a nit figure is only meaningful with its assumption (scene-linear 1.0 at
 * BT.2408's 203 cd/m2 reference white). Every visible peak-nit readout says it.
 * Run: node --test js/tests/viewer_labels.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

test('the header peak badge states BT.2408', () => {
    assert.match(src, /k nit \(BT\.2408\)`/);
    assert.match(src, /\)\} nit \(BT\.2408\)`/);
    assert.doesNotMatch(src, /nitStr = `\$\{Math\.round\(nit\)\} nit`;/);
});

test('the analysis peak stats name BT.2408 and explain it on hover', () => {
    assert.match(src, /mkStat\('PEAK \(BT\.2408\)'/);
    assert.match(src, /label: 'PEAK NIT \(BT\.2408\)'/);
    assert.match(src, /if \(item\.title\) cell\.title = item\.title;/);
    assert.doesNotMatch(src, /label: 'PEAK NIT',/);
    assert.doesNotMatch(src, /mkStat\('PEAK',/);
});
