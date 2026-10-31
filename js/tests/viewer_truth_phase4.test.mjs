/**
 * Phase 4 viewer-truth pins (source assertions; the viewer cannot be imported
 * in Node, same approach as scope_wiring.test.mjs).
 *
 * Run: node --test js/tests/viewer_truth_phase4.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');
const webgl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');

test('window overlay and show-matte are skipped for scopes and graded exports', () => {
    assert.match(webgl, /bool previewAids = !u_scopeSignal && !u_exportSceneLinear;/);
    assert.match(webgl, /if \(previewAids && u_maskType > 0 && u_maskShowOverlay\)/);
    assert.match(webgl, /if \(previewAids && u_qualifierShowMask\)/);
    // the aids must not be tested bare any more
    assert.doesNotMatch(webgl, /if \(u_maskType > 0 && u_maskShowOverlay\)/);
    assert.doesNotMatch(webgl, /if \(u_qualifierShowMask\) \{/);
});

test('Deliver says when a window or qualifier is not baked', () => {
    // 4.0.0: the mattes are read when the job is made (a queued job keeps them) and named here.
    assert.match(viewer, /Not baked: \$\{job\.aids\.join\(' and '\)\}/);
    assert.match(viewer, /aids: unbakedAids\(\)/);
    assert.match(viewer, /_r\?\.maskType > 0/);
    assert.match(viewer, /_r\?\.qualifierEnabled/);
    // it feeds the same warnings list that turns the completion amber
    const i = viewer.indexOf('Not baked:');
    const j = viewer.indexOf("EXPORT COMPLETE WITH WARNINGS");
    assert.ok(i > 0 && j > i, 'notice must be added before the status label is chosen');
});

test('Save PNG (Result) labels the ungraded fallback', () => {
    assert.match(viewer, /const ungraded = !sig\?\.data;/);
    assert.match(viewer, /_ungraded_preview/);
    assert.match(viewer, /not the graded result/);
});
