/**
 * The shared colour code (radiance_palette.js).
 * Run: node --test js/tests/palette.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { DISCIPLINES, disciplineForCategory, disciplineForAnyCategory, dominantDiscipline } from '../radiance_palette.js';

test('Radiance categories map to their family; other packs are not Radiance', () => {
    assert.equal(disciplineForCategory('FXTD STUDIOS/Radiance/Load & Save'), 'io');
    assert.equal(disciplineForCategory('FXTD STUDIOS/Radiance/◎ HDR'), 'hdr');
    assert.equal(disciplineForCategory('FXTD STUDIOS/Radiance/Generate'), 'generate');
    assert.equal(disciplineForCategory('FXTD STUDIOS/Radiance/Review'), 'review');
    assert.equal(disciplineForCategory('FXTD STUDIOS/Radiance/◎ Gizmos'), 'pipeline');
    assert.equal(disciplineForCategory('sampling'), null);
    assert.equal(disciplineForCategory('some_pack/HDR'), null);
});

test('Smart Backdrop: Read + Write are I/O, the Prompt is generation, native nodes count', () => {
    assert.equal(dominantDiscipline(['FXTD STUDIOS/Radiance/Load & Save', 'FXTD STUDIOS/Radiance/Load & Save']), 'io');
    assert.equal(dominantDiscipline(['FXTD STUDIOS/Radiance/Generate', 'sampling', 'conditioning']), 'generate');
    assert.equal(disciplineForAnyCategory('image/upscaling'), 'upscale');
    assert.equal(dominantDiscipline(['utils']), null);
});

test('groups use ComfyUI\'s own group colours for the main families', () => {
    assert.equal(DISCIPLINES.generate.group, '#3f789e');
    assert.equal(DISCIPLINES.hdr.group, '#b58b2a');
    assert.equal(DISCIPLINES.sdr.group, '#8A8');
    assert.equal(DISCIPLINES.optional.group, '#a1309b');
    assert.equal(DISCIPLINES.review.group, '#444');
    for (const d of Object.values(DISCIPLINES)) {
        assert.match(d.header, /^#[0-9a-f]{6}$/i);
        assert.ok(d.title.startsWith('◎ '));
    }
});
