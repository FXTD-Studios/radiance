/**
 * The committed Python parity fixture is what radiance_grade.js computes today.
 *
 * tests/fixtures/viewer_grade_parity.json holds grade payloads, pixels and the
 * values gradePixelFull gives them; tests/test_viewer_grade_parity.py checks the
 * Python delivery grade against it. If the JS grade changes, this fails until
 * the fixture is regenerated (node js/tests/viewer_grade_fixture.mjs --write),
 * and then the Python test fails until the delivery grade follows.
 *
 * Run: node --test js/tests/viewer_grade_fixture.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildFixture, FIXTURE_PATH } from './viewer_grade_fixture.mjs';

test('the committed parity fixture matches radiance_grade.js', () => {
    const committed = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8'));
    const fresh = JSON.parse(JSON.stringify(buildFixture()));
    assert.deepEqual(committed.pixels, fresh.pixels);
    assert.equal(committed.cases.length, fresh.cases.length);
    fresh.cases.forEach((c, i) => {
        const k = committed.cases[i];
        assert.equal(k.name, c.name);
        assert.deepEqual(k.grading, c.grading, `${c.name}: grading differs`);
        c.expected.forEach((px, j) => px.forEach((v, ch) => {
            assert.ok(Math.abs(v - k.expected[j][ch]) <= 1e-12 * Math.max(1, Math.abs(v)),
                `${c.name} pixel ${j}: fixture ${k.expected[j][ch]} vs radiance_grade.js ${v}. `
                + 'Regenerate with node js/tests/viewer_grade_fixture.mjs --write');
        }));
    });
});
