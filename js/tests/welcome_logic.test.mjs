/**
 * When the Radiance welcome shows, and what it says (radiance_welcome_logic.js).
 * Run: node --test js/tests/welcome_logic.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TOUR_PAGES, WHATS_NEW, LINKS, compareVersions, startupView, notesFor, stepPage } from '../radiance_welcome_logic.js';
import { RADIANCE_VERSION } from '../radiance_version.js';

test('first install opens the tour; a newer version opens What\'s new; the same version opens nothing', () => {
    assert.equal(startupView({ lastVersion: '', current: '4.0.0' }), 'tour');
    assert.equal(startupView({ lastVersion: '3.5.4', current: '4.0.0' }), 'update');
    assert.equal(startupView({ lastVersion: '4.0.0', current: '4.0.0' }), null);
    assert.equal(startupView({ lastVersion: '4.0.1', current: '4.0.0' }), null, 'a downgrade is not news');
    assert.equal(startupView({ lastVersion: '', current: '4.0.0', showAtStartup: false }), null);
    assert.equal(startupView({ lastVersion: '3.5.4', current: '4.0.0', showAtStartup: false }), null);
});

test('versions compare by number, not as text', () => {
    assert.equal(compareVersions('4.0.10', '4.0.9'), 1);
    assert.equal(compareVersions('4.0', '4.0.0'), 0);
    assert.equal(compareVersions('4.0.0-beta', '4.0.0'), 0);
    assert.equal(compareVersions('3.5.4', '4.0.0'), -1);
});

test('the tour has six pages and Next and Back stop at its ends', () => {
    assert.deepEqual(TOUR_PAGES, ['welcome', 'workflows', 'viewer', 'backdrops', 'gizmos', 'learn']);
    assert.equal(stepPage('welcome', -1), 'welcome');
    assert.equal(stepPage('welcome', 1), 'workflows');
    assert.equal(stepPage('learn', 1), 'learn');
});

test('the current version has release notes for the update page', () => {
    assert.ok(notesFor(RADIANCE_VERSION).length > 0, `WHATS_NEW has no entry for ${RADIANCE_VERSION}`);
    assert.deepEqual(notesFor('0.0.1'), []);
    assert.ok(WHATS_NEW.every((e) => e.items.every((t) => typeof t === 'string' && t.length < 140)));
});

test('the tour does not mention LTX', () => {
    const src = readFileSync(new URL('../radiance_welcome.js', import.meta.url), 'utf8')
        + readFileSync(new URL('../radiance_welcome_logic.js', import.meta.url), 'utf8');
    assert.doesNotMatch(src, /LTX/i);
});

test('links point at the FXTD site and the GitHub issues', () => {
    assert.match(LINKS.radiance, /^https:\/\/www\.fxtdstudios\.com\//);
    assert.match(LINKS.issues, /^https:\/\/github\.com\/FXTD-Studios\/radiance\/issues$/);
});
