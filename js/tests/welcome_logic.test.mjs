/**
 * When the Radiance welcome shows, and what it says (radiance_welcome_logic.js).
 * Run: node --test js/tests/welcome_logic.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TOUR_PAGES, WHATS_NEW, LINKS, compareVersions, startupView, notesFor, stepPage, setupSummary, sizeText } from '../radiance_welcome_logic.js';
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

test('the tour has seven pages, Set up second, and Next and Back stop at its ends', () => {
    assert.deepEqual(TOUR_PAGES, ['welcome', 'setup', 'workflows', 'viewer', 'backdrops', 'gizmos', 'learn']);
    assert.equal(stepPage('welcome', -1), 'welcome');
    assert.equal(stepPage('welcome', 1), 'setup');
    assert.equal(stepPage('setup', 1), 'workflows');
    assert.equal(stepPage('learn', 1), 'learn');
});

test('the Set up page sums up a scan: GPU, memory, disk, RUDRA and ready workflows', () => {
    const scan = {
        hardware: { gpus: [{ name: 'NVIDIA GeForce RTX 4090', kind: 'cuda', vram: 25_757_220_864 }], ram: 68_719_476_736, disk_free: 812_000_000_000, torch: '2.12.1', cuda: '13.0' },
        paths: { radiance: 'E:/ComfyUI/models/radiance' }, downloads_allowed: true,
        core: [{ id: 'radiance/RUDRA_ACES_2', installed: false }],
        summary: { workflows: 21, ready: 5 },
    };
    const s = setupSummary(scan);
    assert.equal(s.gpu, 'NVIDIA GeForce RTX 4090 · 24 GB');
    assert.equal(s.memory, '64 GB RAM');
    assert.equal(s.disk, '812 GB free for models');
    assert.equal(s.tier.tone, 'ok');
    assert.deepEqual(s.rudra, { installed: false, path: 'E:/ComfyUI/models/radiance' });
    assert.equal(s.ready, '5 of 21 local workflows are ready to run');
    assert.equal(s.downloadsOff, false);
    const none = setupSummary({ hardware: { gpus: [] }, summary: { workflows: 21, ready: 3 }, downloads_allowed: false });
    assert.equal(none.gpu, 'No GPU found');
    assert.equal(none.tier.tone, 'warn');
    assert.equal(none.downloadsOff, true);
    assert.equal(sizeText(3_000_000), '3 MB');
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

// ── First-start display check (4.0) ─────────────────────────────────────────
import { displayCheck, probeDisplay } from '../radiance_welcome_logic.js';

test('display check: no WebGL is bad, WebGL 1 or no float targets warn, WebGL 2 is ok', () => {
    assert.equal(displayCheck({ webgl: 0 }).status, 'bad');
    assert.equal(displayCheck(null).status, 'bad');
    assert.equal(displayCheck({ webgl: 1 }).status, 'warn');
    assert.equal(displayCheck({ webgl: 2, floatTargets: false }).status, 'warn');
    const sdr = displayCheck({ webgl: 2, floatTargets: true, hdr: false, p3: false });
    assert.equal(sdr.status, 'ok');
    assert.match(sdr.detail, /SDR display \(Rec\.709\).*tone-mapped/);
    assert.match(displayCheck({ webgl: 2, floatTargets: true, hdr: true }).detail, /HDR display/);
    assert.match(displayCheck({ webgl: 2, floatTargets: true, p3: true }).detail, /P3/);
    assert.equal(sdr.id, 'display');
    assert.equal(sdr.fix, null);
});

test('display probe: reads media queries and WebGL, and survives a browser without either', () => {
    const lost = [];
    const gl2 = {
        getExtension: (n) => (n === 'EXT_color_buffer_float' ? {} : n === 'WEBGL_lose_context' ? { loseContext: () => lost.push(1) } : null),
        getParameter: () => 'Fake GPU', RENDERER: 1,
    };
    const win = {
        matchMedia: (q) => ({ matches: q === '(dynamic-range: high)' }),
        document: { createElement: () => ({ getContext: (k) => (k === 'webgl2' ? gl2 : null) }) },
    };
    assert.deepEqual(probeDisplay(win), { hdr: true, p3: false, webgl: 2, floatTargets: true, renderer: 'Fake GPU' });
    assert.equal(lost.length, 1, 'the probe context is released');
    const bare = probeDisplay({ document: { createElement: () => { throw new Error('no canvas'); } } });
    assert.deepEqual(bare, { hdr: false, p3: false, webgl: 0, floatTargets: false, renderer: '' });
});
