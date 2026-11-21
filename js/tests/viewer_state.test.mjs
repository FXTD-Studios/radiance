import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
    PREFS, STATE_VERSION, VERSION_KEY, loadPrefs, savePref, parsePref, serializePref, migrate,
} from '../radiance_viewer_state.js';

function store(init = {}) {
    const m = new Map(Object.entries(init));
    return {
        m,
        getItem: (k) => (m.has(k) ? m.get(k) : null),
        setItem: (k, v) => { m.set(k, String(v)); },
    };
}

test('an empty store loads every default', () => {
    const p = loadPrefs(store());
    for (const pref of PREFS) assert.deepEqual(p[pref.field], pref.default, pref.field);
    assert.equal(p.scopeMode, 'parade');
    assert.equal(p.scopeTransformed, true);
    assert.equal(p.scopeHlgPeak, 1000);
});

test('valid stored values are used as they are', () => {
    const p = loadPrefs(store({
        radiance_scope_mode: 'vectorscope', radiance_scope_xform: '0', radiance_scope_hlg_peak: '4000',
        radiance_pixel_filter: 'linear', radiance_time_display: 'timecode', radiance_compare_fit: 'fit',
        radiance_scope_levels: 'video', radiance_matte: '2.39:1',
    }));
    assert.equal(p.scopeMode, 'vectorscope');
    assert.equal(p.scopeTransformed, false);
    assert.equal(p.scopeHlgPeak, 4000);
    assert.equal(p.pixelFilter, 'linear');
    assert.equal(p.timeDisplay, 'timecode');
    assert.equal(p.compareFit, 'fit');
    assert.equal(p.scopeLevels, 'video');
    assert.equal(p.matteMode, '2.39:1');
});

test('stale, hand-edited or junk values load as the default', () => {
    const p = loadPrefs(store({
        radiance_scope_mode: 'oscilloscope-9000', radiance_scope_scale: 'nope', radiance_scope_levels: '',
        radiance_scope_hlg_peak: 'NaN', radiance_pixel_filter: 'cubic', radiance_time_display: 'beats',
        radiance_compare_fit: 'stretch', radiance_safe_preset: 'x'.repeat(500),
    }));
    assert.equal(p.scopeMode, 'parade');
    assert.equal(p.scopeScale, 'cv10');
    assert.equal(p.scopeLevels, 'data');
    assert.equal(p.scopeHlgPeak, 1000);
    assert.equal(p.pixelFilter, 'nearest');
    assert.equal(p.timeDisplay, 'frames');
    assert.equal(p.compareFit, 'pixel');
    assert.equal(p.safeAreaPreset, 'modern');
});

test('the scope scale and level lists come from the scope units, so they cannot drift', () => {
    const scale = PREFS.find((p) => p.field === 'scopeScale');
    assert.ok(scale.values.includes('cv10') && scale.values.length > 1);
    assert.deepEqual(PREFS.find((p) => p.field === 'scopeLevels').values, ['data', 'video']);
});

test("a saved 'manual' view mode is never restored", () => {
    assert.equal(loadPrefs(store({ radiance_view_mode: 'manual' })).viewMode, 'auto');
    assert.equal(loadPrefs(store({ radiance_view_mode: 'aces2' })).viewMode, 'aces2');
});

test('a peak below the floor is rejected, not clamped silently into range', () => {
    assert.equal(loadPrefs(store({ radiance_scope_hlg_peak: '5' })).scopeHlgPeak, 1000);
    assert.equal(loadPrefs(store({ radiance_scope_hlg_peak: '100' })).scopeHlgPeak, 100);
});

test('a storage that throws never breaks loading', () => {
    const hostile = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
    const p = loadPrefs(hostile);
    assert.equal(p.scopeMode, 'parade');
    assert.equal(savePref(hostile, 'scopeMode', 'waveform'), false);
    assert.doesNotThrow(() => loadPrefs(null));
});

test('savePref validates, writes the same keys as before, and refuses bad input', () => {
    const s = store();
    assert.equal(savePref(s, 'scopeMode', 'waveform'), true);
    assert.equal(s.m.get('radiance_scope_mode'), 'waveform');
    assert.equal(savePref(s, 'scopeTransformed', false), true);
    assert.equal(s.m.get('radiance_scope_xform'), '0');
    assert.equal(savePref(s, 'scopeTransformed', true), true);
    assert.equal(s.m.get('radiance_scope_xform'), '1');
    assert.equal(savePref(s, 'scopeMode', 'bogus'), false);
    assert.equal(s.m.get('radiance_scope_mode'), 'waveform');   // unchanged
    assert.equal(savePref(s, 'scopeHlgPeak', NaN), false);
    assert.equal(savePref(s, 'viewMode', 'manual'), false);
    assert.equal(savePref(s, 'unknownField', 'x'), false);
});

test('round trip: what savePref writes is what loadPrefs returns', () => {
    const s = store();
    savePref(s, 'timeDisplay', 'seconds');
    savePref(s, 'scopeHlgPeak', 2000);
    const p = loadPrefs(s);
    assert.equal(p.timeDisplay, 'seconds');
    assert.equal(p.scopeHlgPeak, 2000);
});

test('the version marker is recorded, and old stores count as version 1', () => {
    const s = store();
    loadPrefs(s);
    assert.equal(s.m.get(VERSION_KEY), String(STATE_VERSION));
    assert.equal(migrate(store(), {}, 1).version, 1);
});

test('migrations run in order up to the target, and a failure stops the run', () => {
    const order = [];
    const s = store({ [VERSION_KEY]: '1' });
    const r = migrate(s, { 1: () => order.push('1->2'), 2: () => order.push('2->3') }, 3);
    assert.deepEqual(order, ['1->2', '2->3']);
    assert.equal(r.version, 3);
    assert.equal(s.m.get(VERSION_KEY), '3');

    const s2 = store({ [VERSION_KEY]: '1' });
    const bad = migrate(s2, { 1: () => { throw new Error('boom'); }, 2: () => order.push('never') }, 3);
    assert.ok(bad.error);
    assert.equal(s2.m.get(VERSION_KEY), '1');           // not advanced past the failed step
    assert.ok(!order.includes('never'));
});

test('parse/serialize are inverses for every preference default', () => {
    for (const p of PREFS) {
        const stored = serializePref(p, p.default);
        assert.notEqual(stored, null, p.field);
        assert.deepEqual(parsePref(p, stored), p.default, p.field);
    }
});

test('the viewer reads its preferences through the module, not key by key', () => {
    const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');
    assert.match(src, /_loadViewerPrefs\(\(\(\) => \{ try \{ return localStorage; \} catch \{ return null; \} \}\)\(\)\)/);
    for (const key of ['radiance_scope_mode', 'radiance_scope_scale', 'radiance_pixel_filter', 'radiance_time_display']) {
        assert.doesNotMatch(src, new RegExp(`=\\s*localStorage\\.getItem\\('${key}'\\)`), `${key} is still read directly`);
    }
});
