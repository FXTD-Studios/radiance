/**
 * The Viewer's keyboard map (M4).
 *
 * The help overlay, the menus and the timeline tooltips each kept their own
 * list of keys, and they disagreed with the handler: help said H was
 * Histogram, the menus showed S, A/B and 0 against actions those keys did not
 * run, and two keys did two things at once. These pin the table that now
 * drives all four, so a label cannot be written for a key that is not bound
 * and two actions cannot share a key.
 *
 * Run: node --test js/tests/keymap.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    KEYMAP, MENU_ORDER, matchKey, keyLabel, shortcutLabel, eventCode, isTextEntry,
    helpSections, menuEntries, keyEntry,
} from '../radiance_keymap.js';

const ev = (code, mods = {}) => ({ code, key: '', shiftKey: false, altKey: false, ctrlKey: false, metaKey: false, ...mods });

test('no two actions share a key', () => {
    const seen = new Map();
    for (const e of KEYMAP) {
        for (const k of e.keys) {
            const shifts = k.shift === 'any' ? [false, true] : [!!k.shift];
            for (const s of shifts) {
                const sig = `${k.code}|s${s}|a${!!k.alt}|c${!!k.ctrl}`;
                assert.ok(!seen.has(sig), `${keyLabel(k)} is bound to both ${seen.get(sig)} and ${e.id}`);
                seen.set(sig, e.id);
            }
        }
    }
});

test('every binding finds its own action, with the modifiers it is labelled with', () => {
    for (const e of KEYMAP) {
        for (const k of e.keys) {
            const m = matchKey(ev(k.code, { shiftKey: k.shift === true, altKey: !!k.alt, ctrlKey: !!k.ctrl }));
            assert.equal(m?.entry.id, e.id, `${keyLabel(k)} did not find ${e.id}`);
            if (k.shift === true) assert.match(keyLabel(k), /Shift\+|^\?$/, `${e.id} needs Shift and its label does not say so`);
            if (k.alt) assert.match(keyLabel(k), /Alt\+/);
        }
    }
});

test('the collisions the review found are gone', () => {
    // H is help, and help says so.
    assert.equal(matchKey(ev('KeyH')).entry.id, 'help.toggle');
    const help = helpSections().flatMap((s) => s.items);
    assert.ok(help.some(([k, t]) => /\bH\b/.test(k) && /help/i.test(t)), 'help does not list H as help');
    assert.ok(!help.some(([, t]) => /histogram/i.test(t)), 'help still promises a histogram key');
    // Shift+V does at most one thing (it used to switch the track and the vectorscope).
    const shiftV = KEYMAP.filter((e) => e.keys.some((k) => k.code === 'KeyV' && k.shift !== false && k.shift));
    assert.ok(shiftV.length <= 1);
    assert.equal(matchKey(ev('KeyV', { shiftKey: true }))?.entry.id ?? null, null, 'Shift+V should not also toggle the vectorscope');
    // 2 and full screen are bound, and Alt+2 is a panel tab, not a hidden wheel mode.
    assert.equal(matchKey(ev('Digit2')).entry.id, 'view.zoom200');
    assert.equal(matchKey(ev('Digit2', { altKey: true })).entry.id, 'panel.grade');
    assert.equal(shortcutLabel('window.fullscreen'), 'Shift+F');
    // Numpad digits are zoom presets, not a grade offset.
    assert.deepEqual([1, 2, 4].map((n) => matchKey(ev(`Numpad${n}`)).key.arg), [1, 2, 4]);
    assert.equal(matchKey(ev('Numpad0')).entry.id, 'view.zoomPreset');
    // P is not bound to anything (there is no prompt editor to open).
    assert.equal(matchKey(ev('KeyP')), null);
    assert.ok(!help.some(([k]) => k === 'P'));
    // Snapshot and Pin have no key, so the menus show none (they showed S and A/B).
    assert.equal(shortcutLabel('file.export'), '');
    assert.equal(shortcutLabel('file.pinB'), '');
    assert.equal(shortcutLabel('edit.resetGrade'), '', 'Reset Grade was labelled 0, which resets the viewer');
});

test('matching is by physical key, so macOS Option shortcuts work', () => {
    // Option+X on a Mac types "≈"; the old e.key test never saw an "x".
    assert.equal(matchKey({ code: 'KeyX', key: '≈', altKey: true, shiftKey: false, ctrlKey: false, metaKey: false }).entry.id,
        'play.clearInOut');
    assert.equal(matchKey({ code: 'KeyB', key: '∫', altKey: true, shiftKey: false, ctrlKey: false, metaKey: false }).entry.id,
        'tools.precision');
    // Cmd counts as Ctrl.
    assert.equal(matchKey(ev('KeyZ', { metaKey: true })).entry.id, 'edit.undo');
    // A synthetic event with only `key` still resolves.
    assert.equal(eventCode({ key: '=' }), 'Equal');
    assert.equal(eventCode({ key: 'a' }), 'KeyA');
    assert.equal(matchKey({ key: 'Home', code: '' }).entry.id, 'play.first');
});

test('only text entry keeps the keys; sliders and selects do not', () => {
    const el = (tagName, type) => ({ tagName, type, getAttribute: () => null });
    assert.equal(isTextEntry(el('INPUT', 'text')), true);
    assert.equal(isTextEntry(el('INPUT', 'number')), true);
    assert.equal(isTextEntry(el('TEXTAREA')), true);
    assert.equal(isTextEntry({ isContentEditable: true }), true);
    assert.equal(isTextEntry(el('INPUT', 'range')), false, 'the scrubber swallowed Space and J/K/L');
    assert.equal(isTextEntry(el('SELECT')), false);
    assert.equal(isTextEntry(el('BUTTON')), false);
});

test('menus are made from the table and show only bound keys', () => {
    assert.deepEqual(MENU_ORDER, ['File', 'Edit', 'View', 'Color', 'Tools', 'Window', 'Help']);
    for (const m of MENU_ORDER) assert.ok(menuEntries(m).length, `${m} is empty`);
    assert.ok(menuEntries('File').length >= 4, 'File had two items');
    // Masks, curves, OCIO and presets have menu entries (M18).
    const color = menuEntries('Color').map((e) => e.id);
    for (const id of ['panel.masks', 'panel.curves', 'panel.ocio', 'panel.presets']) assert.ok(color.includes(id), id);
    for (const m of MENU_ORDER) {
        for (const e of menuEntries(m)) {
            const label = shortcutLabel(e.id);
            if (label) assert.equal(label, keyLabel(keyEntry(e.id).keys[0]));
        }
    }
});
