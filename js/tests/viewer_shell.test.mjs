import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    TOKENS, INPUT_SETTINGS, INPUT_WIDGET_NAMES, inputSettingsState, setInputSetting, shortSpace, formatEV, exposureBadge, shellMode, DOCK_LAYOUT, fontFaceCSS, shellCSS, svgIcon, RADIANCE_ICON_SVG,
} from '../radiance_viewer_shell.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const viewerSrc = readFileSync(here('../radiance_viewer.js'), 'utf8');

test('viewer f-stops read as the dock shows them, with a real minus sign', () => {
    assert.equal(formatEV(0), '±0.0');
    assert.equal(formatEV(0.5), '+0.5');
    assert.equal(formatEV(-1.5), '−1.5');
    assert.equal(formatEV('nope'), '±0.0');
    assert.equal(formatEV(0.04), '±0.0', 'rounds to a tenth before deciding the sign');
});

test('the picture says when the viewer f-stop is not in delivery, and only then', () => {
    assert.equal(exposureBadge(0), null);
    assert.equal(exposureBadge(0.01), null);
    assert.match(exposureBadge(1), /^Preview exposure \+1\.0 EV · not in delivery$/);
    assert.match(exposureBadge(-0.5), /not in delivery/);
});

test('Simple / Full follows the saved layout key', () => {
    assert.equal(shellMode('simple'), 'simple');
    assert.equal(shellMode('advanced'), 'full');
    assert.equal(shellMode(undefined), 'simple');
});

test('the dock only moves controls the viewer already builds', () => {
    for (const k of [...DOCK_LAYOUT.main, ...DOCK_LAYOUT.more]) {
        assert.match(viewerSrc, new RegExp(`this\\.${k} = `), `${k} is not a Simple-bar field`);
    }
});

test('the viewer installs the shell and keeps it in step', () => {
    assert.match(viewerSrc, /import \{ installShell as _installShell[^}]*\} from "\.\/radiance_viewer_shell\.js"/);
    assert.match(viewerSrc, /_installShell\(this\)/);
    assert.match(viewerSrc, /setUIMode[\s\S]*?this\._shell\?\.sync\(\)/);
    assert.match(viewerSrc, /setViewExposure\(stops\) \{[\s\S]*?this\._shell\?\.syncExposure\(\)/);
    assert.match(viewerSrc, /destroy\(\) \{\s*this\._destroyed = true;\s*this\._shell\?\.destroy\(\);/);
});

test('the theme is the Radiance gray, with one silver accent', () => {
    assert.match(viewerSrc, /accent: '#d8d8de'/);
    assert.match(viewerSrc, /bg: '#111114'/);
    assert.match(viewerSrc, /panelBorder: '#2c2c35'/);
    assert.equal(TOKENS.accent, '#d8d8de');
    assert.equal(TOKENS.line, '#2c2c35');
    assert.match(TOKENS.font, /^'Manrope'/);
    assert.match(TOKENS.mono, /^'DM Mono'/);
    assert.doesNotMatch(viewerSrc, /#4da3ff/i, 'the old blue is gone');
});

function fakeNode(linked = []) {
    const changes = [];
    const w = (name, type, value, options = {}) => ({ name, type, value, options, callback(v) { changes.push([name, v]); } });
    return {
        changes,
        widgets: [
            w('exposure_bracketing', 'toggle', false),
            w('input_space', 'combo', 'Auto', { values: ['Auto', 'sRGB (ComfyUI IMAGE)', 'ACEScg'] }),
            w('float_precision', 'combo', 'Half (16-bit)', { values: ['Half (16-bit)', 'Full (32-bit)'] }),
            w('fps', 'number', 0, { min: 0, max: 240 }),
        ],
        inputs: linked.map((name) => ({ name, widget: { name }, link: 7 })),
        setDirtyCanvas() {}, graph: { change() {} },
    };
}

test('the Input menu holds exactly the node settings that sat above the viewer', () => {
    const py = readFileSync(here('../../nodes/monitor/viewer.py'), 'utf8');
    for (const name of INPUT_WIDGET_NAMES) assert.match(py, new RegExp(`"${name}": \\(`), `${name} is not a Viewer input`);
    assert.deepEqual([...INPUT_WIDGET_NAMES].sort(), ['exposure_bracketing', 'float_precision', 'fps', 'input_space']);
    assert.equal(INPUT_SETTINGS.length, 4);
});

test('Input settings read and write the node widgets, and respect links', () => {
    const n = fakeNode(['fps']);
    const st = Object.fromEntries(inputSettingsState(n).map((s) => [s.name, s]));
    assert.equal(st.input_space.value, 'Auto');
    assert.deepEqual(st.float_precision.options, ['Half (16-bit)', 'Full (32-bit)']);
    assert.equal(st.fps.linked, true);
    assert.equal(st.input_space.linked, false);
    assert.equal(setInputSetting(n, 'input_space', 'ACEScg'), true);
    assert.equal(n.widgets[1].value, 'ACEScg');
    assert.equal(setInputSetting(n, 'input_space', 'Nonsense'), false, 'only the widget\'s own choices');
    assert.equal(setInputSetting(n, 'fps', 25), false, 'a linked setting is left to the link');
    assert.equal(setInputSetting(n, 'exposure_bracketing', true), true);
    assert.deepEqual(n.changes, [['input_space', 'ACEScg'], ['exposure_bracketing', true]], 'the widget callback runs');
    const m = fakeNode();
    assert.equal(setInputSetting(m, 'fps', '500'), true);
    assert.equal(m.widgets[3].value, 240, 'clamped to the widget range');
    assert.equal(setInputSetting(m, 'fps', 'abc'), false);
    assert.equal(setInputSetting(m, 'missing', 1), false);
});

test('the header names the input space briefly', () => {
    assert.equal(shortSpace('sRGB (ComfyUI IMAGE)'), 'sRGB');
    assert.equal(shortSpace('Linear Rec.709 (sRGB)'), 'Linear Rec.709');
    assert.equal(shortSpace('ACEScg'), 'ACEScg');
    assert.equal(shortSpace(undefined), 'Auto');
});

test('the node hides those widget rows, so nothing sits outside the viewer', () => {
    assert.match(viewerSrc, /const hideInputSettingWidgets = \(node\) =>[\s\S]*?_setWidgetVisible\(w, false, node\)/);
    assert.match(viewerSrc, /scheduleHiddenViewerDefaults = \(node\) => \{\s*if \(!isRadianceViewerNode\(node\)\) return;\s*hideInputSettingWidgets\(node\);/);
});

test('fonts are bundled, not fetched', () => {
    const css = fontFaceCSS('https://host/extensions/radiance/radiance_viewer_shell.js');
    for (const f of ['manrope-latin-wght.woff2', 'dm-mono-latin-400.woff2', 'dm-mono-latin-500.woff2']) {
        assert.ok(css.includes(`https://host/extensions/radiance/fonts/${f}`), f);
        assert.ok(existsSync(here(`../fonts/${f}`)), `js/fonts/${f} is missing`);
    }
    assert.ok(existsSync(here('../fonts/OFL-Manrope.txt')) && existsSync(here('../fonts/OFL-DM-Mono.txt')), 'font licences');
    assert.doesNotMatch(css + shellCSS(), /fonts\.googleapis|fonts\.gstatic/);
});

test('shell styles are scoped to the viewer', () => {
    const rules = shellCSS().replace(/\/\*[\s\S]*?\*\//g, '').split('}').map((r) => r.trim()).filter((r) => r && !r.startsWith('@'));
    for (const r of rules) {
        const sel = r.split('{')[0];
        for (const part of sel.split(',')) assert.match(part.trim(), /^\.radiance-v4/, `unscoped: ${part.trim()}`);
    }
});

test('icons are decorative: hidden from assistive tech', () => {
    assert.match(svgIcon('play'), /aria-hidden="true"/);
    assert.match(RADIANCE_ICON_SVG, /aria-hidden="true"/);
    assert.match(RADIANCE_ICON_SVG, /#0A84FF/);
});
