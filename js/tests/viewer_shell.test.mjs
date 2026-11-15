import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    TOKENS, INPUT_SETTINGS, INPUT_WIDGET_NAMES, inputSettingsState, setInputSetting, shortSpace, formatEV, brightnessBadge, shellMode,
    fontFaceCSS, shellCSS, svgIcon, RADIANCE_ICON_SVG, TASKS, normalizeTask, LOOKS, lookFor, OVERLAYS, currentOverlay, setOverlay,
    EXPOSURE_LEGEND, checkItems, gradeSummary, regionShares, REGION_CLASSES, REGION_LEGEND, issueCount, sizeName, lengthName, MORE_TOOLS,
} from '../radiance_viewer_shell.js';
import { workspaceById, PANEL_TABS } from '../radiance_viewer_modes.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const viewerSrc = readFileSync(here('../radiance_viewer.js'), 'utf8');
const webglSrc = readFileSync(here('../radiance_webgl.js'), 'utf8');

test('preview brightness reads as the bar shows it, with a real minus sign', () => {
    assert.equal(formatEV(0), '0.0');
    assert.equal(formatEV(0.5), '+0.5');
    assert.equal(formatEV(-1.5), '\u22121.5');
    assert.equal(formatEV('nope'), '0.0');
    assert.equal(formatEV(0.04), '0.0', 'rounds to a tenth before deciding the sign');
});

test('the picture says preview brightness is not saved, and only when it is on', () => {
    assert.equal(brightnessBadge(0), null);
    assert.equal(brightnessBadge(0.01), null);
    assert.equal(brightnessBadge(1), 'Preview brightness +1.0 \u00b7 not saved');
    assert.match(brightnessBadge(-0.5), /not saved$/);
});

test('three tasks in plain words; anything else is View', () => {
    assert.deepEqual(TASKS.map((t) => t.label), ['View', 'Compare', 'Check']);
    assert.equal(normalizeTask('check'), 'check');
    assert.equal(normalizeTask('qc'), 'view');
    assert.equal(normalizeTask(undefined), 'view');
    assert.equal(shellMode('advanced'), 'full');
    assert.equal(shellMode(undefined), 'simple');
});

test('every view transform has a name a person picks from', () => {
    const ids = [...viewerSrc.matchAll(/^\s*\['([a-z0-9]+)', '[^']+'\],$/gm)].map((m) => m[1]);
    const modes = ids.slice(0, ids.indexOf('manual') + 1);
    assert.ok(modes.length >= 6, modes.join());
    for (const id of modes) assert.ok(LOOKS[id]?.name && LOOKS[id]?.desc, `${id} has no Look name`);
    assert.equal(lookFor('aces2').name, 'Film');
    assert.equal(lookFor('ocio-config', 'OCIO: My show').name, 'OCIO: My show', 'an unknown view keeps its own label');
    assert.equal(lookFor('x', 'View: Something').name, 'Something');
});

test('overlays are the viewer\'s own flags, one at a time', () => {
    const v = { falseColor: false, clippingMonitor: true, gamutWarning: false, zebra: true, renders: 0, render() { this.renders++; } };
    assert.equal(currentOverlay(v), 'clip');
    setOverlay(v, 'exposure');
    assert.deepEqual([v.falseColor, v.clippingMonitor, v.zebra], [true, false, false]);
    assert.equal(currentOverlay(v), 'exposure');
    setOverlay(v, 'off');
    assert.equal(currentOverlay(v), 'off');
    assert.equal(v.renders, 2);
    for (const o of OVERLAYS) if (o.flag) assert.match(viewerSrc, new RegExp(`this\\.${o.flag} = `), o.flag);
});

test('the exposure legend matches the shader\'s bands', () => {
    const fn = webglSrc.slice(webglSrc.indexOf('vec3 getFalseColorMap'), webglSrc.indexOf('vec3 getFalseColorMap') + 700);
    const colours = [...fn.matchAll(/return vec3\(([\d.]+), ([\d.]+), ([\d.]+)\)/g)]
        .map((m) => '#' + m.slice(1).map((x) => Math.round(Number(x) * 255).toString(16).padStart(2, '0')).join(''));
    for (const g of EXPOSURE_LEGEND) assert.ok(colours.includes(g.c), `${g.label} ${g.c} not in ${colours}`);
});

test('Check says what is wrong in sentences, worst first', () => {
    const stats = { nitPeak: 3190, clippedPct: 1.3, above1000Pct: 0.4, evRange: 9.2 };
    const qc = { checks: [
        { id: 'nan_inf', status: 'PASS' },
        { id: 'flicker', status: 'WARN', frames: [4, 9] },
        { id: 'alpha_range', status: 'NOT_APPLICABLE', reason: 'No alpha channel.' },
    ] };
    const items = checkItems(qc, stats, 100);
    assert.deepEqual(items.map((i) => [i.id, i.tone]),
        [['bright', 'warn'], ['flicker', 'warn'], ['exposure', 'ok'], ['nan_inf', 'ok'], ['alpha_range', 'skip']]);
    assert.match(items[0].text, /^1\.3% of the frame is above what this SDR view shows \(peak about 3,190 nits\)/);
    assert.equal(items[0].overlay, 'clip');
    assert.match(items[1].text, /2 frames\.$/);
    assert.equal(items[4].text, 'No alpha channel.');
    assert.equal(issueCount(items), 2);
    const hdr = checkItems(null, stats, 1000);
    assert.equal(hdr[0].title, 'Brighter than 1000 nits');
    assert.match(hdr[0].text, /^0\.4%/);
    assert.deepEqual(checkItems(null, null), []);
    assert.equal(checkItems(null, { ...stats, clippedPct: 0 })[0].tone, 'ok');
});

test('what RUDRA rebuilt is read back from the regions picture\'s colours', () => {
    const MIX = 0.85, grey = 0.6;
    const px = (rgb, lum = 0.5) => rgb.map((c) => Math.round(((1 - MIX) * lum * grey + MIX * c) * 255));
    const pix = [px([0, 0, 0], 0.4).map(() => 100), px(REGION_CLASSES[0].rgb), px(REGION_CLASSES[1].rgb, 0.9), px(REGION_CLASSES[2].rgb, 0.1),
        [255, 255, 255], px(REGION_CLASSES[0].rgb, 0.2), [20, 20, 22], [130, 128, 126]];
    const rgba = new Uint8ClampedArray(pix.flatMap((p) => [...p, 255]));
    const s = regionShares(rgba);
    assert.equal(s.reconstructed, 25);
    assert.equal(s.estimated, 12.5);
    assert.equal(s.generated, 12.5);
    assert.equal(s.rebuilt, 50);
    assert.deepEqual(regionShares(new Uint8ClampedArray(0)), { reconstructed: 0, estimated: 0, generated: 0, rebuilt: 0 });
    const py = readFileSync(here('../../model/rudra_analytic.py'), 'utf8');
    const m = py.match(/PIXEL_CLASS_COLOUR = np\.array\(\[([\s\S]*?)\], np\.float32\)/);
    const rows = [...m[1].matchAll(/\[([\d.]+), ([\d.]+), ([\d.]+)\]/g)].map((r) => r.slice(1).map(Number));
    assert.deepEqual(REGION_CLASSES.map((c) => c.rgb), rows.slice(2), 'classes 2-4 as the model colours them');
    assert.deepEqual(REGION_LEGEND.map((l) => l.c), ['#40c7db', '#eb4ddb', '#fac733']);
    const items = checkItems(null, null, 100, s);
    assert.equal(items[0].id, 'ai');
    assert.equal(items[0].overlay, 'ai');
    assert.match(items[0].text, /^RUDRA rebuilt 50\.0% of the frame: 25\.0% reconstructed, 12\.5% estimated, 12\.5% generated by AI\.$/);
    assert.equal(checkItems(null, null, 100, { reconstructed: 0, estimated: 0, generated: 0, rebuilt: 0 })[0].tone, 'ok');
    assert.equal(checkItems(null, null, 100, 'loading')[0].tone, 'skip', 'still loading');
});

test('a grade that is not neutral is named, so a dark picture has a reason on screen', () => {
    const D = { exposure: 0, contrast: 1, gain: [1, 1, 1], lift: [0, 0, 0], logShadow: [0, 0, 0], logMidtone: [0, 0, 0],
        grain: 0, dofEnabled: false, maskState: { type: 0 }, qualifierState: { enabled: false } };
    assert.deepEqual(gradeSummary({ ...D }, D), []);
    assert.deepEqual(gradeSummary({ ...D, exposure: -4.1, gain: [0.82, 1.17, 1.02] }, D), ['Exposure \u22124.1', 'Gain']);
    assert.deepEqual(gradeSummary({ ...D, logShadow: [0.1, 0, 0], logMidtone: [0, 0.1, 0], dofEnabled: true, curves: {}, manualLut: 'x.cube' }, D),
        ['Log wheels', 'Depth of field', 'Curves', 'LUT']);
    assert.deepEqual(gradeSummary({ ...D, exposure: 0.00001 }, D), [], 'rounding noise is not a grade');
    assert.deepEqual(gradeSummary(null, D), []);
    const viewerSrc2 = readFileSync(here('../radiance_viewer.js'), 'utf8');
    assert.match(viewerSrc2, /_schedulePersistGrade\(\) \{\s*this\._shell\?\.syncGrade\?\.\(\);/);
});

test('sizes and lengths read as a person says them', () => {
    assert.equal(sizeName(4096, 2160), '4K');
    assert.equal(sizeName(1920, 1080), 'HD');
    assert.equal(sizeName(640, 360), 'Small');
    assert.equal(sizeName(0, 0), '\u2014');
    assert.equal(lengthName(240, 24), '10 s');
    assert.equal(lengthName(36, 24), '1.5 s');
    assert.equal(lengthName(1, 24), 'Still');
    assert.equal(lengthName(24 * 64, 24), '1 min 4 s');
});

test('More tools only opens workspaces and panel tabs that exist', () => {
    for (const m of MORE_TOOLS) {
        const ws = workspaceById(m.workspace);
        assert.ok(ws?.available && ws.layout === 'advanced', m.id);
        assert.ok(PANEL_TABS.includes(m.tab) && ws.tabs.includes(m.tab), `${m.id}: ${m.tab}`);
    }
});

test('the viewer installs the shell and keeps it in step', () => {
    assert.match(viewerSrc, /import \{ installShell as _installShell[^}]*\} from "\.\/radiance_viewer_shell\.js"/);
    assert.match(viewerSrc, /_installShell\(this\)/);
    assert.match(viewerSrc, /setUIMode[\s\S]*?this\._shell\?\.sync\(\)/);
    assert.match(viewerSrc, /setViewExposure\(stops\) \{[\s\S]*?this\._shell\?\.syncExposure\(\)/);
    assert.match(viewerSrc, /_syncCompareUI\(\) \{[\s\S]*?this\._shell\?\.syncCompare\?\.\(\);\n    \}/);
    assert.match(viewerSrc, /_refreshQCPanel\(\) \{\s*this\._shell\?\.syncChecks\?\.\(\);/);
    assert.match(viewerSrc, /_updateHDRPeakBadge\(\) \{[\s\S]{0,200}this\._shell\?\.syncDetails\?\.\(\)/);
    assert.match(viewerSrc, /setViewMode\(mode\) \{[\s\S]*?this\._shell\?\.syncLook\?\.\(\);/);
    assert.match(viewerSrc, /destroy\(\) \{\s*this\._destroyed = true;\s*this\._shell\?\.destroy\(\);/);
});

test('the V5 colours: near-black, one blue for state, amber only for warnings', () => {
    assert.equal(TOKENS.canvas, '#08080a');
    assert.equal(TOKENS.accent, '#5aa9ff');
    assert.equal(TOKENS.accentInk, '#06101f');
    assert.equal(TOKENS.warn, '#f5b544');
    assert.match(TOKENS.font, /^'Inter'/);
    assert.match(TOKENS.mono, /^'JetBrains Mono'/);
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
    for (const f of ['inter-latin-wght.woff2', 'jetbrains-mono-latin-400.woff2', 'jetbrains-mono-latin-500.woff2']) {
        assert.ok(css.includes(`https://host/extensions/radiance/fonts/${f}`), f);
        assert.ok(existsSync(here(`../fonts/${f}`)), `js/fonts/${f} is missing`);
    }
    for (const l of ['OFL-Inter.txt', 'OFL-JetBrains-Mono.txt']) assert.ok(existsSync(here(`../fonts/${l}`)), l);
    assert.doesNotMatch(css + shellCSS(), /fonts\.googleapis|fonts\.gstatic/);
});

test('shell styles are scoped to the viewer', () => {
    const rules = shellCSS().replace(/\/\*[\s\S]*?\*\//g, '').split('}').map((r) => r.trim()).filter((r) => r && !r.startsWith('@'));
    for (const r of rules) {
        const sel = r.split('{')[0];
        for (const part of sel.split(',')) assert.match(part.trim(), /^\.radiance-v5/, `unscoped: ${part.trim()}`);
    }
});

test('icons are decorative: hidden from assistive tech', () => {
    assert.match(svgIcon('play'), /aria-hidden="true"/);
    assert.match(RADIANCE_ICON_SVG, /aria-hidden="true"/);
    assert.match(RADIANCE_ICON_SVG, /#0A84FF/);
    for (const t of TASKS) assert.ok(svgIcon(t.id).includes('<path'), `${t.id} has an icon`);
});
