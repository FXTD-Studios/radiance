import test from 'node:test';
import assert from 'node:assert/strict';
import {
    WORKSPACES, availableWorkspaces, normalizeLayout, defaultWorkspaceFor,
    resolveSaved, propsFor, suggestWorkspace, workspaceById, tabsFor, workspaceForTab, PANEL_TABS,
} from '../radiance_viewer_modes.js';

test('QC is offered, on the Advanced layout, opening its own tab', () => {
    assert.ok(workspaceById('qc'));
    assert.deepEqual(availableWorkspaces().map((w) => w.id), ['review', 'color', 'inspect', 'compare', 'qc']);
    assert.equal(workspaceById('qc').layout, 'advanced');
    assert.deepEqual(tabsFor('qc'), ['scopes', 'analysis', 'qc']);
    assert.equal(workspaceForTab('qc', 'color'), 'qc');
    assert.ok(workspaceById('qc').tabs.includes(workspaceById('qc').tab));
});

test('workspace ids are unique and every one names a known layout', () => {
    const ids = WORKSPACES.map((w) => w.id);
    assert.equal(new Set(ids).size, ids.length);
    for (const w of WORKSPACES) assert.ok(['simple', 'advanced'].includes(w.layout), w.id);
});

test('an old graph opens exactly as before', () => {
    assert.deepEqual(resolveSaved({ radiance_viewer_mode: 'simple' }), { layout: 'simple', workspace: 'review' });
    assert.deepEqual(resolveSaved({ radiance_viewer_mode: 'advanced' }), { layout: 'advanced', workspace: 'color' });
    assert.deepEqual(resolveSaved({}), { layout: 'simple', workspace: 'review' });
    assert.deepEqual(resolveSaved(undefined), { layout: 'simple', workspace: 'review' });
});

test('a recorded workspace is honoured when it agrees with the layout', () => {
    const r = resolveSaved({ radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'inspect' });
    assert.deepEqual(r, { layout: 'advanced', workspace: 'inspect' });
});

test('a stale, mismatched or unavailable workspace falls back to the layout default', () => {
    // layout says simple, workspace says color: an older Radiance rewrote the layout
    assert.equal(resolveSaved({ radiance_viewer_mode: 'simple', radiance_viewer_workspace: 'color' }).workspace, 'review');
    assert.equal(resolveSaved({ radiance_viewer_mode: 'simple', radiance_viewer_workspace: 'qc' }).workspace, 'review');
    assert.equal(resolveSaved({ radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'qc' }).workspace, 'qc');
    assert.equal(resolveSaved({ radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'nope' }).workspace, 'color');
});

test('stored props keep the old layout key, so older versions still read it', () => {
    assert.deepEqual(propsFor('review'), { radiance_viewer_mode: 'simple', radiance_viewer_workspace: 'review' });
    assert.deepEqual(propsFor('compare'), { radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'compare' });
    assert.deepEqual(propsFor('qc'), { radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'qc' });
    assert.equal(propsFor('missing'), null);
});

test('normalizeLayout and defaults', () => {
    assert.equal(normalizeLayout('weird'), 'simple');
    assert.equal(defaultWorkspaceFor('advanced'), 'color');
    assert.equal(defaultWorkspaceFor('x'), 'review');
});

test('a suggestion appears only when it would help, and is never forced', () => {
    assert.deepEqual(suggestWorkspace({ hasB: true }, 'review').workspace, 'compare');
    assert.equal(suggestWorkspace({ hasB: true }, 'compare'), null);
    assert.equal(suggestWorkspace({ hasB: false }, 'review'), null);
    assert.equal(suggestWorkspace(null, 'review'), null);
});

// ── the viewer's use of the module (methods lifted from source) ─────────────
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { workspaceById as _workspaceById, defaultWorkspaceFor as _defaultWorkspaceFor } from '../radiance_viewer_modes.js';

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    return rest.slice(0, rest.search(/\n    \}\n/)) + '\n    }';
}

const Proto = new Function('_workspaceById', '_defaultWorkspaceFor', '_suggestWorkspace', '_tabsFor', '_workspaceForTab',
    `return class { ${['setWorkspace', '_visibleReferenceTab', '_setReferenceTab', 'setUIMode', '_syncWorkspaceHint'].map(methodSource).join('\n')} }`)(
    _workspaceById, _defaultWorkspaceFor, suggestWorkspace, tabsFor, workspaceForTab);

globalThis.requestAnimationFrame = () => 0;

function viewer({ hasB = false, compareMode = 'none' } = {}) {
    const v = new Proto();
    const cls = new Set();
    Object.assign(v, {
        node: { properties: {}, size: [900, 600], setSize(s) { this.size = s; } },
        container: { classList: { toggle: (c, on) => (on ? cls.add(c) : cls.delete(c)), has: (c) => cls.has(c) } },
        _modeBtns: new Map(['review', 'color', 'inspect', 'compare'].map((id) => {
            const b = { setAttribute() {}, classes: new Set(), title: '' };
            b.classList = { toggle(c, on) { if (c === 'is-active') b.on = on; if (on) b.classes.add(c); else b.classes.delete(c); } };
            return [id, b];
        })),
        compareMode, activeTab: 'primaries', showZdepth: true, renderer: { setShowDepth() {} },
        calls: [], _syncCompareUI() {}, _syncSimpleTransport() {}, render() { this.calls.push('render'); },
        _triggerRenderTabs() { this.calls.push('tabs'); },
        _renderReferenceRightHUD() { this.calls.push('hud'); }, _syncPanelButton() {}, _syncRail() {},
        _hasCompareB: () => hasB,
        setCompareMode(m) { this.compareMode = m; },
    });
    v._cls = cls;
    return v;
}

test('Review is the Simple layout and stores the old key', () => {
    const v = viewer();
    v.setWorkspace('review');
    assert.equal(v.workspace, 'review');
    assert.ok(v._cls.has('radiance-mode-simple'));
    assert.equal(v.node.properties.radiance_viewer_mode, 'simple');
    assert.equal(v.node.properties.radiance_viewer_workspace, 'review');
});

test('Inspect uses the Advanced layout, opens the Inspector tab, and grows the node', () => {
    const v = viewer();
    v.setWorkspace('inspect', { grow: true });
    assert.ok(!v._cls.has('radiance-mode-simple'));
    assert.equal(v._referenceRightTab, 'inspector');
    assert.ok(v.calls.includes('hud'), 'the right panel was not redrawn for the workspace');
    assert.equal(v.node.properties.radiance_viewer_mode, 'advanced');
    assert.deepEqual(v.node.size, [1180, 760]);
});

test('Compare never changes the compare mode by itself', () => {
    for (const [hasB, mode] of [[true, 'none'], [false, 'none'], [true, 'blink']]) {
        const v = viewer({ hasB, compareMode: mode });
        v.setWorkspace('compare');
        assert.equal(v.compareMode, mode);
        assert.equal(v.workspace, 'compare');
    }
});

test('unknown workspaces do nothing, and QC opens its tab', () => {
    const v = viewer();
    v.setWorkspace('nope');
    assert.equal(v.workspace, undefined);
    assert.deepEqual(v.node.properties, {});
    v.setWorkspace('qc');
    assert.equal(v.workspace, 'qc');
    assert.equal(v._referenceRightTab, 'qc');
    assert.equal(v.node.properties.radiance_viewer_workspace, 'qc');
});

test('setUIMode on its own (old callers) keeps the workspace consistent', () => {
    const v = viewer();
    v.setUIMode('advanced');
    assert.equal(v.workspace, 'color');
    v.setUIMode('simple');
    assert.equal(v.workspace, 'review');
    v.workspace = 'inspect';
    v.setUIMode('advanced');                         // already consistent: kept
    assert.equal(v.workspace, 'inspect');
});

test('a switch keeps exactly one button active', () => {
    const v = viewer();
    v.setWorkspace('color');
    const on = [...v._modeBtns].filter(([, b]) => b.on).map(([id]) => id);
    assert.deepEqual(on, ['color']);
});

test('the switcher is built from the module, and the restore path is unchanged', () => {
    assert.match(src, /_availableWorkspaces\(\)\.forEach\(\(w\) =>/);
    assert.match(src, /_resolveSavedMode\(this\.node\?\.properties\)/);
    assert.match(src, /setUIMode\?\.\(saved \|\| 'advanced'\)/);
});

test('a connected B puts a hint on Compare, never switches, and clears once Compare is opened', () => {
    const v = viewer();
    v.setWorkspace('review');
    v._hasCompareInput = true;
    v._compareHintSeen = false;
    v._syncWorkspaceHint();
    const cmp = v._modeBtns.get('compare');
    assert.ok(cmp.classes.has('has-hint'));
    assert.match(cmp.title, /A reference \(B\) is connected/);
    assert.equal(v.workspace, 'review');                    // nothing switched
    assert.equal(v.compareMode, 'none');
    v.setWorkspace('compare');
    v.setWorkspace('color');
    assert.ok(!cmp.classes.has('has-hint'));                // seen for this run
});

test('no B, no hint', () => {
    const v = viewer();
    v._hasCompareInput = false;
    v._syncWorkspaceHint();
    assert.ok([...v._modeBtns.values()].every((b) => !b.classes.has('has-hint')));
});

test('a new run resets the hint unless Compare is already open', () => {
    const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');
    assert.match(src, /viewer\._hasCompareInput = compareImages\.length > 0;/);
    assert.match(src, /viewer\._compareHintSeen = viewer\.workspace === 'compare';/);
});

test('each Advanced workspace shows its own panel tabs, in panel order', () => {
    assert.deepEqual(tabsFor('color'), ['grade', 'masks', 'effects', 'scopes']);
    assert.deepEqual(tabsFor('inspect'), ['inspector', 'scopes', 'analysis']);
    assert.deepEqual(tabsFor('compare'), ['inspector', 'scopes', 'analysis']);
    assert.deepEqual(tabsFor('nope'), [...PANEL_TABS]);
    for (const w of WORKSPACES.filter((x) => x.tabs)) assert.ok(w.tabs.includes(w.tab), `${w.id} opens on a tab it hides`);
    // every tab is reachable from some workspace
    for (const t of PANEL_TABS) assert.ok(workspaceForTab(t, 'review'), `${t} is in no workspace`);
});

test('a shortcut to a hidden tab moves to the workspace that has it', () => {
    const v = viewer();
    v.setWorkspace('inspect');
    v._setReferenceTab('grade');                        // Alt+2 from Inspect
    assert.equal(v.workspace, 'color');
    assert.equal(v._referenceRightTab, 'grade');
    v._setReferenceTab('scopes');                       // shown in Color: stays
    assert.equal(v.workspace, 'color');
    assert.equal(v._referenceRightTab, 'scopes');
    assert.equal(v.compareMode, 'none');
});

test('a saved tab the workspace hides falls back to its first tab', () => {
    const v = viewer();
    v.setWorkspace('color');
    assert.equal(v._visibleReferenceTab('inspector'), 'grade');
    assert.equal(v._visibleReferenceTab('masks'), 'masks');
});

test('the panel only builds the buttons of the workspace\'s tabs', () => {
    const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');
    assert.match(src, /\]\.filter\(\(\[id\]\) => !shown \|\| shown\.includes\(id\)\)\.forEach\(\(\[id, label, action\]\)/);
    assert.doesNotMatch(src, /_focusTab\(/, 'the dead legacy-HUD tab focus is back');
});
