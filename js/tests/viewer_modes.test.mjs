import test from 'node:test';
import assert from 'node:assert/strict';
import {
    WORKSPACES, availableWorkspaces, normalizeLayout, defaultWorkspaceFor,
    resolveSaved, propsFor, suggestWorkspace, workspaceById,
} from '../radiance_viewer_modes.js';

test('QC is defined but never offered', () => {
    assert.ok(workspaceById('qc'));
    assert.deepEqual(availableWorkspaces().map((w) => w.id), ['review', 'color', 'inspect', 'compare']);
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
    assert.equal(resolveSaved({ radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'qc' }).workspace, 'color');
    assert.equal(resolveSaved({ radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'nope' }).workspace, 'color');
});

test('stored props keep the old layout key, so older versions still read it', () => {
    assert.deepEqual(propsFor('review'), { radiance_viewer_mode: 'simple', radiance_viewer_workspace: 'review' });
    assert.deepEqual(propsFor('compare'), { radiance_viewer_mode: 'advanced', radiance_viewer_workspace: 'compare' });
    assert.equal(propsFor('qc'), null);
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

const Proto = new Function('_workspaceById', '_defaultWorkspaceFor',
    `return class { ${['setWorkspace', '_focusTab', 'setUIMode'].map(methodSource).join('\n')} }`)(_workspaceById, _defaultWorkspaceFor);

globalThis.requestAnimationFrame = () => 0;

function viewer({ hasB = false, compareMode = 'none' } = {}) {
    const v = new Proto();
    const cls = new Set();
    Object.assign(v, {
        node: { properties: {}, size: [900, 600], setSize(s) { this.size = s; } },
        container: { classList: { toggle: (c, on) => (on ? cls.add(c) : cls.delete(c)), has: (c) => cls.has(c) } },
        _modeBtns: new Map(['review', 'color', 'inspect', 'compare'].map((id) => {
            const b = { setAttribute() {} };
            b.classList = { toggle(c, on) { b.on = on; } };
            return [id, b];
        })),
        compareMode, activeTab: 'primaries', showZdepth: true, renderer: { setShowDepth() {} },
        calls: [], _syncCompareUI() {}, _syncSimpleTransport() {}, render() { this.calls.push('render'); },
        _triggerRenderTabs() { this.calls.push('tabs'); },
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

test('Inspect uses the Advanced layout, focuses the probe tab, and grows the node', () => {
    const v = viewer();
    v.setWorkspace('inspect', { grow: true });
    assert.ok(!v._cls.has('radiance-mode-simple'));
    assert.equal(v.activeTab, 'probe');
    assert.equal(v.showZdepth, false);
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

test('QC and unknown workspaces do nothing', () => {
    const v = viewer();
    v.setWorkspace('qc');
    v.setWorkspace('nope');
    assert.equal(v.workspace, undefined);
    assert.deepEqual(v.node.properties, {});
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
