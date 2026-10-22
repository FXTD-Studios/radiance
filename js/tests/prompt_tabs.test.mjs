import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_prompt.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

const context = vm.createContext({
    app: { registerExtension() {} },
    api: { fetchApi: () => new Promise(() => {}) },
    console: { log() {}, warn() {} },
});
vm.runInContext(source + `_refreshLiveState = () => {};
    globalThis.tabs = { read: _readTabs, edit: _editTabs, values: _tabValues, select: _selectTab, add: _newTab,
        duplicate: _duplicateTab, remove: _deleteTab, move: _moveTab, chain: _chainTab, runAll: _runAllTabs };`, context);
const { tabs } = context;

const DEFAULTS = { base_prompt: '', action: '', framing: 'None', negative_prompt: '' };

function makeNode(values) {
    const node = {
        properties: {},
        setDirtyCanvas() {},
        widgets: [{ name: 'prompt_tabs', serialize: false, element: { firstChild: null } },
            ...Object.entries(DEFAULTS).map(([name, value]) => ({ name, value }))],
    };
    node._radFreshTab = tabs.values(node);
    for (const w of node.widgets) if (w.name in values) w.value = values[w.name];
    return node;
}

const shown = (node) => JSON.parse(JSON.stringify(tabs.values(node)));
const names = (node) => JSON.parse(JSON.stringify(tabs.read(node).tabs.map(t => t.name)));
const A = { base_prompt: 'a woman holding a wand', framing: 'Close-Up (CU)', negative_prompt: 'blurry' };

test('a node starts with Prompt A and Prompt B, and saves nothing until a tab is used', () => {
    const node = makeNode(A);
    assert.deepEqual(names(node), ['Prompt A', 'Prompt B']);
    assert.equal(tabs.read(node).active, 0);
    assert.deepEqual(node.properties, {});
});

test('a new tab shows the node defaults, and the first one comes back as it was', () => {
    const node = makeNode(A);
    tabs.select(node, 1);
    assert.deepEqual(shown(node), DEFAULTS);
    node.widgets.find(w => w.name === 'base_prompt').value = 'a snowy forest';
    tabs.select(node, 0);
    assert.deepEqual(shown(node), { ...DEFAULTS, ...A });
    assert.equal(tabs.read(node).tabs[1].values.base_prompt, 'a snowy forest');
    assert.equal(tabs.read(node).tabs[0].values, undefined);
});

test('"+" names a tab after its place, skipping the names in use', () => {
    const node = makeNode(A);
    tabs.edit(node).tabs[0].name = 'Wand close-up';
    tabs.add(node);
    assert.deepEqual(names(node), ['Wand close-up', 'Prompt B', 'Prompt C']);
    assert.equal(tabs.read(node).active, 2);
    assert.deepEqual(shown(node), DEFAULTS);
    tabs.remove(node, 1);
    tabs.add(node);
    assert.deepEqual(names(node), ['Wand close-up', 'Prompt C', 'Prompt D']);
});

test('duplicate copies the shown values, or a hidden tab\'s, and shows the copy', () => {
    const node = makeNode(A);
    tabs.duplicate(node, 0);
    assert.deepEqual(names(node), ['Prompt A', 'Prompt A copy', 'Prompt B']);
    assert.equal(tabs.read(node).active, 1);
    assert.deepEqual(shown(node), { ...DEFAULTS, ...A });
    node.widgets.find(w => w.name === 'framing').value = 'Wide Shot (WS)';
    tabs.duplicate(node, 0);
    assert.equal(tabs.read(node).active, 1);
    assert.equal(shown(node).framing, 'Close-Up (CU)');
    assert.equal(tabs.read(node).tabs[2].values.framing, 'Wide Shot (WS)');
});

test('deleting the shown tab shows its neighbour, and the last tab stays', () => {
    const node = makeNode(A);
    tabs.remove(node, 0);
    assert.deepEqual(names(node), ['Prompt B']);
    assert.deepEqual(shown(node), DEFAULTS);
    tabs.remove(node, 0);
    assert.deepEqual(names(node), ['Prompt B']);
});

test('"next tab" moves to the next tab with a prompt, and "fixed" stays', () => {
    const node = makeNode(A);
    tabs.add(node);
    node.widgets.find(w => w.name === 'base_prompt').value = 'a snowy forest';
    tabs.select(node, 0);
    tabs.chain(node);
    assert.equal(tabs.read(node).active, 0);
    tabs.edit(node).chain = true;
    tabs.chain(node);
    assert.equal(tabs.read(node).active, 2, 'Prompt B has no prompt');
    tabs.chain(node);
    assert.equal(tabs.read(node).active, 0);
    assert.equal(shown(node).base_prompt, A.base_prompt);
    tabs.remove(node, 2);
    tabs.chain(node);
    assert.equal(tabs.read(node).active, 0, 'no other tab has a prompt');
});

test('moving a tab keeps the shown one shown, with its values', () => {
    const node = makeNode(A);
    tabs.add(node);
    tabs.select(node, 0);
    tabs.move(node, 0, 2);
    assert.deepEqual(names(node), ['Prompt B', 'Prompt C', 'Prompt A']);
    assert.equal(tabs.read(node).active, 2);
    tabs.move(node, 1, 0);
    assert.deepEqual(names(node), ['Prompt C', 'Prompt B', 'Prompt A']);
    assert.equal(tabs.read(node).active, 2);
    assert.deepEqual(shown(node), { ...DEFAULTS, ...A });
});

test('"Run all tabs" runs each tab that has a prompt, then shows the first one again', async () => {
    const node = makeNode(A);
    tabs.add(node);
    node.widgets.find(w => w.name === 'base_prompt').value = 'a snowy forest';
    tabs.select(node, 0);
    tabs.edit(node).chain = true;
    const sent = [];
    context.app.queuePrompt = async () => {
        sent.push(node.widgets.find(w => w.name === 'base_prompt').value);
        for (const w of node.widgets) w.afterQueued?.();
        return true;
    };
    await tabs.runAll(node);
    assert.deepEqual(sent, [A.base_prompt, 'a snowy forest'], 'Prompt B has no prompt');
    assert.equal(tabs.read(node).active, 0);
    assert.deepEqual(shown(node), { ...DEFAULTS, ...A });
});

test('a run that fails to build still brings the first tab back and frees "next tab"', async () => {
    const node = makeNode(A);
    tabs.add(node);
    node.widgets.find(w => w.name === 'base_prompt').value = 'a snowy forest';
    tabs.select(node, 0);
    context.app.queuePrompt = async () => { throw new Error('graphToPrompt failed'); };
    await assert.rejects(tabs.runAll(node), /graphToPrompt failed/);
    assert.equal(tabs.read(node).active, 0);
    assert.equal(node._radRunningTabs, false);
});
