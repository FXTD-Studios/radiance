import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const context = vm.createContext({
    app: { registerExtension() {} },
    api: { fetchApi: () => new Promise(() => {}) },
    console: { log() {}, warn() {} },
});
vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_prompt.js')
    + 'globalThis.applyBase = _applyBasePromptState;', context);
// The real setWidgetVisible touches layout the test does not need.
context.setWidgetVisible = (widget, visible) => { widget.hidden = !visible; };

const MENUS = ['style_preset', 'framing', 'camera_type', 'lens_focal', 'aperture_dof', 'lighting',
    'style_aesthetic', 'color_grading'];

// A Prompt with text_to_video set to `own`, wired to `source` ({cls, mode, value}) when given;
// `lastRun` is the value its last run reported.
function prompt({ own = true, source, imageModel = false, lastRun } = {}) {
    const nodes = {};
    const graph = { links: {}, getNodeById: id => nodes[id] };
    const node = { id: 1, graph, _radTextToVideo: lastRun, inputs: [{ name: 'text_to_video', type: 'BOOLEAN', link: source ? 3 : null }],
        widgets: [{ name: 'text_to_video', value: own }, { name: 'base_prompt', value: 'a woman on a dock' },
                  { name: 'action', value: '', _radIgnored: imageModel },
                  ...MENUS.map(name => ({ name, value: 'None' }))] };
    if (source) {
        nodes[2] = { id: 2, graph, comfyClass: source.cls, mode: source.mode ?? 0, inputs: [],
                     widgets: 'value' in source ? [{ name: 'value', value: source.value }] : [] };
        graph.links[3] = { origin_id: 2, target_id: 1, type: 'BOOLEAN' };
    }
    context.applyBase(node);
    return node;
}

const widget = (node, name) => node.widgets.find(w => w.name === name);
const menusHidden = node => MENUS.map(name => !!widget(node, name).hidden);
const all = value => MENUS.map(() => value);

test('off dims base_prompt and hides the menus, by hand or from a Boolean node', () => {
    for (const node of [prompt({ own: false }), prompt({ source: { cls: 'PrimitiveBoolean', value: false } })]) {
        const base = widget(node, 'base_prompt');
        assert.equal(base._radIgnored, true);
        assert.equal(base.label, 'base_prompt ⚠ image-to-video, ignored');
        assert.equal(base.value, 'a woman on a dock');
        assert.deepEqual(menusHidden(node), all(true));
    }
});

test('a wire wins over the switch, and an unreadable or muted one leaves the node as is', () => {
    for (const source of [{ cls: 'PrimitiveBoolean', value: true }, { cls: 'SomeLogicNode' },
                          { cls: 'PrimitiveBoolean', value: false, mode: 2 }]) {
        const node = prompt({ own: false, source });
        assert.equal(widget(node, 'base_prompt')._radIgnored, false);
        assert.equal(widget(node, 'base_prompt').label ?? 'base_prompt', 'base_prompt');
        assert.deepEqual(menusHidden(node), all(false));
    }
});

test('an unreadable wire follows the last run, a readable one does not', () => {
    assert.equal(widget(prompt({ source: { cls: 'SomeLogicNode' }, lastRun: false }), 'base_prompt')._radIgnored, true);
    assert.equal(widget(prompt({ source: { cls: 'SomeLogicNode' }, lastRun: true }), 'base_prompt')._radIgnored, false);
    assert.equal(widget(prompt({ source: { cls: 'PrimitiveBoolean', value: true }, lastRun: false }), 'base_prompt')._radIgnored, false);
});

test('turning it back on clears the marker and shows the menus', () => {
    const node = prompt({ own: false });
    widget(node, 'text_to_video').value = true;
    context.applyBase(node);
    assert.equal(widget(node, 'base_prompt').label, 'base_prompt');
    assert.deepEqual(menusHidden(node), all(false));
});

test('the switch hides under an image model, unless it is off', () => {
    assert.equal(!!widget(prompt({ imageModel: true }), 'text_to_video').hidden, true);
    assert.equal(!!widget(prompt({ imageModel: true, own: false }), 'text_to_video').hidden, false);
    assert.equal(!!widget(prompt(), 'text_to_video').hidden, false);
});

test('a wired switch is left to the frontend, which shows its socket row', () => {
    const node = prompt({ imageModel: true, source: { cls: 'PrimitiveBoolean', value: true } });
    assert.equal(widget(node, 'text_to_video').hidden, undefined);
});
