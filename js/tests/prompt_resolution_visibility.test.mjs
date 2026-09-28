import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const context = vm.createContext({
    app: { registerExtension() {}, graph: { setDirtyCanvas() {} } },
    api: { fetchApi: () => new Promise(() => {}) },
    console: { log() {}, warn() {} },
});
vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_prompt.js')
    + 'globalThis.applyResolution = _applyResolutionVisibility;', context);
// The real setWidgetVisible touches layout the test does not need.
context.setWidgetVisible = (widget, visible) => { widget.hidden = !visible; };

// A Prompt whose image_1 comes from `source`; `through` puts a node between them.
function prompt(source, through, loader) {
    const nodes = {};
    const links = {};
    const add = (id, spec) => (nodes[id] = { id, mode: spec.mode ?? 0, inputs: spec.inputs ?? [] });
    const load = add(2, { mode: source });
    let feed = load;
    if (through !== undefined) {
        links[5] = { origin_id: load.id, target_id: 3, type: 'IMAGE' };
        feed = add(3, { mode: through, inputs: [{ name: 'image', type: 'IMAGE', link: 5 }] });
    }
    links[7] = { origin_id: feed.id, target_id: 1, type: 'IMAGE' };
    if (loader) {
        nodes[4] = { id: 4, mode: 0, inputs: [],
                     widgets: Object.entries(loader).map(([name, value]) => ({ name, value })) };
        links[8] = { origin_id: 4, target_id: 1, type: 'STRING' };
    }
    const graph = { links, getNodeById: id => nodes[id] };
    Object.values(nodes).forEach(n => { n.graph = graph; });   // as LiteGraph sets it
    return {
        graph,
        widgets: [{ name: 'resolution', value: 1024, hidden: true }],
        inputs: [{ name: 'clip', link: null }, { name: 'images.image_1', type: 'IMAGE', link: 7 },
                 { name: 'images.image_2', type: 'IMAGE', link: null },
                 { name: 'model_meta', type: 'STRING', link: loader ? 8 : null }],
    };
}

const shown = (node) => { context.applyResolution(node); return !node.widgets[0].hidden; };

test('an image from a running node shows resolution', () => {
    assert.equal(shown(prompt(0)), true);
});

test('a muted or bypassed Load Image hides it', () => {
    assert.equal(shown(prompt(2)), false);
    assert.equal(shown(prompt(4)), false);
});

test('a bypassed node in between forwards the image, as ComfyUI does', () => {
    assert.equal(shown(prompt(0, 4)), true);
    assert.equal(shown(prompt(2, 4)), false);
});

test('no image connected hides it, a later disconnect hides it again', () => {
    const node = prompt(0);
    assert.equal(shown(node), true);
    node.inputs[1].link = null;
    assert.equal(shown(node), false);
});

test('a Flux.2 or Qwen-Image Edit Loader hides it: they size their references themselves', () => {
    assert.equal(shown(prompt(0, undefined, { preset: 'Flux.2' })), false);
    assert.equal(shown(prompt(0, undefined, { preset: 'Custom', model_type: 'flux2-klein' })), false);
    assert.equal(shown(prompt(0, undefined, { preset: 'Qwen-Image Edit 2511' })), false);
    assert.equal(shown(prompt(0, undefined, { preset: 'Custom', model_type: 'qwen_image' })), false);
});

test('a Qwen-Image 2.1 Loader, or one still auto-detecting, keeps it', () => {
    assert.equal(shown(prompt(0, undefined, { preset: 'Qwen-Image 2.1' })), true);
    assert.equal(shown(prompt(0, undefined, { preset: 'Custom', model_type: 'Auto-Detect' })), true);
});
