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
    LiteGraph: { INPUT: 1 },
});
vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_prompt.js')
    + 'globalThis.applySockets = _applyImageSockets;', context);

// A Prompt fed by a Loader with `loader` widgets (none: no Loader), `image` linked on image_1
// (or 'image_2', image_1 then left empty), and video_latent / latent present, linked or not.
function prompt({ loader, image = true, input = 'free', output = 'free' }) {
    const nodes = { 50: { id: 50, mode: 0 } };
    const links = { 51: { origin_id: 50, target_id: 1, type: 'IMAGE' } };
    const inputs = [];
    if (loader) {
        nodes[4] = { id: 4, mode: 0, inputs: [],
                     widgets: Object.entries(loader).map(([name, value]) => ({ name, value })) };
        links[8] = { origin_id: 4, target_id: 1, type: 'STRING' };
        inputs.push({ name: 'model_meta', type: 'STRING', link: 8 });
    }
    inputs.push({ name: 'images.image_1', type: 'IMAGE', link: image === true ? 51 : null });
    if (image === 'image_2') inputs.push({ name: 'images.image_2', type: 'IMAGE', link: 51 });
    if (input !== 'gone') inputs.push({ name: 'video_latent', type: 'LATENT', link: input === 'linked' ? 60 : null });
    inputs.push({ name: 'text_to_video', type: 'BOOLEAN', link: null });
    const outputs = [{ name: 'positive', type: 'CONDITIONING', links: [] }];
    if (output !== 'gone') outputs.push({ name: 'latent', type: 'LATENT', links: output === 'linked' ? [70] : [] });
    const graph = { links, getNodeById: id => nodes[id] ?? { id, mode: 0 } };
    if (loader) nodes[4].graph = graph;
    return {
        graph, inputs, outputs,
        _radImageSlots: { input: { localized_name: 'video_latent', shape: 7 }, output: { localized_name: 'latent' } },
        addInput(name, type, extra) { this.inputs.push({ name, type, link: null, ...extra }); },
        removeInput(index) { this.inputs.splice(index, 1); },
        addOutput(name, type, extra) { this.outputs.push({ name, type, links: [], ...extra }); },
        removeOutput(index) { this.outputs.splice(index, 1); },
        setSize() {}, size: [300, 200],
    };
}

const names = slots => slots.map(s => s.name);

test('without an image, video_latent and the latent output go', () => {
    const node = prompt({ loader: { preset: 'MiniMax H3' }, image: false });
    context.applySockets(node);
    assert.deepEqual(names(node.inputs), ['model_meta', 'images.image_1', 'text_to_video', 'first_frame', 'last_frame']);
    assert.deepEqual(names(node.outputs), ['positive']);
});

test('under MiniMax H3, a first or last frame alone brings video_latent and the latent output', () => {
    for (const name of ['first_frame', 'last_frame']) {
        const node = prompt({ loader: { preset: 'MiniMax H3' }, image: false, input: 'gone', output: 'gone' });
        context.applySockets(node);
        node.inputs.find(i => i.name === name).link = 51;
        context.applySockets(node);
        assert.equal(node.inputs.at(-1).name, 'video_latent');
        assert.deepEqual(names(node.outputs), ['positive', 'latent']);
    }
});

test('another family drops the free frame sockets and keeps a linked one', () => {
    const node = prompt({ loader: { preset: 'Flux.2' }, image: false, input: 'gone', output: 'gone' });
    node.inputs.push({ name: 'first_frame', type: 'IMAGE', link: null }, { name: 'last_frame', type: 'IMAGE', link: 51 });
    context.applySockets(node);
    assert.deepEqual(names(node.inputs).filter(n => n.endsWith('_frame')), ['last_frame']);
});

test('an image under MiniMax H3, or an unknown family, brings both back, last', () => {
    for (const loader of [{ preset: 'MiniMax H3' }, { preset: 'Custom', model_type: 'Auto-Detect' }, undefined]) {
        for (const image of [true, 'image_2']) {
            const node = prompt({ loader, image, input: 'gone', output: 'gone' });
            context.applySockets(node);
            assert.deepEqual(node.inputs.at(-1), { name: 'video_latent', type: 'LATENT', link: null, localized_name: 'video_latent', shape: 7 });
            assert.deepEqual(names(node.outputs), ['positive', 'latent']);
        }
    }
});

test('another family keeps the latent output and drops video_latent', () => {
    const node = prompt({ loader: { preset: 'Qwen-Image 2.1' } });
    context.applySockets(node);
    assert.ok(!names(node.inputs).includes('video_latent'));
    assert.deepEqual(names(node.outputs), ['positive', 'latent']);
});

test('a linked socket stays without an image', () => {
    const node = prompt({ loader: { preset: 'MiniMax H3' }, image: false, input: 'linked', output: 'linked' });
    context.applySockets(node);
    assert.ok(names(node.inputs).includes('video_latent'));
    assert.deepEqual(names(node.outputs), ['positive', 'latent']);
});
