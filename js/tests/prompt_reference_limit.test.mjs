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
    + 'globalThis.applyLimit = _applyReferenceLimit;', context);

// A Prompt fed by a Loader with `loader` widgets; `connected` image slots are
// linked, then one empty slot follows, as the frontend's Autogrow leaves it.
function prompt(loader, connected, max = 16) {
    const nodes = { 4: { id: 4, mode: 0, inputs: [],
                         widgets: Object.entries(loader).map(([name, value]) => ({ name, value })) } };
    const links = { 8: { origin_id: 4, target_id: 1, type: 'STRING' } };
    const inputs = [{ name: 'model_meta', type: 'STRING', link: 8 }];
    for (let n = 1; n <= connected + 1 && n <= max; n++) {
        const link = n <= connected ? 100 + n : null;
        if (link) links[link] = { origin_id: 50 + n, target_id: 1, type: 'IMAGE' };
        inputs.push({ name: `images.image_${n}`, type: 'IMAGE', link });
    }
    const graph = { links, getNodeById: id => nodes[id] ?? { id, mode: 0 } };
    nodes[4].graph = graph;
    const replayed = [];
    return {
        graph, inputs, replayed,
        comfyDynamic: { autogrow: { images: { names: Array.from({ length: 16 }, (_, i) => `image_${i + 1}`), max } } },
        removeInput(index) { this.inputs.splice(index, 1); },
        onConnectionsChange(type, index, connected, link) { replayed.push([type, index, connected, link]); },
        setSize() {}, size: [300, 200],
    };
}

const slots = node => node.inputs.filter(i => i.name.startsWith('images.')).map(i => i.name.slice(7));

test('a Qwen-Image Edit Loader stops the slots at image_3', () => {
    const node = prompt({ preset: 'Qwen-Image Edit 2511' }, 3);
    context.applyLimit(node);
    assert.equal(node.comfyDynamic.autogrow.images.max, 3);
    assert.deepEqual(slots(node), ['image_1', 'image_2', 'image_3']);
});

test('a connected slot past the limit stays, for the run to report it', () => {
    const node = prompt({ preset: 'Custom', model_type: 'qwen_image' }, 4);
    context.applyLimit(node);
    assert.deepEqual(slots(node), ['image_1', 'image_2', 'image_3', 'image_4']);
});

test('the other models keep the 16 slots, Qwen-Image 2.1 included', () => {
    for (const loader of [{ preset: 'Flux.2' }, { preset: 'Qwen-Image 2.1' }, { preset: 'Custom', model_type: 'Auto-Detect' }]) {
        const node = prompt(loader, 3);
        context.applyLimit(node);
        assert.equal(node.comfyDynamic.autogrow.images.max, 16);
        assert.deepEqual(slots(node), ['image_1', 'image_2', 'image_3', 'image_4']);
        assert.deepEqual(node.replayed, []);
    }
});

test('leaving Qwen-Image Edit replays the last connection, so image_4 comes back', () => {
    const node = prompt({ preset: 'Flux.2' }, 3, 3);
    context.applyLimit(node);
    assert.equal(node.comfyDynamic.autogrow.images.max, 16);
    assert.deepEqual(node.replayed, [[1, 3, true, node.graph.links[103]]]);
});
