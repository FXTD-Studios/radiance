import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_widget_utils.js', import.meta.url), 'utf8')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');
const context = vm.createContext({});
vm.runInContext(source + 'globalThis.liveSource = liveSourceNode;', context);

// A Loader feeding a Sampler's model_meta, with `middle` nodes (their modes) in between.
function chain(loaderMode, ...middle) {
    const nodes = {};
    const links = {};
    const graph = { links, getNodeById: id => nodes[id] };
    nodes[1] = { id: 1, mode: loaderMode, inputs: [], graph };
    let upstream = 1;
    middle.forEach((mode, i) => {
        const id = 10 + i;
        links[id] = { origin_id: upstream, type: 'STRING' };
        nodes[id] = { id, mode, graph, inputs: [{ name: 'text', type: 'STRING', link: id }] };
        upstream = id;
    });
    links[99] = { origin_id: upstream, type: 'STRING' };
    const sampler = { graph, inputs: [{ name: 'model_meta', type: 'STRING', link: 99 }] };
    return context.liveSource(sampler, sampler.inputs[0]);
}

test('the node feeding the input is the source', () => {
    assert.equal(chain(0).id, 1);
    assert.equal(chain(0, 0).id, 10);
});

test('a bypassed node in between hands over to what feeds it', () => {
    assert.equal(chain(0, 4).id, 1);
    assert.equal(chain(0, 4, 4).id, 1);
});

test('a muted source, or a muted node in between, feeds nothing', () => {
    assert.equal(chain(2), null);
    assert.equal(chain(0, 2), null);
    assert.equal(chain(2, 4), null);
});

// A bypassed node with two STRING inputs and two STRING outputs, fed by
// Loaders 1 and 2; the target reads its output `slot`.
function twoInputBypass(slot, inputTypes = ['STRING', 'STRING']) {
    const nodes = {};
    const links = {};
    const graph = { links, getNodeById: id => nodes[id] };
    nodes[1] = { id: 1, mode: 0, inputs: [], graph };
    nodes[2] = { id: 2, mode: 0, inputs: [], graph };
    links[31] = { origin_id: 1, origin_slot: 0, type: 'STRING' };
    links[32] = { origin_id: 2, origin_slot: 0, type: 'STRING' };
    nodes[5] = {
        id: 5, mode: 4, graph,
        inputs: [{ name: 'a', type: inputTypes[0], link: 31 }, { name: 'b', type: inputTypes[1], link: 32 }],
        outputs: [{ type: 'STRING' }, { type: 'STRING' }],
    };
    links[99] = { origin_id: 5, origin_slot: slot, type: 'STRING' };
    const target = { graph, inputs: [{ name: 'model_meta', type: 'STRING', link: 99 }] };
    return context.liveSource(target, target.inputs[0]);
}

test('a bypassed node forwards the input at the output slot index, as ComfyUI does', () => {
    assert.equal(twoInputBypass(0).id, 1);
    assert.equal(twoInputBypass(1).id, 2);
});

test('without a same-index match, the first input of the exact type is forwarded', () => {
    assert.equal(twoInputBypass(0, ['INT', 'STRING']).id, 2);
});

test('modelMetaSourceNode reads the model_meta input', () => {
    vm.runInContext('globalThis.metaSource = modelMetaSourceNode;', context);
    const loader = { id: 1, mode: 0, inputs: [] };
    const graph = { links: { 7: { origin_id: 1, origin_slot: 0, type: 'STRING' } }, getNodeById: () => loader };
    const node = { graph, inputs: [{ name: 'clip', type: 'CLIP', link: null }, { name: 'model_meta', type: 'STRING', link: 7 }] };
    assert.equal(context.metaSource(node), loader);
    assert.equal(context.metaSource({ graph, inputs: [] }), null);
});
