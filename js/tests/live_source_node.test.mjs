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
