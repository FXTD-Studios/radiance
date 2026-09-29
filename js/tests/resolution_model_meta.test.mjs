// Resolution's model_type follows the Loader on model_meta (🧲) until the user
// picks another one (✎), as the Sampler's widgets do.
//
// Run: node --test js/tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} } });
vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_resolution.js')
    + 'globalThis.sync = syncModelTypeFromLoader;', context);

// A Resolution whose model_meta comes from a Loader with 'preset'.
function resolution(preset, { unet = '', modelType = 'Manual' } = {}) {
    const loader = { id: 1, mode: 0, widgets: [{ name: 'preset', value: preset }, { name: 'unet_name', value: unet },
                                              { name: 'model_type', value: 'Auto-Detect' }] };
    const graph = { links: { 5: { origin_id: 1, origin_slot: 0, target_id: 2, type: 'STRING' } }, getNodeById: id => ({ 1: loader })[id] };
    loader.graph = graph;
    const calls = [];
    const modelTypeW = { name: 'model_type', value: modelType, callback(value) { calls.push(value); } };
    const node = { id: 2, graph, widgets: [modelTypeW], inputs: [{ name: 'model_meta', type: 'STRING', link: 5 }] };
    return { node, loader, modelTypeW, calls };
}

test('model_type follows the Loader and runs its callback, marked 🧲', () => {
    const { node, modelTypeW, calls } = resolution('Qwen-Image');
    assert.equal(context.sync(node), true);
    assert.equal(modelTypeW.value, 'Qwen-Image / Krea 2 (16ch)');
    assert.deepEqual(calls, ['Qwen-Image / Krea 2 (16ch)']);
    assert.equal(modelTypeW.label, 'model_type 🧲');
    assert.equal(context.sync(node), false);
});

test('a later Loader change follows too, Flux.2 Klein from its unet_name', () => {
    const { node, loader, modelTypeW } = resolution('Wan 2.2');
    context.sync(node);
    assert.equal(modelTypeW.value, 'WAN (16ch)');
    loader.widgets[0].value = 'Flux.2';
    loader.widgets[1].value = 'flux-2-klein-9b.safetensors';
    context.sync(node);
    assert.equal(modelTypeW.value, 'Flux.2 / Flux.2 Klein (128ch)');
});

test('a model_type picked by hand stays, marked ✎, until it matches the Loader again', () => {
    const { node, loader, modelTypeW } = resolution('Qwen-Image');
    context.sync(node);
    modelTypeW.value = 'Chroma (16ch)';
    context.sync(node);
    assert.equal(modelTypeW.label, 'model_type ✎');
    loader.widgets[0].value = 'SDXL';
    context.sync(node);
    assert.equal(modelTypeW.value, 'Chroma (16ch)');
    modelTypeW.value = 'SDXL / SD 1.5 / PixArt / Aura Flow (4ch)';
    context.sync(node);
    assert.equal(modelTypeW.label, 'model_type 🧲');
});

test('Manual picked by hand follows the Loader again, as at execution', () => {
    const { node, modelTypeW } = resolution('Qwen-Image');
    context.sync(node);
    modelTypeW.value = 'Manual';
    context.sync(node);
    assert.equal(modelTypeW.value, 'Qwen-Image / Krea 2 (16ch)');
    assert.equal(modelTypeW.label, 'model_type 🧲');
});

test('no Loader, or one still auto-detecting, leaves model_type alone and unmarked', () => {
    const detecting = resolution('Custom', { modelType: 'LTXV (128ch)' });
    context.sync(detecting.node);
    assert.equal(detecting.modelTypeW.value, 'LTXV (128ch)');
    assert.deepEqual(detecting.calls, []);
    assert.equal(detecting.modelTypeW.label, undefined);
    const unlinked = resolution('Qwen-Image');
    unlinked.node.inputs[0].link = null;
    context.sync(unlinked.node);
    assert.equal(unlinked.modelTypeW.value, 'Manual');
});
