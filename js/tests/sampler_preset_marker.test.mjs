import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// The Sampler with the shared helpers it imports from radiance_widget_utils.js.
const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');
const source = strip('radiance_widget_utils.js') + strip('radiance_sampler.js');

function samplerContext() {
    const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} } });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => ({ widgets: [{ name: 'unet_name', value: 'ltx-2.3-22b-dev.safetensors' }] });
        loaderModelType = (loader) => (loader ? 'ltxav' : null);
        _isLtxAvHighResStage = () => false;
        _modelLoraNames = () => [];
        _isSdTurboActive = () => false;
        globalThis.markers = updatePresetDivergenceMarkers;
        globalThis.metaDefaults = updateModelMetaDefaults;
    `, context);
    return context;
}

// The 250 ms poll, in its order, counting every label write after the first cycle.
function poll(context, node, cycles) {
    let writes = 0;
    for (let i = 0; i < cycles; i++) {
        const before = node.widgets.map(w => w.label);
        context.markers(node);
        context.metaDefaults(node);
        if (i > 0) writes += node.widgets.filter((w, k) => w.label !== before[k]).length;
    }
    return writes;
}

function ltxNode(preset, cfg) {
    return {
        widgets: [
            { name: 'preset', value: preset },
            { name: 'cfg', value: cfg },
            { name: 'sampler', value: 'euler' },
            { name: 'steps', value: 20 },
        ],
        setDirtyCanvas() {},
    };
}

test('a cfg edited away from a named preset keeps its pencil, with no label churn', () => {
    const context = samplerContext();
    const node = ltxNode('[V] LTX 2.3 LowRes (20 steps)', 4.0);
    const writes = poll(context, node, 4);
    assert.equal(node.widgets[1].label, 'cfg ✎');
    assert.equal(node.widgets[2].label, undefined);
    assert.equal(writes, 0);
});

test('Auto on LTX 2.5 Dev follows the 30-step LowRes preset, audio_cfg included', () => {
    const context = samplerContext();
    vm.runInContext(`_findModelMetaSourceNode = () => ({ widgets: [{ name: 'unet_name', value: 'ltx-2.5-22b-dev-transformer-bf16.safetensors' }] });`, context);
    const node = ltxNode('Auto', 3.0);
    node.widgets[2].value = 'euler_ancestral';
    node.widgets.push({ name: 'audio_cfg', value: 0.0 });
    context.metaDefaults(node);
    const value = (name) => node.widgets.find(w => w.name === name).value;
    assert.equal(value('steps'), 30);
    assert.equal(value('sampler'), 'euler');
    assert.equal(value('audio_cfg'), 1.0);
});

test('Auto on an LTX Distilled file follows the Distilled presets, 8 steps then 3', () => {
    const context = samplerContext();
    vm.runInContext(`_findModelMetaSourceNode = () => ({ widgets: [{ name: 'unet_name', value: 'ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors' }] });`, context);
    for (const [highRes, steps, denoise] of [[false, 8, 1.0], [true, 3, 0.375]]) {
        vm.runInContext(`_isLtxAvHighResStage = () => ${highRes};`, context);
        const node = ltxNode('Auto', 3.0);
        node.widgets.push({ name: 'scheduler', value: 'simple' }, { name: 'denoise', value: 1.0 });
        context.metaDefaults(node);
        const value = (name) => node.widgets.find(w => w.name === name).value;
        assert.deepEqual([value('steps'), value('cfg'), value('sampler'), value('scheduler'), value('denoise')],
            [steps, 1.0, 'euler_ancestral', 'linear_quadratic', denoise]);
    }
});

test('Auto takes the LTX stage schedule and guidance too', () => {
    const context = samplerContext();
    vm.runInContext(`_findModelMetaSourceNode = () => ({ widgets: [{ name: 'unet_name', value: 'ltx-2.5-22b-dev-transformer-bf16.safetensors' }] });`, context);
    const names = ['steps', 'flux_shift', 'force_exact_steps', 'terminal_sigma_to_zero', 'ltx_guidance', 'guidance_rescale_phi'];
    for (const [highRes, expected] of [[false, [30, 3.0, true, true, 'Modality + STG (early)', 0.45]],
                                       [true, [40, 6.0, true, true, 'Off', 0.0]]]) {
        vm.runInContext(`_isLtxAvHighResStage = () => ${highRes};`, context);
        const node = ltxNode('Auto', 1.0);
        node.widgets.push({ name: 'flux_shift', value: 1.0 }, { name: 'force_exact_steps', value: false },
            { name: 'terminal_sigma_to_zero', value: false }, { name: 'ltx_guidance', value: 'Off' },
            { name: 'guidance_rescale_phi', value: 0.0 });
        context.metaDefaults(node);
        assert.deepEqual(names.map(name => node.widgets.find(w => w.name === name).value), expected);
    }
});

test('leaving Auto clears the magnet, then the preset pencil takes over', () => {
    const context = samplerContext();
    const node = ltxNode('Auto', 3.0);
    poll(context, node, 2);
    assert.equal(node.widgets[1].label, 'cfg 🧲');
    node.widgets[0].value = '[V] LTX 2.3 LowRes (20 steps)';
    node.widgets[1].value = 4.0;
    poll(context, node, 3);
    assert.equal(node.widgets[1].label, 'cfg ✎');
});
