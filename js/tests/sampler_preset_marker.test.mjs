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
