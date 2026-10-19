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
        _setWidgetVisible = (w, visible) => { const changed = !w.hidden !== visible; w.hidden = !visible; return changed; };
        _forceWidgetReinsert = () => {};
        fitNodeSize = () => {};
        _isSdTurboActive = () => false;
        globalThis.fold = applyFolding;
        globalThis.fill = fillLtxGuidanceSettings;
        globalThis.markers = updatePresetDivergenceMarkers;
    `, context);
    return context;
}

function samplerNode(preset, ltxGuidance, [stg, end, modality] = [0.0, 1.0, 1.0]) {
    return {
        widgets: [
            { name: 'preset', value: preset },
            { name: 'model_type', value: 'auto' },
            { name: 'cfg', value: 3.0 },
            { name: 'guidance_rescale_phi', value: 0.45 },
            { name: 'ltx_guidance', value: ltxGuidance },
            { name: 'stg_scale', value: stg },
            { name: 'stg_end', value: end },
            { name: 'modality_scale', value: modality },
        ],
        setDirtyCanvas() {},
    };
}

const LTX25_LOWRES = '[V] LTX 2.5 LowRes (30 steps)';
const shown = (node) => node.widgets.filter(w => !w.hidden).map(w => w.name);
const value = (node, name) => node.widgets.find(w => w.name === name).value;
const label = (node, name) => node.widgets.find(w => w.name === name).label;

test('ltx_guidance shows its three settings only on Custom, and only for LTX 2.3 / 2.5', () => {
    const context = samplerContext();
    const node = samplerNode(LTX25_LOWRES, 'Modality + STG (early)', [1.0, 0.5, 3.0]);
    context.fold(node);
    assert.deepEqual(shown(node).slice(-1), ['ltx_guidance']);
    node.widgets[4].value = 'Custom';
    context.fold(node);
    assert.deepEqual(shown(node).slice(-4), ['ltx_guidance', 'stg_scale', 'stg_end', 'modality_scale']);
    node.widgets[0].value = '[F] Flux txt2img';
    context.fold(node);
    assert.ok(!shown(node).includes('ltx_guidance') && !shown(node).includes('stg_scale'));
});

test('Custom starts from the recipe it replaces', () => {
    const context = samplerContext();
    const node = samplerNode(LTX25_LOWRES, 'Modality', [2.0, 0.2, 7.0]);
    context.fold(node);
    node.widgets[4].value = 'Custom';
    context.fill(node);
    assert.deepEqual(['stg_scale', 'stg_end', 'modality_scale'].map(n => value(node, n)), [0.0, 1.0, 3.0]);
});

test('the LTX 2.5 LowRes preset carries Modality + STG (early); leaving it marks the change', () => {
    const context = samplerContext();
    const node = samplerNode(LTX25_LOWRES, 'Modality + STG (early)', [1.0, 0.5, 3.0]);
    context.markers(node);
    assert.equal(label(node, 'ltx_guidance'), undefined);
    assert.equal(label(node, 'guidance_rescale_phi'), undefined);
    node.widgets[4].value = 'Custom';
    context.markers(node);
    assert.equal(label(node, 'ltx_guidance'), 'ltx_guidance ✎');
    assert.equal(label(node, 'stg_end'), undefined);
    node.widgets[6].value = 0.6;
    context.markers(node);
    assert.equal(label(node, 'stg_end'), 'stg_end ✎');
});
