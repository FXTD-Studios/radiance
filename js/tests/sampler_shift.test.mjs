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

function samplerContext(arch) {
    const context = vm.createContext({
        app: { registerExtension(ext) { context.extension = ext; } }, console: { log() {} }, arch,
        setTimeout() {},
    });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => ({ widgets: [] });
        loaderModelType = () => arch;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
        globalThis.presets = PRESET_CONFIGS;
    `, context);
    return context;
}

for (const arch of ['wan', 'wan_ti2v', 'ltxv', 'lumina2']) {
    for (const preset of ['Auto', 'Custom']) {
        test(`${arch} ${preset} keeps flux_shift and marks an extra shift`, () => {
            const context = samplerContext(arch);
            for (const [shift, label] of [[1, 'flux_shift 🧲'], [2, 'flux_shift ✎']]) {
                const widget = { name: 'flux_shift', value: shift };
                const node = {
                    widgets: [{ name: 'preset', value: preset }, widget],
                    setDirtyCanvas() {},
                };
                context.updateDefaults(node);
                context.updateDefaults(node);
                assert.equal(widget.value, shift);
                assert.equal(widget.label, label);
            }
        });
    }
}

test('a live sigmas_override supplies the schedule: flux_shift gets no marker', () => {
    const widget = { name: 'flux_shift', value: 3 };
    const graph = { links: { 7: { origin_id: 5, type: 'SIGMAS' } }, getNodeById: id => (id === 5 ? { id, mode: 0 } : null) };
    const node = {
        graph, inputs: [{ name: 'sigmas_override', type: 'SIGMAS', link: 7 }],
        widgets: [{ name: 'preset', value: 'Auto' }, widget], setDirtyCanvas() {},
    };
    samplerContext('wan').updateDefaults(node);
    assert.equal(widget.value, 3);
    assert.ok(!widget._radMetaLinked);
    node.inputs[0].link = null;
    samplerContext('wan').updateDefaults(node);
    assert.equal(widget.label, 'flux_shift ✎');
});

test('LTX-AV: a live sigmas_override leaves the stage shift unmarked, and the marker comes back without it', () => {
    const context = vm.createContext({
        app: { registerExtension() {} }, console: { log() {} }, setTimeout() {},
    });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => ({ widgets: [{ name: "unet_name", value: "ltx-2.5-22b-dev-transformer-bf16.safetensors" }] });
        loaderModelType = () => "ltxav";
        _isSdTurboActive = () => false;
        _isLtxAvHighResStage = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
    `, context);
    const widget = { name: 'flux_shift', value: 1 };
    const graph = { links: { 7: { origin_id: 5, type: 'SIGMAS' } }, getNodeById: id => (id === 5 ? { id, mode: 0 } : null) };
    const node = {
        graph, inputs: [{ name: 'sigmas_override', type: 'SIGMAS', link: 7 }],
        widgets: [{ name: 'preset', value: 'Auto' }, widget], setDirtyCanvas() {},
    };
    context.updateDefaults(node);
    const stageShift = widget.value;
    assert.notEqual(stageShift, 1, 'the LowRes stage shift is still synced');
    assert.ok(!widget._radMetaLinked);
    node.inputs[0].link = null;
    context.updateDefaults(node);
    assert.ok(widget._radMetaLinked);
});

for (const arch of ['sdxl', 'ltxav']) {
    test(`${arch} Auto leaves flux_shift unmarked without its own shift`, () => {
        const widget = { name: 'flux_shift', value: 2 };
        samplerContext(arch).updateDefaults({
            widgets: [{ name: 'preset', value: 'Auto' }, widget], setDirtyCanvas() {},
        });
        assert.equal(widget.value, 2);
        assert.ok(!widget._radMetaLinked);
    });
}

test('presets do not add another native shift', () => {
    const { presets } = samplerContext('wan');
    for (const name of ['▶ WAN txt2vid (30 steps)', '▶ WAN img2vid (20 steps)',
        '▶ LTX-Video (25 steps)', '◈ z_image (25 steps)', '◈ Lumina2 (25 steps)']) {
        assert.equal(presets[name].flux_shift, 1, name);
    }
});

test('the HunyuanVideo preset runs on its embedded guidance, without another shift', () => {
    const preset = samplerContext('hunyuan_video').presets['▶ HunyuanVideo (30 steps)'];
    assert.deepEqual([preset.cfg, preset.flux_guidance, preset.flux_shift], [1, 6, 1]);
});

test('a graph saved with the former HunyuanVideo preset loads with its new values', async () => {
    const context = samplerContext('hunyuan_video');
    const nodeType = { prototype: {} };
    await context.extension.beforeRegisterNodeDef(nodeType, { name: 'RadianceSamplerPro' });
    const loaded = (cfg, flux_guidance, flux_shift) => {
        const node = { widgets: [{ name: 'preset', value: '[V] HunyuanVideo (30 steps)' },
            { name: 'cfg', value: cfg }, { name: 'flux_guidance', value: flux_guidance },
            { name: 'flux_shift', value: flux_shift }] };
        nodeType.prototype.onConfigure.call(node, {});
        return node.widgets.slice(1).map(w => w.value);
    };
    assert.deepEqual(loaded(6, 0, 7), [1, 6, 1]);
    assert.deepEqual(loaded(4, 0, 7), [4, 0, 7]);
});

test('a graph saved with a former doubled shift loads at the model\'s own shift', async () => {
    const context = samplerContext('z_image');
    const nodeType = { prototype: {} };
    await context.extension.beforeRegisterNodeDef(nodeType, { name: 'RadianceSamplerPro' });
    const loaded = (preset, flux_shift) => {
        const node = { widgets: [{ name: 'preset', value: preset }, { name: 'flux_shift', value: flux_shift }] };
        nodeType.prototype.onConfigure.call(node, {});
        return node.widgets[1].value;
    };
    assert.equal(loaded('[V] LTX-Video (25 steps)', 2.37), 1);
    assert.equal(loaded('[Q] z_image (25 steps)', 3), 1);
    assert.equal(loaded('[Q] Lumina2 (25 steps)', 6), 1);
    assert.equal(loaded('[Q] z_image (25 steps)', 2), 2);
});
