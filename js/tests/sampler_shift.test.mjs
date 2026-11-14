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
        test(`${arch} ${preset} preserves identity and explicit extra shifts`, () => {
            const context = samplerContext(arch);
            for (const shift of [1, 2]) {
                const widget = { name: 'flux_shift', value: shift };
                const node = {
                    widgets: [{ name: 'preset', value: preset }, widget],
                    setDirtyCanvas() {},
                };
                context.updateDefaults(node);
                context.updateDefaults(node);
                assert.equal(widget.value, shift);
                assert.ok(!widget._radMetaLinked);
            }
        });
    }
}

test('Wan presets do not add another native shift', () => {
    const { presets } = samplerContext('wan');
    assert.equal(presets['▶ WAN txt2vid (30 steps)'].flux_shift, 1);
    assert.equal(presets['▶ WAN img2vid (20 steps)'].flux_shift, 1);
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
