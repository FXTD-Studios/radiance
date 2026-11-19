import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Wan 2.2 14B in the Sampler: its templates' recipe, the low-noise expert's hand-over, its input's name.
const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');
const source = strip('radiance_widget_utils.js') + strip('radiance_sampler.js');

function sampler(unetName, { preset = 'Wan 2.2', samplerPreset = 'Auto', refiner = true } = {}) {
    const loader = { comfyClass: 'RadianceUnifiedLoader', mode: 0,
                     widgets: [{ name: 'preset', value: preset }, { name: 'unet_name', value: unetName }] };
    const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} }, loader });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => loader;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
    `, context);
    const w = (name, value, extra = {}) => ({ name, value, ...extra });
    const node = {
        inputs: [{ name: 'model', link: 1 }, { name: 'refiner_model', link: refiner ? 2 : null }],
        widgets: [
            w('preset', samplerPreset),
            w('model_type', 'auto', { options: { values: ['auto', 'wan', 'wan_ti2v'] } }),
            w('steps', 20), w('cfg', 1.0), w('sampler', 'euler'), w('scheduler', 'normal'),
            w('flux_guidance', 3.5), w('denoise', 1.0), w('refiner_start_step', 20),
        ],
        setDirtyCanvas() {},
    };
    context.updateDefaults(node);
    const value = name => node.widgets.find(x => x.name === name).value;
    return { values: ['steps', 'cfg', 'sampler', 'scheduler', 'refiner_start_step'].map(value), label: node.inputs[1].label };
}

const HIGH = 'wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors';

test("Wan 2.2's experts run their templates' euler at cfg 3.5, the low-noise one from half the steps", () => {
    assert.deepEqual(sampler(HIGH).values, [20, 3.5, 'euler', 'simple', 10]);
    assert.deepEqual(sampler('Wan/wan2.2_i2v_low_noise_14B_fp16.safetensors').values, [20, 3.5, 'euler', 'simple', 10]);
});

test("refiner_model shows as model_low_noise under Wan 2.2, under any Sampler preset", () => {
    assert.equal(sampler(HIGH).label, 'model_low_noise');
    assert.equal(sampler(HIGH, { samplerPreset: '[V] WAN txt2vid (30 steps)' }).label, 'model_low_noise');
    assert.equal(sampler('wan2.1_t2v_14B_fp16.safetensors', { preset: 'Wan 2.1' }).label, undefined);
});

test('Wan 2.1 keeps uni_pc at cfg 6, and the start step stays without a refiner', () => {
    assert.deepEqual(sampler('wan2.1_t2v_14B_fp16.safetensors', { preset: 'Wan 2.1' }).values, [20, 6.0, 'uni_pc', 'simple', 20]);
    assert.deepEqual(sampler(HIGH, { refiner: false }).values, [20, 3.5, 'euler', 'simple', 20]);
});
