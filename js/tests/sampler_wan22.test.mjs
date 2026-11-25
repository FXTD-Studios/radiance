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

// Loader 1, an optional LoraLoaderModelOnly 3 on its model, then the Sampler 9.
function sampler(unetName, { preset = 'Wan 2.2', samplerPreset = 'Auto', refiner = true, lora = null, then = null, override = false } = {}) {
    const nodes = {}, links = {};
    const graph = { links, getNodeById: id => nodes[id] };
    const loader = nodes[1] = { id: 1, comfyClass: 'RadianceUnifiedLoader', mode: 0, graph, inputs: [],
                                widgets: [{ name: 'preset', value: preset }, { name: 'unet_name', value: unetName }] };
    if (lora) {
        nodes[3] = { id: 3, comfyClass: 'LoraLoaderModelOnly', mode: 0, graph, inputs: [{ name: 'model', link: 30 }],
                     widgets: [{ name: 'lora_name', value: lora }, { name: 'strength_model', value: 1 }] };
        links[30] = { origin_id: 1, origin_slot: 0, target_id: 3 };
    }
    links[1] = { origin_id: lora ? 3 : 1, origin_slot: 0, target_id: 9 };
    if (override) {                     // a sigmas node 5 on sigmas_override
        nodes[5] = { id: 5, mode: 0, graph, inputs: [], widgets: [] };
        links[50] = { origin_id: 5, origin_slot: 0, target_id: 9 };
    }
    const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} }, loader });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => loader;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
    `, context);
    const w = (name, value, extra = {}) => ({ name, value, ...extra });
    const node = {
        id: 9, graph,
        inputs: [{ name: 'model', link: 1 }, { name: 'refiner_model', link: refiner ? 2 : null },
                 { name: 'sigmas_override', link: override ? 50 : null }],
        widgets: [
            w('preset', samplerPreset),
            w('model_type', 'auto', { options: { values: ['auto', 'wan', 'wan_ti2v'] } }),
            w('steps', 20), w('cfg', 1.0), w('sampler', 'euler'), w('scheduler', 'normal'),
            w('flux_guidance', 3.5), w('denoise', 1.0), w('refiner_start_step', 20),
        ],
        setDirtyCanvas() {},
    };
    context.updateDefaults(node);
    if (then) {                         // the Loader moves to another model
        loader.widgets[0].value = then.preset;
        loader.widgets[1].value = then.unet;
        context.updateDefaults(node);
    }
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

test('with a sigmas_override, the start step is left to the Sampler, which knows the schedule', () => {
    assert.equal(sampler(HIGH, { override: true }).values[4], 20);
});

test('refiner_model gets its own name back when the Loader moves from Wan 2.2 to Wan 2.1', () => {
    assert.equal(sampler(HIGH, { then: { preset: 'Wan 2.1', unet: 'wan2.1_t2v_14B_fp16.safetensors' } }).label, undefined);
});

test('Wan 2.1 keeps uni_pc at cfg 6, and the start step stays without a refiner', () => {
    assert.deepEqual(sampler('wan2.1_t2v_14B_fp16.safetensors', { preset: 'Wan 2.1' }).values, [20, 6.0, 'uni_pc', 'simple', 20]);
    assert.deepEqual(sampler(HIGH, { refiner: false }).values, [20, 3.5, 'euler', 'simple', 20]);
});

test("a lightx2v LoRA on the high-noise expert runs its 4 steps at cfg 1, handing over at step 2", () => {
    assert.deepEqual(sampler(HIGH, { lora: 'wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors' }).values,
                     [4, 1.0, 'euler', 'simple', 2]);
});
