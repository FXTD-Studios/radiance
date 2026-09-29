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

// Loader 1, fed by LoRA Stack 2, then an optional LoraLoaderModelOnly 3, then the Sampler.
function sample(unetName, { preset = 'Qwen-Image Edit 2511', stackLora = 'None', nativeLora = null } = {}) {
    const nodes = {}, links = {};
    const graph = { links, getNodeById: id => nodes[id] };
    const add = (id, comfyClass, widgets, input, from) => {
        nodes[id] = { id, comfyClass, mode: 0, graph, outputs: [{ type: '*' }],
                      widgets: Object.entries(widgets).map(([name, value]) => ({ name, value })),
                      inputs: [{ name: input, type: '*', link: from && id * 10 }] };
        if (from) links[id * 10] = { origin_id: from, origin_slot: 0, target_id: id, type: '*' };
        return nodes[id];
    };
    add(2, 'RadianceLoraStack', { lora_1: stackLora }, 'lora_stack');
    const loader = add(1, 'RadianceUnifiedLoader', { preset, unet_name: unetName }, 'lora_stack', 2);
    if (nativeLora) add(3, 'LoraLoaderModelOnly', { lora_name: nativeLora }, 'model', 1);
    const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} }, loader });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => loader;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
    `, context);
    const w = (name, value, extra = {}) => ({ name, value, ...extra });
    const node = Object.assign(add(9, 'RadianceSamplerPro', {}, 'model', nativeLora ? 3 : 1), {
        widgets: [
            w('preset', 'Auto'),
            w('model_type', 'auto', { options: { values: ['auto', 'flux', 'qwen_image'] } }),
            w('steps', 20), w('cfg', 1.0), w('sampler', 'euler'), w('scheduler', 'normal'),
            w('flux_guidance', 3.5), w('denoise', 1.0),
        ],
        setDirtyCanvas() {},
    });
    context.updateDefaults(node);
    const value = name => node.widgets.find(x => x.name === name).value;
    return ['model_type', 'steps', 'cfg', 'sampler', 'scheduler'].map(value);
}

test('the "Qwen-Image Edit 2511" Loader preset drives the Sampler to the template values', () => {
    for (const unet of ['qwen_image_edit_2511_int8_convrot.safetensors', 'qwen_image_edit_2511_fp8mixed.safetensors']) {
        assert.deepEqual(sample(unet), ['qwen_image', 40, 4.0, 'euler', 'simple']);
    }
});

const LIGHTNING = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors';

test("a Lightning LoRA in the Loader's LoRA Stack sets its 4 steps at cfg 1", () => {
    assert.deepEqual(sample('qwen_image_edit_2511_int8_convrot.safetensors', { stackLora: LIGHTNING }),
                     ['qwen_image', 4, 1.0, 'euler', 'simple']);
});

test('so does a native LoraLoaderModelOnly between the Loader and the Sampler', () => {
    assert.deepEqual(sample('qwen_image_edit_2511_int8_convrot.safetensors', { nativeLora: LIGHTNING }),
                     ['qwen_image', 4, 1.0, 'euler', 'simple']);
    assert.deepEqual(sample('qwen_image_edit_2511_int8_convrot.safetensors', { nativeLora: 'style.safetensors' }),
                     ['qwen_image', 40, 4.0, 'euler', 'simple']);
});

test('the "Qwen-Image" preset runs the Text to Image templates\' steps at cfg 4', () => {
    assert.deepEqual(sample('qwen_image_fp8_e4m3fn.safetensors', { preset: 'Qwen-Image' }),
                     ['qwen_image', 20, 4.0, 'euler', 'simple']);
    assert.deepEqual(sample('qwen_image_2512_fp8_e4m3fn.safetensors', { preset: 'Qwen-Image' }),
                     ['qwen_image', 50, 4.0, 'euler', 'simple']);
});

test("Qwen-Image 2512's Lightning and Turbo LoRAs run their steps at cfg 1", () => {
    for (const [lora, steps] of [['Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors', 4],
                                 ['Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors', 2]]) {
        assert.deepEqual(sample('qwen_image_2512_fp8_e4m3fn.safetensors', { preset: 'Qwen-Image', stackLora: lora }),
                         ['qwen_image', steps, 1.0, 'euler', 'simple']);
    }
});
