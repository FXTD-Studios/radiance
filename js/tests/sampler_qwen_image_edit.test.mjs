import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_sampler.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

function sample(unetName) {
    const loader = { widgets: [
        { name: 'preset', value: 'Qwen-Image Edit 2511' },
        { name: 'unet_name', value: unetName },
    ] };
    const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} }, loader });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => loader;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
    `, context);
    const w = (name, value, extra = {}) => ({ name, value, ...extra });
    const node = {
        widgets: [
            w('preset', 'Auto'),
            w('model_type', 'auto', { options: { values: ['auto', 'flux', 'qwen_image'] } }),
            w('steps', 20), w('cfg', 1.0), w('sampler', 'euler'), w('scheduler', 'normal'),
            w('flux_guidance', 3.5), w('denoise', 1.0),
        ],
        setDirtyCanvas() {},
    };
    context.updateDefaults(node);
    const value = name => node.widgets.find(x => x.name === name).value;
    return ['model_type', 'steps', 'cfg', 'sampler', 'scheduler'].map(value);
}

test('the "Qwen-Image Edit 2511" Loader preset drives the Sampler to the template values', () => {
    for (const unet of ['qwen_image_edit_2511_int8_convrot.safetensors', 'qwen_image_edit_2511_fp8mixed.safetensors']) {
        assert.deepEqual(sample(unet), ['qwen_image', 40, 4.0, 'euler', 'simple']);
    }
});
