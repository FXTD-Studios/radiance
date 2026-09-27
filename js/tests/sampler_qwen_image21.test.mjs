import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_sampler.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

for (const preset of ['Qwen-Image 2.1', 'Qwen-Image 2.1 (Low VRAM)']) {
    test(`the "${preset}" Loader preset drives the Sampler to the template defaults`, () => {
        const loader = { widgets: [
            { name: 'preset', value: preset },
            { name: 'unet_name', value: 'qwen_image_2.1_int8_convrot.safetensors' },
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
                w('model_type', 'auto', { options: { values: ['auto', 'flux', 'qwen_image21'] } }),
                w('steps', 20), w('cfg', 1.0), w('sampler', 'euler'), w('scheduler', 'normal'),
                w('flux_guidance', 3.5), w('denoise', 1.0),
            ],
            setDirtyCanvas() {},
        };
        context.updateDefaults(node);
        const value = name => node.widgets.find(x => x.name === name).value;
        assert.deepEqual(
            ['model_type', 'steps', 'cfg', 'sampler', 'scheduler'].map(value),
            ['qwen_image21', 25, 1.0, 'euler', 'simple'],
        );
    });
}
