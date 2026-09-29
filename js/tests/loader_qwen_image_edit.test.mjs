import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

import { getWidget } from '../radiance_widget_utils.js';

const source = readFileSync(new URL('../radiance_loader.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} }, getWidget });
vm.runInContext(source + `
    globalThis.find = findMatchingFile;
    globalThis.presets = PRESET_CONFIGS;
    globalThis.slots = PRESET_SLOTS;
    globalThis.autoFill = autoFillPresetFiles;
`, context);

const PRESET = 'Qwen-Image Edit 2511';
const pick = (field, files) => {
    const config = context.presets[PRESET];
    const hints = field === 'llm_encoder' ? config.clip_hints.llm_encoder : config[field];
    return context.find(hints, files);
};

test('Qwen-Image Edit 2511 takes one text encoder file', () => {
    assert.deepEqual([...context.slots[PRESET]], ['llm_encoder']);
});

test('the templates\' transformers come first, int8 then fp8mixed, bf16 as fallback', () => {
    const int8 = 'qwen_image_edit_2511_int8_convrot.safetensors';
    const fp8 = 'qwen_image_edit_2511_fp8mixed.safetensors';
    const bf16 = 'qwen_image_edit_2511_bf16.safetensors';
    assert.equal(pick('unet_hints', [bf16, fp8, int8]), int8);
    assert.equal(pick('unet_hints', [bf16, fp8]), fp8);
    assert.equal(pick('unet_hints', [bf16]), bf16);
    assert.equal(pick('unet_hints', ['qwen_image_2.1_bf16.safetensors', 'qwen_image_edit_2509_fp8_e4m3fn.safetensors']), null);
});

test('the VAE is Qwen-Image\'s own 16-channel VAE, never Qwen-Image 2.1\'s', () => {
    const v1 = 'qwen_image_vae.safetensors';
    assert.equal(pick('vae_hints', ['qwen_image_2.1_vae_bf16.safetensors']), null);
    assert.equal(pick('vae_hints', ['qwen_image_2.1_vae_bf16.safetensors', v1]), v1);
});

test('the text encoder is Qwen2.5-VL-7B, never Qwen-Image 2.1\'s Qwen3-VL-8B', () => {
    const te = 'qwen_2.5_vl_7b_fp8_scaled.safetensors';
    assert.equal(pick('llm_encoder', ['qwen3vl_8b_bf16.safetensors']), null);
    assert.equal(pick('llm_encoder', ['qwen3vl_8b_bf16.safetensors', te]), te);
});

test("the preset shows model_shift at the templates' 3.1; another preset resets it to 0", () => {
    assert.ok(context.presets[PRESET].extra_widgets.includes('model_shift'));
    const node = { widgets: [{ name: 'model_shift', value: 0 }] };
    context.autoFill(node, PRESET);
    assert.equal(node.widgets[0].value, 3.1);
    context.autoFill(node, 'Flux.1');
    assert.equal(node.widgets[0].value, 0);
});

test('the "Qwen-Image" preset picks the Text to Image templates\' fp8 files, 2512 first', () => {
    const config = context.presets['Qwen-Image'];
    const base = 'qwen_image_fp8_e4m3fn.safetensors';
    const q2512 = 'qwen_image_2512_fp8_e4m3fn.safetensors';
    const others = ['qwen_image_edit_2511_int8_convrot.safetensors', 'qwen_image_2.1_bf16.safetensors'];
    assert.equal(context.find(config.unet_hints, [...others, base, q2512]), q2512);
    assert.equal(context.find(config.unet_hints, [...others, base]), base);
    assert.equal(context.find(config.unet_hints, [...others, 'qwen_image_bf16.safetensors']), 'qwen_image_bf16.safetensors');
    assert.equal(context.find(config.unet_hints, others), null);
    assert.deepEqual(config.vae_hints, context.presets[PRESET].vae_hints);
    assert.deepEqual(config.clip_hints, context.presets[PRESET].clip_hints);
    assert.deepEqual([...context.slots['Qwen-Image']], ['llm_encoder']);
    assert.equal(config.model_shift, 3.1);
});
