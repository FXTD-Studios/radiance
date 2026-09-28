import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_loader.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} } });
vm.runInContext(source + `
    globalThis.find = findMatchingFile;
    globalThis.presets = PRESET_CONFIGS;
    globalThis.slots = PRESET_SLOTS;
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
