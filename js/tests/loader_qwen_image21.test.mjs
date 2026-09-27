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
`, context);

const pick = (preset, field, files) => {
    const config = context.presets[preset];
    const hints = field === 'llm_encoder' ? config.clip_hints.llm_encoder : config[field];
    return context.find(hints, files);
};

const DIT_BF16 = 'qwen_image_2.1_bf16.safetensors';
const DIT_INT8 = 'qwen_image_2.1_int8_convrot.safetensors';
const PE = [
    'qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors',
    'qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors',
];

test('Qwen-Image 2.1 prefers the bf16 transformer and falls back to int8', () => {
    assert.equal(pick('Qwen-Image 2.1', 'unet_hints', [DIT_INT8, DIT_BF16]), DIT_BF16);
    assert.equal(pick('Qwen-Image 2.1', 'unet_hints', [DIT_INT8]), DIT_INT8);
});

test('the Low VRAM preset never picks the bf16 transformer', () => {
    assert.equal(pick('Qwen-Image 2.1 (Low VRAM)', 'unet_hints', [DIT_BF16, DIT_INT8]), DIT_INT8);
    assert.equal(pick('Qwen-Image 2.1 (Low VRAM)', 'unet_hints', [DIT_BF16]), null);
});

test('the text encoder is never a prompt enhancer or MiniMax H3\'s 32B encoder', () => {
    for (const preset of ['Qwen-Image 2.1', 'Qwen-Image 2.1 (Low VRAM)']) {
        const files = [...PE, 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'];
        assert.equal(pick(preset, 'llm_encoder', files), null);
        assert.equal(pick(preset, 'llm_encoder', [...files, 'qwen3vl_8b_int8_convrot.safetensors']),
            'qwen3vl_8b_int8_convrot.safetensors');
    }
    assert.equal(pick('Qwen-Image 2.1', 'llm_encoder',
        ['qwen3vl_8b_w4a8.safetensors', 'qwen3vl_8b_bf16.safetensors']), 'qwen3vl_8b_bf16.safetensors');
});

test('the VAE is never Qwen-Image\'s own 16-channel VAE', () => {
    const v1 = 'qwen_image_vae.safetensors';
    const v21 = 'qwen_image_2.1_vae_bf16.safetensors';
    assert.equal(pick('Qwen-Image 2.1', 'vae_hints', [v1]), null);
    assert.equal(pick('Qwen-Image 2.1', 'vae_hints', [v1, v21]), v21);
});
