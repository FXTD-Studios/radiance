// FLUX 3 Image Klein in the Loader, Sampler, Resolution and Prompt UIs.
// Run: node --test js/tests/flux3_klein.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const loaderCtx = vm.createContext({ app: { registerExtension() {} }, console: { log() {} } });
vm.runInContext(strip('radiance_loader.js') + `
    globalThis.find = findMatchingFile;
    globalThis.unetMatch = _resolveUnetMatch;
    globalThis.presets = PRESET_CONFIGS;
    globalThis.slots = PRESET_SLOTS;
`, loaderCtx);

const BASE_BF16 = 'flux3_image_klein_bf16.safetensors';
const SD_BF16 = 'flux3_image_klein_sd_bf16.safetensors';
const BASE_INT8 = 'flux3_image_klein_int8_convrot.safetensors';
const SD_INT8 = 'flux3_image_klein_sd_int8_convrot.safetensors';
const TE_BF16 = 'qwen3vl_4b_bf16.safetensors';
const TE_INT8 = 'qwen3vl_4b_int8_convrot.safetensors';
const PUP = 'qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors';
const KLEIN2 = 'flux-2-klein-4b-fp8.safetensors';

test('Flux.3 Klein picks the FLUX 3 transformer, Qwen3-VL-4B and the Flux.2 VAE', () => {
    const c = loaderCtx.presets['Flux.3 Klein'];
    assert.equal(loaderCtx.unetMatch(c, [KLEIN2, BASE_INT8, SD_BF16]), SD_BF16);
    assert.equal(loaderCtx.unetMatch(c, [KLEIN2, BASE_INT8]), BASE_INT8);
    assert.equal(loaderCtx.unetMatch(c, [KLEIN2]), null);
    assert.equal(loaderCtx.find(c.vae_hints, ['ae.safetensors', 'flux2-vae.safetensors']), 'flux2-vae.safetensors');
    assert.deepEqual([...loaderCtx.slots['Flux.3 Klein']], ['llm_encoder']);
});

test('the text encoder is never the prompt upsampler or a Qwen3 file of Flux.2 Klein', () => {
    for (const preset of ['Flux.3 Klein', 'Flux.3 Klein (Low VRAM)']) {
        const hints = loaderCtx.presets[preset].clip_hints.llm_encoder;
        assert.equal(loaderCtx.find(hints, [PUP, 'qwen_3_4b.safetensors', 'qwen3vl_8b_bf16.safetensors']), null);
        assert.equal(loaderCtx.find(hints, [PUP, TE_INT8]), TE_INT8);
    }
    assert.equal(loaderCtx.find(loaderCtx.presets['Flux.3 Klein'].clip_hints.llm_encoder, [TE_INT8, TE_BF16]), TE_BF16);
});

test('the Low VRAM preset never picks a bf16 file', () => {
    const c = loaderCtx.presets['Flux.3 Klein (Low VRAM)'];
    assert.equal(loaderCtx.unetMatch(c, [SD_BF16, BASE_BF16]), null);
    assert.equal(loaderCtx.unetMatch(c, [SD_BF16, SD_INT8]), SD_INT8);
    assert.equal(loaderCtx.find(c.clip_hints.llm_encoder, [TE_BF16]), null);
    assert.equal(c.offload_mode, 'cpu_offload');
});

test('the Flux.2 presets no longer grab a FLUX 3 Klein file through their bare "klein" hint', () => {
    for (const preset of ['Flux.2', 'Flux.2 (Low VRAM)']) {
        const c = loaderCtx.presets[preset];
        assert.equal(loaderCtx.unetMatch(c, [SD_INT8, BASE_INT8]), null, preset);
    }
    assert.equal(loaderCtx.unetMatch(loaderCtx.presets['Flux.2'], [SD_INT8, KLEIN2]), KLEIN2);
});

// ── Sampler ────────────────────────────────────────────────────────────────

function samplerDefaults(preset, unet) {
    const loader = { widgets: [{ name: 'preset', value: preset }, { name: 'unet_name', value: unet }] };
    const ctx = vm.createContext({ app: { registerExtension() {} }, console: { log() {} }, loader });
    vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_sampler.js') + `
        _findModelMetaSourceNode = () => loader;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
    `, ctx);
    const w = (name, value, extra = {}) => ({ name, value, ...extra });
    const node = {
        widgets: [
            w('preset', 'Auto'),
            w('model_type', 'auto', { options: { values: ['auto', 'flux', 'flux2-klein', 'flux3'] } }),
            w('steps', 20), w('cfg', 1.0), w('sampler', 'euler'), w('scheduler', 'normal'),
            w('flux_guidance', 3.5), w('denoise', 1.0),
        ],
        setDirtyCanvas() {},
    };
    ctx.updateDefaults(node);
    const value = name => node.widgets.find(x => x.name === name).value;
    return ['model_type', 'steps', 'cfg', 'flux_guidance'].map(value);
}

test('a FLUX 3 base file drives the Sampler to 50 steps at cfg 3.5', () => {
    assert.deepEqual(samplerDefaults('Flux.3 Klein', BASE_INT8), ['flux3', 50, 3.5, 0.0]);
});

test('a FLUX 3 step-distilled file drives it to 16 steps at cfg 1', () => {
    assert.deepEqual(samplerDefaults('Flux.3 Klein', SD_INT8), ['flux3', 16, 1.0, 0.0]);
    assert.deepEqual(samplerDefaults('Flux.3 Klein (Low VRAM)', 'flux3_image_klein_sd_nvfp4.safetensors'),
        ['flux3', 16, 1.0, 0.0]);
});

test('a FLUX 3 file under a Flux.2 preset is FLUX 3, not Flux.2 Klein', () => {
    assert.deepEqual(samplerDefaults('Flux.2', SD_INT8), ['flux3', 16, 1.0, 0.0]);
    // Flux.2 Klein itself is unchanged.
    assert.deepEqual(samplerDefaults('Flux.2', 'flux-2-klein-base-4b.safetensors'), ['flux2-klein', 50, 1.0, 4.0]);
});

// ── Resolution and Prompt mirrors ──────────────────────────────────────────

test('Resolution offers Flux.3 Klein at 128ch, 16x, from the flux3 arch', () => {
    const src = readFileSync(new URL('../radiance_resolution.js', import.meta.url), 'utf8');
    assert.match(src, /"Flux\.3 Klein \(128ch\)": \["flux3"\]/);
    assert.match(src, /"Flux\.3 Klein \(128ch\)": 16/);
});

test('the Prompt allows 10 FLUX 3 references and shows resolution for them', () => {
    const src = readFileSync(new URL('../radiance_prompt.js', import.meta.url), 'utf8');
    assert.match(src, /REFERENCE_LIMITS = \{ qwen_image: 3, flux3: 10 \}/);
    const hide = src.match(/!\[([^\]]*)\]\.includes\(loaderModelType\(source\)\)/)[1];
    assert.ok(!hide.includes('flux3'), 'resolution sizes FLUX 3 references, it must not be hidden');
});

// ── Prompt upsampler widgets (4.0.0) ───────────────────────────────────────

test('the upsampler switch and seed show only with an upsampler connected, or while on', () => {
    const ctx = vm.createContext({
        app: { registerExtension() {} }, console: { log() {}, warn() {} }, LiteGraph: {},
        api: { fetchApi: () => Promise.resolve({ ok: false }), addEventListener() {} },
        isInputLive: (node, input) => input.link != null,
        setWidgetVisible: (w, visible) => { w.hidden = !visible; },
        fitNodeSize() {}, loaderModelType() { return null; }, modelMetaSourceNode() { return null; },
        setInterval() {}, clearInterval() {},
    });
    vm.runInContext(strip('radiance_prompt.js') + '\nglobalThis.apply = _applyUpsamplerVisibility;', ctx);
    const control = { name: 'control_after_generate', hidden: false };
    const node = {
        inputs: [{ name: 'upsampler_clip', link: null }],
        widgets: [{ name: 'upsample_prompt', value: false, hidden: false },
            { name: 'upsampler_seed', value: 0, hidden: false, linkedWidgets: [control] }],
    };
    const shown = () => [node.widgets[0].hidden, node.widgets[1].hidden, control.hidden].map((h) => !h);
    ctx.apply(node);
    assert.deepEqual(shown(), [false, false, false]);
    node.inputs[0].link = 3; ctx.apply(node);
    assert.deepEqual(shown(), [true, true, true]);
    node.widgets[0].value = true; node.inputs[0].link = null; ctx.apply(node);
    assert.deepEqual(shown(), [true, true, true], 'a switch left on must stay reachable');
});
