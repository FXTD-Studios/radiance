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

const loaderCtx = vm.createContext({
    app: { registerExtension(ext) { loaderCtx.ext = ext; } }, console: { log() {} },
    getWidget: (node, name) => node.widgets.find((w) => w.name === name) ?? null,
    setTimeout() {}, setInterval() {},
});
vm.runInContext(strip('radiance_loader.js') + `
    globalThis.find = findMatchingFile;
    globalThis.unetMatch = _resolveUnetMatch;
    globalThis.presets = PRESET_CONFIGS;
    globalThis.slots = PRESET_SLOTS;
    globalThis.autoFill = autoFillPresetFiles;
    globalThis.sync = _syncUpsampler;
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

test("the Loader's upsampler is picked only while a Prompt it feeds has upsample_prompt on", () => {
    const PUP_BF16 = 'qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors';
    const PE_T2I = 'qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors';
    const PE_I2I = 'qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors';
    const files = ['None', TE_BF16, TE_INT8, 'qwen3vl_8b_bf16.safetensors', PE_I2I, PE_T2I, PUP, PUP_BF16];
    const w = { name: 'upsampler', value: PUP, options: { values: files } };
    const preset = { name: 'preset', value: 'Flux.3 Klein' };
    const prompt = { mode: 0, widgets: [{ name: 'upsample_prompt', value: false }] };
    const loader = {
        widgets: [preset, w], outputs: [{ name: 'upsampler_clip', links: [] }], setDirtyCanvas() {},
        graph: { links: { 7: { target_id: 2 } }, getNodeById: () => prompt },
    };
    const sync = (presetName, values = files) => {
        preset.value = presetName;
        w.options.values = values;
        loaderCtx.sync(loader);
        return w.value;
    };
    loaderCtx.autoFill(loader, 'Flux.3 Klein');
    assert.equal(w.value, 'None', 'a preset no longer picks it on its own');
    assert.equal(sync('Flux.3 Klein'), 'None', 'nothing connected');
    loader.outputs[0].links = [7];
    assert.equal(sync('Flux.3 Klein'), 'None', 'connected, switch off');
    prompt.widgets[0].value = true;
    assert.equal(sync('Flux.3 Klein'), PUP_BF16);
    w.value = 'None';
    assert.equal(sync('Flux.3 Klein (Low VRAM)'), PUP);
    w.value = 'None';
    assert.equal(sync('Qwen-Image 2.1'), PE_T2I);
    w.value = PE_I2I;
    assert.equal(sync('Qwen-Image 2.1'), PE_I2I, 'a file picked by hand stays while it is used');
    w.value = 'None';
    assert.equal(sync('Qwen-Image 2.1', ['None', PE_I2I]), PE_I2I, "editing's own when it is the one on disk");
    prompt.mode = 4;
    assert.equal(sync('Qwen-Image 2.1'), 'None', 'a bypassed Prompt does not use it');
    prompt.mode = 0; prompt.widgets = [];
    w.value = 'None';
    assert.equal(sync('Flux.3 Klein'), PUP_BF16, 'another consumer (Generate Text) always uses it');
    w.value = TE_BF16;
    assert.equal(sync('Flux.2'), TE_BF16, 'presets without an upsampler leave the slot alone');
    for (const name of ['Flux.3 Klein', 'Flux.3 Klein (Low VRAM)', 'Qwen-Image 2.1', 'Qwen-Image 2.1 (Low VRAM)']) {
        assert.ok(loaderCtx.presets[name].extra_widgets.includes('upsampler'), name);
    }
});

test("a bypassed node between the Loader and the Prompt passes the upsampler on, a muted one does not", () => {
    const w = { name: 'upsampler', value: 'None', options: { values: ['None', PUP] } };
    const prompt = { mode: 0, widgets: [{ name: 'upsample_prompt', value: true }] };
    const between = { mode: 4, outputs: [{ type: 'CLIP', links: [8] }] };
    const nodes = { 2: between, 3: prompt };
    const loader = {
        widgets: [{ name: 'preset', value: 'Flux.3 Klein' }, w],
        outputs: [{ name: 'upsampler_clip', links: [7] }], setDirtyCanvas() {},
        graph: { links: { 7: { target_id: 2 }, 8: { target_id: 3 } }, getNodeById: (id) => nodes[id] },
    };
    loaderCtx.sync(loader);
    assert.equal(w.value, PUP);
    between.mode = 2;
    loaderCtx.sync(loader);
    assert.equal(w.value, 'None');
});

test('the upsampler syncs right before a run is queued, not only on the 250 ms poll', async () => {
    const nodeType = function () {};
    await loaderCtx.ext.beforeRegisterNodeDef(nodeType, { name: 'RadianceUnifiedLoader' });
    const w = { name: 'upsampler', value: 'None', options: { values: ['None', PUP] } };
    const prompt = { mode: 0, widgets: [{ name: 'upsample_prompt', value: true }] };
    const node = Object.assign(Object.create(nodeType.prototype), {
        widgets: [{ name: 'preset', value: 'Flux.3 Klein' }, w],
        outputs: [{ name: 'upsampler_clip', links: [7] }], setDirtyCanvas() {},
        graph: { links: { 7: { target_id: 2 } }, getNodeById: () => prompt },
    });
    node.onNodeCreated();
    w.beforeQueued();
    assert.equal(w.value, PUP);
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

function upsamplerPrompt(archOf) {
    const ctx = vm.createContext({
        app: { registerExtension() {} }, console: { log() {}, warn() {} }, LiteGraph: {},
        api: { fetchApi: () => Promise.resolve({ ok: false }), addEventListener() {} },
        isInputLive: (node, input) => input.link != null,
        setWidgetVisible: (w, visible) => { w.hidden = !visible; },
        fitNodeSize() {}, loaderModelType: archOf, modelMetaSourceNode() { return null; },
        setInterval() {}, clearInterval() {},
    });
    vm.runInContext(strip('radiance_prompt.js')
        + '\nglobalThis.apply = _applyUpsamplerVisibility; globalThis.hold = _holdHiddenSeed;', ctx);
    const control = { name: 'control_after_generate', hidden: false };
    const node = {
        inputs: [{ name: 'upsampler_clip', link: null }],
        widgets: [{ name: 'upsample_prompt', value: false, hidden: false },
            { name: 'upsampler_seed', value: 0, hidden: false, linkedWidgets: [control] }],
    };
    return { ctx, node, control };
}

test('the upsampler switch shows with an upsampler under a family that has one, or while on', () => {
    let arch = null;
    const { ctx, node, control } = upsamplerPrompt(() => arch);
    const shown = () => [node.widgets[0].hidden, node.widgets[1].hidden, control.hidden].map((h) => !h);
    ctx.apply(node);
    assert.deepEqual(shown(), [false, false, false]);
    node.inputs[0].link = 3; ctx.apply(node);
    assert.deepEqual(shown(), [true, false, false], 'Auto-Detect: the switch, the seed waits for it');
    arch = 'ltxav'; ctx.apply(node);
    assert.deepEqual(shown(), [false, false, false], 'a family without an upsampler hides it');
    for (arch of ['flux3', 'qwen_image21']) {
        ctx.apply(node);
        assert.deepEqual(shown(), [true, false, false]);
    }
    node.widgets[0].value = true; ctx.apply(node);
    assert.deepEqual(shown(), [true, true, true], 'the seed shows while the switch is on');
    arch = 'ltxav'; node.inputs[0].link = null; ctx.apply(node);
    assert.deepEqual(shown(), [true, true, true], 'a switch left on must stay reachable');
});

test('a hidden upsampler seed keeps its value through "randomize"', () => {
    const { ctx, node, control } = upsamplerPrompt(() => 'ltxav');
    const seed = node.widgets[1];
    control.beforeQueued = () => {};
    control.afterQueued = () => { seed.value += 1; };
    ctx.hold(node);
    ctx.apply(node);
    control.beforeQueued(); control.afterQueued();
    assert.equal(seed.value, 0, 'a new seed each run re-ran the Prompt and everything after it');
    node.widgets[0].value = true; ctx.apply(node);
    control.afterQueued();
    assert.equal(seed.value, 1);
});

test('the layout input shows under a FLUX 3 Loader, or while it holds a table', () => {
    let arch = 'flux2';
    const ctx = vm.createContext({
        app: { registerExtension() {} }, console: { log() {}, warn() {} }, LiteGraph: {},
        api: { fetchApi: () => Promise.resolve({ ok: false }), addEventListener() {} },
        isInputLive: () => false, setWidgetVisible: (w, visible) => { w.hidden = !visible; },
        fitNodeSize() {}, loaderModelType: () => arch, modelMetaSourceNode: () => ({}),
    });
    vm.runInContext(strip('radiance_prompt.js') + '\nglobalThis.apply = _applyLayoutVisibility;', ctx);
    const w = { name: 'layout', value: '', hidden: false };
    const node = { widgets: [w] };
    ctx.apply(node); assert.equal(w.hidden, true);
    arch = 'flux3'; ctx.apply(node); assert.equal(w.hidden, false);
    arch = 'qwen_image'; w.value = '[{"id":"obj_1","bbox":[0,0,500,500],"desc":"x"}]'; ctx.apply(node);
    assert.equal(w.hidden, false, 'a table under another family must stay visible to be cleared');
});
