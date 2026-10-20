// The LTX 2.x Loader presets: Dev and Distilled files apart, old names renamed on load.
// Run: node --test js/tests/loader_ltx_presets.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_loader.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

const context = vm.createContext({
    app: { registerExtension(ext) { context.ext = ext; } }, console: { log() {} },
    setTimeout() {}, setInterval() {},
    getWidget: (node, name) => node.widgets?.find(w => w.name === name),
});
vm.runInContext(source + `
    globalThis.find = findMatchingFile;
    globalThis.presets = PRESET_CONFIGS;
    globalThis.slots = PRESET_SLOTS;
`, context);

const unet = (preset, files) => context.find(context.presets[preset].unet_hints, files);

const LTX23 = {
    dev: 'ltx-2.3-22b-dev.safetensors',
    devFp8: 'ltx-2.3-22b-dev-fp8.safetensors',
    distilled: 'ltx-2.3-22b-distilled.safetensors',
    distilled11: 'ltx-2.3-22b-distilled-1.1.safetensors',
};
const LTX25 = {
    dev: 'ltx-2.5-22b-dev-transformer-bf16.safetensors',
    devInt8: 'ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors',
    distilled: 'ltx-2.5-22b-distilled-transformer-bf16.safetensors',
    distilledInt8: 'ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors',
    distilledNvfp4: 'ltx-2.5-22b-distilled-transformer-nvfp4.safetensors',
};

test('each LTX 2.3 preset picks its own file, the distilled 1.1 first', () => {
    const all = Object.values(LTX23);
    assert.equal(unet('LTX Video 2.3 Dev', all), LTX23.dev);
    assert.equal(unet('LTX Video 2.3 Dev (Low VRAM)', all), LTX23.devFp8);
    assert.equal(unet('LTX Video 2.3 Distilled', all), LTX23.distilled11);
    assert.equal(unet('LTX Video 2.3 Distilled', [LTX23.dev, LTX23.distilled]), LTX23.distilled);
});

test('each LTX 2.5 preset picks its own file, int8 first when Low VRAM', () => {
    const all = Object.values(LTX25);
    assert.equal(unet('LTX Video 2.5 Dev', all), LTX25.dev);
    assert.equal(unet('LTX Video 2.5 Dev', [LTX25.devInt8]), LTX25.devInt8);
    assert.equal(unet('LTX Video 2.5 Dev (Low VRAM)', all), LTX25.devInt8);
    assert.equal(unet('LTX Video 2.5 Distilled', all), LTX25.distilled);
    assert.equal(unet('LTX Video 2.5 Distilled (Low VRAM)', all), LTX25.distilledInt8);
    assert.equal(unet('LTX Video 2.5 Distilled (Low VRAM)', [LTX25.distilled, LTX25.distilledNvfp4]), LTX25.distilledNvfp4);
});

test('a Dev preset never falls back to a Distilled file, nor the other way round', () => {
    const distilled = [LTX23.distilled, LTX23.distilled11, LTX25.distilled, LTX25.distilledInt8, LTX25.distilledNvfp4];
    const dev = [LTX23.dev, LTX23.devFp8, LTX25.dev, LTX25.devInt8];
    for (const preset of Object.keys(context.presets).filter(p => /^LTX Video 2\.[35] /.test(p))) {
        assert.ok(context.slots[preset], preset);
        assert.equal(unet(preset, preset.includes('Distilled') ? dev : distilled), null, preset);
    }
});

async function loadedPreset(preset, unetName) {
    const nodeType = { prototype: {} };
    await context.ext.beforeRegisterNodeDef(nodeType, { name: 'RadianceUnifiedLoader' }, context.app);
    const node = { widgets: [{ name: 'preset', value: preset }, { name: 'unet_name', value: unetName }] };
    nodeType.prototype.onConfigure.call(node, {});
    return node.widgets[0].value;
}

test('a saved old LTX preset is renamed after the file it loads', async () => {
    assert.equal(await loadedPreset('LTX Video 2.3', LTX23.dev), 'LTX Video 2.3 Dev');
    assert.equal(await loadedPreset('LTX Video 2.3 (Low VRAM)', LTX23.devFp8), 'LTX Video 2.3 Dev (Low VRAM)');
    assert.equal(await loadedPreset('LTX Video 2.3 (Low VRAM)', LTX23.distilled11), 'LTX Video 2.3 Distilled');
    assert.equal(await loadedPreset('LTX Video 2.5', LTX25.dev), 'LTX Video 2.5 Dev');
    assert.equal(await loadedPreset('LTX Video 2.5 (Low VRAM)', LTX25.distilledInt8), 'LTX Video 2.5 Distilled (Low VRAM)');
    assert.equal(await loadedPreset('LTX Video 2.5 Distilled', LTX25.distilled), 'LTX Video 2.5 Distilled');
});
