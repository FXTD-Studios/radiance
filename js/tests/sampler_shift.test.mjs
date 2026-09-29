import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

import { liveSourceNode } from '../radiance_widget_utils.js';

const source = readFileSync(new URL('../radiance_sampler.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

function samplerContext(arch) {
    const context = vm.createContext({
        app: { registerExtension() {} }, console: { log() {} }, liveSourceNode, arch,
    });
    vm.runInContext(source + `
        _findModelMetaSourceNode = () => ({ widgets: [] });
        _resolveLoaderModelType = () => arch;
        _isSdTurboActive = () => false;
        globalThis.updateDefaults = updateModelMetaDefaults;
        globalThis.presets = PRESET_CONFIGS;
    `, context);
    return context;
}

for (const arch of ['wan', 'wan_ti2v', 'ltxv', 'lumina2']) {
    for (const preset of ['Auto', 'Custom']) {
        test(`${arch} ${preset} preserves identity and explicit extra shifts`, () => {
            const context = samplerContext(arch);
            for (const shift of [1, 2]) {
                const widget = { name: 'flux_shift', value: shift };
                const node = {
                    widgets: [{ name: 'preset', value: preset }, widget],
                    setDirtyCanvas() {},
                };
                context.updateDefaults(node);
                context.updateDefaults(node);
                assert.equal(widget.value, shift);
                assert.equal(widget._radMetaLinked, undefined);
            }
        });
    }
}

test('Wan presets do not add another native shift', () => {
    const { presets } = samplerContext('wan');
    assert.equal(presets['▶ WAN txt2vid (30 steps)'].flux_shift, 1);
    assert.equal(presets['▶ WAN img2vid (20 steps)'].flux_shift, 1);
});
