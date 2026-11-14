import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// The Sampler with the shared helpers it imports from radiance_widget_utils.js.
const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} } });
vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_sampler.js')
    + 'globalThis.override = _deriveDistillationOverride;', context);
const override = (file) => JSON.parse(JSON.stringify(context.override(file, 'flux', []) ?? null));

test('Auto gives FLUX.1 Kontext Dev guidance 2.5, as Python does', () => {
    assert.deepEqual(override('flux1-dev-kontext_fp8_scaled.safetensors'), { flux_guidance: 2.5 });
    assert.deepEqual(override('flux1-krea-dev_fp8_scaled.safetensors'), { flux_guidance: 4.5 });
    assert.equal(override('flux1-dev-fp8.safetensors'), null);
});
