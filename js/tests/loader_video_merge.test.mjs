// The Loader took over the Video Loader: saved Loaders are restored by name.
// Run: node --test js/tests/loader_video_merge.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const ctx = vm.createContext({
    app: { registerExtension(ext) { ctx.ext = ext; } }, console: { log() {} },
    setTimeout() {}, setInterval() {},
});
vm.runInContext(strip('radiance_loader.js'), ctx);

async function register(name) {
    const nodeData = { name };
    await ctx.ext.beforeRegisterNodeDef({ prototype: {} }, nodeData, ctx.app);
    return nodeData;
}

test('the Loader and the old Image Loader key are restored by their saved order', async () => {
    for (const name of ['RadianceUnifiedLoader', 'RadianceImageLoader']) {
        const names = (await register(name)).fallbackWidgetsValuesNames;
        assert.equal(names.length, 18, name);
        assert.deepEqual([...names.slice(4, 6)], ['vae_name', 'clip_l'], name);
        assert.deepEqual([...names.slice(-2)], ['model_shift', 'upsampler'], name);
    }
});

test('the old Video Loader key keeps its positional order', async () => {
    assert.equal((await register('RadianceVideoLoader')).fallbackWidgetsValuesNames, undefined);
});
