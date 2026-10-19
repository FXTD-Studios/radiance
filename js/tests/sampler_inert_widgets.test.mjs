import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// The Sampler with the shared helpers it imports from radiance_widget_utils.js.
const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');
const source = strip('radiance_widget_utils.js') + strip('radiance_sampler.js');

function samplerContext() {
    const context = vm.createContext({ app: { registerExtension() {} }, console: { log() {} } });
    vm.runInContext(source + `
        _setWidgetVisible = (w, visible) => { const changed = !w.hidden !== visible; w.hidden = !visible; return changed; };
        _forceWidgetReinsert = () => {};
        fitNodeSize = () => {};
        _isSdTurboActive = () => false;
        globalThis.fold = applyFolding;
    `, context);
    return context;
}

function samplerNode(preset, values = {}, inputs = []) {
    const widgets = Object.entries({
        preset, model_type: 'auto', denoise: 1.0, noise_type: 'Gaussian', force_exact_steps: true,
        restart_count: 0, temporal_window: 0, temporal_overlap: 4, ...values,
    }).map(([name, value]) => ({ name, value }));
    return { widgets, inputs, setDirtyCanvas() {} };
}

const hidden = (node, name) => node.widgets.find(w => w.name === name).hidden;

test('settings that do nothing on this run fold away', () => {
    const context = samplerContext();
    const node = samplerNode('[F] Flux txt2img');
    context.fold(node);
    for (const name of ['restart_count', 'temporal_overlap', 'force_exact_steps']) assert.equal(hidden(node, name), true, name);
    assert.equal(hidden(node, 'temporal_window'), false);
    assert.equal(hidden(node, 'noise_type'), false);
});

test('they come back once they apply', () => {
    const context = samplerContext();
    const node = samplerNode('[F] Flux img2img', { denoise: 0.75, temporal_window: 16 },
        [{ name: 'restart_schedule', link: 7 }]);
    context.fold(node);
    for (const name of ['restart_count', 'temporal_overlap', 'force_exact_steps']) assert.equal(hidden(node, name), false, name);
});

test('LTX 2.3 / 2.5 hide windowing and noise_type, and keep Gaussian noise', () => {
    const context = samplerContext();
    const node = samplerNode('[V] LTX 2.5 LowRes (30 steps)', { noise_type: 'Perlin', temporal_window: 16 });
    context.fold(node);
    for (const name of ['temporal_window', 'temporal_overlap', 'noise_type']) assert.equal(hidden(node, name), true, name);
    assert.equal(node.widgets.find(w => w.name === 'noise_type').value, 'Gaussian');
});

test('Custom still shows every setting', () => {
    const context = samplerContext();
    const node = samplerNode('Custom');
    context.fold(node);
    for (const name of ['restart_count', 'temporal_overlap', 'force_exact_steps', 'noise_type']) assert.equal(hidden(node, name), false, name);
});
