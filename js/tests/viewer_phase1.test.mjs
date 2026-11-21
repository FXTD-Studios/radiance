/**
 * Phase 1 renderer-state pins (source assertions; the viewer cannot be imported
 * in Node and no GPU is available to these tests).
 *
 * Run: node --test js/tests/viewer_phase1.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');
const gpu = readFileSync(join(JS, 'radiance_webgpu.js'), 'utf8');

function method(src, head) {
    const i = src.indexOf(head);
    assert.ok(i > 0, `${head} not found`);
    return src.slice(i, src.indexOf('\n    }\n', i));
}

test('a throw during the WebGPU swap restores the WebGL renderer', () => {
    const body = method(viewer, 'async _tryWebGPUUpgrade() {');
    assert.match(body, /_rollback = \(\) => \{/);
    assert.match(body, /this\.renderer = oldRenderer;/);
    assert.match(body, /this\._gpuBackend = oldBackend;/);
    assert.match(body, /if \(_rollback\)/);
    // the old renderer must only be destroyed after the swap succeeded
    assert.ok(body.indexOf('oldRenderer.destroy') > body.indexOf('loadCompareTexture'));
});

test('the WebGPU renderer reports a lost device, but not one it destroyed', () => {
    assert.match(gpu, /lostDevice\.lost\?\.then/);
    assert.match(gpu, /if \(this\.device !== lostDevice\) return;/);
    assert.match(gpu, /this\.onDeviceLost\?\.\(info\)/);
    // destroy() clears this.device, which is what makes the identity check work
    assert.match(method(gpu, '    destroy() {'), /this\.device = null;/);
});

test('the viewer says so when the device is lost and turns WebGPU off', () => {
    const body = method(viewer, '    _onGPUDeviceLost(info) {');
    assert.match(body, /removeItem\('radiance_prefer_webgpu'\)/);
    assert.match(body, /_showViewerMessage\('context'/);
    assert.match(body, /reload the page/);
    assert.match(viewer, /wgpu\.onDeviceLost = \(info\) => this\._onGPUDeviceLost\(info\);/);
});

test('the engine badge marks WebGPU as experimental', () => {
    const body = method(viewer, '    _updateEngineBadge() {');
    assert.match(body, /GPU: WEBGPU \(experimental\)/);
    assert.doesNotMatch(body, /#4ade80/);
});

// ── C10: the reference shelf survives a lost WebGL context ─────────────────
const webgl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');

function lift(name) {
    const start = webgl.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `${name} not found`);
    const end = webgl.indexOf('\n    }\n', start);
    const body = webgl.slice(start, end + 7).replace(/^\n/, '');
    return new Function(`return ({ ${body.trim()} })`)()[name];
}

function fakeGL() {
    let n = 0;
    const uploads = [];
    return {
        uploads,
        TEXTURE_2D: 1, UNPACK_ALIGNMENT: 2, RGBA: 3, UNSIGNED_BYTE: 4, TEXTURE_WRAP_S: 5, TEXTURE_WRAP_T: 6,
        CLAMP_TO_EDGE: 7, TEXTURE_MIN_FILTER: 8, TEXTURE_MAG_FILTER: 9, LINEAR: 10,
        createTexture: () => ({ id: ++n }),
        bindTexture() {}, pixelStorei() {}, texParameteri() {},
        texImage2D: (...a) => uploads.push({ w: a[3], h: a[4], data: a[8] }),
    };
}

test('the kept still is uploaded again after a context restore', () => {
    const r = {
        gl: fakeGL(), referenceShelf: [], textures: {}, wipeRefEnabled: true,
        _lastShelfSlot: 0,
        _shelfStills: new Map([[0, { data: new Uint8Array([1, 2, 3, 4]), w: 1, h: 1 }]]),
    };
    r._uploadShelfTexture = lift('_uploadShelfTexture');
    r._restoreShelfStills = lift('_restoreShelfStills');
    r._restoreShelfStills();
    assert.equal(r.gl.uploads.length, 1);
    assert.deepEqual([...r.gl.uploads[0].data], [1, 2, 3, 4]);
    assert.ok(r.referenceShelf[0]);
    assert.equal(r.textures.reference, r.referenceShelf[0]);   // wipe has its B again
});

test('with nothing pinned, a restore uploads nothing', () => {
    const r = { gl: fakeGL(), referenceShelf: [{ stale: true }], textures: {}, wipeRefEnabled: false };
    r._uploadShelfTexture = lift('_uploadShelfTexture');
    r._restoreShelfStills = lift('_restoreShelfStills');
    r._restoreShelfStills();
    assert.equal(r.gl.uploads.length, 0);
    assert.deepEqual(r.referenceShelf, []);                    // dead handles are dropped
});

test('grab keeps a CPU copy, clear drops it, restore runs after programs exist', () => {
    assert.match(webgl, /this\._shelfStills\.set\(slot, \{ data: flipped, w, h \}\)/);
    const clear = webgl.slice(webgl.indexOf('    clearReferenceShelf() {'));
    assert.match(clear.slice(0, 200), /this\._shelfStills = null;/);
    const i = webgl.indexOf('this._restoreShelfStills();');
    assert.ok(i > webgl.indexOf('this.createPrograms();'), 'restore must follow createPrograms()');
    assert.ok(i < webgl.indexOf('this.onContextRestored?.()'), 'restore must precede the viewer callback');
});
