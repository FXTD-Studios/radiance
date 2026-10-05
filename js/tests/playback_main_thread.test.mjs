/**
 * Viewer playback: per-frame work that blocked the page's main thread.
 *
 * Profiled in Edge on a 1080p HDR clip in a large ComfyUI workflow:
 *  - gl.getError after every float texture upload waited for the GPU each
 *    frame, a quarter of the main thread during playback;
 *  - the sequence dock read its own offsetHeight on every frame, which laid
 *    out the whole ComfyUI page each time.
 *
 * Run: node --test js/tests/playback_main_thread.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');

function methodSource(src, name) {
    const start = src.search(new RegExp(`\\n    ${name}\\(`));
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    return rest.slice(0, rest.search(/\n    \}\n/)) + '\n    }';
}

test('a float upload checks for GL errors once per format, and keeps checking after a failure', () => {
    const Renderer = new Function(`return class { ${methodSource(read('radiance_webgl.js'), '_uploadError')} }`)();
    let calls = 0, next = 0;
    const r = new Renderer();
    r.gl = { NO_ERROR: 0, getError() { calls++; return next; } };
    next = 1281;                                            // GL_INVALID_VALUE on the first upload
    assert.equal(r._uploadError('f16:1920x1080x4'), 1281);
    next = 0;
    assert.equal(r._uploadError('f16:1920x1080x4'), 0, 'a failed format is checked again');
    for (let i = 0; i < 10; i++) assert.equal(r._uploadError('f16:1920x1080x4'), 0);
    assert.equal(calls, 2, 'once the format uploaded cleanly, getError is not called again');
    r._uploadError('f16:3840x2160x4');
    assert.equal(calls, 3, 'a new format is checked');
});

test('the sequence dock does not read its height on every refresh', () => {
    const body = methodSource(read('radiance_viewer.js'), '_refreshSequenceDock')
        .split('\n').filter((l) => !l.trim().startsWith('//')).join('\n');
    const reads = [...body.matchAll(/offsetHeight/g)].length;
    assert.ok(reads <= 1 && /new ResizeObserver\(\(\) => \{\s*this\.canvasWrapper\.style\.setProperty\('--sequence-dock-height', this\.sequenceDock\.offsetHeight/.test(body),
        'the dock height is read outside a ResizeObserver');
});
