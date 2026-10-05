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

test('a float upload reads the GL error on the first and every 24th upload of a format', () => {
    const Renderer = new Function(`return class { ${methodSource(read('radiance_webgl.js'), '_uploadError')} }`)();
    let calls = 0, next = 0, purges = 0;
    const r = new Renderer();
    r.gl = { NO_ERROR: 0, getError() { calls++; return next; } };
    r.clearFrameCache = () => { purges++; };
    next = 1281;                                            // GL_INVALID_VALUE on the first upload
    assert.equal(r._uploadError('f16:1920x1080x4'), 1281);
    assert.equal(purges, 0, 'nothing went unchecked before the first upload');
    next = 0;
    assert.equal(r._uploadError('f16:1920x1080x4'), 0, 'a failed format is checked on its next upload');
    for (let i = 0; i < 23; i++) r._uploadError('f16:1920x1080x4');
    assert.equal(calls, 2, 'a clean format is not read on every upload');
    next = 1285;                                            // GL_OUT_OF_MEMORY on the 24th upload after it
    assert.equal(r._uploadError('f16:1920x1080x4'), 1285, 'a later failure is still caught');
    assert.equal(purges, 1, 'frames uploaded since the last check are purged');
    next = 0;
    r._uploadError('f16:1920x1080x4');
    assert.equal(calls, 4, 'after a failure the format is checked again at once');
    r._uploadError('f16:3840x2160x4');
    assert.equal(calls, 5, 'a new format is checked');
});

test('the sequence dock does not read its height on every refresh', () => {
    const body = methodSource(read('radiance_viewer.js'), '_refreshSequenceDock')
        .split('\n').filter((l) => !l.trim().startsWith('//')).join('\n');
    const reads = [...body.matchAll(/offsetHeight/g)].length;
    assert.ok(reads <= 1 && /new ResizeObserver\(\(\) => \{\s*this\.canvasWrapper\.style\.setProperty\('--sequence-dock-height', this\.sequenceDock\.offsetHeight/.test(body),
        'the dock height is read outside a ResizeObserver');
});
