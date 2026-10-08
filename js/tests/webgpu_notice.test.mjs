// The Masks and Qualifiers tabs warn on WebGPU that they do nothing there.
// The warning ended "Radiance prefers WebGPU whenever the browser offers it",
// which stopped being true when WebGPU became opt-in: it has to say how to
// get the tabs back.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');

function noticeBody() {
    const start = viewer.indexOf('_backendUnsupportedNotice(container, what) {');
    assert.ok(start > 0, '_backendUnsupportedNotice not found');
    return viewer.slice(start, viewer.indexOf('\n    }\n', start));
}

test('the WebGPU notice does not claim WebGPU is the default', () => {
    assert.doesNotMatch(noticeBody(), /prefers WebGPU whenever/);
});

test('the WebGPU notice says how to switch back to WebGL', () => {
    assert.match(noticeBody(), /Backend/);
    assert.match(noticeBody(), /WebGL/);
});
