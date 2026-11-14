/**
 * The Deliver panel offers exactly what /radiance/deliver accepts, in both
 * directions: every format and colour space in the panel is a key of the
 * server's map, and every key is offered. Read from the two sources.
 * Run: node --test js/tests/deliver_options.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const viewer = readFileSync(join(ROOT, 'js', 'radiance_viewer.js'), 'utf8');
const handler = readFileSync(join(ROOT, 'delivery', 'handler.py'), 'utf8');

function pyKeys(name) {
    const start = handler.indexOf(`${name} = {`);
    assert.notEqual(start, -1, `${name} not found`);
    const body = handler.slice(start, handler.indexOf('\n}', start));
    return [...body.matchAll(/^\s*'([^']+)'\s*:/gm)].map((m) => m[1]).sort();
}

function panelFormats() {
    const start = viewer.indexOf("const dvFormat = document.createElement('select');");
    const list = viewer.slice(start, viewer.indexOf('].forEach(f =>', start));
    return [...list.matchAll(/^\s*'([^']+)',$/gm)].map((m) => m[1]).sort();
}

function panelColourSpaces() {
    const start = viewer.indexOf('const _csGroups = [');
    const list = viewer.slice(start, viewer.indexOf('];', start));
    return [...list.matchAll(/value:\s*'([^']+)'/g)].map((m) => m[1]).sort();
}

test('every Deliver format is one the server maps, and the other way round', () => {
    assert.deepEqual(panelFormats(), pyKeys('_UI_TO_WRITE_FORMAT'));
});

test('every Deliver colour space is one the server maps, and the other way round', () => {
    assert.deepEqual(panelColourSpaces(), pyKeys('_UI_TO_WRITE_COLORSPACE'));
});

test('PQ and HLG are offered, and the 8-bit formats are disabled for them', () => {
    const cs = panelColourSpaces();
    assert.ok(cs.includes('PQ (HDR10 / ST.2084)') && cs.includes('HLG (Hybrid Log-Gamma)'));
    assert.match(viewer, /const _EIGHT_BIT = \['Video — MP4 \(H\.264\)', 'Image Sequence — PNG \(8-bit\)'\];/);
    assert.match(viewer, /if \(hdr && _EIGHT_BIT\.includes\(dvFormat\.value\)\) dvFormat\.value = 'Video — MP4 \(H\.265 10-bit\)';/);
});

test('every HDR10 mastering display in the panel is one the writer knows, and the other way round', () => {
    const writer = readFileSync(join(ROOT, 'io', 'writer.py'), 'utf8');
    const start = writer.indexOf('MASTERING_DISPLAYS: Dict[str, Dict[str, Any]] = {');
    const body = writer.slice(start, writer.indexOf('\n}', start));
    const server = [...body.matchAll(/^\s*"([^"]+)":\s*\{/gm)].map((m) => m[1]).sort();
    const pstart = viewer.indexOf("const dvMastering = document.createElement('select');");
    const list = viewer.slice(pstart, viewer.indexOf('].forEach(f =>', pstart));
    const panel = [...list.matchAll(/^\s*'([^']+)',$/gm)].map((m) => m[1]).sort();
    assert.equal(server.length, 4);
    assert.deepEqual(panel, server);
    assert.match(viewer, /mastering_display: dvMastering\.value/);
});

test('a Deliver job names the frame set it is for (package audit R2)', () => {
    const viewer = readFileSync(new URL('../radiance_viewer.js', import.meta.url), 'utf8');
    const build = viewer.slice(viewer.indexOf('const buildPayload = () => ('), viewer.indexOf('const unbakedAids'));
    assert.match(build, /source_generation: this\.cacheGeneration \?\? null/);
    assert.match(viewer, /viewer\.cacheGeneration = message\.cache_generation\?\.\[0\] \?\? null;/);
});
