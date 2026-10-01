/**
 * Viewer playback: how an RHDR frame is decoded.
 *
 * Measured in Edge on 1080p HDR clips, decoding on the main thread held
 * sequence playback near 14 frames/s and cut its sound:
 *  - _parseRHDR converted every fp16 sample with _halfToFloat (135 ms per
 *    frame). It now reads a table of all 65536 half values, which must give
 *    exactly what the function gives, NaN, infinities, subnormals and -0 included.
 *  - frames are now fetched and decoded in workers built from _parseRHDR's own
 *    source. A worker must hand back exactly what the main thread would get.
 *
 * Run: node --test js/tests/rhdr_decode.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { deflateSync } from 'node:zlib';
import vm from 'node:vm';

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.search(new RegExp(`\\n    (static )?(async )?${name}\\(`));
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    // A blank line after the brace: the worker script inside _sidecarWorkers
    // has a line with a four-space closing brace of its own.
    const end = rest.search(/\n    \}\n\n/);
    assert.notEqual(end, -1, `could not find the end of ${name}`);
    return rest.slice(0, end) + '\n    }';
}

const RadianceViewer = new Function(`return class RadianceViewer {
    static frameWindowConcurrency() { return 2; }
    ${['_sidecarWorkers', '_parseRHDR', '_zlibInflateAsync', '_halfToFloat'].map(methodSource).join('\n')}
}`)();

/** An fp16 RHDR file holding every half value once (256 x 64 RGBA). */
function makeRHDR() {
    const width = 256, height = 64, channels = 4;
    const halves = new Uint16Array(width * height * channels);
    for (let h = 0; h < halves.length; h++) halves[h] = h;
    const header = Buffer.alloc(12);
    header.write('RHDR', 0);
    header.writeUInt16LE(width, 4);
    header.writeUInt16LE(height, 6);
    header.writeUInt16LE(channels, 8);
    header.writeUInt16LE(0, 10);                            // flags 0: fp16 payload
    const file = Buffer.concat([header, deflateSync(Buffer.from(halves.buffer))]);
    return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
}

test('an fp16 RHDR frame decodes every half value exactly as _halfToFloat does', async () => {
    const viewer = new RadianceViewer();
    const parsed = await viewer._parseRHDR(makeRHDR());
    assert.equal(parsed.format, 'rhdr');
    assert.equal(parsed.data.length, 65536);
    for (let h = 0; h < 65536; h++) {
        const expected = Math.fround(viewer._halfToFloat(h));
        if (!Object.is(parsed.data[h], expected)) {
            assert.fail(`half 0x${h.toString(16)}: got ${parsed.data[h]}, expected ${expected}`);
        }
    }
});

test('a sidecar worker decodes a frame exactly as the main thread does', async () => {
    let blob = null;
    const createObjectURL = URL.createObjectURL;
    globalThis.Worker = class { constructor(url) { this.url = url; } postMessage() {} };
    URL.createObjectURL = (b) => { blob = b; return 'blob:radiance-test'; };
    try {
        RadianceViewer._sidecarPool = undefined;
        assert.ok(RadianceViewer._sidecarWorkers(), 'no worker pool was built');
    } finally {
        URL.createObjectURL = createObjectURL;
        delete globalThis.Worker;
    }

    // Run the worker script in a realm of its own, as a Worker would.
    const file = makeRHDR();
    const posted = [];
    const scope = vm.createContext({
        TextDecoder, DecompressionStream, console,
        fetch: async () => ({ arrayBuffer: async () => file.slice(0) }),
    });
    scope.self = { postMessage: (message, transfer) => posted.push({ message, transfer }) };
    vm.runInContext(await blob.text(), scope);
    await scope.self.onmessage({ data: { id: 7, url: 'http://127.0.0.1/api/view?filename=f.rhdr' } });

    assert.equal(posted.length, 1);
    const { message, transfer } = posted[0];
    assert.equal(message.id, 7);
    assert.equal(message.error, undefined, message.error);
    assert.equal(transfer.length, 2, 'both arrays are transferred, not copied');
    const expected = await new RadianceViewer()._parseRHDR(makeRHDR());
    assert.deepEqual([...message.parsed.shape], expected.shape);     // an array of the worker's realm
    assert.equal(message.parsed.format, expected.format);
    assert.deepEqual([...message.parsed.fp16data], [...expected.fp16data]);
    for (let i = 0; i < expected.data.length; i++) {
        if (!Object.is(message.parsed.data[i], expected.data[i])) assert.fail(`sample ${i} differs`);
    }
});

test('without workers, frames are decoded on the main thread', () => {
    const warn = console.warn;
    console.warn = () => {};
    try {
        RadianceViewer._sidecarPool = undefined;
        assert.equal(RadianceViewer._sidecarWorkers(), null);   // no Worker in Node
    } finally {
        console.warn = warn;
    }
});
