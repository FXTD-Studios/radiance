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
    ${['_sidecarWorkers', '_parseRHDR', '_zlibInflateAsync', '_halfToFloat', '_lazyHalfFloats',
        '_halfFloatTable', '_halfFloats', '_zoneStatsFromFloats'].map(methodSource).join('\n')}
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
    // The frame holds its half floats only. Every frame used to carry a
    // Float32Array copy too: 50 MB per 1080p frame instead of 17.
    assert.ok(!Object.keys(parsed).includes('data'), 'the frame carries a float copy');
    assert.equal(parsed.fp16data.length, 65536);
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
        fetch: async () => ({ ok: true, status: 200, arrayBuffer: async () => file.slice(0) }),
    });
    scope.self = { postMessage: (message, transfer) => posted.push({ message, transfer }) };
    vm.runInContext(await blob.text(), scope);
    await scope.self.onmessage({ data: { id: 7, url: 'http://127.0.0.1/api/view?filename=f.rhdr' } });

    assert.equal(posted.length, 1);
    const { message, transfer } = posted[0];
    assert.equal(message.id, 7);
    assert.equal(message.error, undefined, message.error);
    // One array crosses: the halves, transferred. The floats are decoded on
    // the main thread only when something reads them.
    assert.equal(transfer.length, 1, 'the half floats are transferred, not copied');
    assert.equal(message.parsed.data, undefined, 'the worker sent a float copy as well');
    const expected = await new RadianceViewer()._parseRHDR(makeRHDR());
    assert.deepEqual([...message.parsed.shape], expected.shape);     // an array of the worker's realm
    assert.equal(message.parsed.format, expected.format);
    assert.deepEqual([...message.parsed.fp16data], [...expected.fp16data]);
    const floats = RadianceViewer._lazyHalfFloats(message.parsed).data;
    for (let i = 0; i < expected.data.length; i++) {
        if (!Object.is(floats[i], expected.data[i])) assert.fail(`sample ${i} differs`);
    }
    // The frame's statistics come from the worker, not a sort on the main thread.
    assert.equal(message.parsed.zoneStats.samples, 65536 / 4);
});

test('a missing sidecar is reported as missing, not as a decode failure', async () => {
    // After a ComfyUI restart the saved frames are gone: the server answers
    // 404, and the viewer said "missing DecompressionStream, or integrity
    // mismatch".
    let blob = null;
    const createObjectURL = URL.createObjectURL;
    globalThis.Worker = class { constructor(url) { this.url = url; } postMessage() {} };
    URL.createObjectURL = (b) => { blob = b; return 'blob:radiance-test'; };
    try {
        RadianceViewer._sidecarPool = undefined;
        RadianceViewer._sidecarWorkers();
    } finally {
        URL.createObjectURL = createObjectURL;
        delete globalThis.Worker;
        RadianceViewer._sidecarPool = undefined;
    }
    const posted = [];
    const scope = vm.createContext({
        TextDecoder, DecompressionStream, console,
        fetch: async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(13) }),
    });
    scope.self = { postMessage: (message) => posted.push(message) };
    vm.runInContext(await blob.text(), scope);
    await scope.self.onmessage({ data: { id: 3, url: 'http://127.0.0.1/api/view?filename=gone.rhdr' } });
    assert.equal(posted[0].missing, true);
    assert.match(posted[0].error, /404/);
    assert.equal(posted[0].parsed, undefined);
});

test('the zone statistics match a sort of the same samples', () => {
    // A histogram 1/128 of a stop wide stands in for the sort the main
    // thread used to do per frame. It must agree with it to within a bin.
    const viewer = new RadianceViewer();
    const n = 50000, data = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        const v = i < n / 4 ? 0 : 0.001 * Math.pow(2, 16 * ((i * 7919) % n) / n);   // a quarter black
        data[i * 3] = data[i * 3 + 1] = data[i * 3 + 2] = v;
    }
    const raw = viewer._zoneStatsFromFloats(data, 3);
    const luma = [];
    for (let i = 0; i < n; i++) luma.push(0.2126 * data[i * 3] + 0.7152 * data[i * 3 + 1] + 0.0722 * data[i * 3 + 2]);
    luma.sort((a, b) => a - b);
    const at = (arr, p) => arr[Math.floor(p * 0.01 * (arr.length - 1))];
    for (const p of [1, 10, 30, 50, 70, 90, 99, 99.9]) {
        const want = at(luma, p), got = raw['p' + String(p).replace('.', '')];
        if (want === 0) assert.equal(got, 0, `p${p}`);
        else assert.ok(Math.abs(Math.log2(got / want)) < 1 / 64, `p${p}: ${got} against ${want}`);
    }
    const positive = luma.filter((y) => y > 0);
    assert.ok(Math.abs(Math.log2(raw.p1NonZero / at(positive, 1))) < 1 / 64, 'lowest non-zero percentile');
    assert.ok(Math.abs(raw.meanLuma - luma.reduce((a, b) => a + b, 0) / n) < 1e-6);
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
