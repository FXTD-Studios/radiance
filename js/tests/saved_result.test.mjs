/**
 * What a saved workflow keeps of a Viewer result.
 *
 * The whole result list was written into node.properties: about 1.4 KB a
 * frame, so 6 MB for a 5,000-frame shot, copied into every saved PNG's
 * metadata, every prompt and the clipboard. The node now names every file of
 * a run with one token (nodes/monitor/viewer.py), and the workflow keeps the
 * token, one template entry per kind of frame and the counts. These check the
 * round trip, the size, and that a result which does not rebuild exactly is
 * kept whole, as before.
 *
 * Run: node --test js/tests/saved_result.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'radiance_viewer.js'), 'utf8');

function methodSource(name) {
    const start = src.search(new RegExp(`\\n    static ${name}\\(`));
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    return rest.slice(0, end) + '\n    }';
}

const RadianceViewer = new Function(`return class RadianceViewer {
    ${methodSource('_compactResult')}
    ${methodSource('_expandResult')}
}`)();

/** A result shaped like the node's, n frames with compare and depth. */
function result(n, token = 'a1b2c3d4e5f6', namer = null) {
    const name = namer || ((prefix, i, tail) => `${prefix}_${token}_${i}${tail}`);
    const frame = (prefix, i, extra = {}) => ({
        filename: name(prefix, i, '_thumb.png'), subfolder: '', type: 'temp',
        source_width: 1920, source_height: 1080, preview_width: 1920, preview_height: 1080,
        preview_tonemapped: true, preview_encoding: 'display',
        data_range: [0.001 * i, 4 + i / 100], has_hdr: true,
        hdr_stats: { p1: 0.01, p50: 0.18 + i / 1e4, p99: 3.2, p999: 4.1, mean_luma: 0.2, clipped_pct: 1.5,
            negative_pct: 0, scene_linear_peak: 4.1, ev_range: 8.3, max_nit_est: 832 },
        channel_names: ['R', 'G', 'B', 'A'],
        metadata: { container: 'RHDR', compression: 'ZLIB', pixelType: 'HALF', source: 'ComfyUI tensor',
            exrCompression: 'ZIP', channels: ['R', 'G', 'B', 'A'].map((c) => ({ name: c, pixelType: 'HALF', xSampling: 1, ySampling: 1 })) },
        hdr_filename: name(prefix, i, '.rhdr'), exr_filename: name(prefix, i, '.exr'),
        exr_subfolder: '', exr_type: 'temp', hdr_sidecar: name(prefix, i, '.rhdr'),
        hdr_primary: true, hdr_fp32: false, frame: i,
        source_encoding: 'linear', source_colorspace: 'ACEScg', ...extra,
    });
    const images = [];
    for (let i = 0; i < n; i++) images.push(frame('Radiance_viewer', i, { total_frames: n }));
    for (let i = 0; i < n; i++) images.push(frame('Radiance_compare', i, { is_compare: true }));
    for (let i = 0; i < n; i++) {
        images.push({ filename: name('Radiance_zdepth', i, '.png'), subfolder: '', type: 'temp', is_zdepth: true,
            frame: i, bit_depth: 16, depth_range: [0, i], hdr_sidecar: name('Radiance_zdepth', i, '_float.rhdr') });
    }
    return { radiance_images: images, source_encoding: ['linear'], source_colorspace: ['ACEScg'], fps: [24],
        audio: [], audio_fps: [24], instance_id: ['wf:5'], batch_size: [n], bit_depth: ['16-bit'],
        flicker_data: Array.from({ length: n }, (_, i) => (i % 7) / 9), cut_indices: [3], file_token: [token] };
}

const PER_FRAME = ['data_range', 'hdr_stats', 'depth_range'];
const withoutStats = (e) => Object.fromEntries(Object.entries(e).filter(([k]) => !PER_FRAME.includes(k)));

test('a 5,000-frame result is saved as a few kilobytes and rebuilds every entry', () => {
    const message = result(5000);
    const saved = RadianceViewer._compactResult(message);
    assert.ok(saved, 'the result was not compacted');
    const full = JSON.stringify(message).length, small = JSON.stringify(saved).length;
    assert.ok(full > 5e6, `test premise: the full list is ${full} bytes`);
    // The flicker list is one number per frame and stays; the entries go.
    assert.ok(small < 40e3, `the saved copy is ${small} bytes`);
    const back = RadianceViewer._expandResult(JSON.parse(JSON.stringify(saved)));
    assert.equal(back.radiance_images.length, message.radiance_images.length);
    message.radiance_images.forEach((e, i) => assert.deepEqual(back.radiance_images[i], withoutStats(e)));
    for (const k of ['fps', 'source_colorspace', 'instance_id', 'cut_indices', 'file_token']) {
        assert.deepEqual(back[k], message[k], k);
    }
});

test('a result whose names do not follow the token is kept whole', () => {
    // An older node named every file with its own uuid.
    let n = 0;
    const message = result(4, 'tok000000000', (p, i, tail) => `${p}_${(n++).toString(16).padStart(12, '0')}_${i}${tail}`);
    assert.equal(RadianceViewer._compactResult(message), null);
    const noToken = result(3);
    delete noToken.file_token;
    assert.equal(RadianceViewer._compactResult(noToken), null);
});

test('workflows saved with the whole list still load, and new ones use the compact copy', () => {
    const at = src.indexOf('nodeType.prototype.onConfigure = function');
    const body = src.slice(at, src.indexOf('\n        };', at));
    assert.match(body, /_expandResult\(info\?\.properties\?\.radiance_viewer_saved\)/);
    assert.match(body, /radiance_viewer_result\?\.radiance_images\?\.length/);
    assert.equal(RadianceViewer._expandResult(undefined), null);
    assert.equal(RadianceViewer._expandResult({ v: 2 }), null);
});
