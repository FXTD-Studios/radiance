// "Write" showed widgets its writer ignores for the chosen format or colour
// settings: quality on ProRes/DNxHR, broadcast_safe on video, the reference
// white and ocio_config when nothing reads them.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const strip = (file) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '')
    .replace(/^export default .*$/m, '')
    .replace(/^export /gm, '');

const context = vm.createContext({
    app: { registerExtension() {}, graph: { setDirtyCanvas() {} } },
    document: { addEventListener() {} },
    console: { log() {}, warn() {} },
});
vm.runInContext(strip('radiance_widget_utils.js') + strip('radiance_io.js')
    + 'globalThis.visibility = writeWidgetVisibility;', context);

const shown = (values) => Object.entries(context.visibility(values))
    .filter(([, visible]) => visible).map(([name]) => name).sort();

test('an EXR sequence shows its compression and numbering, no video or display settings', () => {
    assert.deepEqual(shown({ format: 'SEQ │ EXR (32-bit float)', color_space: 'Linear (pass-through)' }),
        ['color_space', 'exr_compression', 'frame_padding', 'start_frame']);
});

test('ProRes takes no CRF and no broadcast clamp; PQ shows the reference white', () => {
    assert.deepEqual(shown({ format: 'VID │ MOV (ProRes 4444)', color_space: 'PQ (HDR10 / ST.2084)' }),
        ['audio_source', 'color_space', 'fps', 'hdr_reference_nits']);
});

test('H.264 and H.265 show their CRF', () => {
    for (const format of ['VID │ MP4 (H.264)', 'VID │ MP4 (H.265 10-bit)']) {
        assert.ok(shown({ format, color_space: 'sRGB' }).includes('quality'), format);
    }
});

test('an 8/16-bit still shows broadcast_safe, and quality only for JPEG and WEBP', () => {
    assert.deepEqual(shown({ format: 'IMG │ PNG (16-bit)', color_space: 'sRGB' }),
        ['broadcast_safe', 'color_space']);
    assert.ok(shown({ format: 'IMG │ JPEG', color_space: 'sRGB' }).includes('quality'));
});

test('an OCIO colorspace replaces color_space and shows its config', () => {
    assert.deepEqual(shown({ format: 'SEQ │ EXR (16-bit half)', color_space: 'PQ (HDR10 / ST.2084)',
                             ocio_colorspace: 'ACEScct' }),
        ['exr_compression', 'frame_padding', 'ocio_config', 'start_frame']);
});
