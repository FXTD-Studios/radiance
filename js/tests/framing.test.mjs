/**
 * Framing guides, safe areas, timecode.
 *
 * The safe-area figures are the reason this file is careful. A safe-area box is
 * a delivery-QC guide: a graphic placed against the wrong one gets rejected, or
 * gets needlessly inset for the life of a show. Radiance's own planning note
 * had them attributed wrongly in both directions, and the shipped viewer drew a
 * pairing that no published standard specifies — the modern 93% action box with
 * the legacy 80% title box.
 *
 * What the standards actually say, checked against the documents:
 *
 *   SMPTE ST 2046-1 (and RP 218): safe action 93%, safe title 90%, with a
 *   legacy 90%/80% pair retained for 480-line material from RP 8 / RP 13.
 *
 *   EBU R 95: action safe at a 3.5% inset, graphics safe at a 5% inset — the
 *   same 93% and 90% boxes, named differently.
 *
 * So the two bodies agree on the geometry. These tests pin that, because it is
 * exactly the kind of fact that gets "corrected" from memory into something
 * plausible and wrong.
 *
 * Run: node --test js/tests/framing.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');

function code(src) {
    return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
}
function methodBody(src, name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    assert.notEqual(end, -1, `could not find the end of ${name}`);
    return rest.slice(0, end);
}

/** Pull the preset table out of the source, since the class cannot be imported. */
function presets() {
    const m = viewer.match(/static SAFE_AREA_PRESETS = \[([\s\S]*?)\n    \];/);
    assert.ok(m, 'SAFE_AREA_PRESETS not found');
    return m[1];
}

// ── the standards ───────────────────────────────────────────────────────────

test('the modern preset is 93% action and 90% title', () => {
    const table = presets();
    const modern = table.slice(table.indexOf("id: 'modern'"), table.indexOf("id: 'legacy'"));
    assert.match(modern, /outer: 0\.93/, 'safe action must be 93%');
    assert.match(modern, /inner: 0\.90/, 'safe title / graphics must be 90%');
});

test('the modern preset credits both bodies, because they agree', () => {
    // SMPTE ST 2046-1 gives 93/90 directly; EBU R 95 gives 3.5% and 5% insets,
    // which are the same two boxes. Offering them as rival options would imply
    // a choice that does not exist.
    const table = presets();
    const modern = table.slice(table.indexOf("id: 'modern'"), table.indexOf("id: 'legacy'"));
    assert.match(modern, /ST 2046-1/, 'the SMPTE standard must be named');
    assert.match(modern, /EBU R ?95/, 'the EBU recommendation must be named');
    assert.match(modern, /graphics/i, 'EBU calls the inner box graphics safe; say so');
});

test('90/80 is labelled legacy SMPTE, not EBU', () => {
    // The specific error in the planning note. 90/80 is SMPTE's 480-line pair
    // carried forward from RP 8 (1961) and RP 13 (1963). EBU R 95 does not
    // specify it at all, and attributing it to them would send someone to the
    // wrong document.
    const table = presets();
    const legacy = table.slice(table.indexOf("id: 'legacy'"));
    assert.match(legacy, /outer: 0\.90/);
    assert.match(legacy, /inner: 0\.80/);
    assert.match(legacy, /RP 218|480/, 'the legacy pair must name where it comes from');
    assert.doesNotMatch(legacy, /EBU/, '90/80 is not an EBU figure');
});

test('the action and title boxes always come from the same preset', () => {
    // The shipped viewer drew the modern 93% action box against the legacy 80%
    // title box — a pairing no standard publishes, and one that insets graphics
    // by 10% for no reason.
    const body = code(methodBody(viewer, 'drawGrid'));
    assert.match(body, /box\(preset\.outer/, 'the action box must come from the preset');
    assert.match(body, /box\(preset\.inner/, 'the title box must come from the preset');
    assert.doesNotMatch(body, /0\.10|titleMargin/, 'a hardcoded 10% title margin is back');
    assert.doesNotMatch(body, /Title Safe 80%/, 'the legacy title box is being drawn unconditionally');
});

test('every safe-area box is labelled, and the standard is named on screen', () => {
    const body = code(methodBody(viewer, 'drawGrid'));
    assert.match(body, /fillText\(label/, 'the boxes are drawn without their labels');
    assert.match(body, /fillText\(preset\.label/,
        'an unlabelled safe-area box is not usable for QC — name the standard');
});

// ── geometry ────────────────────────────────────────────────────────────────

test('guides are placed against the picture, not the canvas', () => {
    // They used to be drawn across the full canvas, so at any zoom or pan other
    // than an exact fit the "93%" box bore no relationship to the image.
    assert.match(code(methodBody(viewer, '_imageRect')), /this\.imageWidth.*this\.zoom/s,
        '_imageRect must be derived from the image size and zoom');
    for (const m of ['drawGrid', 'drawAspectMatte']) {
        assert.match(code(methodBody(viewer, m)), /_imageRect\(\)/,
            `${m} must place its guides against the picture`);
    }
});

test('an image with no size does not produce a rectangle', () => {
    assert.match(code(methodBody(viewer, '_imageRect')), /if \(!\(w > 0 && h > 0\)\) return null/,
        'a zero-size image must yield null rather than a degenerate rect');
});

test('the matte is a separate control from the safe areas', () => {
    // Conflating a creative framing guide with a delivery QC guide is how a
    // graphic ends up placed against the wrong box.
    assert.match(viewer, /static MATTE_PRESETS = \[/);
    assert.match(code(methodBody(viewer, 'drawAspectMatte')), /MATTE_PRESETS/);
    assert.doesNotMatch(code(methodBody(viewer, 'drawGrid')), /MATTE_PRESETS/,
        'the matte must not be drawn by the safe-area code');
});

test('the matte offers the ratios a finish actually targets', () => {
    const table = viewer.match(/static MATTE_PRESETS = \[([\s\S]*?)\n    \];/)[1];
    for (const r of ['2.39', '1.85', '16 / 9', '4 / 3', '9 / 16']) {
        assert.ok(table.includes(r), `the matte list is missing ${r}`);
    }
});

test('the matte dims rather than blanking', () => {
    // The reason to frame to a matte is usually to check what is just outside
    // it, so a solid black bar defeats the purpose.
    const body = code(methodBody(viewer, 'drawAspectMatte'));
    assert.match(body, /rgba\(0, 0, 0, \$\{this\.matteOpacity/, 'the matte must be translucent');
    assert.match(body, /fillRect[\s\S]*fillRect[\s\S]*fillRect[\s\S]*fillRect/,
        'four bars, so the framed area is left completely untouched');
});

test('the matte handles both wider and taller targets', () => {
    // 9:16 on a 16:9 image needs side bars, not top and bottom. A single branch
    // would produce a matte outside the picture.
    const body = code(methodBody(viewer, 'drawAspectMatte'));
    assert.match(body, /preset\.ratio > current/, 'the matte must branch on which axis is constrained');
});

// ── magnification filter ────────────────────────────────────────────────────

test('the magnification filter is bound to N, as RV binds it', () => {
    assert.match(viewer, /case 'n': this\.togglePixelFilter\(\)/,
        'nearest-neighbour should be on N — that is where people reach for it');
});

test('the filter choice survives a new frame', () => {
    // Every texture upload resets the parameter, so a toggle that only set it
    // once would silently revert on the next frame of a sequence.
    const gl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');
    assert.match(code(methodBody(gl, 'loadImageTexture')), /_applyPixelFilter\(\)/,
        'the filter must be re-applied after an image upload');
    assert.match(viewer, /localStorage\.setItem\('radiance_pixel_filter'/, 'the choice is not persisted');
});

test('only magnification changes, never minification', () => {
    // Nearest on a downscaled image aliases and shows detail that is not there,
    // which is the opposite of what the toggle is for.
    const gl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');
    const body = code(methodBody(gl, '_applyPixelFilter'));
    assert.match(body, /TEXTURE_MAG_FILTER/);
    assert.doesNotMatch(body, /TEXTURE_MIN_FILTER/,
        'minification must stay interpolated');
});

test('a float texture without the linear extension falls back rather than sampling black', () => {
    const gl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');
    assert.match(code(methodBody(gl, '_applyPixelFilter')), /extColorFloatLinear/,
        'linear on an unfilterable float texture returns zero — guard it');
});

// ── time display ────────────────────────────────────────────────────────────

test('timecode is SMPTE: integer timebase, zero-padded, through one formatter', async () => {
    // 3.5.0: every timecode in the viewer goes through radiance_timecode.js.
    // The old per-site maths used the real rate as the frame modulus, so
    // 29.97 printed frame 30 ("00:00:17:30" for frame 539).
    const body = code(methodBody(viewer, 'formatFramePosition'));
    assert.match(body, /_smpteTC\(/, 'formatFramePosition must use the shared SMPTE formatter');
    const { smpteTimecode } = await import('../radiance_timecode.js');
    assert.equal(smpteTimecode(0, 24), '00:00:00:00');
    assert.equal(smpteTimecode(86400, 24), '01:00:00:00');
    assert.equal(smpteTimecode(47, 23.976), '00:00:01:23', '23.976 counts against 24');
    for (let f = 0; f < 2000; f++) {
        const ff = Number(smpteTimecode(f, 29.97).slice(-2));
        assert.ok(ff < 30, `frame ${f} printed frames field ${ff} at 29.97`);
    }
});

test('drop-frame at 29.97 / 59.94 renumbers, and says so with a semicolon', async () => {
    const { smpteTimecode } = await import('../radiance_timecode.js');
    // SMPTE ST 12: frame numbers 00 and 01 are skipped at each minute except
    // every tenth, so 1800 frames in is 00:01:00;02 and 17982 is 00:10:00;00.
    assert.equal(smpteTimecode(1799, 29.97), '00:00:59;29');
    assert.equal(smpteTimecode(1800, 29.97), '00:01:00;02');
    assert.equal(smpteTimecode(17982, 29.97), '00:10:00;00');
    assert.equal(smpteTimecode(107892, 29.97), '01:00:00;00', 'one wall-clock hour');
    assert.equal(smpteTimecode(3600, 59.94), '00:01:00;04');
    assert.equal(smpteTimecode(1800, 29.97, { dropFrame: false }), '00:01:00:00', 'non-drop on request');
    assert.equal(smpteTimecode(1800, 25), '00:01:12:00', 'PAL is never drop-frame');
});

test('the three display modes are all reachable', () => {
    const body = code(methodBody(viewer, 'cycleTimeDisplay'));
    assert.match(body, /\['frames', 'seconds', 'timecode'\]/);
    assert.match(body, /localStorage\.setItem\('radiance_time_display'/, 'the choice is not persisted');
});

test('the frame rate is only used for conversion, and says so', () => {
    // It must not imply the viewer is playing at that rate — it is an
    // inspector, not a transport.
    const section = code(methodBody(viewer, 'renderFramingSection'));
    assert.match(section, /Used only to convert/, 'the frame-rate control must state its scope');
    assert.match(section, /23\.976/, 'the broadcast rates must be offered, not just integers');
    assert.match(section, /29\.97/);
});

// ── review findings, pinned ─────────────────────────────────────────────────
// Three defects a self-review of this session's own work turned up. None of
// them failed a test, which is why they are here now.

test('the filter is applied to a named texture unit', () => {
    // _applyPixelFilter bound TEXTURE_2D without selecting a unit first, so it
    // operated on whichever unit happened to be active — unit 3 is the depth
    // map, unit 4 the reference — and then left the image texture bound there.
    const gl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');
    const body = code(methodBody(gl, '_applyPixelFilter'));
    assert.match(body, /gl\.activeTexture\(gl\.TEXTURE0\)/,
        'bind to unit 0 explicitly, not to whatever was current');
    assert.doesNotMatch(body, /bindTexture\(gl\.TEXTURE_2D, null\)/,
        'unit 0 holds the image; unbinding it just forces the next draw to rebind');
});

test('OCIO LUT textures are freed on teardown', () => {
    // Everything else in destroy() with a lifetime longer than a frame is freed
    // explicitly rather than left to context loss; these were not.
    const gl = readFileSync(join(JS, 'radiance_webgl.js'), 'utf8');
    const start = gl.indexOf('    destroy() {');
    const body = gl.slice(start, gl.indexOf('\n    }\n', start));
    assert.match(body, /_releaseOCIOTextures\(\)/, 'OCIO textures leak on destroy');
});

// ── backend default ─────────────────────────────────────────────────────────

test('WebGPU is opt-in, not automatic', () => {
    // It used to upgrade wherever navigator.gpu existed, so nobody chose it —
    // and the backend it switched people to is the one missing masks,
    // qualifiers, the HDR heatmap and OpenColorIO. Defaulting to the backend
    // with the missing features, then explaining the gaps with four in-panel
    // banners, is a worse product than defaulting to the one that works.
    assert.match(viewer, /localStorage\.getItem\('radiance_prefer_webgpu'\) === '1'\s*\n\s*&& navigator\.gpu/,
        'the WebGPU upgrade must be gated on an explicit preference');
    const auto = /if \(navigator\.gpu && typeof RadianceWebGPURenderer !== 'undefined'/;
    assert.doesNotMatch(code(viewer), auto, 'the automatic WebGPU upgrade is back');
});

test('turning WebGPU on says what stops working', () => {
    // A user who opts in should know exactly what breaks, rather than
    // discovering it one inert panel at a time.
    const section = code(methodBody(viewer, 'renderFramingSection'));
    assert.match(section, /radiance_prefer_webgpu/, 'no control to change the backend');
    for (const feature of ['[Mm]asks', '[Qq]ualifiers', 'HDR heatmap', 'OpenColorIO']) {
        assert.match(section, new RegExp(feature),
            `the backend control does not mention ${feature} as unavailable on WebGPU`);
    }
});
