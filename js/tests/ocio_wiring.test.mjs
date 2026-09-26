/**
 * The OCIO panel's wiring.
 *
 * The GPU test proves the transform is right. These hold the two properties
 * that make it *honest*, neither of which a rendering test can see:
 *
 *   - OCIO is a capability, not a dependency. With no config loaded the
 *     viewer's own ACES 1.3 pipeline runs exactly as it did before.
 *   - There is no half-applied state. Every failure turns OCIO fully off and
 *     says why, because the alternative is a Display/View menu that says one
 *     thing while the picture shows another — the precise failure this feature
 *     exists to remove.
 *
 * Run: node --test js/tests/ocio_wiring.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');
const viewer = read('radiance_viewer.js');
const webgl = read('radiance_webgl.js');
const base = read('radiance_renderer.js');

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

test('the vendored library is present and documented', () => {
    for (const f of ['index.js', 'ocio-wasm.js', 'ocio-wasm.node.js', 'ocio-wasm.wasm', 'LICENSE', 'README.md']) {
        assert.ok(existsSync(join(JS, 'vendor/ocio', f)), `js/vendor/ocio/${f} is missing`);
    }
    const readme = read('vendor/ocio/README.md');
    assert.match(readme, /BSD-3-Clause/, 'the vendor README must record the licence');
    assert.match(readme, /0\.0\.\d+/, 'the vendor README must pin a version');
    assert.match(readme, /patch/i, 'the vendor README must describe the local patch');
});

test('the WASM is loaded lazily, not on import', () => {
    // 4.7 MB of WebAssembly must not be instantiated for a user who never opens
    // a config. The viewer imports the façade; the façade dynamic-imports the
    // library on first use.
    const ocio = read('radiance_ocio.js');
    assert.doesNotMatch(code(ocio), /^import .*vendor\/ocio/m,
        'the façade must dynamic-import the vendored library, not import it statically');
    assert.match(code(ocio), /await import\(VENDOR\)/, 'no dynamic import of the vendored library');
    assert.match(code(viewer), /from "\.\/radiance_ocio\.js"/,
        'the viewer must go through the façade');
    assert.doesNotMatch(code(viewer), /vendor\/ocio/,
        'the viewer must not reach past the façade into the vendored library');
});

test('nothing is applied that was not built', () => {
    // The failure mode this whole feature exists to remove: a populated
    // Display/View menu that does not change the picture.
    const apply = code(methodBody(viewer, '_ocioApply'));
    assert.match(apply, /if \(built\.error\)[\s\S]*_ocioDisable/,
        'a view that fails to build must turn OCIO off, not leave it listed');
    assert.match(apply, /if \(!res \|\| !res\.ok\)[\s\S]*_ocioDisable/,
        'a renderer that refuses must turn OCIO off, not leave it listed');
    assert.match(apply, /this\.ocioActive = true/);
    const activeAt = apply.indexOf('this.ocioActive = true');
    const disableAt = apply.lastIndexOf('_ocioDisable');
    assert.ok(disableAt < activeAt, 'ocioActive is set before the failure paths have run');
});

test('disabling says why, and returns to Radiance\'s own pipeline', () => {
    const off = code(methodBody(viewer, '_ocioDisable'));
    assert.match(off, /this\.ocioActive = false/);
    assert.match(off, /setOCIODisplay\?\.\(null\)/,
        'disabling must clear the renderer, not just the flag');
    assert.match(off, /_ocioSetStatus\('error'/,
        'a forced disable must state its reason');
});

test('a pass-through view is called out rather than looking like a failure', () => {
    // Every ACES config has a Raw view. A user picking it sees nothing change
    // and should not be left wondering whether the config loaded.
    assert.match(code(methodBody(viewer, '_ocioApply')), /built\.isNoOp/,
        'the panel must distinguish a no-op view from a broken one');
});

test('switching display repairs the view rather than failing on a stale one', () => {
    // The previous view usually does not exist on the new display. Keeping the
    // stale name produces an error naming a view the user never chose.
    assert.match(code(methodBody(viewer, 'renderOcioSection')),
        /if \(!views\.includes\(this\.ocioView\)\) this\.ocioView = views\[0\]/,
        'changing display must reset the view to one the new display has');
});

test('the shader always declares the entry point, config or not', () => {
    // The composite shader contains the OCIODisplay() call site unconditionally,
    // so a missing declaration breaks the viewer for every user — including the
    // ones who never touch colour management.
    assert.match(webgl, /vec4 OCIODisplay\(vec4 inPixel\) \{ return inPixel; \}/,
        'no stub entry point for the no-config case');
    assert.match(webgl, /uniform bool u_ocioEnabled/);
});

test('exactly-zero input is nudged before it reaches OCIO', () => {
    // Regression. GLSL leaves pow(x, y) undefined for x == 0, y <= 0 and OCIO's
    // inverse-EOTF chains reach it: measured on a real context, scene-linear
    // black rendered as 1.3e16, 7.7e14 or NaN depending on the display.
    //
    // Pinned as `equal(...)` rather than a max(): anything down to 1e-30 goes
    // through correctly and negatives do too, so a blanket clamp would change
    // legitimately negative scene-linear pixels for no reason.
    const frag = code(webgl);
    assert.match(frag, /mix\(color, vec3\(1e-10\), vec3\(equal\(color, vec3\(0\.0\)\)\)\)/,
        'the exact-zero nudge is gone — black will render as NaN or ~1e16');
    assert.doesNotMatch(frag, /OCIODisplay\(vec4\(color, 1\.0\)\)/,
        'OCIODisplay is being called on unguarded input');
});

test('float LUT filtering is enabled, not merely assumed', () => {
    // Regression. R32F is not filterable in WebGL2 unless this extension is
    // *enabled*; listing it in getSupportedExtensions() does nothing. Without
    // the call, LINEAR sampling returns zero and every HDR view renders solid
    // black with a clean compile and no error anywhere.
    const create = code(methodBody(webgl, '_createOCIOTexture'));
    assert.match(create, /getExtension\('OES_texture_float_linear'\)/,
        'the extension must be enabled before a float LUT is sampled with LINEAR');
    assert.match(create, /_ocioFloatLinear.*\?.*NEAREST.*:.*LINEAR|NEAREST\s*:\s*gl\.LINEAR/s,
        'without the extension the filter must fall back to NEAREST');
});

test('OCIO textures do not collide with the composite shader\'s own', () => {
    // Units 0–6 carry the image, depth, LUT, reference, curves and bloom.
    assert.match(code(methodBody(webgl, 'setOCIODisplay')), /unit: 7 \+ i/,
        'OCIO textures must start at unit 7');
    assert.match(code(methodBody(webgl, 'setOCIODisplay')), /MAX_TEXTURE_IMAGE_UNITS/,
        'a config needing more units than the GPU has must be refused, not truncated');
});

test('a shader that will not compile leaves a working viewer', () => {
    const set = code(methodBody(webgl, 'setOCIODisplay'));
    assert.match(set, /if \(!rebuilt\)[\s\S]*_rebuildCompositeProgram\(\)/,
        'a failed OCIO compile must restore the previous program, not leave a dead one');
    assert.match(set, /ok: false/, 'the failure must be reported to the caller');
});

test('rebuilding the program clears the uniform caches', () => {
    // Both caches are keyed on the program object, so the old program's entries
    // would shadow the new one's and half the uniforms would never be uploaded.
    const rebuild = code(methodBody(webgl, '_rebuildCompositeProgram'));
    assert.match(rebuild, /_uniformValueCache\?\.clear\?\.\(\)/);
    assert.match(rebuild, /_uniformCache\?\.clear\?\.\(\)/);
});

test('the backend without a WGSL path refuses rather than pretending', () => {
    // Storing the state on WebGPU would give a fully interactive Display/View
    // menu that changes nothing — the same defect as the Masks and Qualifiers
    // tabs on that backend.
    const stub = code(methodBody(base, 'setOCIODisplay'));
    assert.match(stub, /ok: false/, 'the base renderer must refuse, not store');
    assert.match(stub, /error:/, 'the refusal must carry a reason the viewer can show');
});

test('OCIO replaces the display transform outright, never partially', () => {
    // Running our tonemap or our sRGB OETF alongside the config's would
    // double-apply a transfer function — the same class of bug the
    // u_lutIsDisplayTransform flag already exists to prevent.
    const frag = code(webgl);
    const start = frag.indexOf('if (u_ocioEnabled) {');
    assert.notEqual(start, -1, 'the OCIO branch is gone');
    const branch = frag.slice(start, frag.indexOf('} else {', start));
    assert.doesNotMatch(branch, /toneMapACES|linearToSRGB|applyDisplayLUT/,
        'the OCIO branch must not also run Radiance\'s display transform');
    // And the converse: the fallback must still be intact for everyone who has
    // no config loaded.
    const fallback = frag.slice(frag.indexOf('} else {', start), frag.indexOf('end !u_ocioEnabled'));
    assert.match(fallback, /toneMapACES/, 'the built-in ACES path was removed along the way');
    assert.match(fallback, /linearToSRGB/, 'the built-in OETF was removed along the way');
});
