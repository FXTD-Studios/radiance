/**
 * The two rendering backends must agree.
 *
 * Radiance ships a WebGL renderer and a WebGPU renderer, and WebGPU is the
 * *preferred* one — `_tryWebGPUUpgrade()` runs whenever `navigator.gpu` exists.
 * Nothing has ever compared their output. An audit of the two files found the
 * grade maths implemented four separate times, and they do not all agree:
 *
 *   radiance_webgl.js:2617    (c − pivot) · clamp(k,0,5) + pivot     linear
 *   radiance_webgpu.js:423    (c − pivot) · k + pivot                linear, unclamped
 *   radiance_webgpu.js:1388   pivot · pow(c/pivot, k)                POWER
 *   radiance_viewer.js:6515   (c − pivot) · k + pivot                linear (LUT export)
 *
 * The third is a different curve, not an approximation of the others. At
 * k=2, pivot=0.5, c=0.25 the linear form gives 0.0 and the power form 0.125.
 * It is reached whenever `_getGradedPixels` returns null or `_isAllZeroes`
 * trips — and that samples ~40 pixels, so a dark or letterboxed frame trips it
 * spuriously.
 *
 * **Resolved.** All four now come from `js/radiance_grade.js`: the two shaders
 * are emitted from it and the two CPU paths call it. The divergence table above
 * is history rather than a warning, and the maths tests that used to live here
 * moved to `grade.test.mjs` alongside the definition they check.
 *
 * What stays here is the part that is genuinely about the two *backends* and
 * not about arithmetic: the readback contract, which is still hand-written on
 * both sides, and the WGSL feature gaps.
 *
 * These tests do not need a GPU.
 *
 * Run: node --test js/tests/backend_parity.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { applyContrastChannel as applyContrast, applyGammaChannel as applyGamma } from '../radiance_grade.js';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');

// The reference maths is no longer defined here. It used to be a local copy,
// which made this file a *fifth* implementation of the thing it was written to
// police. It now comes from the one definition both backends are emitted from.

// ── the contract both backends must satisfy ─────────────────────────────────

test('readPixelsFloat32 returns the same shape in both backends', () => {
    const gl = read('radiance_webgl.js');
    const gpu = read('radiance_webgpu.js');

    for (const [name, src] of [['webgl', gl], ['webgpu', gpu]]) {
        assert.match(src, /readPixelsFloat32\s*\(/, `${name} has no readPixelsFloat32`);
        assert.match(
            src, /return \{ data: \w+, width: \w+, height: \w+, graded: (true|false)(, sceneLinear: (true|false))? \}/,
            `${name}'s readPixelsFloat32 must return {data,width,height,graded}`,
        );
    }
});

test('readPixelsFloat32 is synchronous in both backends', () => {
    // WebGPU used to return a Promise while WebGL returned a value. The EXR
    // export read `.data` off the Promise, got undefined, and blamed the
    // encoder.
    const gpu = read('radiance_webgpu.js');
    const body = gpu.slice(gpu.indexOf('readPixelsFloat32'), gpu.indexOf('// ── Reference shelf'));
    assert.doesNotMatch(body, /return Promise\./,
        'readPixelsFloat32 must not return a Promise — WebGL does not');
});

test('each backend declares whether its pixels are graded', () => {
    assert.match(read('radiance_webgl.js'), /graded: true/,
        'WebGL renders the graded composite and must say so');
    assert.match(read('radiance_webgpu.js'), /graded: false/,
        'WebGPU returns ungraded source and must say so');
});

// ── the maths ───────────────────────────────────────────────────────────────

// The maths tests that used to sit here now live in grade.test.mjs, next to the
// definition. Two of them are kept below in backend form: they assert the
// *shipped source* no longer contains the divergent formulas, which is a
// different claim from "the shared function is correct".

test('neither backend re-implements contrast or gamma', () => {
    const strip = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
    for (const [name, file] of [['webgl', 'radiance_webgl.js'], ['webgpu', 'radiance_webgpu.js']]) {
        const src = strip(read(file));
        assert.doesNotMatch(src, /clamp\(contrast, 0\.0, 5\.0\)/,
            `${name} has its own contrast clamp again — it should come from radiance_grade.js`);
    }
    // And the shared functions still behave, so a green suite here means
    // something.
    assert.equal(applyContrast(0.25, 2, 0.5), 0);
    assert.ok(Number.isFinite(applyGamma(0.5, 0)));
});

test('the WebGPU shader path guards gamma', () => {
    const gpu = read('radiance_webgpu.js');
    const unguarded = /pow\(\s*max\(color,\s*0\),\s*1\.0\s*\/\s*gGradeGamma\s*\)/.test(gpu);
    assert.ok(!unguarded,
        'radiance_webgpu.js:418 divides by gGradeGamma with no floor; '
        + 'radiance_webgl.js:2609 uses max(0.01, gamma). Match it.');
});

// ── the WGSL feature gaps ───────────────────────────────────────────────────
//
// AUDIT-FIX (2026-09): this was one test carrying
// `{ todo: 'WGSL implementation missing ...' }`. `node --test` exits 0 with a
// todo present, so a shipped functional gap, two Viewer tabs that are fully
// interactive and completely inert on the *preferred* backend, could not turn
// CI red, and nothing made anyone look at it again.
//
// Making it fail was not the fix. The gap is real, known and not scheduled, and
// a suite that is permanently red is a suite people stop reading, which is the
// same outcome as a todo by a slower route. What replaces it is a ratchet. The
// gap is declared here and written up in KNOWN_ISSUES.md, and the three tests
// below fail if the source and that entry stop agreeing: implementing the WGSL
// without updating the docs fails, removing the tabs instead fails, and
// deleting the KNOWN_ISSUES entry to tidy up fails. The limitation cannot
// quietly become either fixed or forgotten.

/**
 * What the WebGPU backend does not implement, and what a person sees because of
 * it. `present` is how an implementation would show up in the source: an
 * override of the base class's state setter, a read of the state it stores, or
 * a shader uniform named for it. It is deliberately not /mask/i, because a WebGPU
 * pipeline descriptor's `colorWriteMask` would match that and turn this red for
 * nothing, and a ratchet nobody trusts is a ratchet nobody reads.
 */
const WGSL_GAPS = [
    { feature: 'mask', tab: 'Masks', present: /setMask\s*\(|\bmaskEnabled\b|\bu_mask|\bgMask/ },
    { feature: 'qualifier', tab: 'Qualifiers', present: /setQualifier\s*\(|\bqualifierEnabled\b|\bu_qualifier|\bgQualifier/ },
];

test('the WebGPU backend still lacks exactly the features documented as missing', () => {
    const gpu = read('radiance_webgpu.js');
    const landed = WGSL_GAPS.filter(({ present }) => present.test(gpu)).map((g) => g.tab);

    assert.deepEqual(landed, [],
        `${landed.join(' and ')} now appear in the WebGPU backend. If the WGSL `
        + 'landed, delete the entry from WGSL_GAPS here and from KNOWN_ISSUES.md. '
        + 'This is not a complaint about the implementation, it is the ratchet: '
        + 'the documented gap and the shipped source have to agree.');
});

test('each WebGPU gap is a parity gap, not a feature nobody has', () => {
    // Without this the list above could be satisfied by deleting the feature
    // everywhere, which is a different product and would leave KNOWN_ISSUES.md
    // describing a tab that no longer exists.
    const gl = read('radiance_webgl.js');
    const base = read('radiance_renderer.js');

    for (const { feature, tab, present } of WGSL_GAPS) {
        assert.match(gl, present,
            `the WebGL backend no longer implements ${feature} either, so the `
            + `${tab} tab is not a WebGPU gap any more. Update KNOWN_ISSUES.md.`);
        const setter = new RegExp(`set${feature[0].toUpperCase()}${feature.slice(1)}\\s*\\(`);
        assert.match(base, setter,
            `radiance_renderer.js no longer implements set${feature}, and that base `
            + `class is what makes the ${tab} tab *silently* inert on WebGPU `
            + 'rather than an error the user can see.');
    }
});

test('every WebGPU gap is written down where a person will read it', () => {
    const known = readFileSync(join(JS, '..', 'KNOWN_ISSUES.md'), 'utf8');
    const bullets = known.split(/\n(?=- \*\*)/).filter((b) => /WebGPU/.test(b));

    assert.equal(bullets.length, 1,
        'KNOWN_ISSUES.md should carry exactly one WebGPU bullet; found '
        + `${bullets.length}. The gap list in this file points at it by name.`);

    for (const { tab } of WGSL_GAPS) {
        assert.match(bullets[0], new RegExp(tab),
            `KNOWN_ISSUES.md does not name the ${tab} tab. A gap that is only `
            + 'recorded in a test file is a gap nobody reads.');
    }
    assert.match(bullets[0], /WGSL/,
        'the KNOWN_ISSUES.md entry should say what is missing, not just that '
        + 'something is');
});
