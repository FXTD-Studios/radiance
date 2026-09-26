/**
 * The grade maths.
 *
 * Radiance had four implementations of this — WebGL GLSL, WebGPU WGSL, the
 * WebGPU CPU readback, and the `.cube` export — and no two of them agreed. The
 * fix was not a test that reports the drift; it was collapsing them into one
 * definition that all four are emitted from. These tests hold that definition,
 * and the ones at the bottom hold the collapse itself.
 *
 * Run: node --test js/tests/grade.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
    applyLift, applyGain, applyOffset, applyGamma, applyGammaChannel,
    applyContrast, applyContrastChannel, applySaturation, gradePixel,
    luminance, GAMMA_FLOOR, CONTRAST_MIN, CONTRAST_MAX, GLSL, WGSL,
} from '../radiance_grade.js';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');
const close = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;
const closeAll = (a, b, eps = 1e-12) => a.every((v, i) => close(v, b[i], eps));

// ── contrast ────────────────────────────────────────────────────────────────

test('contrast is linear about the pivot, not a power curve', () => {
    // The exact case that separates the two shipped formulas. The WebGPU CPU
    // path used the power form, so a scope reading through it disagreed with
    // the picture on screen.
    assert.equal(applyContrastChannel(0.25, 2, 0.5), 0.0);
    assert.equal(0.5 * Math.pow(0.25 / 0.5, 2), 0.125);
});

test('contrast leaves the pivot fixed', () => {
    for (const k of [0, 0.5, 1, 2, 5]) {
        assert.ok(close(applyContrastChannel(0.5, k, 0.5), 0.5));
    }
});

test('contrast at 1.0 is the identity', () => {
    for (const c of [0, 0.18, 0.5, 1, 4]) assert.ok(close(applyContrastChannel(c, 1, 0.5), c));
});

test('contrast is clamped, and clamped everywhere', () => {
    // WebGL clamped to 0–5; WGSL and the .cube export did not. Beyond that the
    // separation is not a grade, and the three paths produced three pictures.
    assert.equal(applyContrastChannel(1, 99, 0.5), applyContrastChannel(1, CONTRAST_MAX, 0.5));
    assert.equal(applyContrastChannel(1, -99, 0.5), applyContrastChannel(1, CONTRAST_MIN, 0.5));
});

test('pivot 0 does not produce NaN', () => {
    // The Pivot slider's minimum is 0. The power form computes pow(c/0, k) —
    // Infinity ** k — then 0 * Infinity, and NaN goes through the whole buffer.
    for (const c of [0, 0.25, 1, 10, -0.5]) {
        assert.ok(Number.isFinite(applyContrastChannel(c, 2, 0)), `pivot=0 failed at c=${c}`);
    }
});

// ── gamma ───────────────────────────────────────────────────────────────────

test('gamma 0 does not divide by zero', () => {
    // Unguarded on two of the four old paths: 1/0 = Infinity turns the image
    // into a hard black-and-blown split.
    for (const c of [0.001, 0.5, 1, 2]) {
        assert.ok(Number.isFinite(applyGammaChannel(c, 0)), `gamma=0 failed at c=${c}`);
    }
    assert.equal(applyGammaChannel(0.5, 0), applyGammaChannel(0.5, GAMMA_FLOOR));
});

test('gamma leaves non-positive values alone rather than sending them to pow', () => {
    // pow(negative, fractional) is NaN, and scene-linear carries negatives after
    // a matrix conversion.
    assert.equal(applyGammaChannel(-0.2, 2.2), -0.2);
    assert.equal(applyGammaChannel(0, 2.2), 0);
});

test('gamma is monotonic and fixes 1', () => {
    for (const g of [0.5, 1, 2.2, 2.4]) {
        assert.ok(close(applyGammaChannel(1, g), 1));
        let prev = -Infinity;
        for (let c = 0; c <= 1.0001; c += 0.05) {
            const v = applyGammaChannel(c, g);
            assert.ok(v >= prev - 1e-12, `gamma ${g} not monotonic at ${c}`);
            prev = v;
        }
    }
});

// ── lift ────────────────────────────────────────────────────────────────────

test('lift is pivoted at white, not flat', () => {
    // The divergence the earlier audit missed. WebGL and the .cube export
    // pivoted; both WebGPU paths added flat, which visibly washes out anything
    // with a bright area.
    const dark = applyLift([0, 0, 0], [0.1, 0.1, 0.1]);
    const bright = applyLift([1, 1, 1], [0.1, 0.1, 0.1]);
    assert.ok(closeAll(dark, [0.1, 0.1, 0.1]), 'lift should act fully on black');
    assert.ok(closeAll(bright, [1, 1, 1]), 'lift should do nothing at white');
});

test('lift fades with luminance, not per channel', () => {
    // Per-channel fading would shift hue on a neutral lift.
    const [r, g, b] = applyLift([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]);
    assert.ok(close(r, g) && close(g, b), 'a neutral lift shifted hue');
});

test('lift does not amplify above white on an over-range pixel', () => {
    // luma > 1 would give a negative pivot and *subtract* without the clamp.
    const out = applyLift([4, 4, 4], [0.5, 0.5, 0.5]);
    assert.ok(closeAll(out, [4, 4, 4]), `over-range lift moved the pixel: ${out}`);
});

// ── the rest ────────────────────────────────────────────────────────────────

test('gain and offset are what they say', () => {
    assert.ok(closeAll(applyGain([0.5, 0.25, 0.1], [2, 2, 2]), [1, 0.5, 0.2]));
    assert.ok(closeAll(applyOffset([0.5, 0.25, 0.1], [0.1, 0, -0.1]), [0.6, 0.25, 0]));
});

test('saturation 0 collapses to luminance and 1 is the identity', () => {
    const px = [0.6, 0.3, 0.1];
    const y = luminance(...px);
    assert.ok(closeAll(applySaturation(px, 0), [y, y, y]));
    assert.ok(closeAll(applySaturation(px, 1), px));
});

test('saturation above 1 extrapolates rather than clamping', () => {
    // Clamping here would silently cap a legitimate creative move.
    const out = applySaturation([0.6, 0.3, 0.1], 2);
    assert.ok(out[0] > 0.6 && out[2] < 0.1);
});

// ── the whole grade ─────────────────────────────────────────────────────────

test('a neutral grade is the identity', () => {
    for (const px of [[0, 0, 0], [0.18, 0.18, 0.18], [1, 1, 1], [4, 2, 0.5]]) {
        assert.ok(closeAll(gradePixel(px, {}), px), `neutral grade moved ${px}`);
    }
});

test('the grade order is offset, lift, gain, gamma, contrast, saturation', () => {
    // The order is part of the definition: lift before gain means the lift is
    // scaled by the gain, which is what a Resolve-style wheel set does.
    // Swapping any two gives a different picture from identical sliders.
    const opts = { offset: [0.05, 0, 0], lift: [0.1, 0, 0], gain: [2, 1, 1] };
    const expected = applyGain(applyLift(applyOffset([0.2, 0.2, 0.2], opts.offset), opts.lift), opts.gain);
    assert.ok(closeAll(gradePixel([0.2, 0.2, 0.2], opts), expected));
    // Gain-then-lift would land somewhere else.
    const wrong = applyLift(applyGain(applyOffset([0.2, 0.2, 0.2], opts.offset), opts.gain), opts.lift);
    assert.ok(!closeAll(gradePixel([0.2, 0.2, 0.2], opts), wrong, 1e-6),
        'the test cannot distinguish the two orders — it is not testing anything');
});

test('no grade setting produces a non-finite pixel', () => {
    // Every combination the UI can reach. A single NaN here becomes a NaN
    // frame, and the sliders that reach it are the extremes users actually
    // drag to.
    for (const gamma of [0, 0.01, 1, 4]) {
        for (const contrast of [0, 1, 5, 99]) {
            for (const pivot of [0, 0.18, 1]) {
                for (const px of [[0, 0, 0], [-0.3, 0.2, 4], [1, 1, 1]]) {
                    const out = gradePixel(px, {
                        gamma: [gamma, gamma, gamma], contrast, pivot,
                        lift: [0.2, -0.1, 0], gain: [2, 0.5, 1], saturation: 2,
                    });
                    assert.ok(out.every(Number.isFinite),
                        `γ=${gamma} k=${contrast} pivot=${pivot} px=${px} → ${out}`);
                }
            }
        }
    }
});

// ── the collapse ────────────────────────────────────────────────────────────
//
// These are what stop the four implementations coming back. A test that merely
// compared them would keep passing while someone quietly re-inlined one.

test('both shaders are emitted from this module', () => {
    assert.match(read('radiance_webgl.js'), /import \{ GLSL as GRADE_GLSL \} from "\.\/radiance_grade\.js"/,
        'the WebGL backend must take its grade maths from the module');
    assert.match(read('radiance_webgpu.js'), /WGSL as GRADE_WGSL/,
        'the WebGPU backend must take its grade maths from the module');
    assert.match(read('radiance_webgl.js'), /\$\{GRADE_GLSL\}/, 'GRADE_GLSL is imported but never spliced');
    assert.match(read('radiance_webgpu.js'), /\$\{GRADE_WGSL\}/, 'GRADE_WGSL is imported but never spliced');
});

test('both CPU paths call this module rather than repeating it', () => {
    assert.match(read('radiance_webgpu.js'), /px = gradeLift\(/,
        'the WebGPU CPU readback must use the shared functions');
    assert.match(read('radiance_viewer.js'), /_gradePixel\(\[r, g, b\]/,
        'the .cube export must use the shared grade');
});

test('the old divergent formulas are gone from every backend', () => {
    const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
    const gpu = strip(read('radiance_webgpu.js'));
    // The power-curve contrast, in either the WGSL or the JS spelling.
    assert.doesNotMatch(gpu, /piv \* Math\.pow\(/, 'the power-curve contrast is back in the CPU path');
    assert.doesNotMatch(gpu, /pow\(max\(color, vec3f\(0\.0\)\), 1\.0 \/ gGradeGamma\)/,
        'the unguarded WGSL gamma is back');
    assert.doesNotMatch(gpu, /color \* gGain \+ gLift/, 'the flat additive lift is back in WGSL');
    assert.doesNotMatch(strip(read('radiance_viewer.js')), /lift\[0\] \* lumaPivot/,
        'the .cube export has its own lift again');
});

test('the two shader dialects say the same thing', () => {
    // They cannot be diffed as text — one is GLSL and one is WGSL — but they
    // are emitted side by side from this file, so every function in one must
    // exist in the other. A helper added to only one dialect is exactly how
    // these drifted the first time.
    const names = (src, re) => new Set([...src.matchAll(re)].map((m) => m[1]));
    const glsl = names(GLSL, /^(?:vec3|float)\s+(rad\w+)\s*\(/gm);
    const wgsl = names(WGSL, /^fn\s+(rad\w+)\s*\(/gm);
    assert.ok(glsl.size >= 7, `only ${glsl.size} functions in the GLSL block`);
    assert.deepEqual([...glsl].sort(), [...wgsl].sort(),
        'the two dialects do not define the same set of functions');
});

test('the shader constants come from the module, not from literals', () => {
    // If someone edits 0.01 in the GLSL string without touching GAMMA_FLOOR,
    // the backends part company again.
    for (const src of [GLSL, WGSL]) {
        assert.ok(src.includes(String(GAMMA_FLOOR)), 'the gamma floor is not in the emitted shader');
        assert.ok(src.includes(String(CONTRAST_MAX)), 'the contrast clamp is not in the emitted shader');
        assert.match(src, /0\.2126/, 'the luma coefficients are not in the emitted shader');
    }
});

test('the emitted shaders are marked as generated', () => {
    for (const src of [GLSL, WGSL]) {
        assert.match(src, /generated from js\/radiance_grade\.js/,
            'someone will edit the shader in place unless it says not to');
    }
});
