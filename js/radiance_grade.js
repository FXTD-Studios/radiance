/**
 * The grade maths. One definition, four consumers.
 *
 * Radiance renders through two backends and bakes the same grade into two more
 * places, and until this file existed all four implemented the maths
 * separately. They did not agree:
 *
 * | | lift | gamma | contrast |
 * | :-- | :-- | :-- | :-- |
 * | WebGL GLSL | luma-pivoted | guarded, `max(0.01, γ)` | clamped linear |
 * | WebGPU WGSL | flat additive | **unguarded** | unclamped linear |
 * | WebGPU CPU readback | flat additive | **unguarded** | **power curve** |
 * | Viewer `.cube` export | luma-pivoted | guarded | unclamped linear |
 *
 * Every row differs from the one above it. In practice that meant the picture
 * changed when a user's browser happened to support WebGPU — `_tryWebGPUUpgrade()`
 * runs whenever `navigator.gpu` exists, so nobody chose it — and a `.cube`
 * exported for Resolve did not match either.
 *
 * The contrast row is the worst of them, because the power form is a different
 * curve rather than an approximation of the others: at contrast 2, pivot 0.5,
 * input 0.25, the linear form gives 0.0 and the power form gives 0.125. And at
 * pivot 0 — a legal slider position — the power form computes `0 · ∞` and puts
 * NaN through the whole buffer.
 *
 * A test can only report that these have drifted. Emitting all four from here
 * means they cannot: the two shaders are generated from `GLSL` and `WGSL`
 * below, and the two CPU paths call the functions directly.
 *
 * ## Which behaviour won
 *
 * WebGL's, everywhere it was a real choice — it is what users have been looking
 * at and grading against, so changing it would silently invalidate saved work.
 * The guards are additive: they only change values that were previously
 * undefined, infinite or NaN.
 */

import { LUMA_R, LUMA_G, LUMA_B, luminance } from './radiance_probe.js';

/** Gamma floor. 1/0 is Infinity, which splits an image into hard black and blown. */
export const GAMMA_FLOOR = 0.01;

/** Contrast is clamped before use. Beyond ~5 the separation is not a grade. */
export const CONTRAST_MIN = 0.0;
export const CONTRAST_MAX = 5.0;

export { LUMA_R, LUMA_G, LUMA_B, luminance };

// ── the maths ───────────────────────────────────────────────────────────────

/**
 * Lift — additive shadow shift, pivoted at white.
 *
 * Pivoted, not flat: `lift · (1 − luma)` fades to nothing at white, so lifting
 * the blacks does not also wash out the highlights. Flat addition is what the
 * WebGPU paths did, and it is visibly a different grade in anything with a
 * bright area.
 *
 * Luminance-driven rather than per-channel so a neutral lift does not shift hue.
 */
export function applyLift(rgb, lift) {
    const p = Math.min(Math.max(1 - luminance(rgb[0], rgb[1], rgb[2]), 0), 1);
    return [rgb[0] + lift[0] * p, rgb[1] + lift[1] * p, rgb[2] + lift[2] * p];
}

/** Gain — multiplicative slope, pivoted at black. */
export function applyGain(rgb, gain) {
    return [rgb[0] * gain[0], rgb[1] * gain[1], rgb[2] * gain[2]];
}

/** Offset — global additive. */
export function applyOffset(rgb, offset) {
    return [rgb[0] + offset[0], rgb[1] + offset[1], rgb[2] + offset[2]];
}

/**
 * Gamma — power curve on positives.
 *
 * Two guards, both load-bearing. The floor stops `1/0` becoming Infinity. And
 * negatives pass through untouched rather than into `pow`, which is NaN for a
 * fractional exponent — scene-linear legitimately carries negatives after a
 * matrix conversion.
 */
export function applyGammaChannel(c, gamma) {
    if (!(c > 0)) return c;
    return Math.pow(c, 1 / Math.max(gamma, GAMMA_FLOOR));
}

export function applyGamma(rgb, gamma) {
    return [
        applyGammaChannel(rgb[0], gamma[0]),
        applyGammaChannel(rgb[1], gamma[1]),
        applyGammaChannel(rgb[2], gamma[2]),
    ];
}

/**
 * Contrast — linear interpolation about the pivot.
 *
 * Linear, not `pivot · (c/pivot)^k`. The two are different curves, the linear
 * one is what three of the four shipped paths already used and what the viewer
 * has always displayed, and it is finite at pivot 0 where the power form is
 * NaN.
 */
export function applyContrastChannel(c, contrast, pivot) {
    const k = Math.min(Math.max(contrast, CONTRAST_MIN), CONTRAST_MAX);
    return (c - pivot) * k + pivot;
}

export function applyContrast(rgb, contrast, pivot) {
    return [
        applyContrastChannel(rgb[0], contrast, pivot),
        applyContrastChannel(rgb[1], contrast, pivot),
        applyContrastChannel(rgb[2], contrast, pivot),
    ];
}

/** Saturation — interpolation toward luminance. Above 1 extrapolates. */
export function applySaturation(rgb, sat) {
    const y = luminance(rgb[0], rgb[1], rgb[2]);
    return [y + (rgb[0] - y) * sat, y + (rgb[1] - y) * sat, y + (rgb[2] - y) * sat];
}

/**
 * The whole primary grade, in the order the shaders apply it.
 *
 * Offset → Lift → Gain → Gamma → Contrast → Saturation. The order is part of
 * the definition: lift before gain means the lift is scaled by the gain, which
 * is what a colourist expects from a Resolve-style wheel set, and swapping any
 * two of these produces a different picture from identical slider values.
 */
export function gradePixel(rgb, {
    offset = [0, 0, 0], lift = [0, 0, 0], gain = [1, 1, 1],
    gamma = [1, 1, 1], contrast = 1, pivot = 0.18, saturation = 1,
} = {}) {
    let c = applyOffset(rgb, offset);
    c = applyLift(c, lift);
    c = applyGain(c, gain);
    c = applyGamma(c, gamma);
    if (contrast !== 1) c = applyContrast(c, contrast, pivot);
    if (saturation !== 1) c = applySaturation(c, saturation);
    return c;
}

// ── the shaders ─────────────────────────────────────────────────────────────
//
// Emitted from the constants above so a change here reaches every backend. The
// bodies are transliterations of the functions above and are covered by tests
// that compare them line for line against this file rather than by eye.

/**
 * A JS number as a shader float literal.
 *
 * `String(5.0)` is `"5"`, and GLSL has no implicit int-to-float conversion —
 * `clamp(contrast, 0, 5)` is "no matching overloaded function found", which is a
 * link-time failure of the whole composite shader from a constant that looks
 * perfectly fine in JS. Every interpolated number goes through here.
 */
const f = (v) => (Number.isInteger(v) ? `${v}.0` : String(v));

const LUMA_VEC_GLSL = `vec3(${f(LUMA_R)}, ${f(LUMA_G)}, ${f(LUMA_B)})`;
const LUMA_VEC_WGSL = `vec3f(${f(LUMA_R)}, ${f(LUMA_G)}, ${f(LUMA_B)})`;

/** GLSL (WebGL2, `#version 300 es`). */
export const GLSL = `
// ── Radiance grade maths — generated from js/radiance_grade.js ──────────────
// Do not edit here. Both backends and both CPU paths are emitted from that one
// file precisely so they cannot drift apart again.

float radLuma(vec3 c) { return dot(c, ${LUMA_VEC_GLSL}); }

vec3 radLift(vec3 color, vec3 lift) {
    // Pivoted at white: fades to nothing as luma rises, so lifting blacks does
    // not wash out highlights.
    float p = clamp(1.0 - radLuma(color), 0.0, 1.0);
    return color + lift * p;
}

vec3 radGain(vec3 color, vec3 gain) { return color * gain; }
vec3 radOffset(vec3 color, vec3 offset) { return color + offset; }

vec3 radGamma(vec3 color, vec3 gamma) {
    // max(gamma, ${f(GAMMA_FLOOR)}) stops 1/0 = Infinity. Negatives skip pow(),
    // which is NaN for a fractional exponent.
    vec3 g = max(gamma, vec3(${f(GAMMA_FLOOR)}));
    vec3 lifted = pow(max(color, vec3(0.0)), vec3(1.0) / g);
    return mix(color, lifted, vec3(greaterThan(color, vec3(0.0))));
}

vec3 radContrast(vec3 color, float contrast, float pivot) {
    // Linear about the pivot, not pivot * pow(c/pivot, k). Different curve, and
    // finite at pivot 0 where the power form is NaN.
    float k = clamp(contrast, ${f(CONTRAST_MIN)}, ${f(CONTRAST_MAX)});
    return (color - pivot) * k + pivot;
}

vec3 radSaturation(vec3 color, float sat) {
    float y = radLuma(color);
    return vec3(y) + (color - vec3(y)) * sat;
}

vec3 radGradeOrder(vec3 color, vec3 offset, vec3 lift, vec3 gain, vec3 gamma) {
    // Offset, Lift, Gain, Gamma -- the order is part of the definition. Lift
    // before gain means the lift is scaled by the gain, which is what a
    // Resolve-style wheel set does; swapping any two gives a different picture
    // from identical slider values.
    return radGamma(radGain(radLift(radOffset(color, offset), lift), gain), gamma);
}
// ── end generated ───────────────────────────────────────────────────────────
`;

/** WGSL (WebGPU). */
export const WGSL = `
// ── Radiance grade maths — generated from js/radiance_grade.js ──────────────
// Do not edit here. See the note in the GLSL block; this is the same maths.

fn radLuma(c: vec3f) -> f32 { return dot(c, ${LUMA_VEC_WGSL}); }

fn radLift(color: vec3f, lift: vec3f) -> vec3f {
    let p = clamp(1.0 - radLuma(color), 0.0, 1.0);
    return color + lift * p;
}

fn radGain(color: vec3f, gain: vec3f) -> vec3f { return color * gain; }
fn radOffset(color: vec3f, offset: vec3f) -> vec3f { return color + offset; }

fn radGamma(color: vec3f, gamma: vec3f) -> vec3f {
    let g = max(gamma, vec3f(${f(GAMMA_FLOOR)}));
    let lifted = pow(max(color, vec3f(0.0)), vec3f(1.0) / g);
    return select(color, lifted, color > vec3f(0.0));
}

fn radContrast(color: vec3f, contrast: f32, pivot: f32) -> vec3f {
    let k = clamp(contrast, ${f(CONTRAST_MIN)}, ${f(CONTRAST_MAX)});
    return (color - pivot) * k + pivot;
}

fn radSaturation(color: vec3f, sat: f32) -> vec3f {
    let y = radLuma(color);
    return vec3f(y) + (color - vec3f(y)) * sat;
}

fn radGradeOrder(color: vec3f, offset: vec3f, lift: vec3f, gain: vec3f, gamma: vec3f) -> vec3f {
    return radGamma(radGain(radLift(radOffset(color, offset), lift), gain), gamma);
}
// ── end generated ───────────────────────────────────────────────────────────
`;
