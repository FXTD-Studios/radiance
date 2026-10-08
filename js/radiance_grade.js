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

/**
 * The contrast pivot: 18% grey in linear light, the same default the node-side
 * grade uses. The viewer pivoted on 0.5, which in linear light is a stop above
 * mid grey, so contrast 1.2 darkened grey by 0.63 stop.
 */
export const PIVOT_DEFAULT = 0.18;

/**
 * Pivot floor. The power curve divides by the pivot, and the Pivot slider
 * reaches 0. A thousandth is ten stops under mid grey, far below anything a
 * pivot is placed on.
 */
export const PIVOT_FLOOR = 0.001;

/**
 * Contrast output ceiling. A power of a power (gamma 0.01, then contrast 5)
 * leaves even float64 behind, and an Infinity turns into NaN at the next
 * luminance sum. 1e30 is far above any scene value and inside float32.
 */
export const CONTRAST_CEILING = 1e30;

/**
 * White balance scale: one unit of Temperature is one stop of red against
 * blue, and one unit of Tint is one stop of green against magenta.
 */
export const WB_TEMPERATURE_STOPS = 1.0;
export const WB_TINT_STOPS = 1.0;

/** Exposure range, in stops. The shader clamps to it and the slider covers it. */
export const EXPOSURE_MIN = -12.0;
export const EXPOSURE_MAX = 12.0;

/** ACEScct (S-2016-001) constants. */
export const ACESCCT_A = 10.5402377416545;
export const ACESCCT_B = 0.0729055341958355;
export const ACESCCT_LIN_BREAK = 0.0078125;
export const ACESCCT_CCT_BREAK = 0.155251141552511;

/**
 * Source gamut to ACEScg (AP1), Bradford D65 to ACES white, row major, indexed
 * by the renderer's u_sourceGamut: 0 linear Rec.709 / sRGB, 1 ACEScg,
 * 2 ACES2065-1, 3 linear Rec.2020, 4 linear P3-D65. Derived from the published
 * primaries and white points; each entry agrees with OpenColorIO's ACES studio
 * config to better than 1e-7. The ACEScct grade mode used tone-mapper fit
 * matrices here, which are not sRGB to AP1 and are not inverses of each other,
 * so the picture changed with every control at neutral.
 */
export const TO_AP1 = [
    [[0.6130974024011878, 0.3395231461841061, 0.04737945141470665],
     [0.07019372246958168, 0.9163538790573436, 0.01345239847307412],
     [0.02061559288222693, 0.1095697729381354, 0.8698146341796377]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1.451439316145666, -0.2365107468937401, -0.2149285692519255],
     [-0.0765537733960206, 1.176229699833573, -0.09967592643755213],
     [0.008316148425697719, -0.006032449791021028, 0.9977163013653231]],
    [[0.9748949779244189, 0.01959910863700534, 0.005505913438576188],
     [0.002179562797703918, 0.9955354688932204, 0.002284968309075179],
     [0.004797239683772727, 0.02453201663458945, 0.9706707436816377]],
    [[0.735797914028892, 0.2121664852931746, 0.05203560067793385],
     [0.04717988497673042, 0.9380457009217151, 0.01477441410155395],
     [0.003563664639098911, 0.04114188562513292, 0.955294449735768]],
];

/** ACEScg (AP1) back to each source gamut: the exact inverses of TO_AP1. */
export const FROM_AP1 = [
    [[1.705050992657983, -0.6217921206570056, -0.08325887200097853],
     [-0.1302564175070434, 1.140804736575402, -0.01054831906835806],
     [-0.02400335680461803, -0.1289689760649706, 1.152972332869588]],
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0.6954522413574517, 0.1406786964702941, 0.1638690621722542],
     [0.04479456337203774, 0.859671118456422, 0.0955343181715404],
     [-0.005525882558113543, 0.004025210305978663, 1.001500672252135]],
    [[1.02582474766601, -0.02005319083821517, -0.005771556827795564],
     [-0.002234369519975978, 1.00458650188848, -0.002352132368503649],
     [-0.005013351468089286, -0.02529007181078517, 1.030303423278875]],
    [[1.379214128253342, -0.3088641446737119, -0.07034998357963095],
     [-0.06933485838138222, 1.08229674600235, -0.01296188762096683],
     [-0.002159009513570322, -0.04545932483731564, 1.047618334350886]],
];

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
 * Contrast — a power curve about the pivot, `pivot · (c/pivot)^k`.
 *
 * In linear light this is a straight line of slope k through the pivot on a
 * log axis, so every stop above and below the pivot is spread by the same
 * factor and mid grey stays put. The linear form `(c − pivot)·k + pivot` it
 * replaces is a different curve: at contrast 2 it sent grey and black negative.
 *
 * Values at or below zero pass through unchanged (the power of a negative is
 * NaN, and the curve meets zero there anyway). The pivot is floored so pivot 0
 * stays finite, and k = 1 returns the input exactly.
 */
export function applyContrastChannel(c, contrast, pivot = PIVOT_DEFAULT) {
    const k = Math.min(Math.max(contrast, CONTRAST_MIN), CONTRAST_MAX);
    if (k === 1 || !(c > 0)) return c;
    const p = Math.max(pivot, PIVOT_FLOOR);
    return Math.min(p * Math.pow(c / p, k), CONTRAST_CEILING);
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

// ── white balance ───────────────────────────────────────────────────────────

/**
 * White balance as per-channel gains: Temperature moves red against blue,
 * Tint moves green against both, and the three are scaled together so a
 * neutral keeps its BT.709 luminance.
 *
 * Gains, not an added offset. Adding up to ±2 to the channels tinted black
 * (Temperature 0.1 turned black dark red) and, through the clamp that came with
 * it, zeroed every negative scene value. A gain leaves black black, scales a
 * colour cast the same at every level, and needs no clamp.
 */
export function whiteBalanceGains(temperature, tint) {
    if (!temperature && !tint) return [1, 1, 1];
    const t = (temperature || 0) * WB_TEMPERATURE_STOPS;
    const n = (tint || 0) * WB_TINT_STOPS;
    const r = Math.pow(2, t / 2), g = Math.pow(2, -n), b = Math.pow(2, -t / 2);
    const y = luminance(r, g, b);
    return [r / y, g / y, b / y];
}

export function applyWhiteBalance(rgb, temperature, tint) {
    if (!temperature && !tint) return rgb;
    const w = whiteBalanceGains(temperature, tint);
    return [rgb[0] * w[0], rgb[1] * w[1], rgb[2] * w[2]];
}

/**
 * The Temperature and Tint that make `rgb` neutral: the eyedropper's solve.
 * Exact, because the gains above span every ratio of three positive channels,
 * and independent of level, so a cast picked at a tenth of the brightness
 * gives the same answer. Null when a channel is not positive.
 */
export function whiteBalanceForNeutral(rgb) {
    const [r, g, b] = rgb;
    if (!(r > 0 && g > 0 && b > 0)) return null;
    return {
        temperature: Math.log2(b / r) / WB_TEMPERATURE_STOPS,
        tint: Math.log2(g / Math.sqrt(r * b)) / WB_TINT_STOPS,
    };
}

// ── luma mix ────────────────────────────────────────────────────────────────

/**
 * Luma Mix: at 0 the pixel keeps the luminance `refLuma` it had after the
 * primaries, so later saturation, hue and secondary moves cannot change its
 * brightness; at 1 it is untouched. The reference is taken after exposure,
 * gain, contrast and curves, not from the ungraded source, which made those
 * controls stop changing brightness at 0.
 */
export function applyLumaMix(rgb, refLuma, mix) {
    if (mix >= 1) return rgb;
    const m = Math.min(Math.max(mix, 0), 1);
    const y = luminance(rgb[0], rgb[1], rgb[2]);
    const held = y > 1e-4
        ? rgb.map((v) => v * (refLuma / y))
        : rgb.map((v) => v + (refLuma - y));
    return held.map((v, i) => v + (rgb[i] - v) * m);
}

// ── ACEScct ─────────────────────────────────────────────────────────────────

export function linToACEScct(x) {
    return x <= ACESCCT_LIN_BREAK ? ACESCCT_A * x + ACESCCT_B : (Math.log2(x) + 9.72) / 17.52;
}

export function acescctToLin(y) {
    return y > ACESCCT_CCT_BREAK ? Math.pow(2, y * 17.52 - 9.72) : (y - ACESCCT_B) / ACESCCT_A;
}

const mat3 = (m, v) => [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
];

/** Source gamut (u_sourceGamut index) to ACEScg. ACEScg itself is untouched. */
export function toAP1(rgb, gamut = 0) {
    return gamut === 1 ? rgb : mat3(TO_AP1[gamut] || TO_AP1[0], rgb);
}

export function fromAP1(rgb, gamut = 0) {
    return gamut === 1 ? rgb : mat3(FROM_AP1[gamut] || FROM_AP1[0], rgb);
}

// ── the rest of the shader's grade, on the CPU ──────────────────────────────
//
// These mirror the GLSL in radiance_webgl.js op for op, so that an export can
// reproduce the screen. They are checked against the real shader by
// grade_render.test.mjs, which renders a frame through the viewer and compares.

const smoothstep = (e0, e1, x) => {
    const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
    return t * t * (3 - 2 * t);
};

export function applyLogWheels(rgb, shadow, midtone, highlight) {
    const z = (v) => !v || (v[0] === 0 && v[1] === 0 && v[2] === 0);
    if (z(shadow) && z(midtone) && z(highlight)) return rgb;
    const s3 = shadow || [0, 0, 0], m3 = midtone || [0, 0, 0], h3 = highlight || [0, 0, 0];
    const y = luminance(rgb[0], rgb[1], rgb[2]);
    const sw = 1 - smoothstep(0, 0.45, y);
    const hw = smoothstep(0.55, 1, y);
    const mw = 1 - sw - hw;
    return rgb.map((v, i) => v * (1 + s3[i] * sw + m3[i] * mw + h3[i] * hw));
}

export function applyPrinterLights(rgb, r, g, b) {
    if (!r && !g && !b) return rgb;
    return [rgb[0] * Math.pow(2, (r || 0) / 50), rgb[1] * Math.pow(2, (g || 0) / 50), rgb[2] * Math.pow(2, (b || 0) / 50)];
}

export function applyShadowsHighlights(rgb, shadows, highlights) {
    if (!shadows && !highlights) return rgb;
    const y = luminance(rgb[0], rgb[1], rgb[2]);
    const sw = Math.pow(1 - smoothstep(0, 0.5, y), 2);
    const hw = Math.pow(smoothstep(0.5, 1, y), 2);
    const k = (1 + (shadows || 0) * sw * 0.5) * (1 + (highlights || 0) * hw * 0.5);
    return rgb.map((v) => v * k);
}

export function applyColorBoost(rgb, boost) {
    if (!boost) return rgb;
    const y = luminance(rgb[0], rgb[1], rgb[2]);
    const sat = Math.max(...rgb) - Math.min(...rgb);
    const f = 1 + (1 - sat * 0.8) * boost;
    return rgb.map((v) => Math.max(y + (v - y) * f, 0));
}

function rgb2hsv([r, g, b]) {
    // The shader's branchless form, so hue at the seams lands where it does there.
    const K = [0, -1 / 3, 2 / 3, -1];
    const st1 = g >= b ? 1 : 0;   // step(c.b, c.g)
    const p = [b + (g - b) * st1, g + (b - g) * st1, K[3] + (K[0] - K[3]) * st1, K[2] + (K[1] - K[2]) * st1];
    const st2 = r >= p[0] ? 1 : 0;   // step(p.x, c.r)
    const q = [p[0] + (r - p[0]) * st2, p[1] + (p[1] - p[1]) * st2, p[3] + (p[2] - p[3]) * st2, r + (p[0] - r) * st2];
    const d = q[0] - Math.min(q[3], q[1]);
    const e = 1e-10;
    return [Math.abs(q[2] + (q[3] - q[1]) / (6 * d + e)), d / (q[0] + e), q[0]];
}

function hsv2rgb([h, s, v]) {
    const fract = (x) => x - Math.floor(x);
    return [1, 2 / 3, 1 / 3].map((k) => {
        const p = Math.abs(fract(h + k) * 6 - 3);
        return v * (1 + (Math.min(Math.max(p - 1, 0), 1) - 1) * s);
    });
}

export function applyHueShift(rgb, degrees) {
    if (!degrees) return rgb;
    const hsv = rgb2hsv(rgb);
    hsv[0] += degrees / 360;
    if (hsv[0] > 1) hsv[0] -= 1;
    if (hsv[0] < 0) hsv[0] += 1;
    return hsv2rgb(hsv);
}

/**
 * One channel of a 256-entry curve table, read the way the GPU reads it:
 * linear filtering between texel centres, so input c lands on entry c·255.
 * The shader sampled at c itself, half a texel off at both ends.
 */
export function sampleCurveTable(table, c, channel = 0, stride = 4) {
    const f = Math.min(Math.max(c, 0), 1) * 255;
    const k = Math.min(Math.floor(f), 254);
    const t = f - k;
    return table[k * stride + channel] * (1 - t) + table[(k + 1) * stride + channel] * t;
}

/**
 * The custom curves, with the shader's extensions past the table: above 1.0 a
 * value is scaled by the curve's output at 1.0, below 0 it is offset by the
 * curve's output at 0.
 */
export function applyCurves(rgb, table, mix = 1) {
    if (!table || !(mix > 0)) return rgb;
    // Below 0, a value keeps its distance under the curve's black output.
    const curved = rgb.map((v, i) => (v >= 1
        ? v * Math.max(table[255 * 4 + i], 0)
        : v < 0 ? v + table[i] : sampleCurveTable(table, v, i)));
    return rgb.map((v, i) => v + (curved[i] - v) * mix);
}

/** Hue vs Hue / Sat / Luma, from the secondary 256-entry table. */
export function applySecondaryCurves(rgb, table, mix = 1, isLinear = true) {
    if (!table || !(mix > 0)) return rgb;
    const hsv = rgb2hsv(rgb);
    const look = [0, 1, 2].map((ch) => sampleCurveTable(table, hsv[0], ch));
    hsv[0] = (hsv[0] + (look[0] - 0.5)) - Math.floor(hsv[0] + (look[0] - 0.5));
    hsv[1] = Math.min(Math.max(hsv[1] * look[1] * 2, 0), 1);
    hsv[2] = Math.min(Math.max(hsv[2] + (look[2] - 0.5) * 2 * 0.5, 0), isLinear ? 65504 : 1);
    const curved = hsv2rgb(hsv);
    return rgb.map((v, i) => v + (curved[i] - v) * mix);
}

/** True when a 256-entry curve table is the identity. */
export function isIdentityCurveTable(table) {
    if (!table) return true;
    for (let i = 0; i < 256; i++) {
        for (let ch = 0; ch < 3; ch++) if (Math.abs(table[i * 4 + ch] - i / 255) > 1e-6) return false;
    }
    return true;
}

/** True when a 256-entry hue-curve table leaves every hue alone. */
export function isIdentitySecondaryTable(table) {
    if (!table) return true;
    for (let i = 0; i < 256; i++) {
        for (let ch = 0; ch < 3; ch++) if (Math.abs(table[i * 4 + ch] - 0.5) > 1e-3) return false;
    }
    return true;
}

/**
 * The whole grade the viewer shows, in the shader's order, on one pixel of
 * working scene-linear (after the input transform, before the view).
 *
 *   exposure → white balance → offset/lift/gain/gamma (in ACEScct when the
 *   ACEScct grade space is on) → contrast → log wheels → printer lights →
 *   shadows/highlights → colour boost → curves → hue curves → saturation →
 *   hue shift → luma mix
 *
 * Mid Detail is spatial and is not part of a per-pixel grade. `state` uses the
 * viewer's field names (the same object undo and the workflow save).
 */
export function gradePixelFull(rgb, state = {}, { gamut = 0, isLinear = true } = {}) {
    const s = state;
    const exposure = Math.min(Math.max(s.exposure ?? 0, EXPOSURE_MIN), EXPOSURE_MAX);
    const k = Math.pow(2, exposure);
    let c = [rgb[0] * k, rgb[1] * k, rgb[2] * k];
    c = applyWhiteBalance(c, s.temperature ?? 0, s.tint ?? 0);

    const order = (x) => applyGamma(applyGain(applyLift(applyOffset(x,
        s.offset ?? [0, 0, 0]), s.lift ?? [0, 0, 0]), s.gain ?? [1, 1, 1]), s.gamma ?? [1, 1, 1]);
    if ((s.colorScience ?? 0) === 1) {
        let cct = toAP1(c, gamut).map(linToACEScct);
        cct = order(cct);
        c = fromAP1(cct.map(acescctToLin), gamut);
    } else {
        c = order(c);
    }

    if ((s.contrast ?? 1) !== 1) c = applyContrast(c, s.contrast ?? 1, s.pivot ?? PIVOT_DEFAULT);
    c = applyLogWheels(c, s.logShadow, s.logMidtone, s.logHighlight);
    c = applyPrinterLights(c, s.printerR, s.printerG, s.printerB);
    c = applyShadowsHighlights(c, s.shadows ?? 0, s.highlights ?? 0);
    c = applyColorBoost(c, s.colorBoost ?? 0);
    c = applyCurves(c, s.curveTable, s.curveMix ?? 1);
    const refLuma = luminance(c[0], c[1], c[2]);
    c = applySecondaryCurves(c, s.secondaryCurveTable, s.secondaryCurveMix ?? 0, isLinear);
    if ((s.saturation ?? 1) !== 1) c = applySaturation(c, s.saturation);
    c = applyHueShift(c, s.hueShift ?? 0);
    c = applyLumaMix(c, refLuma, s.lumaMix ?? 1);
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

// Row-major JS matrices as shader constructors, which take columns.
const cols = (m) => [0, 1, 2].flatMap((j) => [m[0][j], m[1][j], m[2][j]]).map(f).join(', ');
const glslMat = (m) => `mat3(${cols(m)})`;
const wgslMat = (m) => `mat3x3f(${cols(m)})`;
const gamutBranches = (mats, mat, ret, brace) => [2, 3, 4].map((g) => (brace
    ? `    if (gamut == ${g}) { return ${mat(mats[g])} * c; }`
    : `    if (gamut == ${g}) return ${mat(mats[g])} * c;`)).join('\n')
    + `\n    ${ret} ${mat(mats[0])} * c;`;

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
    // pivot * (c / pivot)^k on positives, the identity at k = 1, negatives
    // untouched. pow() only ever sees a positive base (pow(0, 0) is undefined)
    // and the result is capped, so neither half of the sum below can carry an
    // Infinity or a NaN into the other.
    float k = clamp(contrast, ${f(CONTRAST_MIN)}, ${f(CONTRAST_MAX)});
    if (k == 1.0) return color;
    float p = max(pivot, ${f(PIVOT_FLOOR)});
    vec3 curved = min(p * pow(max(color, vec3(1e-30)) / p, vec3(k)), vec3(${CONTRAST_CEILING}));
    return min(color, vec3(0.0)) + curved * vec3(greaterThan(color, vec3(0.0)));
}

vec3 radWhiteBalanceGains(float temperature, float tint) {
    // Red against blue, green against both, scaled so a neutral keeps its
    // luminance.
    float t = temperature * ${f(WB_TEMPERATURE_STOPS)};
    float n = tint * ${f(WB_TINT_STOPS)};
    vec3 g = exp2(vec3(0.5 * t, -n, -0.5 * t));
    return g / radLuma(g);
}

vec3 radWhiteBalance(vec3 color, float temperature, float tint) {
    if (temperature == 0.0 && tint == 0.0) return color;
    return color * radWhiteBalanceGains(temperature, tint);
}

vec3 radLumaMix(vec3 color, float refLuma, float m) {
    // refLuma is the luminance after the primaries; 0 keeps it, 1 is a no-op.
    if (m >= 1.0) return color;
    float y = radLuma(color);
    vec3 held = y > 1e-4 ? color * (refLuma / y) : color + vec3(refLuma - y);
    return mix(held, color, clamp(m, 0.0, 1.0));
}

vec3 radLinToACEScct(vec3 c) {
    vec3 lo = ${f(ACESCCT_A)} * c + ${f(ACESCCT_B)};
    vec3 hi = (log2(max(c, vec3(1e-30))) + 9.72) / 17.52;
    return mix(lo, hi, vec3(greaterThan(c, vec3(${f(ACESCCT_LIN_BREAK)}))));
}

vec3 radACEScctToLin(vec3 c) {
    vec3 lo = (c - ${f(ACESCCT_B)}) / ${f(ACESCCT_A)};
    vec3 hi = exp2(c * 17.52 - 9.72);
    return mix(lo, hi, vec3(greaterThan(c, vec3(${f(ACESCCT_CCT_BREAK)}))));
}

vec3 radToAP1(vec3 c, int gamut) {
    // Source gamut to ACEScg; an ACEScg source is already there.
    if (gamut == 1) return c;
${gamutBranches(TO_AP1, glslMat, 'return', false)}
}

vec3 radFromAP1(vec3 c, int gamut) {
    if (gamut == 1) return c;
${gamutBranches(FROM_AP1, glslMat, 'return', false)}
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
    if (k == 1.0) { return color; }
    let p = max(pivot, ${f(PIVOT_FLOOR)});
    let curved = min(p * pow(max(color, vec3f(1e-30)) / p, vec3f(k)), vec3f(${CONTRAST_CEILING}));
    return select(color, curved, color > vec3f(0.0));
}

fn radWhiteBalanceGains(temperature: f32, tint: f32) -> vec3f {
    let t = temperature * ${f(WB_TEMPERATURE_STOPS)};
    let n = tint * ${f(WB_TINT_STOPS)};
    let g = exp2(vec3f(0.5 * t, -n, -0.5 * t));
    return g / radLuma(g);
}

fn radWhiteBalance(color: vec3f, temperature: f32, tint: f32) -> vec3f {
    if (temperature == 0.0 && tint == 0.0) { return color; }
    return color * radWhiteBalanceGains(temperature, tint);
}

fn radLumaMix(color: vec3f, refLuma: f32, m: f32) -> vec3f {
    if (m >= 1.0) { return color; }
    let y = radLuma(color);
    var held = color + vec3f(refLuma - y);
    if (y > 1e-4) { held = color * (refLuma / y); }
    return mix(held, color, clamp(m, 0.0, 1.0));
}

fn radLinToACEScct(c: vec3f) -> vec3f {
    let lo = ${f(ACESCCT_A)} * c + ${f(ACESCCT_B)};
    let hi = (log2(max(c, vec3f(1e-30))) + 9.72) / 17.52;
    return select(lo, hi, c > vec3f(${f(ACESCCT_LIN_BREAK)}));
}

fn radACEScctToLin(c: vec3f) -> vec3f {
    let lo = (c - ${f(ACESCCT_B)}) / ${f(ACESCCT_A)};
    let hi = exp2(c * 17.52 - 9.72);
    return select(lo, hi, c > vec3f(${f(ACESCCT_CCT_BREAK)}));
}

fn radToAP1(c: vec3f, gamut: i32) -> vec3f {
    if (gamut == 1) { return c; }
${gamutBranches(TO_AP1, wgslMat, 'return', true)}
}

fn radFromAP1(c: vec3f, gamut: i32) -> vec3f {
    if (gamut == 1) { return c; }
${gamutBranches(FROM_AP1, wgslMat, 'return', true)}
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
