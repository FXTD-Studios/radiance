/**
 * Scope scales and measurement point.
 *
 * A scope that does not say what it measures is ambiguous, and an ambiguous
 * instrument is an untrusted one. Radiance's scopes drew a graticule labelled
 * `0 / 25 / 50 / 75 / 100` under a caption that read "Linear · 0–255" — two
 * different units in the same panel, neither of them stated. This module is the
 * fix: one place that answers "what number is this line, and in what unit".
 *
 * The set of scales is Resolve's, deliberately:
 *
 *   10-bit code value 0–1023   (the default; what a delivery spec is written in)
 *   12-bit code value 0–4095
 *   Percent
 *   Millivolts 0–700
 *   Nits, ST.2084 (PQ)
 *   Nits, HLG
 *
 * **IRE is deliberately absent.** Resolve's manual does not list it; it is a
 * legacy analogue-composite unit, and offering it signals the opposite of
 * expertise. If someone asks for it, the answer is percent.
 *
 * Two things a reader should understand before trusting a number out of here:
 *
 * 1. **Code value is not levels-dependent; percent and millivolts are.** Code
 *    64 is code 64 whether the signal is narrow- or full-range. What changes is
 *    where 0% and 100% sit — at 64 and 940 in video levels, at 0 and 1023 in
 *    data levels. Conflating the two is the classic way to misread a legal-range
 *    error as a grading choice.
 *
 * 2. **A nit reading is an interpretation, not a measurement.** Nits exist only
 *    once you declare what the code values encode. These scales apply the ST.2084
 *    or HLG transfer function to the plotted signal; if the viewer is not
 *    outputting that encoding, the number is meaningless and the panel says so
 *    rather than printing it confidently.
 */

// ── Signal ranges (BT.709 / BT.2100 narrow range) ───────────────────────────

/**
 * Black and white code values per bit depth, narrow ("video") range. Full
 * ("data") range is always 0 … 2^n − 1.
 */
export const NARROW_RANGE = {
    8: { black: 16, white: 235 },
    10: { black: 64, white: 940 },
    12: { black: 256, white: 3760 },
};

export const LEVELS = [
    { id: 'data', label: 'Data Levels', hint: 'Full range: 0 … 2ⁿ−1. Percent is measured across the whole code range.' },
    { id: 'video', label: 'Video Levels', hint: 'Narrow range: 64 … 940 at 10-bit. 0% and 100% sit at black and white, with footroom and headroom visible outside them.' },
];

/** The code value a normalised 0–1 signal lands on, at a given bit depth. */
export function codeValue(norm, bits = 10) {
    const max = (1 << bits) - 1;
    return norm * max;
}

/**
 * Normalised signal, re-referenced so 0 is black and 1 is white.
 *
 * In data levels this is the identity. In video levels it rescales 64…940 onto
 * 0…1, which is what lets percent read 0 at black rather than at code 0, and
 * what a transfer function has to be fed before it means anything.
 */
export function toSignal(norm, levels = 'data', bits = 10) {
    if (levels !== 'video') return norm;
    const max = (1 << bits) - 1;
    const { black, white } = NARROW_RANGE[bits] || NARROW_RANGE[10];
    return (norm * max - black) / (white - black);
}

// ── SMPTE ST.2084 (PQ) ──────────────────────────────────────────────────────
// Constants exactly as published. Writing them as the ratios they are, rather
// than as decimals, keeps them checkable against the standard.

const PQ_M1 = 2610 / 16384;          // 0.1593017578125
const PQ_M2 = (2523 / 4096) * 128;   // 78.84375
const PQ_C1 = 3424 / 4096;           // 0.8359375
const PQ_C2 = (2413 / 4096) * 32;    // 18.8515625
const PQ_C3 = (2392 / 4096) * 32;    // 18.6875

/** PQ EOTF: a 0–1 signal to display luminance in cd/m², peak 10 000. */
export function pqToNits(signal) {
    if (!Number.isFinite(signal)) return NaN;
    const n = Math.min(Math.max(signal, 0), 1);
    const p = Math.pow(n, 1 / PQ_M2);
    const num = Math.max(p - PQ_C1, 0);
    const den = PQ_C2 - PQ_C3 * p;
    if (den <= 0) return 10000;
    return 10000 * Math.pow(num / den, 1 / PQ_M1);
}

/** Its inverse, for placing a graticule line at a chosen nit value. */
export function nitsToPq(cdm2) {
    if (!Number.isFinite(cdm2)) return NaN;
    const y = Math.min(Math.max(cdm2, 0), 10000) / 10000;
    const ym = Math.pow(y, PQ_M1);
    return Math.pow((PQ_C1 + PQ_C2 * ym) / (1 + PQ_C3 * ym), PQ_M2);
}

// ── ITU-R BT.2100 HLG ───────────────────────────────────────────────────────

const HLG_A = 0.17883277;
const HLG_B = 1 - 4 * HLG_A;              // 0.28466892
const HLG_C = 0.5 - HLG_A * Math.log(4 * HLG_A);   // 0.55991073

/** HLG inverse OETF: signal to *scene* linear, normalised so 1.0 is peak. */
export function hlgToScene(signal) {
    if (!Number.isFinite(signal)) return NaN;
    const e = Math.min(Math.max(signal, 0), 1);
    if (e <= 0.5) return (e * e) / 3;
    return (Math.exp((e - HLG_C) / HLG_A) + HLG_B) / 12;
}

/**
 * The HLG system gamma for a given nominal peak (BT.2100 Table 5 note):
 * 1.2 at 1000 cd/m², moving by 0.42 per decade of peak luminance.
 */
export function hlgSystemGamma(peakNits = 1000) {
    return 1.2 + 0.42 * Math.log10(Math.max(peakNits, 1) / 1000);
}

/**
 * HLG signal to display luminance, through the inverse OETF and the OOTF.
 *
 * Applied to the plotted value directly. On a per-channel parade that is an
 * approximation — the OOTF is defined on luminance, not per channel — and the
 * panel says so. On a luma waveform it is exact.
 */
export function hlgToNits(signal, peakNits = 1000) {
    const scene = hlgToScene(signal);
    if (!Number.isFinite(scene)) return NaN;
    return peakNits * Math.pow(scene, hlgSystemGamma(peakNits));
}

/** Inverse, for placing a graticule line at a chosen nit value. */
export function nitsToHlg(cdm2, peakNits = 1000) {
    if (!Number.isFinite(cdm2) || cdm2 < 0) return NaN;
    const scene = Math.pow(Math.min(cdm2, peakNits) / peakNits, 1 / hlgSystemGamma(peakNits));
    if (scene <= 1 / 12) return Math.sqrt(3 * scene);
    return HLG_A * Math.log(12 * scene - HLG_B) + HLG_C;
}

// ── The scales ──────────────────────────────────────────────────────────────

/**
 * HDR Reference White, ITU-R BT.2408. 203 cd/m² is 58% PQ and 75% HLG — the two
 * numbers below are not hand-tuned, they fall out of the transfer functions, and
 * there is a test that proves it.
 */
export const HDR_REFERENCE_WHITE_NITS = 203;

/**
 * Every scale, in the order they should appear in the menu.
 *
 * `value(norm, ctx)` converts a normalised plot position to the scale's number.
 * `position(value, ctx)` is the inverse, used to place graticule lines.
 * `ticks(ctx)` returns the lines to draw, as `{ at, label, key }` where `at` is
 * a normalised 0–1 height.
 */
export const SCOPE_SCALES = [
    {
        id: 'cv10', label: '10-bit', unit: 'CV', digits: 0,
        describe: () => '10-bit code value, 0–1023.',
        value: (norm) => codeValue(norm, 10),
        position: (v) => v / 1023,
        ticks: (ctx) => (ctx.levels === 'video'
            ? [0, 64, 256, 512, 720, 940, 1023]
            : [0, 128, 256, 512, 768, 896, 1023]
        ).map((cv) => ({ at: cv / 1023, label: String(cv), key: cv })),
    },
    {
        id: 'cv12', label: '12-bit', unit: 'CV', digits: 0,
        describe: () => '12-bit code value, 0–4095.',
        value: (norm) => codeValue(norm, 12),
        position: (v) => v / 4095,
        ticks: (ctx) => (ctx.levels === 'video'
            ? [0, 256, 1024, 2048, 2880, 3760, 4095]
            : [0, 512, 1024, 2048, 3072, 3584, 4095]
        ).map((cv) => ({ at: cv / 4095, label: String(cv), key: cv })),
    },
    {
        id: 'percent', label: 'Percent', unit: '%', digits: 1,
        describe: (ctx) => (ctx.levels === 'video'
            ? '0% at code 64, 100% at code 940. Footroom and headroom are visible outside them.'
            : '0% at code 0, 100% at code 1023.'),
        value: (norm, ctx) => toSignal(norm, ctx.levels, 10) * 100,
        position: (v, ctx) => fromSignal(v / 100, ctx.levels, 10),
        ticks: (ctx) => [-7, 0, 25, 50, 75, 100, 109]
            .filter((p) => ctx.levels === 'video' || (p >= 0 && p <= 100))
            .map((p) => ({ at: fromSignal(p / 100, ctx.levels, 10), label: `${p}`, key: p })),
    },
    {
        id: 'mv', label: 'Millivolts', unit: 'mV', digits: 0,
        describe: () => 'Analogue reference: 0 mV at black, 700 mV at white.',
        value: (norm, ctx) => toSignal(norm, ctx.levels, 10) * 700,
        position: (v, ctx) => fromSignal(v / 700, ctx.levels, 10),
        ticks: (ctx) => [0, 175, 350, 525, 700]
            .map((mv) => ({ at: fromSignal(mv / 700, ctx.levels, 10), label: String(mv), key: mv })),
    },
    {
        id: 'nits-pq', label: 'Nits (ST.2084)', unit: 'cd/m²', digits: 1,
        interprets: 'PQ',
        describe: () => 'Reads the signal as ST.2084 (PQ). Only meaningful if the viewer is outputting PQ.',
        value: (norm, ctx) => pqToNits(toSignal(norm, ctx.levels, 10)),
        position: (v, ctx) => fromSignal(nitsToPq(v), ctx.levels, 10),
        ticks: (ctx) => [0.01, 1, 10, 100, HDR_REFERENCE_WHITE_NITS, 1000, 4000, 10000]
            .map((n) => ({
                at: fromSignal(nitsToPq(n), ctx.levels, 10),
                label: n === HDR_REFERENCE_WHITE_NITS ? '203 ref' : formatNits(n),
                key: n,
                emphasis: n === HDR_REFERENCE_WHITE_NITS,
            })),
    },
    {
        id: 'nits-hlg', label: 'Nits (HLG)', unit: 'cd/m²', digits: 1,
        interprets: 'HLG',
        describe: (ctx) => `Reads the signal as HLG at a ${ctx.peakNits} cd/m² nominal peak `
            + `(system gamma ${hlgSystemGamma(ctx.peakNits).toFixed(2)}). Only meaningful if the viewer is outputting HLG.`,
        value: (norm, ctx) => hlgToNits(toSignal(norm, ctx.levels, 10), ctx.peakNits),
        position: (v, ctx) => fromSignal(nitsToHlg(v, ctx.peakNits), ctx.levels, 10),
        ticks: (ctx) => [0.01, 1, 10, 100, HDR_REFERENCE_WHITE_NITS, 1000]
            .filter((n) => n <= ctx.peakNits)
            .map((n) => ({
                at: fromSignal(nitsToHlg(n, ctx.peakNits), ctx.levels, 10),
                label: n === HDR_REFERENCE_WHITE_NITS ? '203 ref' : formatNits(n),
                key: n,
                emphasis: n === HDR_REFERENCE_WHITE_NITS,
            })),
    },
];

/** Inverse of `toSignal`: black-to-white 0–1 back to a normalised code position. */
export function fromSignal(signal, levels = 'data', bits = 10) {
    if (levels !== 'video') return signal;
    const max = (1 << bits) - 1;
    const { black, white } = NARROW_RANGE[bits] || NARROW_RANGE[10];
    return (black + signal * (white - black)) / max;
}

/**
 * The LogC-assist reshaping, as a pair.
 *
 * The scopes offer a "log view" that pushes the plotted pixels through
 * `log(1 + v·C) / log(1 + C)` with C = 299, roughly LogC3-shaped, so shadows
 * open up. The graticule has to travel through the same curve or every label
 * slides off its line — which is the difference between a log view that still
 * reads true and one that merely warns that it does not.
 */
export const LOG_ASSIST_C = 299.0;

export function logAssistPos(v) {
    const q = Math.min(Math.max(v, 0), 1);
    return Math.log(1 + q * LOG_ASSIST_C) / Math.log(1 + LOG_ASSIST_C);
}

export function logAssistInv(p) {
    const q = Math.min(Math.max(p, 0), 1);
    return (Math.pow(1 + LOG_ASSIST_C, q) - 1) / LOG_ASSIST_C;
}

export function getScale(id) {
    return SCOPE_SCALES.find((s) => s.id === id) || SCOPE_SCALES[0];
}

/** Compact nit labels: 0.01, 1, 100, 1k, 10k. */
export function formatNits(n) {
    if (!Number.isFinite(n)) return '—';
    if (n >= 1000) return `${n / 1000}k`;
    if (n >= 1) return String(Math.round(n));
    return String(n);
}

/**
 * Ticks for a scale, clipped to the visible axis and de-duplicated.
 *
 * A tick outside 0–1 would be drawn off the canvas or, worse, clamped onto the
 * edge where it would read as a real measurement at the wrong place.
 */
export function scaleTicks(scaleId, ctx = {}) {
    const scale = getScale(scaleId);
    const full = { levels: 'data', peakNits: 1000, ...ctx };
    const seen = new Set();
    return scale.ticks(full)
        .filter((t) => Number.isFinite(t.at) && t.at >= -1e-9 && t.at <= 1 + 1e-9)
        .filter((t) => {
            const k = t.at.toFixed(6);
            if (seen.has(k)) return false;
            seen.add(k);
            return true;
        })
        .map((t) => ({ ...t, at: Math.min(Math.max(t.at, 0), 1) }));
}

/** The number at a plot position, formatted for the axis. */
export function scaleValue(norm, scaleId, ctx = {}) {
    const scale = getScale(scaleId);
    const full = { levels: 'data', peakNits: 1000, ...ctx };
    return scale.value(norm, full);
}

/**
 * The one-line statement of what the scope is measuring.
 *
 * Assembled rather than stored so it cannot fall out of sync with the controls,
 * and never empty — the whole point of this module is that no number on a scope
 * is unlabelled.
 */
export function describeMeasurement(scaleId, ctx = {}) {
    const scale = getScale(scaleId);
    const full = { levels: 'data', peakNits: 1000, transformed: true, ...ctx };
    const levels = LEVELS.find((l) => l.id === full.levels) || LEVELS[0];
    const where = full.transformed
        ? 'after the viewer colour transforms — what the display receives'
        : 'before the viewer colour transforms — the source as loaded';
    return {
        unit: scale.unit,
        label: scale.label,
        levels: levels.label,
        detail: scale.describe(full),
        measuredAt: where,
        // A nit scale on an untransformed source is reading a transfer function
        // off pixels that were never encoded with it.
        warn: !!scale.interprets && !full.transformed
            ? `${scale.interprets} is a display encoding. Measuring before the viewer transforms means the signal is not ${scale.interprets}-encoded, so these numbers are not luminance.`
            : null,
    };
}
