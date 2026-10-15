/**
 * Onion skin and difference settings. Pure logic: which neighbouring frames to
 * show and how strongly, and how a difference is measured. No rendering here;
 * a renderer binding consumes `onionLayers()` and `differenceSpec()`.
 */
export const ONION_LIMITS = Object.freeze({ maxSide: 4, minOpacity: 0.02 });
export const ONION_DEFAULTS = Object.freeze({
    before: 1, after: 1, opacity: 0.4, falloff: 0.5, tint: true, step: 1,
});
export const DIFF_DOMAINS = Object.freeze(['scene-linear', 'display']);
export const DIFF_GAINS = Object.freeze([1, 2, 4, 8, 16]);
// Display-referred is what the renderer measures today (A and B after the same view).
export const DIFF_DEFAULTS = Object.freeze({ domain: 'display', gain: 4, mode: 'abs' });

const clampInt = (v, lo, hi, d) => {
    const n = Math.trunc(Number(v));
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
};
const clamp01 = (v, d) => (Number.isFinite(Number(v)) ? Math.min(1, Math.max(0, Number(v))) : d);

/** Anything unusable falls back to the default for that field, never throws. */
export function normalizeOnion(o = {}) {
    const d = ONION_DEFAULTS;
    return {
        before: clampInt(o.before, 0, ONION_LIMITS.maxSide, d.before),
        after: clampInt(o.after, 0, ONION_LIMITS.maxSide, d.after),
        step: clampInt(o.step, 1, 100, d.step),
        opacity: clamp01(o.opacity, d.opacity),
        falloff: clamp01(o.falloff, d.falloff),
        tint: typeof o.tint === 'boolean' ? o.tint : d.tint,
    };
}

/**
 * Layers to draw under or over frame `frame` of `count`. Frames outside
 * 0..count-1 are skipped, never wrapped or clamped to a duplicate. Opacity
 * falls off by `falloff` per step away; layers below minOpacity are dropped.
 * `side` is 'before' (past) or 'after' (future) so a renderer can tint them.
 */
export function onionLayers(frame, count, options = {}) {
    const o = normalizeOnion(options);
    if (!Number.isInteger(frame) || !Number.isInteger(count) || count < 1 || frame < 0 || frame >= count) return [];
    const out = [];
    for (const [side, n, dir] of [['before', o.before, -1], ['after', o.after, 1]]) {
        for (let i = 1; i <= n; i++) {
            const f = frame + dir * i * o.step;
            if (f < 0 || f >= count) break;
            const opacity = o.opacity * Math.pow(o.falloff, i - 1);
            if (opacity < ONION_LIMITS.minOpacity) break;
            out.push({ frame: f, side, opacity, tint: o.tint ? (side === 'before' ? [1, 0.35, 0.35] : [0.35, 1, 0.35]) : null });
        }
    }
    return out;
}

/** What a difference view measures. Unknown values fall back, and the result says what it uses. */
export function differenceSpec(s = {}) {
    const domain = DIFF_DOMAINS.includes(s.domain) ? s.domain : DIFF_DEFAULTS.domain;
    const gain = DIFF_GAINS.includes(Number(s.gain)) ? Number(s.gain) : DIFF_DEFAULTS.gain;
    const mode = s.mode === 'signed' ? 'signed' : 'abs';
    return {
        domain, gain, mode,
        label: `${mode === 'abs' ? '|A - B|' : 'A - B (grey = equal)'} x${gain}, ${domain === 'display' ? 'display-referred' : 'scene-linear'}`,
    };
}

/** One difference sample on already-domain-converted values; the shader must match this. */
export function differenceValue(a, b, spec) {
    const d = a - b;
    return (spec.mode === 'signed' ? d : Math.abs(d)) * spec.gain;
}
