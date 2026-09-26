/**
 * Pixel probe statistics.
 *
 * Pure maths — no DOM, no WebGL, no viewer state. Everything here takes plain
 * typed arrays and returns plain objects, so it is testable in Node and so the
 * probe panel and the scopes cannot drift into computing "the average" two
 * different ways.
 *
 * The vocabulary deliberately matches the references a colourist already knows:
 * Nuke's Pixel Analyzer (Current / Min / Max / Average / Median, and its
 * Pixel Selection vs Full Frame sampling modes), RV's Color Inspector (source
 * value vs final rendered value), and mrv2's Color Area panel (rectangle
 * selection with min/max/mean). Inventing new words here would only make the
 * panel harder to trust.
 *
 * Two things this module is strict about, because they are the ones that make a
 * probe worth having rather than decorative:
 *
 *   1. Non-finite values never enter a statistic. Scene-linear EXR carries NaN
 *      from a bad matrix, division, or an un-premultiplied alpha, and a single
 *      NaN in a naive `Math.min` chain turns every number in the panel into NaN.
 *      They are excluded and *counted*, and the panel reports the count — a
 *      probe that silently hides a NaN is worse than no probe.
 *   2. Negative values are counted too. They are legal in scene-linear after a
 *      colour-space conversion, and they are also the signature of a mistake.
 *      The probe does not judge; it reports.
 */

/** Rec.709 / sRGB luminance coefficients (ITU-R BT.709-6). */
export const LUMA_R = 0.2126;
export const LUMA_G = 0.7152;
export const LUMA_B = 0.0722;

/**
 * HDR Reference White, ITU-R BT.2408: 203 cd/m². Diffuse white sits here, at
 * 58% PQ / 75% HLG. The same anchor the HDR heatmap and the SDR→HDR node use —
 * three places reading the same number is the point.
 */
export const HDR_REFERENCE_WHITE_NITS = 203;

/** Mid-grey, scene-linear. The zero of the EV scale. */
export const MIDDLE_GREY = 0.18;

/**
 * Above this many samples the median is estimated from a histogram rather than
 * a full sort. 4K RGBA is 8.3M pixels per channel; sorting that four times per
 * mouse move is not a real option, and the estimate is exact to one part in
 * `MEDIAN_BINS` of the channel's range.
 */
export const MEDIAN_EXACT_LIMIT = 1 << 16;
export const MEDIAN_BINS = 16384;

/**
 * Per-channel statistics over a set of samples.
 *
 * @param {ArrayLike<number>} data      interleaved samples
 * @param {object}            opts
 * @param {number}            opts.width      pixels per row in `data`
 * @param {number}            opts.height     rows in `data`
 * @param {number}            opts.channels   values per pixel in `data`
 * @param {object}           [opts.rect]      {x, y, w, h} in pixels; omitted = full frame
 * @returns {{
 *   count: number, sampled: number,
 *   rect: {x:number,y:number,w:number,h:number},
 *   channels: Array<{min:number,max:number,mean:number,median:number,medianExact:boolean}|null>,
 *   nan: number, inf: number, negative: number,
 * }}
 */
export function sampleStats(data, { width, height, channels, rect } = {}) {
    const w = Math.max(0, Math.floor(width || 0));
    const h = Math.max(0, Math.floor(height || 0));
    const ch = Math.max(1, Math.floor(channels || 1));

    const box = clampRect(rect ?? { x: 0, y: 0, w, h }, w, h);
    const empty = {
        count: 0, sampled: 0, rect: box,
        channels: new Array(ch).fill(null),
        nan: 0, inf: 0, negative: 0,
    };
    if (!data || box.w <= 0 || box.h <= 0) return empty;

    const total = box.w * box.h;

    // Pass 1 — extent, sum, and the counts that matter. Sum in a plain JS
    // number (a double): 8.3M float32 samples cannot overflow 53 bits of
    // mantissa at any sane magnitude, so no compensated summation is needed.
    const min = new Float64Array(ch).fill(Infinity);
    const max = new Float64Array(ch).fill(-Infinity);
    const sum = new Float64Array(ch);
    const finite = new Float64Array(ch);
    let nan = 0, inf = 0, negative = 0;

    for (let row = 0; row < box.h; row++) {
        let i = ((box.y + row) * w + box.x) * ch;
        for (let col = 0; col < box.w; col++) {
            for (let c = 0; c < ch; c++, i++) {
                const v = data[i];
                if (Number.isNaN(v)) { nan++; continue; }
                if (v === Infinity || v === -Infinity) { inf++; continue; }
                if (v < 0) negative++;
                if (v < min[c]) min[c] = v;
                if (v > max[c]) max[c] = v;
                sum[c] += v;
                finite[c]++;
            }
        }
    }

    const out = {
        count: total, sampled: total,
        rect: box,
        channels: new Array(ch).fill(null),
        nan, inf, negative,
    };

    // Pass 2 — median. Exact by sort when the sample is small, histogram
    // otherwise. 'medianExact' travels with the number so the panel can say
    // which one it is showing rather than implying a precision it does not have.
    const exact = total <= MEDIAN_EXACT_LIMIT;
    const buckets = exact ? null : new Float64Array(ch * MEDIAN_BINS);
    const lists = exact ? Array.from({ length: ch }, () => []) : null;

    for (let row = 0; row < box.h; row++) {
        let i = ((box.y + row) * w + box.x) * ch;
        for (let col = 0; col < box.w; col++) {
            for (let c = 0; c < ch; c++, i++) {
                const v = data[i];
                if (!Number.isFinite(v)) continue;
                if (exact) {
                    lists[c].push(v);
                } else {
                    const lo = min[c], hi = max[c];
                    const span = hi - lo;
                    const bin = span > 0
                        ? Math.min(MEDIAN_BINS - 1, Math.floor(((v - lo) / span) * MEDIAN_BINS))
                        : 0;
                    buckets[c * MEDIAN_BINS + bin]++;
                }
            }
        }
    }

    for (let c = 0; c < ch; c++) {
        if (finite[c] === 0) { out.channels[c] = null; continue; }
        let median;
        if (exact) {
            const list = lists[c];
            list.sort((a, b) => a - b);
            const mid = list.length >> 1;
            median = list.length % 2 ? list[mid] : (list[mid - 1] + list[mid]) / 2;
        } else {
            median = medianFromHistogram(
                buckets, c * MEDIAN_BINS, MEDIAN_BINS, min[c], max[c], finite[c],
            );
        }
        out.channels[c] = {
            min: min[c],
            max: max[c],
            mean: sum[c] / finite[c],
            median,
            medianExact: exact,
        };
    }
    return out;
}

/**
 * The value at the half-way count of a histogram, interpolated within the bin
 * that contains it. Interpolating rather than returning the bin's centre keeps
 * the estimate stable as the bin boundaries move with the data's range.
 */
export function medianFromHistogram(bins, offset, binCount, lo, hi, total) {
    if (!(total > 0)) return NaN;
    if (!(hi > lo)) return lo;
    const target = total / 2;
    let seen = 0;
    for (let b = 0; b < binCount; b++) {
        const n = bins[offset + b];
        if (n === 0) continue;
        if (seen + n >= target) {
            const width = (hi - lo) / binCount;
            const within = n > 0 ? (target - seen) / n : 0;
            return lo + (b + within) * width;
        }
        seen += n;
    }
    return hi;
}

/** Clamp a rectangle to the image, in pixels. Never returns negative extent. */
export function clampRect(rect, width, height) {
    const x0 = Math.max(0, Math.min(Math.floor(rect?.x ?? 0), width));
    const y0 = Math.max(0, Math.min(Math.floor(rect?.y ?? 0), height));
    const x1 = Math.max(x0, Math.min(x0 + Math.floor(rect?.w ?? 0), width));
    const y1 = Math.max(y0, Math.min(y0 + Math.floor(rect?.h ?? 0), height));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** A rectangle from two corners in any order, inclusive of both. */
export function rectFromCorners(ax, ay, bx, by) {
    const x = Math.min(ax, bx), y = Math.min(ay, by);
    return { x, y, w: Math.abs(bx - ax) + 1, h: Math.abs(by - ay) + 1 };
}

/** One pixel, as an interleaved read. Returns null outside the image. */
export function pixelAt(data, { width, height, channels }, x, y) {
    if (!data) return null;
    if (x < 0 || y < 0 || x >= width || y >= height) return null;
    const ch = Math.max(1, channels || 1);
    const i = (y * width + x) * ch;
    const r = data[i];
    const g = ch > 1 ? data[i + 1] : r;
    const b = ch > 2 ? data[i + 2] : r;
    const a = ch > 3 ? data[i + 3] : 1;
    return { r, g, b, a };
}

/** BT.709 luminance. */
export function luminance(r, g, b) {
    return r * LUMA_R + g * LUMA_G + b * LUMA_B;
}

/**
 * Scene-linear luminance as cd/m², on the convention that 1.0 is diffuse white
 * and diffuse white is BT.2408's 203 cd/m². This is the mapping the HDR heatmap
 * bands against, so the probe and the heatmap agree by construction.
 */
export function nits(linearLuma, referenceWhite = HDR_REFERENCE_WHITE_NITS) {
    if (!Number.isFinite(linearLuma)) return NaN;
    return Math.max(linearLuma, 0) * referenceWhite;
}

/** Stops relative to 18% mid-grey. */
export function exposureValue(linearLuma) {
    if (!Number.isFinite(linearLuma) || linearLuma <= 0) return -Infinity;
    return Math.log2(linearLuma / MIDDLE_GREY);
}

/**
 * HSV from linear or display RGB, whichever the caller sampled. Hue in degrees
 * 0–360, S and V in 0–1. V is max(r,g,b) and is *not* clamped to 1 — an HDR
 * probe that clamped its own readout would be lying about the highlight it
 * exists to measure.
 */
export function rgbToHsv(r, g, b) {
    if (![r, g, b].every(Number.isFinite)) return { h: NaN, s: NaN, v: NaN };
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d > 0) {
        if (max === r) h = ((g - b) / d) % 6;
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h *= 60;
        if (h < 0) h += 360;
    }
    const s = max > 0 ? d / max : 0;
    return { h, s, v: max };
}

/**
 * The sRGB breakpoint, on both sides of the curve.
 *
 * IEC 61966-2-1 prints these rounded to 0.04045 and 0.0031308, and those two
 * rounded numbers are not each other's image: 0.04045 / 12.92 is 0.00313080…,
 * which falls on the *power* side of 0.0031308, so a round trip through the
 * published constants is discontinuous by about 1e-7 at that one point. These
 * are the exact values, where the linear segment and the power segment meet, so
 * `linearToSrgb(srgbToLinear(c)) === c` to double precision everywhere. The
 * difference from the published pair is far below a 16-bit code value; using
 * the exact pair costs nothing and removes a class of confusing off-by-a-hair.
 */
export const SRGB_ENCODED_BREAK = 0.04044823627710785;
export const SRGB_LINEAR_BREAK = SRGB_ENCODED_BREAK / 12.92;

/**
 * IEC 61966-2-1 sRGB EOTF. Used when there is no float sidecar to sample.
 *
 * Applied oddly — sign carried around the curve rather than through it. Scene
 * linear legitimately carries negatives after a matrix conversion, and
 * `Math.pow(negative, 1/2.4)` is NaN, which would show up as a probe fault on a
 * perfectly valid pixel.
 */
export function srgbToLinear(c) {
    if (!Number.isFinite(c)) return c;
    const s = c < 0 ? -1 : 1;
    const a = Math.abs(c);
    return s * (a <= SRGB_ENCODED_BREAK ? a / 12.92 : Math.pow((a + 0.055) / 1.055, 2.4));
}

/** Its inverse, for the swatch. */
export function linearToSrgb(c) {
    if (!Number.isFinite(c)) return c;
    const s = c < 0 ? -1 : 1;
    const a = Math.abs(c);
    return s * (a <= SRGB_LINEAR_BREAK ? a * 12.92 : 1.055 * Math.pow(a, 1 / 2.4) - 0.055);
}

/** An 8-bit hex swatch for a linear colour. Clamps — a swatch is a swatch. */
export function hexSwatch(r, g, b) {
    const to8 = (c) => {
        if (!Number.isFinite(c)) return 0;
        return Math.max(0, Math.min(255, Math.round(linearToSrgb(c) * 255)));
    };
    return '#' + [to8(r), to8(g), to8(b)]
        .map((v) => v.toString(16).padStart(2, '0')).join('');
}

/**
 * Fixed-width value text. Non-finite values print as words rather than as
 * "NaN" buried in a column of numbers, because that is the case the reader must
 * not skim past.
 */
export function formatValue(v, digits = 5) {
    if (Number.isNaN(v)) return 'NaN';
    if (v === Infinity) return '+Inf';
    if (v === -Infinity) return '-Inf';
    if (v === undefined || v === null) return '—';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e5)) return v.toExponential(2);
    return v.toFixed(digits);
}
