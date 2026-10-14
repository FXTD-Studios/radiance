/**
 * Deterministic QC measurements for the viewer. Pure maths: no DOM, no GPU.
 *
 * Every check states what it measured, in which domain, against what threshold,
 * and what it cannot tell you. A check that could not run is NOT_RUN or
 * INDETERMINATE, never PASS; the overall status is PASS only when every
 * applicable check ran and passed.
 *
 * Domain: values as decoded from the file (scene-referred where the file is),
 * before any view transform, grade or tone mapping. Nothing here is a display
 * measurement, and nothing here is an AI estimate.
 *
 * A frame is { data: Float32Array | Float64Array, width, height, channels } with
 * interleaved channels (1, 3 or 4; 4 means RGBA with alpha last).
 */

export const QC_VERSION = 1;

export const Status = Object.freeze({
    PASS: 'PASS', WARN: 'WARN', FAIL: 'FAIL',
    NOT_RUN: 'NOT_RUN', NOT_APPLICABLE: 'NOT_APPLICABLE', INDETERMINATE: 'INDETERMINATE',
});

export const DOMAIN = 'decoded source values (before view transform and grade)';

export const DEFAULTS = Object.freeze({
    over: { limit: 1.0, warnFraction: 0.001, failFraction: 0.05 },        // values above `limit`
    under: { limit: 0.0, warnFraction: 0.0005, failFraction: 0.05 },      // values below `limit`
    alpha: { tolerance: 1e-3, warnFraction: 0.0, failFraction: 0.01 },
    flicker: { zThreshold: 6, relFloor: 0.02, absFloor: 1e-4, minFrames: 5 },
});

function validFrame(f) {
    return !!f && f.data && Number.isFinite(f.width) && Number.isFinite(f.height)
        && [1, 3, 4].includes(f.channels) && f.data.length >= f.width * f.height * f.channels
        && f.width > 0 && f.height > 0;
}

function result(id, status, extra = {}) {
    return { id, status, domain: DOMAIN, ...extra };
}

function notRun(id, reason) {
    return result(id, Status.NOT_RUN, { reason });
}

function grade(fraction, warnFraction, failFraction) {
    if (fraction > failFraction) return Status.FAIL;
    if (fraction > warnFraction) return Status.WARN;
    return Status.PASS;
}

const LUMA = [0.2126, 0.7152, 0.0722];

/** Mean Rec.709 luma of the finite colour values; null if there are none. */
export function meanLuma(frame) {
    if (!validFrame(frame)) return null;
    const { data, width, height, channels } = frame;
    const n = width * height;
    let sum = 0, count = 0;
    for (let i = 0; i < n; i++) {
        const o = i * channels;
        let y;
        if (channels === 1) y = data[o];
        else y = LUMA[0] * data[o] + LUMA[1] * data[o + 1] + LUMA[2] * data[o + 2];
        if (Number.isFinite(y)) { sum += y; count++; }
    }
    return count ? sum / count : null;
}

/**
 * Spatial checks on one frame. Returns an array of results.
 * `options` overrides DEFAULTS per check.
 */
export function analyzeFrame(frame, options = {}) {
    if (!validFrame(frame)) {
        return ['nan_inf', 'over_range', 'under_range', 'alpha_range', 'alpha_premultiplied']
            .map((id) => notRun(id, 'no usable pixel data for this frame'));
    }
    const o = {
        over: { ...DEFAULTS.over, ...options.over },
        under: { ...DEFAULTS.under, ...options.under },
        alpha: { ...DEFAULTS.alpha, ...options.alpha },
    };
    const { data, width, height, channels } = frame;
    const pixels = width * height;
    const colourCh = channels === 4 ? 3 : channels;
    let bad = 0, over = 0, under = 0, finiteSamples = 0, maxVal = -Infinity, minVal = Infinity;
    let alphaOut = 0, alphaPixels = 0, premultBad = 0;

    for (let i = 0; i < pixels; i++) {
        const base = i * channels;
        for (let c = 0; c < colourCh; c++) {
            const v = data[base + c];
            if (!Number.isFinite(v)) { bad++; continue; }
            finiteSamples++;
            if (v > maxVal) maxVal = v;
            if (v < minVal) minVal = v;
            if (v > o.over.limit) over++;
            if (v < o.under.limit) under++;
        }
        if (channels === 4) {
            const a = data[base + 3];
            if (!Number.isFinite(a)) { bad++; continue; }
            alphaPixels++;
            if (a < -o.alpha.tolerance || a > 1 + o.alpha.tolerance) alphaOut++;
            // premultiplied colour cannot exceed its alpha: a result above it is
            // either straight (unpremultiplied) data or a bad composite
            const r = data[base], g = data[base + 1], b = data[base + 2];
            if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)
                && Math.max(r, g, b) > a + o.alpha.tolerance && a < 1 - o.alpha.tolerance) premultBad++;
        }
    }

    const total = pixels * channels;
    const out = [];
    out.push(result('nan_inf', bad > 0 ? Status.FAIL : Status.PASS, {
        metric: 'count of NaN or Inf samples', value: bad, threshold: 0,
        fraction: bad / total, limitations: 'Reports only that such samples exist, not where from.',
    }));

    if (finiteSamples === 0) {
        out.push(notRun('over_range', 'no finite colour samples'), notRun('under_range', 'no finite colour samples'));
    } else {
        const n = pixels * colourCh;
        out.push(result('over_range', grade(over / n, o.over.warnFraction, o.over.failFraction), {
            metric: `fraction of colour samples above ${o.over.limit}`, value: over / n,
            threshold: { limit: o.over.limit, warn: o.over.warnFraction, fail: o.over.failFraction }, max: maxVal,
            limitations: 'Above 1.0 is normal for scene-linear HDR; raise the limit for HDR deliveries.',
        }));
        out.push(result('under_range', grade(under / n, o.under.warnFraction, o.under.failFraction), {
            metric: `fraction of colour samples below ${o.under.limit}`, value: under / n,
            threshold: { limit: o.under.limit, warn: o.under.warnFraction, fail: o.under.failFraction }, min: minVal,
            limitations: 'Small negatives are legitimate in wide-gamut or filtered data.',
        }));
    }

    if (channels !== 4) {
        out.push(result('alpha_range', Status.NOT_APPLICABLE, { reason: 'no alpha channel' }),
            result('alpha_premultiplied', Status.NOT_APPLICABLE, { reason: 'no alpha channel' }));
    } else {
        out.push(result('alpha_range', grade(alphaOut / alphaPixels, o.alpha.warnFraction, o.alpha.failFraction), {
            metric: `fraction of pixels with alpha outside 0..1 (tolerance ${o.alpha.tolerance})`,
            value: alphaOut / alphaPixels, threshold: { warn: o.alpha.warnFraction, fail: o.alpha.failFraction },
            limitations: 'Says nothing about whether the matte is correct, only that it is in range.',
        }));
        const frac = premultBad / alphaPixels;
        out.push(result('alpha_premultiplied', grade(frac, o.alpha.warnFraction, o.alpha.failFraction), {
            metric: 'fraction of partly transparent pixels whose colour exceeds alpha',
            value: frac, threshold: { warn: o.alpha.warnFraction, fail: o.alpha.failFraction },
            limitations: 'Straight (unpremultiplied) alpha fails this on purpose; it is a test of the premultiplied assumption, not of the image.',
        }));
    }
    return out;
}

function median(xs) {
    const a = [...xs].sort((p, q) => p - q);
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/**
 * Flicker: frame-to-frame jumps in mean luma that are outliers against the
 * sequence's own typical change. `means` is one mean luma per frame (null for a
 * frame that could not be measured). Returns one result.
 */
export function analyzeFlicker(means, options = {}) {
    const o = { ...DEFAULTS.flicker, ...options };
    const n = means.length;
    if (n < 2) return result('flicker', Status.NOT_APPLICABLE, { reason: 'a single frame has no temporal behaviour' });
    const known = means.filter((m) => m !== null && Number.isFinite(m)).length;
    if (known < o.minFrames) {
        return result('flicker', Status.INDETERMINATE, {
            reason: `${known} measurable frame(s); at least ${o.minFrames} are needed to judge what is normal for this sequence`,
        });
    }
    const deltas = [];                         // { frame, d }
    for (let i = 1; i < n; i++) {
        if (means[i] === null || means[i - 1] === null) continue;
        if (!Number.isFinite(means[i]) || !Number.isFinite(means[i - 1])) continue;
        deltas.push({ frame: i, d: means[i] - means[i - 1] });
    }
    const ds = deltas.map((x) => x.d);
    const med = median(ds);
    const mad = median(ds.map((d) => Math.abs(d - med)));
    const level = median(means.filter((m) => m !== null && Number.isFinite(m)).map(Math.abs));
    const floor = Math.max(o.absFloor, o.relFloor * level);
    const scale = Math.max(1.4826 * mad, floor);
    const flagged = deltas
        .filter((x) => Math.abs(x.d - med) / scale > o.zThreshold && Math.abs(x.d - med) > floor)
        .map((x) => ({ frame: x.frame, delta: x.d, z: Math.abs(x.d - med) / scale }));
    return result('flicker', flagged.length ? Status.WARN : Status.PASS, {
        metric: 'frame-to-frame change in mean luma, as a robust z-score against the sequence',
        value: flagged.length, threshold: { z: o.zThreshold, floor }, frames: flagged.map((x) => x.frame),
        detail: flagged,
        limitations: 'Detects sudden brightness jumps, not slow drift, and a deliberate cut or flash is flagged the same as a fault.',
    });
}

const RANK = { FAIL: 5, WARN: 4, INDETERMINATE: 3, NOT_RUN: 2, PASS: 1, NOT_APPLICABLE: 0 };

/**
 * Overall status. FAIL and WARN win. PASS needs at least one check that ran
 * and no check left NOT_RUN or INDETERMINATE; otherwise INDETERMINATE (some ran
 * and passed, others did not) or NOT_RUN (nothing ran).
 */
export function overallStatus(checks) {
    const applicable = checks.filter((c) => c.status !== Status.NOT_APPLICABLE);
    if (!applicable.length) return Status.NOT_APPLICABLE;
    if (applicable.some((c) => c.status === Status.FAIL)) return Status.FAIL;
    if (applicable.some((c) => c.status === Status.WARN)) return Status.WARN;
    const ran = applicable.filter((c) => c.status === Status.PASS).length;
    if (!ran) return applicable.some((c) => c.status === Status.INDETERMINATE) ? Status.INDETERMINATE : Status.NOT_RUN;
    if (applicable.some((c) => c.status === Status.NOT_RUN || c.status === Status.INDETERMINATE)) return Status.INDETERMINATE;
    return Status.PASS;
}

/** Merge per-frame results of one check id into a single result with a frame list. */
function mergeFrameResults(id, perFrame, total) {
    const ran = perFrame.filter((r) => r && r.status !== Status.NOT_RUN && r.status !== Status.NOT_APPLICABLE);
    const first = perFrame.find((r) => r) || {};
    if (!ran.length) {
        const na = perFrame.every((r) => r && r.status === Status.NOT_APPLICABLE);
        return result(id, na ? Status.NOT_APPLICABLE : Status.NOT_RUN, {
            reason: na ? first.reason : 'no frame was analysed', frameCount: total,
        });
    }
    const worst = ran.reduce((a, b) => (RANK[b.status] > RANK[a.status] ? b : a));
    const badFrames = perFrame.map((r, i) => (r && (r.status === Status.WARN || r.status === Status.FAIL) ? i : -1)).filter((i) => i >= 0);
    // A frame with no data counts as not analysed: it must never let the rest pass.
    const complete = perFrame.length === total && perFrame.every((r) => r) && ran.length === perFrame.filter((r) => r && r.status !== Status.NOT_APPLICABLE).length;
    return {
        ...worst, id, frames: badFrames, framesAnalysed: ran.length, frameCount: total,
        status: complete ? worst.status : (worst.status === Status.PASS ? Status.INDETERMINATE : worst.status),
        ...(complete ? {} : { reason: `only ${ran.length} of ${total} frames were analysed` }),
    };
}

/**
 * Run spatial checks over a sequence, and flicker if it has more than one frame.
 *
 *   count            number of frames
 *   getFrame(i)      returns a frame (sync or Promise), or null if unavailable
 *   options          { checks overrides, flicker overrides }
 *   control          { signal, onProgress(done, total), yieldEvery, yieldFn, startAt, previous }
 *
 * Cancellable through `signal` (AbortSignal): the run stops between frames and
 * returns what it measured with `complete: false`. Pass the returned `state`
 * back as `control.previous` with `startAt` to resume without redoing frames.
 * Frames are processed one at a time, so memory does not grow with the length.
 */
export async function analyzeSequence(count, getFrame, options = {}, control = {}) {
    const { signal, onProgress, yieldEvery = 8, yieldFn = () => new Promise((r) => setTimeout(r, 0)) } = control;
    const prev = control.previous || { perCheck: {}, means: [] };
    const perCheck = { ...prev.perCheck };
    for (const k of Object.keys(perCheck)) perCheck[k] = perCheck[k].slice();
    const means = prev.means.slice();
    let i = control.startAt || 0;
    let cancelled = false;
    for (; i < count; i++) {
        if (signal?.aborted) { cancelled = true; break; }
        let frame = null;
        try { frame = await getFrame(i); } catch { frame = null; }
        const results = analyzeFrame(frame, options);
        for (const r of results) {
            (perCheck[r.id] ||= [])[i] = r;
        }
        means[i] = meanLuma(frame);
        onProgress?.(i + 1, count);
        if (yieldEvery && (i + 1) % yieldEvery === 0) await yieldFn();
    }
    const done = cancelled ? i : count;
    const checks = Object.entries(perCheck).map(([id, arr]) => mergeFrameResults(id, Array.from(arr, (r) => r), count));
    // analysed prefix only for flicker when stopped early
    const flicker = analyzeFlicker(means.slice(0, done), options.flicker);
    if (cancelled && flicker.status === Status.PASS) {
        flicker.status = Status.INDETERMINATE;
        flicker.reason = `only ${done} of ${count} frames were analysed`;
    }
    checks.push(flicker);
    return {
        complete: !cancelled, framesDone: done, frameCount: count,
        checks, overall: overallStatus(checks),
        state: { perCheck, means },
    };
}

/**
 * The machine-readable report. `inputs` describes what was measured so the run
 * can be repeated (file names, frame range, dimensions, source tags).
 */
export function buildReport(analysis, inputs = {}, options = {}) {
    return {
        schema: 'radiance.qc.report',
        version: QC_VERSION,
        overall: analysis.overall,
        complete: analysis.complete,
        frames: { analysed: analysis.framesDone, total: analysis.frameCount },
        domain: DOMAIN,
        inputs,
        options: {
            over: { ...DEFAULTS.over, ...options.over }, under: { ...DEFAULTS.under, ...options.under },
            alpha: { ...DEFAULTS.alpha, ...options.alpha }, flicker: { ...DEFAULTS.flicker, ...options.flicker },
        },
        checks: analysis.checks.map(({ detail, ...c }) => c),
        notes: 'Measured values only. NOT_RUN and INDETERMINATE mean the check could not support a verdict, not that the media is fine.',
    };
}
