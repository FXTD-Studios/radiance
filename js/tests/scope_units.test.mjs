/**
 * Scope scales.
 *
 * These are standards, so they are testable against published anchor points
 * rather than against whatever the code happens to do. The two that matter most:
 * ITU-R BT.2408 puts HDR Reference White at 203 cd/m², 58% PQ, 75% HLG — and
 * that relationship must *fall out of* the transfer functions here, not be
 * hard-coded next to them. If a constant is fat-fingered, those two tests fail.
 *
 * Run: node --test js/tests/scope_units.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    NARROW_RANGE, LEVELS, SCOPE_SCALES,
    codeValue, toSignal, fromSignal,
    pqToNits, nitsToPq, hlgToScene, hlgToNits, nitsToHlg, hlgSystemGamma,
    getScale, scaleTicks, scaleValue, describeMeasurement, formatNits,
    logAssistPos, logAssistInv,
    HDR_REFERENCE_WHITE_NITS,
} from '../radiance_scope_units.js';

const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const rel = (a, b, frac) => Math.abs(a - b) <= Math.abs(b) * frac;

// ── ST.2084 (PQ) ────────────────────────────────────────────────────────────

test('PQ hits its published endpoints', () => {
    assert.ok(close(pqToNits(0), 0, 1e-9), 'signal 0 must be 0 cd/m²');
    assert.ok(rel(pqToNits(1), 10000, 1e-9), 'signal 1 must be 10 000 cd/m²');
});

test('BT.2408 HDR Reference White is 58% PQ', () => {
    // The single most quoted number in HDR delivery, and the one that catches a
    // fat-fingered PQ constant.
    //
    // Asserted in this direction on purpose. BT.2408 defines the *luminance* —
    // 203 cd/m² — and prints 58% as its rounded signal value; the exact signal
    // is 0.58069, so feeding 0.58 back in yields 201.7, not 203. Testing
    // `pqToNits(0.58) === 203` would be testing the standard's rounding rather
    // than this code, and would fail a correct implementation.
    assert.ok(rel(nitsToPq(203), 0.58, 0.002),
        `203 cd/m² gave ${nitsToPq(203).toFixed(5)} PQ, expected ≈0.58`);
    assert.equal(nitsToPq(203).toFixed(2), '0.58');
});

test('100 nits PQ is the SDR-white anchor at ~51%', () => {
    assert.ok(rel(nitsToPq(100), 0.5081, 0.002),
        `100 cd/m² gave ${nitsToPq(100).toFixed(4)} PQ`);
});

test('PQ round-trips', () => {
    for (const n of [0, 0.01, 1, 100, 203, 1000, 4000, 10000]) {
        assert.ok(rel(pqToNits(nitsToPq(n)), n, 1e-6) || close(n, 0),
            `round trip failed at ${n} cd/m²`);
    }
});

test('PQ is monotonic', () => {
    let prev = -1;
    for (let s = 0; s <= 1.0001; s += 0.01) {
        const v = pqToNits(s);
        assert.ok(v >= prev, `PQ decreased at signal ${s}`);
        prev = v;
    }
});

test('PQ clamps rather than returning NaN or a negative', () => {
    assert.equal(pqToNits(-0.5), 0);
    assert.ok(rel(pqToNits(1.5), 10000, 1e-9));
    assert.ok(Number.isNaN(pqToNits(NaN)));
});

// ── HLG ─────────────────────────────────────────────────────────────────────

test('BT.2408 HDR Reference White is 75% HLG', () => {
    // Same 203 cd/m² as the PQ case, reached through a completely different
    // curve and a different set of constants. Two independent paths landing on
    // one published number is what makes both trustworthy.
    assert.ok(rel(nitsToHlg(203, 1000), 0.75, 0.002),
        `203 cd/m² gave ${nitsToHlg(203, 1000).toFixed(5)} HLG, expected ≈0.75`);
    assert.equal(nitsToHlg(203, 1000).toFixed(2), '0.75');
});

test('HLG endpoints are black and nominal peak', () => {
    assert.ok(close(hlgToNits(0, 1000), 0, 1e-9));
    assert.ok(rel(hlgToNits(1, 1000), 1000, 1e-6));
});

test('the HLG OETF segments meet at 0.5', () => {
    // The curve is defined piecewise; a mismatched constant shows up as a step
    // exactly here and nowhere else.
    const lower = hlgToScene(0.5);
    const upper = hlgToScene(0.5000001);
    assert.ok(close(lower, 1 / 12, 1e-9), `lower segment gave ${lower} at 0.5, expected 1/12`);
    assert.ok(close(lower, upper, 1e-6), 'the HLG curve steps at its breakpoint');
});

test('HLG system gamma follows the peak', () => {
    assert.ok(close(hlgSystemGamma(1000), 1.2, 1e-12));
    assert.ok(close(hlgSystemGamma(10000), 1.62, 1e-12));   // +0.42 per decade
    assert.ok(hlgSystemGamma(500) < 1.2);
});

test('HLG round-trips at every peak it is offered at', () => {
    for (const peak of [1000, 2000, 4000]) {
        for (const n of [0.1, 1, 100, 203, 500]) {
            if (n > peak) continue;
            assert.ok(rel(hlgToNits(nitsToHlg(n, peak), peak), n, 1e-6),
                `round trip failed at ${n} cd/m², peak ${peak}`);
        }
    }
});

test('HLG is monotonic', () => {
    let prev = -1;
    for (let s = 0; s <= 1.0001; s += 0.01) {
        const v = hlgToNits(s, 1000);
        assert.ok(v >= prev, `HLG decreased at signal ${s}`);
        prev = v;
    }
});

// ── Levels ──────────────────────────────────────────────────────────────────

test('narrow range matches the published code values', () => {
    assert.deepEqual(NARROW_RANGE[8], { black: 16, white: 235 });
    assert.deepEqual(NARROW_RANGE[10], { black: 64, white: 940 });
    assert.deepEqual(NARROW_RANGE[12], { black: 256, white: 3760 });
});

test('video levels put black at code 64 and white at 940', () => {
    assert.ok(close(toSignal(64 / 1023, 'video'), 0, 1e-12));
    assert.ok(close(toSignal(940 / 1023, 'video'), 1, 1e-12));
});

test('data levels are the identity', () => {
    for (const n of [0, 0.25, 0.5, 1]) assert.equal(toSignal(n, 'data'), n);
});

test('video levels expose footroom and headroom as signal outside 0–1', () => {
    // Code 0 is *below* black in video levels. Clamping it to 0 would hide a
    // crushed-blacks error, which is precisely what the scope is for.
    assert.ok(toSignal(0, 'video') < 0, 'code 0 must read below 0% in video levels');
    assert.ok(toSignal(1, 'video') > 1, 'code 1023 must read above 100% in video levels');
});

test('toSignal and fromSignal are inverses', () => {
    for (const levels of ['data', 'video']) {
        for (const n of [0, 0.1, 0.5, 0.9, 1]) {
            assert.ok(close(fromSignal(toSignal(n, levels), levels), n, 1e-12),
                `${levels} round trip failed at ${n}`);
        }
    }
});

test('code value ignores levels — a code is a code', () => {
    // The classic confusion this guards: percent changes with levels, code
    // value does not.
    assert.ok(close(codeValue(0.5, 10), 511.5));
    assert.equal(scaleValue(0.5, 'cv10', { levels: 'data' }),
        scaleValue(0.5, 'cv10', { levels: 'video' }));
});

test('percent does change with levels', () => {
    const atBlack = 64 / 1023;
    assert.ok(close(scaleValue(atBlack, 'percent', { levels: 'video' }), 0, 1e-9));
    assert.ok(scaleValue(atBlack, 'percent', { levels: 'data' }) > 6,
        'in data levels, code 64 is ~6.3%, not 0%');
});

// ── The scales as a set ─────────────────────────────────────────────────────

test('IRE is not offered', () => {
    // A deliberate omission, not an oversight. Resolve does not list it; it is a
    // legacy analogue-composite unit and offering it signals the opposite of
    // expertise.
    for (const s of SCOPE_SCALES) {
        assert.doesNotMatch(s.label, /IRE/i, `${s.id} offers IRE`);
        assert.doesNotMatch(s.unit, /IRE/i, `${s.id} uses IRE as a unit`);
    }
});

test('the default scale is 10-bit code value', () => {
    assert.equal(SCOPE_SCALES[0].id, 'cv10');
    assert.equal(getScale('nonexistent').id, 'cv10', 'an unknown id must fall back, not throw');
});

test('every scale has a unit, a label and a description', () => {
    for (const s of SCOPE_SCALES) {
        assert.ok(s.unit, `${s.id} has no unit`);
        assert.ok(s.label, `${s.id} has no label`);
        assert.ok(s.describe({ levels: 'data', peakNits: 1000 }), `${s.id} has no description`);
    }
});

test('every scale round-trips value and position', () => {
    for (const s of SCOPE_SCALES) {
        for (const norm of [0.1, 0.35, 0.58, 0.9]) {
            const ctx = { levels: 'data', peakNits: 1000 };
            const back = s.position(s.value(norm, ctx), ctx);
            assert.ok(close(back, norm, 1e-6), `${s.id} did not round-trip at ${norm} (got ${back})`);
        }
    }
});

test('mV runs 0 to 700 across black to white', () => {
    assert.ok(close(scaleValue(0, 'mv', { levels: 'data' }), 0));
    assert.ok(close(scaleValue(1, 'mv', { levels: 'data' }), 700));
    assert.ok(close(scaleValue(64 / 1023, 'mv', { levels: 'video' }), 0, 1e-9));
    assert.ok(close(scaleValue(940 / 1023, 'mv', { levels: 'video' }), 700, 1e-9));
});

// ── Ticks ───────────────────────────────────────────────────────────────────

test('no tick is drawn off the axis', () => {
    // A tick outside 0–1 gets clamped onto the edge, where it reads as a real
    // measurement at the wrong place.
    for (const s of SCOPE_SCALES) {
        for (const levels of ['data', 'video']) {
            for (const t of scaleTicks(s.id, { levels, peakNits: 1000 })) {
                assert.ok(t.at >= 0 && t.at <= 1, `${s.id}/${levels} tick "${t.label}" at ${t.at}`);
                assert.ok(t.label, `${s.id}/${levels} has an unlabelled tick`);
            }
        }
    }
});

test('ticks are unique and ordered by position after de-duplication', () => {
    for (const s of SCOPE_SCALES) {
        const ticks = scaleTicks(s.id, { levels: 'video', peakNits: 1000 });
        const at = ticks.map((t) => t.at);
        assert.equal(new Set(at.map((v) => v.toFixed(6))).size, at.length,
            `${s.id} draws two ticks at the same height`);
    }
});

test('video levels expose the footroom and headroom ticks that data levels do not', () => {
    const video = scaleTicks('percent', { levels: 'video' }).map((t) => t.label);
    const data = scaleTicks('percent', { levels: 'data' }).map((t) => t.label);
    assert.ok(video.includes('-7'), 'video levels should show sub-black');
    assert.ok(video.includes('109'), 'video levels should show super-white');
    assert.ok(!data.includes('-7'), 'data levels have no footroom to show');
});

test('both nit scales mark 203 and mark it as the reference', () => {
    for (const id of ['nits-pq', 'nits-hlg']) {
        const ref = scaleTicks(id, { levels: 'data', peakNits: 1000 })
            .find((t) => t.key === HDR_REFERENCE_WHITE_NITS);
        assert.ok(ref, `${id} has no 203 cd/m² reference line`);
        assert.ok(ref.emphasis, `${id}'s reference line is not emphasised`);
        assert.match(ref.label, /203/);
    }
});

test('the PQ 203 tick lands at 58% of the axis', () => {
    const ref = scaleTicks('nits-pq', { levels: 'data' }).find((t) => t.key === 203);
    assert.ok(rel(ref.at, 0.58, 0.01), `203 cd/m² drew at ${ref.at.toFixed(4)}, expected ≈0.58`);
});

test('the HLG 203 tick lands at 75% of the axis', () => {
    const ref = scaleTicks('nits-hlg', { levels: 'data', peakNits: 1000 }).find((t) => t.key === 203);
    assert.ok(rel(ref.at, 0.75, 0.01), `203 cd/m² drew at ${ref.at.toFixed(4)}, expected ≈0.75`);
});

test('HLG ticks above the nominal peak are dropped, not clamped', () => {
    const ticks = scaleTicks('nits-hlg', { levels: 'data', peakNits: 400 });
    assert.ok(!ticks.some((t) => t.key > 400), 'a tick above peak would sit on the top edge and lie');
});

// ── The caption ─────────────────────────────────────────────────────────────

test('the measurement is always described', () => {
    for (const s of SCOPE_SCALES) {
        for (const levels of ['data', 'video']) {
            for (const transformed of [true, false]) {
                const d = describeMeasurement(s.id, { levels, transformed, peakNits: 1000 });
                assert.ok(d.unit && d.label && d.levels && d.detail && d.measuredAt,
                    `${s.id}/${levels}/${transformed} left part of the caption blank`);
            }
        }
    }
});

test('the caption states which side of the viewer transform it measured', () => {
    assert.match(describeMeasurement('cv10', { transformed: true }).measuredAt, /after/);
    assert.match(describeMeasurement('cv10', { transformed: false }).measuredAt, /before/);
});

test('a nit scale on untransformed pixels is flagged, not silently printed', () => {
    // PQ and HLG are display encodings. Applied to source pixels that were never
    // encoded that way, the output is a number with no meaning — and a
    // confident-looking one.
    for (const id of ['nits-pq', 'nits-hlg']) {
        assert.ok(describeMeasurement(id, { transformed: false }).warn,
            `${id} does not warn when measuring before the transform`);
        assert.equal(describeMeasurement(id, { transformed: true }).warn, null,
            `${id} warns when it should not`);
    }
});

test('code and percent scales never warn — they are not interpretations', () => {
    for (const id of ['cv10', 'cv12', 'percent', 'mv']) {
        for (const transformed of [true, false]) {
            assert.equal(describeMeasurement(id, { transformed }).warn, null);
        }
    }
});

test('LEVELS carries a hint for each entry, since the distinction is the confusing part', () => {
    assert.equal(LEVELS.length, 2);
    for (const l of LEVELS) assert.ok(l.hint && l.label && l.id);
});

test('nit labels stay short enough for an axis', () => {
    assert.equal(formatNits(203), '203');
    assert.equal(formatNits(1000), '1k');
    assert.equal(formatNits(10000), '10k');
    assert.equal(formatNits(0.01), '0.01');
    assert.equal(formatNits(NaN), '—');
});

// ── LogC assist ─────────────────────────────────────────────────────────────

test('the log-assist curve fixes both ends', () => {
    assert.ok(close(logAssistPos(0), 0, 1e-12));
    assert.ok(close(logAssistPos(1), 1, 1e-12));
});

test('the log-assist curve opens the shadows', () => {
    // That is the entire point of the mode: a value low in the range should
    // plot much higher than it would linearly.
    assert.ok(logAssistPos(0.05) > 0.4, `0.05 plotted at ${logAssistPos(0.05)}`);
    assert.ok(logAssistPos(0.5) > 0.5, 'the curve must be above the diagonal');
});

test('the log-assist curve is invertible', () => {
    // The graticule is placed with the forward curve and the hover readout is
    // resolved with the inverse. If they disagree, the line and the number
    // under the cursor say different things at the same height.
    for (const v of [0, 0.01, 0.18, 0.5, 0.9, 1]) {
        assert.ok(close(logAssistInv(logAssistPos(v)), v, 1e-9), `round trip failed at ${v}`);
    }
});

test('the log-assist curve is monotonic', () => {
    let prev = -1;
    for (let v = 0; v <= 1.0001; v += 0.02) {
        const p = logAssistPos(v);
        assert.ok(p >= prev, `not monotonic at ${v}`);
        prev = p;
    }
});
