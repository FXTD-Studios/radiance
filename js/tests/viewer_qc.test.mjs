import test from 'node:test';
import assert from 'node:assert/strict';
import {
    Status, analyzeFrame, analyzeFlicker, analyzeSequence, overallStatus, buildReport, meanLuma, DEFAULTS,
} from '../radiance_qc.js';

const W = 8, H = 8;
function frame(fill, channels = 3, mutate) {
    const data = new Float32Array(W * H * channels).fill(fill);
    if (mutate) mutate(data, channels);
    return { data, width: W, height: H, channels };
}
const byId = (rs, id) => rs.find((r) => r.id === id);

// ── spatial ────────────────────────────────────────────────────────────────
test('a clean mid-grey frame passes every applicable check', () => {
    const rs = analyzeFrame(frame(0.5));
    for (const id of ['nan_inf', 'over_range', 'under_range']) assert.equal(byId(rs, id).status, Status.PASS, id);
    assert.equal(byId(rs, 'alpha_range').status, Status.NOT_APPLICABLE);   // RGB: no alpha to test
    assert.equal(overallStatus(rs), Status.PASS);
});

test('NaN and Inf are a FAIL with the count', () => {
    const rs = analyzeFrame(frame(0.5, 3, (d) => { d[0] = NaN; d[5] = Infinity; d[9] = -Infinity; }));
    const r = byId(rs, 'nan_inf');
    assert.equal(r.status, Status.FAIL);
    assert.equal(r.value, 3);
    assert.equal(overallStatus(rs), Status.FAIL);
});

test('clipping is graded by how much of the frame is over the limit', () => {
    const some = analyzeFrame(frame(0.5, 3, (d) => { for (let i = 0; i < 30; i++) d[i] = 2; }));   // 30 of 192
    assert.equal(byId(some, 'over_range').status, Status.FAIL);
    const few = analyzeFrame(frame(0.5, 3, (d) => { d[0] = 2; }));                                   // 1 of 192
    assert.equal(byId(few, 'over_range').status, Status.WARN);
    assert.equal(byId(few, 'over_range').max, 2);
});

test('the over-range limit is a parameter, so HDR content can be judged against its own range', () => {
    const hdr = frame(4);
    assert.equal(byId(analyzeFrame(hdr), 'over_range').status, Status.FAIL);
    assert.equal(byId(analyzeFrame(hdr, { over: { limit: 10 } }), 'over_range').status, Status.PASS);
});

test('negative values are reported as under-range', () => {
    const rs = analyzeFrame(frame(0.2, 3, (d) => { d[3] = -0.5; }));
    assert.equal(byId(rs, 'under_range').status, Status.WARN);
    assert.equal(byId(rs, 'under_range').min, -0.5);
});

test('alpha outside 0..1 fails; straight alpha fails the premultiplied test only', () => {
    const bad = analyzeFrame(frame(0.3, 4, (d) => { for (let i = 3; i < d.length; i += 4) d[i] = 1.5; }));
    assert.equal(byId(bad, 'alpha_range').status, Status.FAIL);
    // colour 0.9 over alpha 0.5: not a valid premultiplied pixel
    const straight = analyzeFrame(frame(0.9, 4, (d) => { for (let i = 3; i < d.length; i += 4) d[i] = 0.5; }));
    assert.equal(byId(straight, 'alpha_range').status, Status.PASS);
    assert.equal(byId(straight, 'alpha_premultiplied').status, Status.FAIL);
    const ok = analyzeFrame(frame(0.2, 4, (d) => { for (let i = 3; i < d.length; i += 4) d[i] = 0.5; }));
    assert.equal(byId(ok, 'alpha_premultiplied').status, Status.PASS);
});

test('missing or malformed data is NOT_RUN, never PASS', () => {
    for (const bad of [null, undefined, {}, { data: new Float32Array(3), width: 8, height: 8, channels: 3 }]) {
        const rs = analyzeFrame(bad);
        assert.ok(rs.every((r) => r.status === Status.NOT_RUN));
        assert.equal(overallStatus(rs), Status.NOT_RUN);
    }
});

test('every result names its measure, domain and limits of what it can say', () => {
    for (const r of analyzeFrame(frame(0.5, 4, (d) => { for (let i = 3; i < d.length; i += 4) d[i] = 1; }))) {
        assert.ok(r.domain && /source/.test(r.domain), r.id);
        if (r.status !== Status.NOT_APPLICABLE) { assert.ok(r.metric, r.id); assert.ok(r.limitations, r.id); }
    }
});

// ── flicker ────────────────────────────────────────────────────────────────
const steady = (n, v = 0.4) => Array.from({ length: n }, (_, i) => v + 0.001 * Math.sin(i));

test('a steady sequence has no flicker', () => {
    const r = analyzeFlicker(steady(40));
    assert.equal(r.status, Status.PASS);
    assert.deepEqual(r.frames, []);
});

test('an injected brightness pop is found on the right frame', () => {
    const m = steady(40);
    m[17] = 0.55;                                   // a one-frame flash
    const r = analyzeFlicker(m);
    assert.equal(r.status, Status.WARN);
    assert.ok(r.frames.includes(17));               // the jump up
    assert.ok(r.frames.includes(18));               // and the return
});

test('a perfectly static sequence does not divide by zero or flag noise', () => {
    const r = analyzeFlicker(Array(30).fill(0.4));
    assert.equal(r.status, Status.PASS);
});

test('too few frames is INDETERMINATE, one frame is NOT_APPLICABLE', () => {
    assert.equal(analyzeFlicker([0.4, 0.4, 0.9]).status, Status.INDETERMINATE);
    assert.equal(analyzeFlicker([0.4]).status, Status.NOT_APPLICABLE);
    assert.equal(analyzeFlicker([]).status, Status.NOT_APPLICABLE);
});

test('unmeasurable frames are skipped, and too many of them is INDETERMINATE', () => {
    const m = steady(12);
    for (let i = 3; i <= 10; i++) m[i] = null;                 // 4 measurable frames left
    assert.equal(analyzeFlicker(m).status, Status.INDETERMINATE);
    const few = steady(12); few[4] = null;                     // one missing is fine
    assert.equal(analyzeFlicker(few).status, Status.PASS);
});

test('slow drift is not flicker (and the limit says so)', () => {
    const m = Array.from({ length: 60 }, (_, i) => 0.2 + i * 0.005);
    const r = analyzeFlicker(m);
    assert.equal(r.status, Status.PASS);
    assert.match(r.limitations, /slow drift/);
});

// ── overall ────────────────────────────────────────────────────────────────
test('overall status never defaults to PASS', () => {
    assert.equal(overallStatus([]), Status.NOT_APPLICABLE);
    assert.equal(overallStatus([{ status: Status.NOT_RUN }]), Status.NOT_RUN);
    assert.equal(overallStatus([{ status: Status.INDETERMINATE }]), Status.INDETERMINATE);
    assert.equal(overallStatus([{ status: Status.PASS }, { status: Status.NOT_RUN }]), Status.INDETERMINATE);
    assert.equal(overallStatus([{ status: Status.PASS }, { status: Status.NOT_APPLICABLE }]), Status.PASS);
    assert.equal(overallStatus([{ status: Status.PASS }, { status: Status.WARN }]), Status.WARN);
    assert.equal(overallStatus([{ status: Status.WARN }, { status: Status.FAIL }, { status: Status.NOT_RUN }]), Status.FAIL);
});

// ── sequence ───────────────────────────────────────────────────────────────
const noYield = { yieldFn: async () => {} };

test('a sequence run finds the bad frame, names it, and fails overall', async () => {
    const getFrame = (i) => frame(0.4, 3, i === 6 ? (d) => { d[0] = NaN; } : undefined);
    const a = await analyzeSequence(10, getFrame, {}, noYield);
    assert.equal(a.complete, true);
    const nan = byId(a.checks, 'nan_inf');
    assert.equal(nan.status, Status.FAIL);
    assert.deepEqual(nan.frames, [6]);
    assert.equal(a.overall, Status.FAIL);
});

test('a clean sequence passes, and the flicker check ran', async () => {
    const a = await analyzeSequence(12, () => frame(0.4), {}, noYield);
    assert.equal(a.overall, Status.PASS);
    assert.equal(byId(a.checks, 'flicker').status, Status.PASS);
});

test('flicker injected into a sequence is reported with its frames', async () => {
    const a = await analyzeSequence(30, (i) => frame(i === 12 ? 0.8 : 0.4), {}, noYield);
    const f = byId(a.checks, 'flicker');
    assert.equal(f.status, Status.WARN);
    assert.ok(f.frames.includes(12));
});

test('a frame that cannot be loaded keeps the run from passing', async () => {
    const a = await analyzeSequence(12, (i) => (i === 5 ? null : frame(0.4)), {}, noYield);
    assert.notEqual(a.overall, Status.PASS);
    assert.equal(byId(a.checks, 'nan_inf').status, Status.INDETERMINATE);
    assert.match(byId(a.checks, 'nan_inf').reason, /11 of 12/);
});

test('getFrame that throws is treated as a missing frame, not a crash', async () => {
    const a = await analyzeSequence(8, (i) => { if (i === 2) throw new Error('decode'); return frame(0.4); }, {}, noYield);
    assert.notEqual(a.overall, Status.PASS);
});

test('cancel stops between frames and the partial result is not a PASS', async () => {
    const ac = new AbortController();
    let seen = 0;
    const a = await analyzeSequence(100, () => { seen++; if (seen === 10) ac.abort(); return frame(0.4); }, {},
        { ...noYield, signal: ac.signal });
    assert.equal(a.complete, false);
    assert.equal(a.framesDone, 10);
    assert.equal(seen, 10);
    assert.notEqual(a.overall, Status.PASS);
});

test('a cancelled run resumes without redoing frames', async () => {
    const ac = new AbortController();
    const calls = [];
    const get = (i) => { calls.push(i); if (i === 4) ac.abort(); return frame(0.4); };
    const first = await analyzeSequence(10, get, {}, { ...noYield, signal: ac.signal });
    assert.equal(first.framesDone, 5);
    const second = await analyzeSequence(10, get, {}, { ...noYield, startAt: first.framesDone, previous: first.state });
    assert.equal(second.complete, true);
    assert.deepEqual(calls, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);      // each frame fetched once
    assert.equal(second.overall, Status.PASS);
});

test('frames are fetched one at a time (memory does not grow with the length)', async () => {
    let inFlight = 0, peak = 0;
    const get = async () => { inFlight++; peak = Math.max(peak, inFlight); await null; inFlight--; return frame(0.4); };
    await analyzeSequence(50, get, {}, noYield);
    assert.equal(peak, 1);
});

test('progress is reported for every frame', async () => {
    const seen = [];
    await analyzeSequence(5, () => frame(0.4), {}, { ...noYield, onProgress: (d, t) => seen.push([d, t]) });
    assert.deepEqual(seen, [[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]]);
});

// ── report ─────────────────────────────────────────────────────────────────
test('the report is plain JSON with inputs, options and a note on what NOT_RUN means', async () => {
    const a = await analyzeSequence(6, () => frame(0.4), {}, noYield);
    const rep = buildReport(a, { file: 'shot.exr', frames: [1001, 1006] });
    const round = JSON.parse(JSON.stringify(rep));
    assert.equal(round.schema, 'radiance.qc.report');
    assert.equal(round.overall, 'PASS');
    assert.deepEqual(round.inputs, { file: 'shot.exr', frames: [1001, 1006] });
    assert.deepEqual(round.options.over, DEFAULTS.over);
    assert.match(round.notes, /NOT_RUN and INDETERMINATE/);
    assert.ok(round.checks.every((c) => c.status && c.id && c.domain));
});

test('the same input gives the same report', async () => {
    const run = () => analyzeSequence(9, (i) => frame(0.4 + (i === 3 ? 0.3 : 0)), {}, noYield).then((a) => JSON.stringify(buildReport(a, {})));
    assert.equal(await run(), await run());
});

test('meanLuma ignores non-finite samples', () => {
    assert.ok(Math.abs(meanLuma(frame(0.5)) - 0.5) < 1e-6);
    assert.equal(meanLuma(frame(NaN)), null);
    assert.equal(meanLuma(null), null);
});
