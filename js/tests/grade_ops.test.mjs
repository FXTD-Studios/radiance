/**
 * White balance, the ACEScct grade space, Luma Mix, the curve table read and
 * the whole-grade function the exporters are built on.
 *
 * Each of these was a separate finding in the viewer review: Temperature and
 * Tint added colour instead of balancing it, ACEScct mode changed the picture
 * with nothing graded, Luma Mix at 0 cancelled exposure, and the curve table
 * was read half a texel off. The maths now lives in js/radiance_grade.js, so
 * it is tested here in Node; grade_gpu.test.mjs checks the emitted GLSL
 * against the same functions and grade_render.test.mjs checks the real
 * composite shader.
 *
 * Run: node --test js/tests/grade_ops.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    applyWhiteBalance, whiteBalanceGains, whiteBalanceForNeutral, applyLumaMix,
    linToACEScct, acescctToLin, toAP1, fromAP1, TO_AP1, FROM_AP1,
    sampleCurveTable, applyCurves, gradePixelFull, luminance,
} from '../radiance_grade.js';

const close = (a, b, eps) => Math.abs(a - b) <= eps;
const closeAll = (a, b, eps) => a.every((v, i) => close(v, b[i], eps));
const neutral = (rgb, eps = 1e-12) => close(rgb[0], rgb[1], eps) && close(rgb[1], rgb[2], eps);

// ── white balance ───────────────────────────────────────────────────────────

test('white balance at zero is exactly the identity', () => {
    for (const px of [[0, 0, 0], [0.18, 0.5, 2], [-0.05, 0.2, 0.2]]) {
        assert.equal(applyWhiteBalance(px, 0, 0), px);
    }
    assert.deepEqual(whiteBalanceGains(0, 0), [1, 1, 1]);
});

test('white balance is a gain: black stays black and negatives are not clamped', () => {
    // The additive version turned black dark red at Temperature 0.1 and
    // clamped every negative to zero for any non-zero Tint.
    assert.deepEqual(applyWhiteBalance([0, 0, 0], 0.1, 0), [0, 0, 0]);
    const out = applyWhiteBalance([-0.05, 0.2, 0.2], 0, 0.001);
    assert.ok(out[0] < 0, `a negative was clamped: ${out}`);
});

test('white balance keeps the luminance of a neutral', () => {
    for (const [t, n] of [[0.5, 0], [-1, 0.3], [2, -2], [0, 1]]) {
        const out = applyWhiteBalance([0.18, 0.18, 0.18], t, n);
        assert.ok(close(luminance(...out), 0.18, 1e-15), `t=${t} n=${n} moved grey's luminance`);
    }
});

test('Temperature warms toward red and cools toward blue; Tint moves green against magenta', () => {
    const warm = whiteBalanceGains(1, 0);
    assert.ok(warm[0] > warm[1] && warm[1] > warm[2]);
    assert.ok(close(warm[0] / warm[2], 2, 1e-12), 'one unit of Temperature is one stop of red against blue');
    const magenta = whiteBalanceGains(0, 1);
    assert.ok(close(magenta[0], magenta[2], 1e-15) && magenta[1] < magenta[0]);
    assert.ok(close(Math.sqrt(magenta[0] * magenta[2]) / magenta[1], 2, 1e-12), 'one unit of Tint is one stop');
});

test('the eyedropper solve makes the picked pixel neutral, at any level', () => {
    // The old pick overcorrected (0.6, 0.5, 0.4) to (0.28, 0.5, 0.72).
    for (const px of [[0.6, 0.5, 0.4], [0.06, 0.05, 0.04], [2.0, 1.1, 0.7], [0.3, 0.45, 0.6]]) {
        const wb = whiteBalanceForNeutral(px);
        const out = applyWhiteBalance(px, wb.temperature, wb.tint);
        assert.ok(neutral(out), `${px} → ${out}`);
    }
    const a = whiteBalanceForNeutral([0.6, 0.5, 0.4]);
    const b = whiteBalanceForNeutral([0.06, 0.05, 0.04]);
    assert.ok(close(a.temperature, b.temperature, 1e-12) && close(a.tint, b.tint, 1e-12),
        'a tenth of the level must give the same correction');
    assert.equal(whiteBalanceForNeutral([0.5, 0, 0.5]), null, 'a zero channel has no neutral solve');
});

// ── ACEScct ─────────────────────────────────────────────────────────────────

// OpenColorIO 2.6, studio-config-v4.0.0_aces-v2.0_ocio-v2.5, processor columns.
const OCIO_TO_AP1 = {
    0: [[0.613097429276, 0.339523136616, 0.047379452735], [0.070193722844, 0.916353881359, 0.013452398591], [0.020615592599, 0.109569773078, 0.869814634323]],
    2: [[1.451439261436, -0.236510753632, -0.21492856741], [-0.07655377686, 1.176229715347, -0.099675923586], [0.008316148072, -0.006032449659, 0.99771630764]],
    3: [[0.974895000458, 0.019599108025, 0.005505913403], [0.002179562813, 0.995535492897, 0.002284968272], [0.004797239788, 0.024532016367, 0.970670759678]],
    4: [[0.735797941685, 0.212166488171, 0.052035599947], [0.047179885209, 0.938045680523, 0.014774413779], [0.003563664621, 0.041141886264, 0.955294430256]],
};
const OCIO_FROM_AP1_709 = [[1.705050945282, -0.621792137623, -0.083258874714], [-0.130256414413, 1.140804767609, -0.010548318736], [-0.024003356695, -0.128968968987, 1.152972340584]];

test('the AP1 matrices match OpenColorIO to 1e-6', () => {
    for (const [g, ref] of Object.entries(OCIO_TO_AP1)) {
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
            assert.ok(close(TO_AP1[g][i][j], ref[i][j], 1e-6), `gamut ${g} [${i}][${j}]`);
        }
    }
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
        assert.ok(close(FROM_AP1[0][i][j], OCIO_FROM_AP1_709[i][j], 1e-6));
    }
});

test('each from-AP1 matrix is the inverse of its to-AP1 matrix', () => {
    for (let g = 0; g < 5; g++) {
        for (const px of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.2, 0.1]]) {
            assert.ok(closeAll(fromAP1(toAP1(px, g), g), px, 1e-14), `gamut ${g}`);
        }
    }
    // An ACEScg source is not converted at all.
    const px = [0.5, 0.2, 0.1];
    assert.equal(toAP1(px, 1), px);
    assert.equal(fromAP1(px, 1), px);
});

test('the ACEScct curve matches the spec and inverts', () => {
    // S-2016-001, and OpenColorIO's float32 output: 0 → 0.0729, 0.18 → 0.4136,
    // 1.0 → 0.5548.
    assert.ok(close(linToACEScct(0), 0.0729055341958355, 1e-15));
    assert.ok(close(linToACEScct(0.18), 0.4135878086, 1e-6));
    assert.ok(close(linToACEScct(1), 0.5547952652, 1e-6));
    for (const x of [-0.01, 0, 0.001, 0.0078125, 0.18, 1, 4, 100]) {
        assert.ok(close(acescctToLin(linToACEScct(x)), x, 1e-12 * Math.max(1, x)), `x=${x}`);
    }
});

test('ACEScct mode with every control at neutral leaves the picture alone', () => {
    // (0.5, 0.2, 0.1) became (0.474, 0.206, 0.117) with nothing graded.
    for (let gamut = 0; gamut < 5; gamut++) {
        for (const px of [[0.5, 0.2, 0.1], [0.18, 0.18, 0.18], [4, 2, 0.5], [0.001, 0.003, 0.0]]) {
            const out = gradePixelFull(px, { colorScience: 1 }, { gamut });
            assert.ok(closeAll(out, px, 1e-5 * Math.max(1, ...px)), `gamut ${gamut}: ${px} → ${out}`);
        }
    }
});

test('the ACEScct grade happens in AP1 log space', () => {
    // Offset 0.1 in ACEScct is about a 2^(0.1·17.52) = 3.37x gain on a mid grey.
    const out = gradePixelFull([0.18, 0.18, 0.18], { colorScience: 1, offset: [0.1, 0.1, 0.1] });
    assert.ok(close(out[1] / 0.18, Math.pow(2, 0.1 * 17.52), 1e-9));
});

// ── luma mix ────────────────────────────────────────────────────────────────

test('Luma Mix 0 keeps the luminance after the primaries, not the source luminance', () => {
    // At 0 the old version restored the ungraded luminance, so Exposure and
    // Gain stopped changing brightness.
    const src = [0.2, 0.18, 0.1];
    const y0 = luminance(...src);
    const brighter = gradePixelFull(src, { exposure: 1, lumaMix: 0 });
    assert.ok(close(luminance(...brighter), 2 * y0, 1e-12), 'exposure +1 must still double the luminance');
    const gained = gradePixelFull(src, { gain: [1.5, 1.5, 1.5], lumaMix: 0 });
    assert.ok(close(luminance(...gained), 1.5 * y0, 1e-12));
    const contrasted = gradePixelFull(src, { contrast: 1.5, lumaMix: 0 });
    assert.ok(!close(luminance(...contrasted), y0, 1e-6), 'contrast must still change the luminance');
});

test('Luma Mix 0 stops saturation and hue changing brightness', () => {
    const src = [0.4, 0.1, 0.05];
    const ref = luminance(...gradePixelFull(src, { exposure: 0.5 }));
    for (const s of [{ saturation: 2 }, { saturation: 0 }, { hueShift: 120 }]) {
        const out = gradePixelFull(src, { exposure: 0.5, lumaMix: 0, ...s });
        assert.ok(close(luminance(...out), ref, 1e-9), `${JSON.stringify(s)} changed luminance`);
    }
    // And at 1 it is a no-op.
    assert.deepEqual(applyLumaMix([0.3, 0.2, 0.1], 5, 1), [0.3, 0.2, 0.1]);
});

// ── curves ──────────────────────────────────────────────────────────────────

const identityTable = () => {
    const t = new Float32Array(1024);
    for (let i = 0; i < 256; i++) { t[i * 4] = t[i * 4 + 1] = t[i * 4 + 2] = i / 255; t[i * 4 + 3] = 1; }
    return t;
};

test('the curve table is read at texel centres', () => {
    // Input c reads entry c * 255. The shader read it at c, which lands on
    // entry c * 256 - 0.5: half a texel low at 0, half a texel high at 1.
    const t = identityTable();
    for (const c of [0, 0.02, 0.1, 0.5, 0.9, 1]) {
        assert.ok(close(sampleCurveTable(t, c, 0), c, 1e-6), `identity curve read ${c} as ${sampleCurveTable(t, c, 0)}`);
    }
    assert.deepEqual(applyCurves([0.25, 0.5, 4], t).map((v) => +v.toFixed(6)), [0.25, 0.5, 4]);
});

// ── the whole grade ─────────────────────────────────────────────────────────

test('an identity curve leaves negative scene values alone', () => {
    // The curve clamp read entry 0 for anything below zero, so once the curve
    // editor had been opened every negative pixel came out 0.
    assert.ok(closeAll(applyCurves([-0.1, -2, 0.5], identityTable()), [-0.1, -2, 0.5], 1e-6));
});

test('the whole grade at neutral is exactly the identity', () => {
    for (const px of [[0, 0, 0], [0.18, 0.18, 0.18], [4, 2, 0.5], [-0.1, 0.3, 0.2]]) {
        assert.deepEqual(gradePixelFull(px, {}), px);
        assert.deepEqual(gradePixelFull(px, { curveTable: identityTable(), contrast: 1, pivot: 0.5 }).map((v) => +v.toFixed(6)),
            px.map((v) => +v.toFixed(6)));
    }
});
