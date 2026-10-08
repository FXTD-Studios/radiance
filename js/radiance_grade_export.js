/**
 * Grade exports: one ASC CDL writer and one .cube writer.
 *
 * The viewer had three copies of this, in radiance_viewer.js and
 * radiance_viewer_export.js, and none matched the screen: one CDL wrote gamma
 * as Power the wrong way round and lift as Offset, one .cube dropped exposure,
 * offset, white balance, curves and log wheels, and every .cube was linear over
 * 0 to 1, so anything brighter than diffuse white was clipped.
 *
 * Both writers here are built on gradePixelFull() from radiance_grade.js, the
 * CPU form of the composite shader's grade in the shader's own order, and both
 * say what they could not carry. Nothing in this file touches the DOM, so the
 * exports are tested in Node against the same function the shader is tested
 * against.
 */

import {
    gradePixelFull, whiteBalanceGains, linToACEScct, acescctToLin, toAP1, fromAP1,
    isIdentityCurveTable, isIdentitySecondaryTable,
    PIVOT_DEFAULT, PIVOT_FLOOR, GAMMA_FLOOR, CONTRAST_MIN, CONTRAST_MAX, EXPOSURE_MIN, EXPOSURE_MAX,
} from './radiance_grade.js';

const v3 = (v, d) => (Array.isArray(v) && v.length >= 3 ? v.map(Number) : [d, d, d]);
const nz3 = (v, d) => v3(v, d).some((x) => Math.abs(x - d) > 1e-9);

/** Grade controls that change the picture but have no place in a per-pixel transform. */
function spatialAndEffects(s) {
    const out = [];
    if (s.midDetail) out.push('Mid Detail');
    if (s.maskActive) out.push('Power window mask');
    if (s.qualifierActive) out.push('Qualifier');
    if (s.creativeLut) out.push('Loaded creative LUT');
    if (s.softClip) out.push('Soft Clip (applied with the view)');
    const fx = [['grain', 'Grain'], ['bloom', 'Bloom'], ['halation', 'Halation'], ['diffusion', 'Diffusion'],
        ['lensFringe', 'Fringe'], ['lensDistortion', 'Lens distortion'], ['vignetteIntensity', 'Vignette'],
        ['anamorphicStreaks', 'Anamorphic streaks'], ['denoise', 'Denoise']];
    for (const [k, label] of fx) if (s[k]) out.push(label);
    if (s.dofEnabled) out.push('Depth of field');
    return out;
}

// ── ASC CDL ─────────────────────────────────────────────────────────────────

/**
 * The grade as Slope / Offset / Power / Saturation, and what did not fit.
 *
 * ASC CDL is out = (in · slope + offset) ^ power, then saturation with BT.709
 * weights. The viewer's grade, in order, is exposure, white balance, offset,
 * lift, gain, gamma, contrast, ... saturation. So:
 *
 *   slope  = 2^exposure · white balance · gain    (both are gains before offset)
 *   offset = offset · gain
 *   power  = 1 / gamma                             (FIX-006: gamma is c^(1/γ))
 *
 * Contrast pivot·(c/pivot)^k folds in exactly, as power·k with slope and offset
 * scaled by pivot^((1−k)/(power·k)); printer lights are gains after the power
 * and fold the same way. Lift (luminance pivoted) and everything between the
 * power and the saturation that is not neutral cannot be expressed; they are
 * listed in `dropped`, and the CDL is still the closest SOP.
 *
 * In the ACEScct grade space the wheels act on ACEScct values, so the CDL is
 * an ACEScct CDL; exposure becomes an ACEScct offset, and the controls applied
 * in linear afterwards are listed as dropped.
 */
export function cdlFromGrade(state = {}) {
    const s = state;
    const dropped = [];
    const notes = [];
    const gain = v3(s.gain, 1);
    const gamma = v3(s.gamma, 1);
    const off = v3(s.offset, 0);
    const exposure = Math.min(Math.max(Number(s.exposure) || 0, EXPOSURE_MIN), EXPOSURE_MAX);
    let slope, offset;
    let power = gamma.map((g) => 1 / Math.max(g, GAMMA_FLOOR));
    let saturation = s.saturation ?? 1;
    const acescct = (s.colorScience ?? 0) === 1;

    if (nz3(s.lift, 0)) dropped.push('Lift (pivoted on luminance, not a CDL offset)');

    if (acescct) {
        notes.push('Apply in ACEScct (AP1): the grade wheels act on ACEScct values.');
        // A linear gain of 2^e is an ACEScct offset of e / 17.52 above the toe.
        const e = exposure / 17.52;
        slope = gain.slice();
        offset = off.map((o, i) => (o + e) * gain[i]);
        if (exposure) notes.push('Exposure is written as an ACEScct offset, exact above linear 0.0078.');
        if (s.temperature || s.tint) dropped.push('Temperature / Tint (applied in linear before ACEScct)');
        if ((s.contrast ?? 1) !== 1) dropped.push('Contrast (applied in linear after the wheels)');
        if (s.printerR || s.printerG || s.printerB) dropped.push('Printer lights (applied in linear)');
        if (saturation !== 1) { dropped.push('Saturation (applied in linear, not ACEScct)'); saturation = 1; }
    } else {
        const k = Math.pow(2, exposure);
        const wb = whiteBalanceGains(s.temperature || 0, s.tint || 0);
        slope = gain.map((g, i) => k * wb[i] * g);
        offset = off.map((o, i) => o * gain[i]);

        const kc = Math.min(Math.max(s.contrast ?? 1, CONTRAST_MIN), CONTRAST_MAX);
        if (kc !== 1) {
            if (kc > 0) {
                const p = Math.max(s.pivot ?? PIVOT_DEFAULT, PIVOT_FLOOR);
                const m = power.map((pw) => Math.pow(p, (1 - kc) / (pw * kc)));
                slope = slope.map((v, i) => v * m[i]);
                offset = offset.map((v, i) => v * m[i]);
                power = power.map((pw) => pw * kc);
            } else {
                dropped.push('Contrast 0 (flattens to the pivot)');
            }
        }
        const pl = [s.printerR, s.printerG, s.printerB].map((x) => Math.pow(2, (Number(x) || 0) / 50));
        if (pl.some((x) => x !== 1)) {
            slope = slope.map((v, i) => v * Math.pow(pl[i], 1 / power[i]));
            offset = offset.map((v, i) => v * Math.pow(pl[i], 1 / power[i]));
        }
    }

    // Between the power and the saturation, and after it.
    if (nz3(s.logShadow, 0) || nz3(s.logMidtone, 0) || nz3(s.logHighlight, 0)) dropped.push('Log wheels');
    if (s.shadows || s.highlights) dropped.push('Shadows / Highlights');
    if (s.colorBoost) dropped.push('Color Boost');
    if (s.curveTable && !isIdentityCurveTable(s.curveTable) && (s.curveMix ?? 1) > 0) dropped.push('Curves');
    const hueCurves = s.secondaryCurveTable && !isIdentitySecondaryTable(s.secondaryCurveTable)
        && (s.secondaryCurveMix ?? 0) > 0;
    if (hueCurves) dropped.push('Hue curves');
    if (s.hueShift) dropped.push('Hue shift');
    if ((s.lumaMix ?? 1) < 1 && ((s.saturation ?? 1) !== 1 || s.hueShift || hueCurves)) dropped.push('Luma Mix');
    dropped.push(...spatialAndEffects(s));

    return { slope, offset, power, saturation, dropped, notes, acescct };
}

/** ASC CDL out = (in · slope + offset) ^ power, then saturation; for tests and previews. */
export function applyCDL(rgb, { slope, offset, power, saturation }) {
    let c = rgb.map((v, i) => {
        const x = v * slope[i] + offset[i];
        return x > 0 ? Math.pow(x, power[i]) : x;
    });
    if (saturation !== 1) {
        const y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        c = c.map((v) => y + (v - y) * saturation);
    }
    return c;
}

const xmlEscape = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** The .cdl file. `dropped` is repeated in Description elements so it travels with the file. */
export function buildCDL(state = {}, { id = 'radiance_grade', date = new Date() } = {}) {
    const cdl = cdlFromGrade(state);
    const num = (a) => a.map((v) => (Number.isFinite(v) ? v : 0).toFixed(6)).join(' ');
    const desc = [
        `Radiance Viewer grade, exported ${date.toISOString()}`,
        ...cdl.notes,
        ...(cdl.dropped.length ? [`Not representable in CDL, left out: ${cdl.dropped.join(', ')}`] : []),
    ];
    const xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ColorDecisionList xmlns="urn:ASC:CDL:v1.01">',
        '  <ColorDecision>',
        `    <ColorCorrection id="${xmlEscape(id)}">`,
        ...desc.map((d) => `      <Description>${xmlEscape(d)}</Description>`),
        '      <SOPNode>',
        `        <Slope>${num(cdl.slope)}</Slope>`,
        `        <Offset>${num(cdl.offset)}</Offset>`,
        `        <Power>${num(cdl.power)}</Power>`,
        '      </SOPNode>',
        '      <SatNode>',
        `        <Saturation>${(cdl.saturation ?? 1).toFixed(6)}</Saturation>`,
        '      </SatNode>',
        '    </ColorCorrection>',
        '  </ColorDecision>',
        '</ColorDecisionList>',
        '',
    ].join('\n');
    return { xml, ...cdl, fileName: `radiance_grade_${date.getTime()}${cdl.acescct ? '_ACEScct' : ''}.cdl` };
}

// ── .cube ───────────────────────────────────────────────────────────────────

/** ACEScct code value 0 and 1 in scene-linear: the LUT's input range. */
export const CUBE_LINEAR_MIN = acescctToLin(0);
export const CUBE_LINEAR_MAX = acescctToLin(1);

const GAMUT_NAMES = ['Linear Rec.709 (sRGB)', 'ACEScg', 'ACES2065-1', 'Linear Rec.2020', 'Linear P3-D65'];

/** What a .cube of the grade leaves out. */
export function cubeNotBaked(state = {}) {
    return spatialAndEffects(state);
}

/**
 * The grade as a 3D LUT that takes and returns ACEScct (AP1).
 *
 * A .cube on scene-linear 0 to 1 cannot hold a value above 1, so a highlight
 * at 4.0 came back at 1.0. ACEScct code 0 to 1 covers scene-linear -0.0069 to
 * 222.9 at an even density per stop, which is what a lattice needs to
 * interpolate a grade made in linear light. Each lattice point is decoded to
 * linear AP1, taken to the gamut the grade was made in, graded by
 * gradePixelFull(), and encoded back. The input space, the domain and what was
 * not baked are written into the file.
 */
export function buildCubeLUT(state = {}, {
    size = 65, gamut = 0, inputTransform = 'None', date = new Date(), isLinear = true,
} = {}) {
    const N = Math.max(2, Math.round(size));
    const notBaked = cubeNotBaked(state);
    const working = GAMUT_NAMES[gamut] || GAMUT_NAMES[0];
    const lines = [
        `TITLE "Radiance grade, ACEScct AP1 in and out"`,
        `# Radiance Viewer grade, exported ${date.toISOString()}`,
        '# Input:  ACEScct code values, AP1 primaries. Domain 0 to 1 is scene-linear '
            + `${CUBE_LINEAR_MIN.toFixed(4)} to ${CUBE_LINEAR_MAX.toFixed(1)}.`,
        '# Output: ACEScct code values, AP1 primaries.',
        `# Graded on: scene-linear ${working}`
            + (inputTransform && inputTransform !== 'None' ? `, after the viewer input transform "${inputTransform}"` : '') + '.',
        `# To use on scene-linear ${working}: convert to ACEScct (AP1), apply this LUT, convert back.`,
        notBaked.length ? `# Not baked (not a per-pixel colour transform): ${notBaked.join(', ')}` : '# Everything in the grade is baked.',
        `LUT_3D_SIZE ${N}`,
        'DOMAIN_MIN 0.0 0.0 0.0',
        'DOMAIN_MAX 1.0 1.0 1.0',
    ];
    const lin = new Float64Array(N);
    for (let i = 0; i < N; i++) lin[i] = acescctToLin(i / (N - 1));
    // R varies fastest, then G, then B.
    for (let b = 0; b < N; b++) {
        for (let g = 0; g < N; g++) {
            for (let r = 0; r < N; r++) {
                const src = fromAP1([lin[r], lin[g], lin[b]], gamut);
                const out = toAP1(gradePixelFull(src, state, { gamut, isLinear }), gamut).map(linToACEScct);
                lines.push(out.map((v) => (Number.isFinite(v) ? v : 0).toFixed(6)).join(' '));
            }
        }
    }
    lines.push('');
    return {
        text: lines.join('\n'),
        size: N,
        notBaked,
        fileName: `radiance_grade_ACEScct_${N}_${date.getTime()}.cube`,
    };
}
