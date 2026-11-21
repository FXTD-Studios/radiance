/**
 * The grade exports reproduce the screen.
 *
 * C4 in the viewer review: three copies of the export maths, none matching the
 * shader. The Inspector's .CDL wrote gamma as Power the wrong way round and
 * lift as Offset and dropped exposure; the .CUBE dropped exposure, offset,
 * white balance, curves and log wheels; every .cube was linear 0 to 1 and
 * clipped anything brighter. Now there is one CDL writer and one LUT writer in
 * radiance_grade_export.js, built on gradePixelFull(), which
 * grade_render.test.mjs holds to the composite shader.
 *
 * Run: node --test js/tests/grade_export.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { gradePixelFull, linToACEScct, acescctToLin, toAP1, fromAP1 } from '../radiance_grade.js';
import { buildCDL, cdlFromGrade, applyCDL, buildCubeLUT } from '../radiance_grade_export.js';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => readFileSync(join(JS, f), 'utf8');
const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');

// ── a .cube reader, for checking what was written ───────────────────────────

function parseCube(text) {
    let size = 0;
    const rows = [];
    for (const line of text.split('\n')) {
        const t = line.trim();
        if (t.startsWith('LUT_3D_SIZE')) size = Number(t.split(/\s+/)[1]);
        else if (/^-?[\d.]+(e-?\d+)?\s+-?[\d.]+(e-?\d+)?\s+-?[\d.]+(e-?\d+)?$/.test(t)) rows.push(t.split(/\s+/).map(Number));
    }
    return { size, rows };
}

/** Trilinear lookup, the interpolation every .cube reader uses. */
function sampleCube({ size, rows }, rgb) {
    const N = size;
    const at = (r, g, b) => rows[b * N * N + g * N + r];
    const idx = rgb.map((v) => Math.min(Math.max(v, 0), 1) * (N - 1));
    const i0 = idx.map((v) => Math.min(Math.floor(v), N - 2));
    const f = idx.map((v, k) => v - i0[k]);
    const out = [0, 0, 0];
    for (let dr = 0; dr < 2; dr++) for (let dg = 0; dg < 2; dg++) for (let db = 0; db < 2; db++) {
        const w = (dr ? f[0] : 1 - f[0]) * (dg ? f[1] : 1 - f[1]) * (db ? f[2] : 1 - f[2]);
        const p = at(i0[0] + dr, i0[1] + dg, i0[2] + db);
        for (let c = 0; c < 3; c++) out[c] += w * p[c];
    }
    return out;
}

/** Scene-linear in the grade's gamut → through the exported LUT → back. */
function throughCube(cube, rgb, gamut = 0) {
    const code = toAP1(rgb, gamut).map(linToACEScct);
    return fromAP1(sampleCube(cube, code).map(acescctToLin), gamut);
}

// A grade with every per-pixel control moved, curves included.
const curve = new Float32Array(1024);
for (let i = 0; i < 256; i++) {
    const x = i / 255;
    curve.set([Math.pow(x, 0.85), x, Math.min(1, x * 1.05), 1], i * 4);
}
const FULL = {
    exposure: 0.5, temperature: 0.3, tint: -0.1, contrast: 1.25, pivot: 0.18, saturation: 1.15,
    lift: [0.02, 0, -0.01], gamma: [1.1, 1, 0.95], gain: [1.05, 1, 0.97], offset: [0.01, 0, 0],
    shadows: 0.2, highlights: -0.2, colorBoost: 0.15, lumaMix: 0.6, hueShift: 8,
    printerR: 1, printerG: 0, printerB: -1,   // about 0.08 stop at 12 points to the stop
    logShadow: [0.05, 0, 0], logMidtone: [0, 0.03, 0], logHighlight: [0, 0, -0.05],
    curveTable: curve, curveMix: 1,
};

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

// ── the LUT ─────────────────────────────────────────────────────────────────

// The primary grade: everything gradePixel() covers, with a moderate move on
// each control.
const PRIMARY = {
    offset: [0.01, 0, 0], lift: [0.02, 0, -0.01], gain: [1.05, 1, 0.97], gamma: [1.1, 1, 0.95],
    contrast: 1.2, pivot: 0.18, saturation: 1.15,
};

/** Worst and median error, in 1/255 steps of linear light, over samples whose graded value is in range. */
function lutError(state, { size = 65, gamut = 0, n = 2000 } = {}) {
    const cube = parseCube(buildCubeLUT(state, { size, gamut }).text);
    const r = rng(7);
    const errs = [];
    for (let i = 0; i < n; i++) {
        const px = [r(), r(), r()];
        const want = gradePixelFull(px, state, { gamut });
        if (!want.every((v) => v >= 0 && v <= 1)) continue;
        const got = throughCube(cube, px, gamut);
        errs.push(Math.max(...got.map((v, c) => Math.abs(v - want[c]) * 255)));
    }
    errs.sort((a, b) => a - b);
    return { n: errs.length, median: errs[errs.length >> 1], max: errs[errs.length - 1] };
}

test('the .cube reproduces the primary grade within 1/255 for in-range values', () => {
    const lut = buildCubeLUT(PRIMARY, { date: new Date(0) });
    const cube = parseCube(lut.text);
    assert.equal(cube.size, 65, 'the default lattice is 65 points per axis');
    assert.equal(cube.rows.length, 65 ** 3);
    const e = lutError(PRIMARY);
    assert.ok(e.n > 400, `only ${e.n} in-range samples`);
    assert.ok(e.max <= 1, `worst ${e.max.toFixed(3)} / 255`);
});

test('the .cube carries the whole grade, curves and hue moves included', () => {
    // Hue shift, colour boost and the curve's own corners are piecewise, so a
    // lattice follows them only between its points; the rest is exact to the
    // interpolation. The old .cube dropped exposure, offset, white balance,
    // curves and log wheels outright: input 0.25 stayed 0.25 while the screen
    // showed 0.5.
    const e = lutError(FULL);
    assert.ok(e.median <= 2, `median ${e.median.toFixed(3)} / 255`);
    assert.ok(e.max <= 16, `worst ${e.max.toFixed(3)} / 255`);
    const exposureOnly = parseCube(buildCubeLUT({ exposure: 1 }, { size: 33 }).text);
    throughCube(exposureOnly, [0.25, 0.25, 0.25]).forEach((v) => assert.ok(Math.abs(v - 0.5) < 1e-3, `0.25 at +1 EV came back ${v}`));
});

test('the .cube carries HDR: 4.0 is not clipped', () => {
    const neutral = parseCube(buildCubeLUT({}).text);
    const out = throughCube(neutral, [4, 4, 4]);
    out.forEach((v) => assert.ok(Math.abs(v - 4) < 4e-3, `4.0 came back as ${v}`));
    const graded = parseCube(buildCubeLUT({ exposure: -1 }).text);
    throughCube(graded, [4, 2, 1]).forEach((v, i) => assert.ok(Math.abs(v - [2, 1, 0.5][i]) < 4e-3 * [2, 1, 0.5][i] * 2, `${v}`));
});

test('the .cube states its domain and input space, and the file name says ACEScct', () => {
    const lut = buildCubeLUT(FULL, { size: 17, gamut: 1, inputTransform: 'IDT: LogC3 → Linear' });
    assert.match(lut.text, /^TITLE ".*ACEScct.*"/m);
    assert.match(lut.text, /^# Input: +ACEScct code values, AP1 primaries\. Domain 0 to 1 is scene-linear -0\.0069 to 222\.9\./m);
    assert.match(lut.text, /Graded on: scene-linear ACEScg, after the viewer input transform "IDT: LogC3 → Linear"/);
    assert.match(lut.fileName, /ACEScct_17/);
    assert.match(lut.text, /^LUT_3D_SIZE 17$/m);
});

test('the .cube works the same for an ACEScg grade', () => {
    // On an ACEScg source the lattice and the grade share primaries. Saturation
    // is left at 1 here: pushed far enough to take a channel through zero, a
    // log-encoded output bends sharply there, which the whole-grade test bounds.
    const e = lutError({ ...PRIMARY, saturation: 1 }, { gamut: 1 });
    assert.ok(e.max <= 1, `worst ${e.max.toFixed(3)} / 255`);
    assert.ok(lutError(PRIMARY, { gamut: 1 }).max <= 4);
});

test('the .cube says what it could not bake', () => {
    const lut = buildCubeLUT({ ...FULL, midDetail: 0.3, grain: 0.2, bloom: 0.1 }, { size: 2 });
    assert.deepEqual(lut.notBaked, ['Mid Detail', 'Grain', 'Bloom']);
    assert.match(lut.text, /# Not baked .*Mid Detail, Grain, Bloom/);
});

// ── the CDL ─────────────────────────────────────────────────────────────────

test('CDL Power is 1/gamma, and exposure is folded into Slope', () => {
    // The Inspector's exporter wrote gamma 2.0 as Power 2.0; the grade raises
    // to 1/gamma, so the CDL power is 0.5.
    const c = cdlFromGrade({ gamma: [2, 2, 2] });
    assert.deepEqual(c.power, [0.5, 0.5, 0.5]);
    assert.deepEqual(cdlFromGrade({ exposure: 1 }).slope, [2, 2, 2]);
    assert.deepEqual(cdlFromGrade({ gain: [2, 1, 1], offset: [0.1, 0, 0] }).offset, [0.2, 0, 0]);
});

test('lift is never written as the CDL offset, and is reported as dropped', () => {
    const c = cdlFromGrade({ lift: [0.1, 0.1, 0.1] });
    assert.deepEqual(c.offset, [0, 0, 0]);
    assert.ok(c.dropped.some((d) => /^Lift/.test(d)));
});

test('a CDL-representable grade exports exactly', () => {
    // exposure, white balance, offset, gain, gamma, contrast (a power about the
    // pivot) and printer lights all fold into Slope / Offset / Power.
    const g = {
        exposure: 0.7, temperature: -0.4, tint: 0.2, offset: [0.01, 0.0, -0.005], gain: [1.1, 0.95, 1.02],
        gamma: [1.2, 1, 0.9], contrast: 1.35, pivot: 0.2, printerR: 6, printerG: -2, printerB: 3, saturation: 1.3,
    };
    const c = cdlFromGrade(g);
    assert.deepEqual(c.dropped, []);
    for (const px of [[0.18, 0.18, 0.18], [0.6, 0.3, 0.1], [0.02, 0.05, 0.1], [3, 2, 1]]) {
        const want = gradePixelFull(px, g);
        const got = applyCDL(px, c);
        got.forEach((v, i) => assert.ok(Math.abs(v - want[i]) <= 1e-9 * Math.max(1, Math.abs(want[i])), `${px}: ${got} vs ${want}`));
    }
});

test('the CDL lists everything it could not represent', () => {
    const c = cdlFromGrade({ ...FULL, midDetail: 0.2 });
    for (const name of ['Lift', 'Log wheels', 'Shadows / Highlights', 'Color Boost', 'Curves', 'Hue shift', 'Luma Mix', 'Mid Detail']) {
        assert.ok(c.dropped.some((d) => d.startsWith(name)), `${name} missing from ${JSON.stringify(c.dropped)}`);
    }
    const { xml } = buildCDL({ ...FULL }, { date: new Date(0) });
    assert.match(xml, /<Description>Not representable in CDL, left out: Lift/);
});

test('an ACEScct grade exports as an ACEScct CDL', () => {
    const c = buildCDL({ colorScience: 1, gain: [1.2, 1, 1], gamma: [2, 2, 2], exposure: 1, temperature: 0.2 });
    assert.ok(c.acescct);
    assert.match(c.xml, /Apply in ACEScct/);
    assert.deepEqual(c.power, [0.5, 0.5, 0.5]);
    assert.ok(Math.abs(c.offset[0] - 1.2 / 17.52) < 1e-12, 'exposure +1 is an ACEScct offset of 1/17.52');
    assert.ok(c.dropped.some((d) => d.startsWith('Temperature')));
    assert.match(c.fileName, /_ACEScct\.cdl$/);
});

test('the CDL file is valid ASC CDL XML with the values', () => {
    const { xml } = buildCDL({ gamma: [2, 2, 2], saturation: 0 }, { date: new Date(0) });
    assert.match(xml, /<Slope>1\.000000 1\.000000 1\.000000<\/Slope>/);
    assert.match(xml, /<Power>0\.500000 0\.500000 0\.500000<\/Power>/);
    // Saturation 0 (black and white) is a value, not "unset".
    assert.match(xml, /<Saturation>0\.000000<\/Saturation>/);
});

// ── one implementation ──────────────────────────────────────────────────────

test('there is one CDL exporter and one LUT exporter', () => {
    const viewer = strip(read('radiance_viewer.js'));
    const exportMod = strip(read('radiance_viewer_export.js'));
    // The prototype overrides in radiance_viewer_export.js replaced the viewer's
    // methods whichever file loaded first; the File menu and the Inspector ran
    // different code.
    assert.doesNotMatch(exportMod, /prototype\._exportCDL\s*=/);
    assert.doesNotMatch(exportMod, /prototype\._exportGradeLUT\s*=/);
    assert.doesNotMatch(viewer, /\bexportToCube\s*\(/, 'the second .cube exporter is back');
    assert.doesNotMatch(viewer, /\bexportToCDL\s*\(/, 'the second CDL exporter is back');
    assert.match(viewer, /buildCDL\(/);
    assert.match(viewer, /buildCubeLUT\(/);
    assert.equal((viewer.match(/^\s{4}_exportCDL\(\)\s*\{/gm) || []).length, 1);
    assert.equal((viewer.match(/^\s{4}_exportGradeLUT\(\)\s*\{/gm) || []).length, 1);
});
