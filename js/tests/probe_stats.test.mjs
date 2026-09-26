/**
 * The pixel probe's maths.
 *
 * A probe is an instrument. Its whole value is that a number it prints can be
 * quoted in a delivery note, so these tests pin the cases where an image
 * viewer's statistics usually go quietly wrong: a NaN poisoning the extent, an
 * Inf inflating the mean, a median estimated over a huge frame, an empty
 * selection, and a single-channel or alpha-less buffer.
 *
 * Run: node --test js/tests/probe_stats.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    sampleStats, medianFromHistogram, clampRect, rectFromCorners, pixelAt,
    luminance, nits, exposureValue, rgbToHsv, srgbToLinear, linearToSrgb,
    hexSwatch, formatValue, SRGB_ENCODED_BREAK,
    HDR_REFERENCE_WHITE_NITS, MIDDLE_GREY, MEDIAN_EXACT_LIMIT,
} from '../radiance_probe.js';

const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// ── statistics ──────────────────────────────────────────────────────────────

test('min, max, mean and median over a flat single-channel image', () => {
    const data = Float32Array.from([1, 2, 3, 4]);
    const s = sampleStats(data, { width: 2, height: 2, channels: 1 });
    assert.equal(s.count, 4);
    assert.equal(s.channels[0].min, 1);
    assert.equal(s.channels[0].max, 4);
    assert.equal(s.channels[0].mean, 2.5);
    assert.equal(s.channels[0].median, 2.5);      // even count → mean of middle two
    assert.equal(s.channels[0].medianExact, true);
});

test('channels are kept apart', () => {
    // R ramps, G is constant, B descends. A statistic that leaked across
    // channels would show up here immediately.
    const data = Float32Array.from([
        0.0, 0.5, 1.0, 1,
        0.5, 0.5, 0.5, 1,
        1.0, 0.5, 0.0, 1,
    ]);
    const s = sampleStats(data, { width: 3, height: 1, channels: 4 });
    assert.deepEqual(
        s.channels.map((c) => [c.min, c.max]),
        [[0, 1], [0.5, 0.5], [0, 1], [1, 1]],
    );
    assert.ok(close(s.channels[1].mean, 0.5));
});

test('NaN is excluded from every statistic and counted', () => {
    // The failure this exists to prevent: `Math.min(NaN, x)` is NaN, so one bad
    // pixel in a 4K plate makes the entire panel read NaN and the user concludes
    // the probe is broken rather than the image.
    const data = Float32Array.from([1, NaN, 3, 5]);
    const s = sampleStats(data, { width: 4, height: 1, channels: 1 });
    assert.equal(s.nan, 1);
    assert.equal(s.channels[0].min, 1);
    assert.equal(s.channels[0].max, 5);
    assert.ok(close(s.channels[0].mean, 3));      // (1+3+5)/3, not /4
    assert.equal(s.channels[0].median, 3);
});

test('Inf is excluded and counted separately from NaN', () => {
    const data = Float32Array.from([1, Infinity, 3, -Infinity]);
    const s = sampleStats(data, { width: 4, height: 1, channels: 1 });
    assert.equal(s.inf, 2);
    assert.equal(s.nan, 0);
    assert.equal(s.channels[0].max, 3, 'Infinity must not become the max');
    assert.equal(s.channels[0].min, 1, '-Infinity must not become the min');
    assert.ok(Number.isFinite(s.channels[0].mean));
});

test('a channel that is entirely non-finite reports null, not NaN statistics', () => {
    const data = Float32Array.from([NaN, NaN, NaN, NaN]);
    const s = sampleStats(data, { width: 2, height: 2, channels: 1 });
    assert.equal(s.channels[0], null);
    assert.equal(s.nan, 4);
});

test('negative scene-linear values are counted, not discarded', () => {
    // Legal after a matrix conversion, and also the signature of a mistake.
    // The probe reports; it does not decide which.
    const data = Float32Array.from([-0.02, 0.5, 1.0]);
    const s = sampleStats(data, { width: 3, height: 1, channels: 1 });
    assert.equal(s.negative, 1);
    assert.ok(close(s.channels[0].min, -0.02, 1e-7), 'negatives still count toward the extent');
});

// ── rectangle selection ─────────────────────────────────────────────────────

test('a rectangle samples only its own pixels', () => {
    // 4×4 single channel: value = x + y*4. The 2×2 box at (1,1) holds 5,6,9,10.
    const data = Float32Array.from({ length: 16 }, (_, i) => i);
    const s = sampleStats(data, { width: 4, height: 4, channels: 1, rect: { x: 1, y: 1, w: 2, h: 2 } });
    assert.equal(s.count, 4);
    assert.equal(s.channels[0].min, 5);
    assert.equal(s.channels[0].max, 10);
    assert.equal(s.channels[0].mean, 7.5);
});

test('a rectangle that runs off the edge is clamped, not wrapped', () => {
    const data = Float32Array.from({ length: 16 }, (_, i) => i);
    const s = sampleStats(data, { width: 4, height: 4, channels: 1, rect: { x: 3, y: 3, w: 10, h: 10 } });
    assert.deepEqual(s.rect, { x: 3, y: 3, w: 1, h: 1 });
    assert.equal(s.channels[0].min, 15);
});

test('an empty selection reports no samples rather than throwing', () => {
    const data = Float32Array.from([1, 2, 3, 4]);
    const s = sampleStats(data, { width: 2, height: 2, channels: 1, rect: { x: 0, y: 0, w: 0, h: 0 } });
    assert.equal(s.count, 0);
    assert.equal(s.channels[0], null);
});

test('a selection entirely outside the image is empty, not negative', () => {
    const data = Float32Array.from([1, 2, 3, 4]);
    const s = sampleStats(data, { width: 2, height: 2, channels: 1, rect: { x: 90, y: 90, w: 4, h: 4 } });
    assert.equal(s.rect.w, 0);
    assert.equal(s.count, 0);
});

test('clampRect never returns a negative extent', () => {
    for (const r of [{ x: -5, y: -5, w: 2, h: 2 }, { x: 0, y: 0, w: -9, h: -9 }, { x: 100, y: 100, w: 5, h: 5 }]) {
        const c = clampRect(r, 10, 10);
        assert.ok(c.w >= 0 && c.h >= 0, `${JSON.stringify(r)} → ${JSON.stringify(c)}`);
        assert.ok(c.x + c.w <= 10 && c.y + c.h <= 10);
    }
});

test('a rectangle drawn in any direction is the same rectangle', () => {
    const a = rectFromCorners(10, 20, 4, 6);
    const b = rectFromCorners(4, 6, 10, 20);
    assert.deepEqual(a, b);
    assert.deepEqual(a, { x: 4, y: 6, w: 7, h: 15 });   // inclusive of both corners
});

test('a one-pixel drag selects one pixel', () => {
    assert.deepEqual(rectFromCorners(7, 7, 7, 7), { x: 7, y: 7, w: 1, h: 1 });
});

// ── the median ──────────────────────────────────────────────────────────────

test('odd counts take the middle sample', () => {
    const data = Float32Array.from([5, 1, 9]);
    const s = sampleStats(data, { width: 3, height: 1, channels: 1 });
    assert.equal(s.channels[0].median, 5);
});

test('the median is robust to an outlier where the mean is not', () => {
    const data = Float32Array.from([0.1, 0.1, 0.1, 0.1, 1000]);
    const s = sampleStats(data, { width: 5, height: 1, channels: 1 });
    assert.ok(close(s.channels[0].median, 0.1, 1e-7));
    assert.ok(s.channels[0].mean > 100, 'the mean should be dragged; that is the point');
});

test('above the exact limit the median is estimated, and says so', () => {
    // A uniform ramp over a frame larger than MEDIAN_EXACT_LIMIT. The estimate
    // must land within one bin of the true midpoint and must not claim to be
    // exact.
    const n = MEDIAN_EXACT_LIMIT + 1024;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = i / (n - 1);
    const s = sampleStats(data, { width: n, height: 1, channels: 1 });
    assert.equal(s.channels[0].medianExact, false);
    assert.ok(Math.abs(s.channels[0].median - 0.5) < 1e-3,
        `estimated median ${s.channels[0].median} is too far from 0.5`);
    assert.equal(s.channels[0].min, 0);
    assert.equal(s.channels[0].max, 1);
});

test('a constant image has a median equal to its value, exact path or not', () => {
    for (const n of [16, MEDIAN_EXACT_LIMIT + 8]) {
        const data = new Float32Array(n).fill(0.42);
        const s = sampleStats(data, { width: n, height: 1, channels: 1 });
        assert.ok(close(s.channels[0].median, 0.42, 1e-6), `n=${n} gave ${s.channels[0].median}`);
    }
});

test('medianFromHistogram interpolates inside the bin it lands in', () => {
    // Two bins over [0,1], both half full: the midpoint is the boundary at 0.5.
    const bins = Float64Array.from([50, 50]);
    assert.ok(close(medianFromHistogram(bins, 0, 2, 0, 1, 100), 0.5, 1e-12));
});

test('medianFromHistogram on an empty histogram is NaN, not 0', () => {
    // 0 would be a plausible-looking lie. NaN forces the panel to print a dash.
    assert.ok(Number.isNaN(medianFromHistogram(new Float64Array(4), 0, 4, 0, 1, 0)));
});

// ── single pixel ────────────────────────────────────────────────────────────

test('pixelAt reads the pixel the coordinates name', () => {
    const data = Float32Array.from([
        0, 0, 0, 1, 1, 0, 0, 1,
        0, 1, 0, 1, 0, 0, 1, 1,
    ]);
    assert.deepEqual(pixelAt(data, { width: 2, height: 2, channels: 4 }, 1, 1), { r: 0, g: 0, b: 1, a: 1 });
});

test('pixelAt outside the image is null rather than a wrapped read', () => {
    const data = Float32Array.from([1, 2, 3, 4]);
    const dim = { width: 2, height: 2, channels: 1 };
    for (const [x, y] of [[-1, 0], [0, -1], [2, 0], [0, 2]]) {
        assert.equal(pixelAt(data, dim, x, y), null, `(${x},${y}) should be outside`);
    }
});

test('a single-channel buffer reads as grey with opaque alpha', () => {
    const p = pixelAt(Float32Array.from([0.25]), { width: 1, height: 1, channels: 1 }, 0, 0);
    assert.deepEqual(p, { r: 0.25, g: 0.25, b: 0.25, a: 1 });
});

test('an RGB buffer with no alpha reads as opaque', () => {
    const p = pixelAt(Float32Array.from([0.1, 0.2, 0.3]), { width: 1, height: 1, channels: 3 }, 0, 0);
    assert.equal(p.a, 1);
});

// ── derived readouts ────────────────────────────────────────────────────────

test('luminance uses BT.709 and sums to unity on white', () => {
    assert.ok(close(luminance(1, 1, 1), 1, 1e-12));
    assert.ok(close(luminance(0, 1, 0), 0.7152, 1e-12));
});

test('diffuse white reads as 203 nits', () => {
    // The number the HDR heatmap draws its white line at, and the number the
    // SDR→HDR node expands to. All three must agree or the panel is misleading.
    assert.equal(nits(1.0), HDR_REFERENCE_WHITE_NITS);
    assert.equal(nits(1.0), 203);
    assert.ok(close(nits(0.5), 101.5));
});

test('nits never reports a negative luminance', () => {
    assert.equal(nits(-0.5), 0);
});

test('mid-grey is 0 EV and a stop is a doubling', () => {
    assert.ok(close(exposureValue(MIDDLE_GREY), 0, 1e-12));
    assert.ok(close(exposureValue(MIDDLE_GREY * 2), 1, 1e-12));
    assert.ok(close(exposureValue(MIDDLE_GREY / 4), -2, 1e-12));
});

test('EV of black is -Infinity, not NaN', () => {
    // log2(0) is -Infinity and that is the honest answer; NaN would print as a
    // fault when the pixel is merely black.
    assert.equal(exposureValue(0), -Infinity);
    assert.equal(exposureValue(-1), -Infinity);
});

test('HSV hue lands on the primaries', () => {
    assert.ok(close(rgbToHsv(1, 0, 0).h, 0));
    assert.ok(close(rgbToHsv(0, 1, 0).h, 120));
    assert.ok(close(rgbToHsv(0, 0, 1).h, 240));
    assert.ok(close(rgbToHsv(1, 1, 0).h, 60));
});

test('HSV of a neutral has zero saturation and no hue jitter', () => {
    for (const v of [0, 0.18, 1, 12]) {
        const hsv = rgbToHsv(v, v, v);
        assert.equal(hsv.s, 0);
        assert.equal(hsv.h, 0);
        assert.equal(hsv.v, v);
    }
});

test('HSV value is not clamped to 1', () => {
    // Clamping here would hide exactly the highlight the probe exists to measure.
    assert.equal(rgbToHsv(4, 0.2, 0.1).v, 4);
});

test('HSV of a non-finite pixel is NaN throughout rather than a fake colour', () => {
    const hsv = rgbToHsv(NaN, 0.5, 0.5);
    assert.ok(Number.isNaN(hsv.h) && Number.isNaN(hsv.s) && Number.isNaN(hsv.v));
});

test('the sRGB transfer function round-trips, breakpoint included', () => {
    // The breakpoint is the interesting one. IEC's published pair (0.04045 and
    // 0.0031308) are rounded independently and are not each other's image, so
    // the round trip through them is discontinuous right here. The module uses
    // the exact pair instead.
    for (const c of [0, 0.001, SRGB_ENCODED_BREAK, 0.04045, 0.5, 1]) {
        assert.ok(close(linearToSrgb(srgbToLinear(c)), c, 1e-12), `round trip failed at ${c}`);
    }
});

test('the two sRGB segments meet at the breakpoint', () => {
    const linearSide = SRGB_ENCODED_BREAK / 12.92;
    const powerSide = Math.pow((SRGB_ENCODED_BREAK + 0.055) / 1.055, 2.4);
    assert.ok(close(linearSide, powerSide, 1e-12),
        `the curve has a step of ${Math.abs(linearSide - powerSide)} at the breakpoint`);
});

test('the sRGB transfer function is odd, so negatives survive it', () => {
    // Scene-linear negatives are legal. pow(negative, 1/2.4) is NaN, so the
    // sign has to be carried around the curve rather than through it.
    assert.ok(Number.isFinite(srgbToLinear(-0.2)));
    assert.ok(close(srgbToLinear(-0.2), -srgbToLinear(0.2), 1e-12));
});

test('the swatch clamps but the numbers beside it do not', () => {
    assert.equal(hexSwatch(1, 1, 1), '#ffffff');
    assert.equal(hexSwatch(50, 50, 50), '#ffffff', 'a swatch is a swatch');
    assert.equal(hexSwatch(0, 0, 0), '#000000');
    assert.equal(hexSwatch(NaN, NaN, NaN), '#000000');
});

test('formatValue names the non-finite cases instead of printing NaN in a column', () => {
    assert.equal(formatValue(NaN), 'NaN');
    assert.equal(formatValue(Infinity), '+Inf');
    assert.equal(formatValue(-Infinity), '-Inf');
    assert.equal(formatValue(null), '—');
});

test('formatValue keeps very small and very large values legible', () => {
    assert.equal(formatValue(0), '0.00000');
    assert.equal(formatValue(0.5), '0.50000');
    assert.match(formatValue(1e-7), /e-7$/);
    assert.match(formatValue(1e9), /e\+9$/);
});
