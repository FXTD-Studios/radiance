/**
 * The grade maths, on a real WebGPU device.
 *
 * `grade_gpu.test.mjs` does this for the emitted GLSL and used to carry a
 * standing `todo` for the WGSL, because headless Chromium has no
 * `navigator.gpu` -- not disabled, absent: `typeof navigator.gpu` is
 * "undefined" under every flag combination tried, including
 * `--enable-unsafe-webgpu` with SwiftShader, with `--use-vulkan=swiftshader`,
 * and with the Blink runtime flag, while WebGL2 works in the same browser. The
 * WebGPU half of Playwright's Chromium build is simply not there.
 *
 * Deno is, and it needs no GPU: its wgpu backend runs on Mesa's lavapipe
 * software Vulkan driver, which is one apt package on any Linux CI box. So the
 * dialect that was structural-only is now compiled and compared, over the same
 * case matrix as the GLSL, against the same JS functions it was generated from.
 *
 * That matters more here than for the GLSL. WebGPU is the backend nobody
 * chooses -- `_tryWebGPUUpgrade()` fires on the presence of `navigator.gpu` --
 * so a WGSL-only defect appears as "the picture changed when I switched
 * browsers", reported by someone who changed nothing.
 *
 * Skips loudly when Deno or a WebGPU adapter is missing. A skipped GPU test is
 * visible in the output; a deleted one is not.
 *
 * Run: node --test js/tests/grade_wgsl_gpu.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const HARNESS = join(HERE, 'wgslharness.mjs');

/** Deno, wherever this machine keeps it. */
function findDeno() {
    const candidates = [
        process.env.RADIANCE_TEST_DENO,
        join(process.env.HOME || '', '.deno', 'bin', 'deno'),
        '/usr/local/bin/deno',
        '/opt/homebrew/bin/deno',
    ].filter(Boolean);
    for (const p of candidates) if (existsSync(p)) return p;
    try {
        return execFileSync('sh', ['-c', 'command -v deno'], { encoding: 'utf8' }).trim() || null;
    } catch { return null; }
}

const deno = findDeno();
let report = null;
let skip = deno ? false
    : 'Deno is not installed — the WGSL half cannot be compiled here. '
      + 'Install Deno and mesa-vulkan-drivers, or set RADIANCE_TEST_DENO.';

if (!skip) {
    let raw;
    try {
        raw = execFileSync(deno, ['run', '--unstable-webgpu', '--allow-all', HARNESS],
            { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 180000 });
    } catch (e) {
        skip = `the WGSL harness could not run: ${String(e.stderr || e.message).slice(0, 400)}`;
    }
    if (!skip) {
        // The harness prints one JSON object last. Deno writes its own notices
        // to stderr, but a --allow-all prompt or a download line could still
        // land on stdout, so take the last line rather than the whole stream.
        const line = raw.trim().split('\n').pop();
        try { report = JSON.parse(line); } catch {
            skip = `the WGSL harness printed something that is not JSON: ${line.slice(0, 200)}`;
        }
    }
    // Environment failures skip; shader failures fail. The distinction matters
    // because this runs on whatever adapter the machine has: a missing Vulkan
    // ICD, a driver that will not hand out a device, a lost device mid-run --
    // none of those are the WGSL being wrong, and turning them red would train
    // people to ignore red. Anything else, including a compile error or a
    // pipeline that will not build, is the thing under test.
    const ENVIRONMENT = /no WebGPU adapter|requestDevice|device is lost|device was lost|Vulkan|adapter/i;
    if (!skip && report && !report.ok && ENVIRONMENT.test(report.error || '')) {
        skip = `no usable WebGPU device here (${report.error}) — install `
             + 'mesa-vulkan-drivers for the lavapipe software driver, or run '
             + 'somewhere with a GPU.';
    }
}

// Same as the GLSL side: fp32 in the shader against fp64 in JS. Looser would
// let a real formula difference through; tighter is float noise.
const TOLERANCE = 2e-6;

test('the emitted WGSL compiles on a real device', { skip }, () => {
    assert.ok(report, 'no report came back from the harness');
    assert.ok(report.ok, report.error);
    assert.equal(report.compiled, true);
    // Warnings are not failures, but an unexplained one is worth reading.
    assert.deepEqual(report.compileMessages, [],
        'the WGSL compiler had something to say:\n'
        + (report.compileMessages || []).map((m) => `${m.type}: ${m.message}`).join('\n'));
});

test('the emitted WGSL computes what the JS computes', { skip }, () => {
    // The claim the single-definition refactor rests on, for the dialect that
    // could not be checked before. The four shipped grade paths disagreed about
    // lift, gamma and contrast; this is what stops them drifting again.
    assert.ok(report.cases.length >= 50, `only ${report.cases.length} comparisons`);
    const comparable = report.cases.filter((c) => c.representable);
    assert.ok(comparable.length >= 50, `only ${comparable.length} comparable cases`);
    for (const c of comparable) {
        assert.ok(c.gpu.every(Number.isFinite),
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)} → ${JSON.stringify(c.gpu)}`);
        assert.ok(c.delta <= TOLERANCE,
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)}: `
            + `GPU ${JSON.stringify(c.gpu)} vs JS ${JSON.stringify(c.cpu)} (Δ ${c.delta})`);
    }
});

test('the only samples WGSL cannot match are ones fp32 cannot hold', { skip }, () => {
    // Gamma 0 floors to 0.01, so the exponent is 100 and a scene-linear 4.0
    // becomes 1.6e60 — past fp32's 3.4e38 ceiling. Identical to the GLSL case,
    // and worth asserting separately: if the two dialects ever disagree about
    // *which* samples overflow, one of them is doing different arithmetic.
    for (const c of report.cases.filter((x) => !x.representable)) {
        assert.ok(c.grade.gamma && c.grade.gamma[0] <= 0.01,
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)} exceeded fp32 without an extreme gamma`);
        assert.ok(Math.max(...c.in.map(Math.abs)) > 1,
            'an in-range pixel should never exceed fp32');
    }
});

test('no grade setting produces a non-finite pixel through WGSL', { skip }, () => {
    // gamma 0, contrast 99 and pivot 0 are all reachable on the sliders, and
    // the unguarded WGSL path produced Infinity or NaN on all three before the
    // grade definition was unified. This is the test that would have caught it.
    const risky = report.cases.filter((c) => c.representable
        && (c.grade.gamma?.[0] === 0 || c.grade.contrast === 99 || c.grade.pivot === 0));
    assert.ok(risky.length > 0, 'the harness stopped covering the extreme settings');
    for (const c of risky) {
        assert.ok(c.gpu.every(Number.isFinite),
            `${JSON.stringify(c.grade)} at ${JSON.stringify(c.in)} → ${JSON.stringify(c.gpu)}`);
    }
});
