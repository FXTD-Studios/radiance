/**
 * OpenColorIO integration.
 *
 * Unlike the probe and scope panels, this one is genuinely testable: the
 * vendored build runs under Node, so these are behavioural tests against real
 * OCIO 2.5 and real ACES configs, not source assertions.
 *
 * What they are for, in order of importance:
 *
 * 1. **The vendored build works.** A bad re-vendor fails here rather than in
 *    someone's browser. `js/vendor/ocio/README.md` points at this file.
 * 2. **The maths is OCIO's.** ACES has published anchor points; if a view ever
 *    starts returning our own tone curve wearing the config's labels, the
 *    numbers move.
 * 3. **Failures are sentences.** Every error path returns text a user can act
 *    on rather than a WASM stack.
 *
 * These load 4.7 MB of WebAssembly, so they are slower than the rest of the
 * suite. That is the cost of testing the real thing.
 *
 * Run: node --test js/tests/ocio.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    initOCIO, isReady, builtinConfigs, loadConfig, buildDisplayView,
    prepareShaderSource, OCIO_FUNCTION_NAME, OCIO_SHADER_LANGUAGE,
    OCIO_TEXTURE_UNIT_BASE,
} from '../radiance_ocio.js';

const boot = await initOCIO();
const cfgs = builtinConfigs();
const aces = boot.ok ? await loadConfig({ builtin: cfgs[0].id }, { name: 'ACES CG' }) : null;

// ── the vendored build ──────────────────────────────────────────────────────

test('the vendored WASM starts, and reports the OCIO version it is', () => {
    assert.equal(boot.ok, true, `OCIO failed to start: ${boot.error}`);
    assert.match(boot.version, /^2\.\d+\.\d+$/, `unexpected version ${boot.version}`);
    assert.ok(isReady());
});

test('the built-in ACES configs are offered', () => {
    // Someone with no config of their own should still get a correct picture.
    assert.ok(cfgs.length >= 4, `only ${cfgs.length} built-in configs`);
    for (const c of cfgs) {
        assert.match(c.id, /^ocio:\/\//, `${c.id} is not an ocio:// URI`);
        assert.ok(c.label, `${c.id} has no label`);
    }
});

// ── reading a config ────────────────────────────────────────────────────────

test('a config yields displays, views and colour spaces', () => {
    assert.ok(aces.displays.length >= 3, 'too few displays');
    assert.ok(aces.displays.includes('sRGB - Display'));
    assert.ok(aces.colorSpaces.length > 10, 'too few colour spaces');
    for (const d of aces.displays) {
        assert.ok(Array.isArray(aces.viewsByDisplay[d]), `${d} has no view list`);
    }
});

test('every display has at least one view', () => {
    // A display with an empty view list would populate a dropdown that cannot
    // be used, which reads as a broken feature rather than a missing entry.
    for (const d of aces.displays) {
        assert.ok(aces.viewsByDisplay[d].length > 0, `${d} has no views`);
    }
});

test('the defaults come from the config, not from us', () => {
    assert.ok(aces.displays.includes(aces.defaultDisplay), 'default display is not in the list');
    assert.ok(aces.viewsByDisplay[aces.defaultDisplay].includes(aces.defaultView),
        'default view is not a view of the default display');
});

test('the suggested input space is a real colour space in the config', () => {
    // Guessing an input space that the config does not define would fail at the
    // first processor build, after the UI had already offered it.
    assert.ok(aces.suggestedSource, 'no input colour space suggested');
    const names = aces.colorSpaces.map((c) => c.name);
    assert.ok(names.includes(aces.suggestedSource),
        `${aces.suggestedSource} is not in the config`);
});

test('the scene_linear role drives the suggestion where it exists', () => {
    if (aces.roles.scene_linear) {
        assert.equal(aces.suggestedSource, aces.roles.scene_linear);
    }
});

test('a config loads from text as well as from a built-in', async () => {
    // The whole point: a show hands you a config file, not a URI.
    const minimal = [
        'ocio_profile_version: 2',
        'search_path: ""',
        'roles:',
        // OCIO v2 refuses a config with neither a `default` role nor a default
        // file rule, so a fixture without one tests the error path instead.
        '  default: lin',
        '  scene_linear: lin',
        'displays:',
        '  sRGB:',
        '    - !<View> {name: Raw, colorspace: lin}',
        'active_displays: [sRGB]',
        'active_views: [Raw]',
        'colorspaces:',
        '  - !<ColorSpace>',
        '    name: lin',
        '    family: ""',
        '    bitdepth: 32f',
        '    isdata: false',
    ].join('\n');
    const r = await loadConfig({ text: minimal }, { name: 'minimal' });
    assert.ok(!r.error, `minimal config failed: ${r.error}`);
    assert.deepEqual(r.displays, ['sRGB']);
    assert.deepEqual(r.viewsByDisplay.sRGB, ['Raw']);
    assert.equal(r.suggestedSource, 'lin');
});

// ── building a view ─────────────────────────────────────────────────────────

test('a view builds and produces OCIO\'s own shader', () => {
    const dv = buildDisplayView(aces.config, {
        source: aces.suggestedSource, display: aces.defaultDisplay, view: aces.defaultView,
    });
    assert.ok(!dv.error, dv.error);
    assert.equal(dv.functionName, OCIO_FUNCTION_NAME);
    assert.ok(dv.shaderText.includes(OCIO_FUNCTION_NAME),
        'the generated shader does not define the entry point we asked for');
    assert.ok(dv.shaderText.length > 500, 'the generated shader is suspiciously short');
    assert.equal(dv.isNoOp, false, 'an ACES SDR view is not a no-op');
});

test('the picture actually changes — this is not our pipeline relabelled', () => {
    // ACEScg 18% grey through the ACES 1.0 SDR Video view. If this ever returns
    // 0.18, or the sRGB OETF of 0.18 (0.46), the config is not being applied
    // and the menu is decorative.
    const dv = buildDisplayView(aces.config, {
        source: 'ACEScg', display: 'sRGB - Display', view: 'ACES 1.0 - SDR Video',
    });
    assert.ok(!dv.error, dv.error);
    const out = dv.cpu(Float32Array.from([0.18, 0.18, 0.18, 1]));
    assert.ok(Math.abs(out[0] - 0.3560) < 0.002,
        `ACES 1.0 SDR maps 18% grey to ~0.356; got ${out[0].toFixed(4)}`);
    assert.ok(Math.abs(out[0] - 0.18) > 0.1, 'the view is not being applied at all');
    assert.equal(out[3], 1, 'alpha must pass through untouched');
});

test('the tone curve rolls off rather than clipping', () => {
    // Scene-linear 4.0 is two stops over white. A view that clipped would
    // return 1.0; ACES rolls it in.
    const dv = buildDisplayView(aces.config, {
        source: 'ACEScg', display: 'sRGB - Display', view: 'ACES 1.0 - SDR Video',
    });
    const out = dv.cpu(Float32Array.from([4, 4, 4, 1]));
    assert.ok(out[0] < 1.0, `4.0 should roll off below 1.0, got ${out[0].toFixed(4)}`);
    assert.ok(out[0] > 0.9, `4.0 should still be near white, got ${out[0].toFixed(4)}`);
});

test('a view is monotonic in luminance', () => {
    const dv = buildDisplayView(aces.config, {
        source: 'ACEScg', display: 'sRGB - Display', view: 'ACES 1.0 - SDR Video',
    });
    let prev = -1;
    for (const v of [0, 0.01, 0.05, 0.18, 0.5, 1, 2, 4, 8, 16]) {
        const out = dv.cpu(Float32Array.from([v, v, v, 1]));
        assert.ok(out[0] >= prev - 1e-6, `not monotonic at scene-linear ${v}`);
        prev = out[0];
    }
});

test('a Raw view is a no-op and says so', () => {
    // Every ACES config has one. A user picking it and seeing no change should
    // be able to tell that from the panel rather than suspecting the config
    // failed to load.
    const raw = buildDisplayView(aces.config, {
        source: 'ACEScg', display: 'sRGB - Display', view: 'Raw',
    });
    assert.ok(!raw.error, raw.error);
    assert.equal(raw.isNoOp, true, 'Raw should report as a no-op');
});

test('different views give different pictures', () => {
    const a = buildDisplayView(aces.config, { source: 'ACEScg', display: 'sRGB - Display', view: 'ACES 1.0 - SDR Video' });
    const b = buildDisplayView(aces.config, { source: 'ACEScg', display: 'sRGB - Display', view: 'Un-tone-mapped' });
    const px = () => Float32Array.from([0.5, 0.25, 0.1, 1]);
    const oa = a.cpu(px()), ob = b.cpu(px());
    assert.ok(Math.abs(oa[0] - ob[0]) > 0.01,
        'two different views produced the same pixel — the view is being ignored');
});

test('an HDR display gives a different picture from an SDR one', () => {
    const sdr = buildDisplayView(aces.config, { source: 'ACEScg', display: 'sRGB - Display', view: 'ACES 1.0 - SDR Video' });
    const pqDisplay = 'Rec.2100-PQ - Display';
    const pq = buildDisplayView(aces.config, {
        source: 'ACEScg', display: pqDisplay, view: aces.viewsByDisplay[pqDisplay][0],
    });
    assert.ok(!pq.error, pq.error);
    const px = () => Float32Array.from([1, 1, 1, 1]);
    assert.ok(Math.abs(sdr.cpu(px())[0] - pq.cpu(px())[0]) > 0.01,
        'PQ and sRGB displays produced the same value');
});

// ── GPU resources ───────────────────────────────────────────────────────────

test('the shader is generated for the dialect the composite shader uses', () => {
    assert.equal(OCIO_SHADER_LANGUAGE, 'glsl_es_3.0',
        'the renderer is WebGL2 / #version 300 es; anything else will not compile');
});

test('OCIO textures carry everything needed to upload them', () => {
    // A view that needs a LUT is the normal case on a show. Missing any of
    // these fields means a silently black lookup.
    const d = 'Rec.2100-PQ - Display';
    const dv = buildDisplayView(aces.config, { source: 'ACEScg', display: d, view: aces.viewsByDisplay[d][0] });
    assert.ok(dv.textures.length > 0, 'expected this view to need at least one LUT texture');
    for (const t of dv.textures) {
        assert.ok(t.samplerName, 'texture has no sampler name to bind to');
        assert.ok([2, 3].includes(t.dimensions),
            `texture ${t.samplerName} is ${t.dimensions}D — WebGL2 has no 1D textures`);
        assert.ok([1, 3].includes(t.channels), `unexpected channel count ${t.channels}`);
        assert.ok(t.values instanceof Float32Array, 'texture values must be Float32Array');
        assert.equal(t.values.length, t.width * t.height * t.depth * t.channels,
            `texture ${t.samplerName} has the wrong number of samples for its dimensions`);
    }
});

test('OCIO texture units start clear of the composite shader\'s own', () => {
    // The composite fragment shader binds units 0–6. Overlapping would replace
    // the image, the depth map or the reference with a LUT.
    assert.ok(OCIO_TEXTURE_UNIT_BASE >= 7,
        `unit ${OCIO_TEXTURE_UNIT_BASE} collides with the composite shader`);
    assert.ok(OCIO_TEXTURE_UNIT_BASE < 16, 'WebGL2 guarantees only 16 fragment texture units');
});

test('the generated shader is safe to concatenate into ours', () => {
    const dv = buildDisplayView(aces.config, {
        source: 'ACEScg', display: aces.defaultDisplay, view: aces.defaultView,
    });
    const prep = prepareShaderSource(dv.shaderText);
    assert.doesNotMatch(prep.source, /^\s*#version/m,
        'a second #version directive is a compile error, not a warning');
    assert.doesNotMatch(prep.source, /^\s*precision\s+\w+\s+\w+\s*;/m,
        'a duplicate precision qualifier will not compile');
    assert.ok(prep.source.includes(OCIO_FUNCTION_NAME), 'the entry point was stripped');
});

test('preparing a shader strips only what it says it strips', () => {
    const { source, stripped } = prepareShaderSource(
        '#version 300 es\nprecision highp float;\nvec3 keep(vec3 c) { return c; }\n');
    assert.deepEqual(stripped, ['#version 300 es', 'precision highp float;']);
    assert.ok(source.includes('vec3 keep'));
});

test('preparing an empty shader does not throw', () => {
    assert.deepEqual(prepareShaderSource(''), { source: '', stripped: [] });
    assert.deepEqual(prepareShaderSource(null), { source: '', stripped: [] });
});

// ── failures are sentences ──────────────────────────────────────────────────

test('a malformed config reports the line', () => {
    // The most common failure by far, and OCIO's own message names the line and
    // column. Replacing it with something of our own would be strictly worse.
    return loadConfig({ text: 'not: a: config' }).then((r) => {
        assert.ok(r.error, 'a malformed config must not load silently');
        assert.match(r.error, /line \d+/, `error does not name a line: ${r.error}`);
    });
});

test('an unknown display or view names what was not found', () => {
    const r = buildDisplayView(aces.config, { source: 'ACEScg', display: 'nope', view: 'nope' });
    assert.ok(r.error);
    assert.match(r.error, /nope/, 'the error should name the display that was missing');
});

test('an unknown input colour space is refused rather than substituted', () => {
    // Quietly falling back to a role here would show a plausible picture that
    // is the wrong one, which is worse than an error.
    const r = buildDisplayView(aces.config, {
        source: 'NotAColourSpace', display: aces.defaultDisplay, view: aces.defaultView,
    });
    assert.ok(r.error, 'an unknown input space must not silently succeed');
});

test('the missing-argument cases each say which argument', () => {
    assert.match(buildDisplayView(null, {}).error, /config/i);
    assert.match(buildDisplayView(aces.config, { display: 'd', view: 'v' }).error, /input colour space/i);
    assert.match(buildDisplayView(aces.config, { source: 'ACEScg' }).error, /display or view/i);
});

test('loading with neither text nor builtin is an error, not a crash', async () => {
    const r = await loadConfig({});
    assert.ok(r.error);
    assert.match(r.error, /text|builtin/);
});
