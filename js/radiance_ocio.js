/**
 * OpenColorIO, as much of it as the viewer needs and no more.
 *
 * This is the item that decides whether the Viewer can be used on a show. Nuke,
 * RV, DJV, mrv2 and cineSync can all load a show's OCIO config; without that,
 * the Viewer's colour is *its own opinion*, which is fine for looking at
 * generations and useless for looking at work.
 *
 * The colour maths is real OpenColorIO 2.5, compiled to WebAssembly and
 * vendored under `js/vendor/ocio/`. Nothing here re-implements a transfer
 * function or a tone curve — the whole point is that the show's config decides,
 * not us.
 *
 * ## Why this façade exists at all
 *
 * The vendored binding is young (0.0.x, one maintainer). Every call into it
 * goes through this file, so the blast radius of an upstream change is one
 * module, and so every failure has somewhere to be turned into a sentence a
 * user can act on. `loadConfig` returning `{ error: 'line 41: unknown key' }`
 * is worth more than a stack trace out of WASM.
 *
 * ## What "applied" means
 *
 * `buildDisplayView` returns OCIO's own generated GLSL — the same code path
 * OCIO uses to drive a GPU viewport in any host. The renderer splices it in and
 * calls `OCIODisplay()`. It is not a bake, not an approximation, and not our
 * ACES path wearing the config's labels. When a view cannot be built, this
 * returns an error and the viewer keeps its own display pipeline and says so;
 * it never silently shows the wrong picture under the right menu entry.
 */

const VENDOR = './vendor/ocio/index.js';

/** OCIO's own texture units start here — 0–6 belong to the composite shader. */
export const OCIO_TEXTURE_UNIT_BASE = 7;

/** The GLSL dialect the WebGL2 composite shader is written in. */
export const OCIO_SHADER_LANGUAGE = 'glsl_es_3.0';

/** The function name spliced into the fragment shader. */
export const OCIO_FUNCTION_NAME = 'OCIODisplay';

let _mod = null;
let _ocio = null;
let _initError = null;
let _initPromise = null;

/**
 * Load the WASM once, ever.
 *
 * 4.7 MB of WebAssembly is not something to instantiate on the chance someone
 * opens the colour panel, so this is called on first use and the result — including
 * the failure — is cached. A second failed attempt would just be a second
 * four-second pause before the same message.
 */
export async function initOCIO() {
    if (_ocio) return { ok: true, ocio: _ocio, version: _ocio.version };
    if (_initError) return { ok: false, error: _initError };
    if (_initPromise) return _initPromise;

    _initPromise = (async () => {
        try {
            _mod = await import(VENDOR);
            _ocio = await _mod.createOCIO();
            return { ok: true, ocio: _ocio, version: _ocio.version };
        } catch (e) {
            _initError = `OpenColorIO could not start: ${e?.message || e}`;
            return { ok: false, error: _initError };
        } finally {
            _initPromise = null;
        }
    })();
    return _initPromise;
}

/** True once the library is up. Never triggers the load. */
export function isReady() {
    return !!_ocio;
}

/** The built-in ACES configs, offered when the user has no config of their own. */
export function builtinConfigs() {
    if (!_mod) return [];
    return [
        { id: _mod.ACES_CG_V2_CONFIG, label: 'ACES CG — ACES 1.3 / OCIO 2.4' },
        { id: _mod.ACES_STUDIO_V2_CONFIG, label: 'ACES Studio — ACES 1.3 / OCIO 2.4' },
        { id: _mod.ACES_CG_V4_CONFIG, label: 'ACES CG — ACES 2.0 / OCIO 2.5' },
        { id: _mod.ACES_STUDIO_V4_CONFIG, label: 'ACES Studio — ACES 2.0 / OCIO 2.5' },
    ];
}

/**
 * A loaded config, flattened into everything the panel needs in one read.
 *
 * Returned rather than left behind a live handle so the UI never holds a
 * reference into WASM memory it might use after `dispose()`. The `config`
 * handle is here too, because the processor has to come from it, but nothing in
 * the panel should reach past `summary`.
 */
export async function loadConfig(source, { name = null } = {}) {
    const boot = await initOCIO();
    if (!boot.ok) return { error: boot.error };

    let config;
    try {
        if (source.builtin) {
            config = _ocio.createBuiltinConfig(source.builtin);
        } else if (typeof source.text === 'string') {
            config = _ocio.createConfigFromString(source.text, source.workingDir
                ? { workingDir: source.workingDir } : undefined);
        } else {
            return { error: 'No config given — pass either { text } or { builtin }.' };
        }
    } catch (e) {
        // A config that will not parse is the single most common failure here,
        // and OCIO's message names the line. Passing it through unchanged is
        // more useful than anything this layer could write instead.
        return { error: `Config could not be read: ${cleanMessage(e)}` };
    }

    try {
        config.validate();
    } catch (e) {
        // Validation failure is not fatal — OCIO will still build processors
        // from a config it complains about — so this is reported rather than
        // refused, and the panel shows it.
        return { ...describeConfig(config, name), config, warning: cleanMessage(e) };
    }

    return { ...describeConfig(config, name), config };
}

/** Everything the panel reads off a config, gathered once. */
function describeConfig(config, name) {
    const displays = config.listDisplays();
    const roles = config.listRoles();
    const colorSpaces = config.listColorSpaces().map((c) => ({
        name: c.name, family: c.family, encoding: c.encoding, isData: c.isData,
    }));
    const viewsByDisplay = {};
    for (const d of displays) {
        try {
            viewsByDisplay[d] = config.listViews(d).map((v) => v.name);
        } catch {
            viewsByDisplay[d] = [];
        }
    }
    const roleMap = Object.fromEntries(roles.map((r) => [r.name, r.colorSpace]));
    const defaultDisplay = safe(() => config.getDefaultDisplay(), displays[0] || '');
    return {
        name: name || 'config',
        version: config.version,
        displays,
        viewsByDisplay,
        defaultDisplay,
        defaultView: safe(() => config.getDefaultView(defaultDisplay), viewsByDisplay[defaultDisplay]?.[0] || ''),
        colorSpaces,
        roles: roleMap,
        // The colour space an image should be read as when the config does not
        // say. 'scene_linear' first because that is what a renderer emits;
        // falling back to 'default' and then to whatever exists.
        suggestedSource: roleMap.scene_linear || roleMap.default
            || colorSpaces.find((c) => /ACEScg/i.test(c.name))?.name
            || colorSpaces[0]?.name || '',
        looks: safe(() => config.listLooks(), []),
        namedTransforms: safe(() => config.listNamedTransforms(), []),
    };
}

/**
 * Build the display transform for a (source, display, view) triple.
 *
 * Returns OCIO's generated shader, its textures and its uniforms. The caller
 * uploads the textures, sets the uniforms and calls `functionName` on a vec4.
 *
 * `cpu` is included so a caller can check a value without a GPU — the tests use
 * it, and so does the probe if it ever wants to report the displayed value
 * rather than the rendered one.
 */
export function buildDisplayView(config, { source, display, view, looksBypass = false } = {}) {
    if (!config) return { error: 'No config loaded.' };
    if (!source) return { error: 'No input colour space selected.' };
    if (!display || !view) return { error: 'No display or view selected.' };

    let processor;
    try {
        processor = config.createDisplayViewProcessor({ source, display, view, looksBypass });
    } catch (e) {
        return { error: `${display} / ${view} could not be built from ${source}: ${cleanMessage(e)}` };
    }

    let gpu;
    try {
        gpu = processor.getGpuShaderInfo({
            language: OCIO_SHADER_LANGUAGE,
            functionName: OCIO_FUNCTION_NAME,
        });
    } catch (e) {
        processor.dispose?.();
        return { error: `${display} / ${view} has no GPU path: ${cleanMessage(e)}` };
    }

    // A view that is genuinely a no-op is worth saying out loud rather than
    // leaving the user to wonder whether the config loaded. "Raw" is exactly
    // this, in every ACES config.
    return {
        processor,
        isNoOp: processor.isNoOp,
        shaderText: gpu.shaderText,
        functionName: gpu.functionName,
        textures: gpu.textures,
        uniforms: gpu.uniforms,
        cacheId: gpu.cacheId,
        label: `${source} → ${display} / ${view}`,
        cpu: (rgba) => processor.applyRGBAF32(rgba, { copy: true }),
    };
}

/**
 * Rewrite OCIO's generated shader so it can be concatenated into ours.
 *
 * OCIO emits a complete top-level block: `#version` is ours, and its samplers
 * are declared without a binding, so they have to be given texture units the
 * composite shader is not already using. Everything else is left alone —
 * rewriting generated colour code would defeat the purpose of generating it.
 */
export function prepareShaderSource(shaderText) {
    if (!shaderText) return { source: '', stripped: [] };
    const stripped = [];
    const source = shaderText
        .split('\n')
        .filter((line) => {
            // Our fragment shader already declares these, and a duplicate
            // '#version' or 'precision' is a compile error rather than a
            // warning.
            if (/^\s*#version/.test(line) || /^\s*precision\s+\w+\s+\w+\s*;/.test(line)) {
                stripped.push(line.trim());
                return false;
            }
            return true;
        })
        .join('\n');
    return { source, stripped };
}

/** Message text from a WASM exception, which is not always an Error. */
function cleanMessage(e) {
    if (!e) return 'unknown error';
    if (typeof e === 'string') return e;
    if (e.message) return String(e.message);
    const last = _ocio?.lastError;
    return last ? String(last) : String(e);
}

function safe(fn, fallback) {
    try {
        const v = fn();
        return v === undefined || v === null ? fallback : v;
    } catch {
        return fallback;
    }
}
