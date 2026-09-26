/**
 * RadianceRenderer — Abstract base class for GPU-accelerated image rendering.
 *
 * Defines the contract that both WebGL (current) and WebGPU (next-gen)
 * renderers must implement. Enables gradual migration by allowing either
 * backend to be selected at runtime via feature detection.
 *
 * Lifecycle:
 *   constructor(canvas)
 *   init()          → boolean
 *   destroy()
 *
 * Per-frame:
 *   setExposure(v) … (30+ setters for grading state)
 *   render(lutStrength)
 *
 * Texture I/O:
 *   loadImageTexture(img)
 *   loadFloat16Texture(data, w, h, ch)
 *   loadFloat32Texture(data, w, h, ch)
 *   loadDepthTexture(img)
 *   loadCompareTexture(img)
 *   loadLUT(data, size)
 *   readPixelsFloat32(w, h, lutStrength) → Float32Array
 *
 * Scopes (optional — WebGPU can use compute shaders):
 *   renderScope(type, targetCanvas, sourceTexture, isLinear, paradeMode)
 *
 * Reference shelf:
 *   grabReferenceStill(slot)
 *   swapReferenceShelf(slot)
 *   clearReferenceShelf()
 *   clearFrameCache()
 */

export class RadianceRenderer {
    // ── 3.5.0: what the current texture holds ─────────────────────────────
    // "isLinearTexture" drives u_isLinear (skip the sRGB decode) and every
    // scope / bloom choice. Float frames used to be linear unconditionally, so
    // an sRGB-encoded ComfyUI IMAGE sent as floats was decoded as linear and
    // gamma-encoded a second time on the way out (washed out). The node now
    // tags each frame; "sourceDisplayEncoded" carries that tag, and the
    // accessor folds it in so every existing reader sees the effective value.
    get isLinearTexture() { return !!this._texIsLinear && !this.sourceDisplayEncoded; }
    set isLinearTexture(v) { this._texIsLinear = !!v; }

    /** An 8-bit (or display-encoded) texture: a finished display image, e.g.
     *  the node's PNG preview, which is already through the view transform. */
    get displayReferredTexture() { return !this._texIsLinear && !this.sourceDisplayEncoded; }

    /** 'srgb' for a display-encoded float source, anything else for linear. */
    setSourceEncoding(encoding) { this.sourceDisplayEncoded = encoding === 'srgb'; }

    constructor(canvas) {
        if (new.target === RadianceRenderer) {
            throw new TypeError('RadianceRenderer is abstract — instantiate a subclass');
        }
        this.canvas = canvas;
        this.textures = { image: null, lut: null, depth: null, compare: null };
        this.isLinearTexture = false;
        this.pipelinePrecision = 'f32'; // 'u8' | 'f16' | 'f32'
        this.imageWidth = 0;
        this.imageHeight = 0;

        // — Grading state (set by viewer every frame) —
        this.exposure = 0.0;
        this.gamma = 2.2;
        this.saturation = 1.0;
        this.lift = [0.0, 0.0, 0.0];
        this.gradingGamma = [1.0, 1.0, 1.0];
        this.gain = [1.0, 1.0, 1.0];
        this.offset = [0.0, 0.0, 0.0];
        this.temperature = 0.0;
        this.tint = 0.0;
        this.contrast = 1.0;
        this.pivot = 0.5;
        this.colorBoost = 0.0;
        this.shadows = 0.0;
        this.highlights = 0.0;
        this.midDetail = 0.0;
        this.hueShift = 0.0;
        this.lumaMix = 1.0;
        this.colorScience = 0;
        this.logShadow = [0.0, 0.0, 0.0];
        this.logMidtone = [0.0, 0.0, 0.0];
        this.logHighlight = [0.0, 0.0, 0.0];
        this.printerLightsR = 0;
        this.printerLightsG = 0;
        this.printerLightsB = 0;
        this.softClip = 0.0;

        // — Analytics —
        this.falseColor = false;
        this.zebra = false;
        this.zebraThreshold = 0.98;
        this.gamutWarning = false;
        this.clippingMonitor = false;
        this.channelMode = 0;
        this.focusPeaking = false;
        this.focusPeakThreshold = 120.0;
        this.displayLutMode = 0;
        this.inputLutMode = 0;
        this.displayLutStrength = 1.0;
        this.lutIsDisplayTransform = false;

        // — Effects —
        this.denoise = 0.0;
        this.showDepth = false;
        this.grainAmount = 0.0;
        this.grainSize = 1.0;
        this.grainColor = 0.0;
        this.grainAnimate = false;
        this.bloom = 0.0;
        this.halation = 0.0;
        this.diffusion = 0.0;
        this.lensDistortion = 0.0;
        this.lensFringe = 0.0;
        this.vignetteIntensity = 0.0;
        this.vignetteFalloff = 0.5;
        this.dofEnabled = false;
        this.focusDistance = 0.5;
        this.aperture = 0.0;
        this.apertureBlades = 0;
        this.apertureRotation = 0.0;
        this.apertureAnamorphic = 1.0;
        this.bokehHighlightBias = 0.0;
        this.bokehSoapBubble = 0.0;
        this.bokehOpticalVig = 0.0;

        // — Wipe / Grid —
        this.wipeEnabled = false;
        this.wipe = 0.5;
        this.wipeRefEnabled = false;
        this.gridMode = 0;
        this.gridColor = [1.0, 1.0, 1.0, 0.3];

        // — Mask —
        this.maskEnabled = false;
        this.maskShowMask = false;
        this.maskType = 0;
        this.maskCenter = [0.5, 0.5];
        this.maskScale = [0.3, 0.3];
        this.maskFeather = 0.2;
        this.maskRotation = 0.0;
        this.maskInvert = false;
        this.maskShowOverlay = false;
        this.qualifierEnabled = false;
        this.qualifierShowMask = false;
        this.qualifier = { h: 0.0, hW: 0.1, hS: 0.05, s: 0.5, sW: 0.5, sS: 0.1, l: 0.5, lW: 0.5, lS: 0.1 };

        // — Frame counter & time —
        this.frame = 0;
        this.time = 0.0;

        // — Reference shelf —
        this.referenceShelf = [];
        this.activeShelfIndex = 0;
        this.curveMix = 0.0;
        this.secondaryCurveMix = 0.0;
        this.curveSlope = [1.0, 1.0, 1.0];
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────
    init() { throw new Error('subclass must implement init()'); }
    destroy() { throw new Error('subclass must implement destroy()'); }

    // ── Texture I/O ────────────────────────────────────────────────────────
    loadImageTexture(img) { throw new Error('subclass must implement loadImageTexture()'); }
    loadFloat16Texture(data, w, h, ch) { throw new Error('subclass must implement loadFloat16Texture()'); }
    loadFloat32Texture(data, w, h, ch) { throw new Error('subclass must implement loadFloat32Texture()'); }
    loadDepthTexture(img) { throw new Error('subclass must implement loadDepthTexture()'); }
    loadCompareTexture(img) { throw new Error('subclass must implement loadCompareTexture()'); }
    loadLUT(data, size) { throw new Error('subclass must implement loadLUT()'); }
    readPixelsFloat32(w, h, lutStrength) { throw new Error('subclass must implement readPixelsFloat32()'); }

    // ── Main render ────────────────────────────────────────────────────────
    render(lutStrength) { throw new Error('subclass must implement render()'); }

    // ── Scopes ─────────────────────────────────────────────────────────────
    renderScope(mode, targetCanvas, sourceTexture, isLinear, paradeMode) {
        throw new Error('subclass must implement renderScope()');
    }
    renderHistogram(targetCanvas, logScale) {
        throw new Error('subclass must implement renderHistogram()');
    }

    // ── Reference shelf ────────────────────────────────────────────────────
    grabReferenceStill(slot) { throw new Error('subclass must implement grabReferenceStill()'); }
    swapReferenceShelf(slot) { throw new Error('subclass must implement swapReferenceShelf()'); }
    clearReferenceShelf() { throw new Error('subclass must implement clearReferenceShelf()'); }
    clearFrameCache() { throw new Error('subclass must implement clearFrameCache()'); }

    // ── Default setters (store values for subclass use) ────────────────────
    setExposure(v) { this.exposure = v; }
    setGamma(r, g, b) { this.gradingGamma = g === undefined ? [r, r, r] : [r, g, b]; }
    setSaturation(v) { this.saturation = v; }
    setFalseColor(v) { this.falseColor = v; }
    // WebGPU has no WGSL implementation yet; it stores the flag and the
    // viewer reports the backend, rather than the toggle looking active
    // and doing nothing.
    setHDRHeatmap(v) { this.hdrHeatmap = v; }
    // Magnification filter. Stored on the base so a backend without an
    // implementation degrades to interpolated rather than throwing.
    setPixelFilter(v) { this.pixelFilter = v === 'nearest' ? 'nearest' : 'linear'; }

    // OpenColorIO. The WebGL backend splices OCIO's generated GLSL into its
    // composite shader; the WebGPU backend has no WGSL path for it. Refusing
    // here rather than storing the state means the viewer can tell the user the
    // config is not being applied on this backend, instead of leaving a
    // populated Display/View menu that changes nothing.
    setOCIODisplay(_info) {
        return { ok: false, enabled: false, error: 'OpenColorIO needs the WebGL backend; this renderer has no WGSL path for it.' };
    }

    setZebra(v) { this.zebra = v; }
    setZebraThreshold(v) { this.zebraThreshold = v; }
    setGamutWarning(v) { this.gamutWarning = v; }
    setClippingMonitor(v) { this.clippingMonitor = v; }
    setMask(state) {
        if (!state) return;
        this.maskEnabled = state.enabled;
        this.maskShowMask = state.showMask;
        if (state.type !== undefined) this.maskType = state.type;
        if (state.center) this.maskCenter = state.center;
        if (state.scale) this.maskScale = state.scale;
        if (state.feather !== undefined) this.maskFeather = state.feather;
        if (state.rotation !== undefined) this.maskRotation = state.rotation;
        if (state.invert !== undefined) this.maskInvert = state.invert;
        if (state.showOverlay !== undefined) this.maskShowOverlay = state.showOverlay;
    }
    setQualifier(data) {
        if (!data) return;
        this.qualifierEnabled = data.enabled;
        this.qualifierShowMask = data.showMask;
        Object.assign(this.qualifier, data);
    }
    setChannelMode(v) { this.channelMode = v; }
    setFocusPeaking(v, t) { this.focusPeaking = v; if (t !== undefined) this.focusPeakThreshold = t; }
    setDisplayLutMode(v) { this.displayLutMode = v; }
    setInputLutMode(v) { this.inputLutMode = v; }
    setDisplayLutStrength(v) { this.displayLutStrength = v; }
    setLutIsDisplayTransform(v) { this.lutIsDisplayTransform = v; }
    setDenoise(v) { this.denoise = v; }
    setShowDepth(v) { this.showDepth = v; }
    setPrinterLights(r, g, b) { this.printerLightsR = r; this.printerLightsG = g; this.printerLightsB = b; }
    setSoftClip(v) { this.softClip = v; }
    setGrain(v) { this.grainAmount = v; }
    setGrainSize(v) { this.grainSize = v; }
    setGrainColor(v) { this.grainColor = v; }
    setGrainAnimate(v) { this.grainAnimate = v; }
    setBloom(v) { this.bloom = v; }
    setHalation(v) { this.halation = v; }
    setDiffusion(v) { this.diffusion = v; }
    setLensDistortion(dist, fringe) { this.lensDistortion = dist; this.lensFringe = fringe; }
    setVignette(intensity, falloff) { this.vignetteIntensity = intensity; this.vignetteFalloff = falloff; }
    setBokehPhysics(bias, soap, vig) {
        this.bokehHighlightBias = bias;
        this.bokehSoapBubble = soap;
        this.bokehOpticalVig = vig;
    }
    setApertureShape(blades, rot, ana) { this.apertureBlades = blades; this.apertureRotation = rot; this.apertureAnamorphic = ana; }
    setAnamorphicStreaks(v) { /* subclass may override */ }
    setTime(v) { this.time = v; }
    setFrame(v) { this.frame = v; }
    setDoFEnabled(v) { this.dofEnabled = v; }
    setFocusDistance(v) { this.focusDistance = v; }
    setAperture(v) { this.aperture = v; }
    setWipe(pos, enabled) { this.wipe = pos; this.wipeEnabled = enabled; }
    setWipeRef(v) { this.wipeRefEnabled = v; }
    setTemperature(v) { this.temperature = v; }
    setTint(v) { this.tint = v; }
    setContrast(v) { this.contrast = v; }
    setPivot(v) { this.pivot = v; }
    setLift(r, g, b) { this.lift = g === undefined ? [r, r, r] : [r, g, b]; }
    setGain(r, g, b) { this.gain = g === undefined ? [r, r, r] : [r, g, b]; }
    setOffset(r, g, b) { this.offset = g === undefined ? [r, r, r] : [r, g, b]; }
    setPipelinePrecision(v) { this.pipelinePrecision = v; }
    setCurveMix(v) { this.curveMix = v; }
    setSecondaryCurveMix(v) { this.secondaryCurveMix = v; }
    setColorScience(v) { this.colorScience = v; }
    setColorBoost(v) { this.colorBoost = v; }
    setHueShift(v) { this.hueShift = v; }
    setHighlights(v) { this.highlights = v; }
    setShadows(v) { this.shadows = v; }
    setMidDetail(v) { this.midDetail = v; }
    setLumaMix(v) { this.lumaMix = v; }
    setGridMode(v) { this.gridMode = v; }
    setLogShadow(r, g, b) { this.logShadow = g === undefined ? [r, r, r] : [r, g, b]; }
    setLogMidtone(r, g, b) { this.logMidtone = g === undefined ? [r, r, r] : [r, g, b]; }
    setLogHighlight(r, g, b) { this.logHighlight = g === undefined ? [r, r, r] : [r, g, b]; }
    setGamutCompression(v) { /* subclass may override */ }
    updateCurveLut(data) { throw new Error('subclass must implement updateCurveLut()'); }
    updateSecondaryCurveLut(data) { throw new Error('subclass must implement updateSecondaryCurveLut()'); }

    getPipelineInfo() { return { api: 'abstract', precision: this.pipelinePrecision }; }

    // Display-P3 detection helper (static)
    static initDisplayP3(canvas) {
        try {
            const gl = canvas.getContext('webgl2', { alpha: false, desynchronized: true });
            if (!gl) return { supported: false, gamut: 'srgb' };
            const ext = gl.getExtension('EXT_sRGB');
            const p3 = gl.getExtension('WEBGL_draw_buffers'); // proxy check
            gl.getExtension('EXT_disjoint_timer_query_webgl2');
            return { supported: !!ext, gamut: 'display-p3' };
        } catch (e) {
            return { supported: false, gamut: 'srgb' };
        }
    }
}
