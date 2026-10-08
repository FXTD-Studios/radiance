// WebGL Context Manager — extends abstract RadianceRenderer base class
import { RadianceRenderer } from "./radiance_renderer.js";
// The grade maths, emitted from one file for both backends and both CPU
// paths. See js/radiance_grade.js for what used to be four implementations.
import { GLSL as GRADE_GLSL } from "./radiance_grade.js";

/**
 * The GPU frame-texture budget, for the whole page.
 *
 * It was per renderer, so every viewer got up to 3 GB of cached frame
 * textures on a 16 GB machine, and three viewers playing at once asked the
 * GPU for about 8 GB. The limits are now the page's, scaled to (an
 * approximation of) machine size, and each live renderer gets an equal share.
 */
const _DEVICE_MEMORY_GB = (typeof navigator !== 'undefined' && navigator.deviceMemory) || 4;
export const _SHARED_FRAME_CACHE = {
    maxFrames: _DEVICE_MEMORY_GB >= 16 ? 24 : _DEVICE_MEMORY_GB >= 8 ? 16 : _DEVICE_MEMORY_GB >= 4 ? 8 : 4,
    maxBytes: (_DEVICE_MEMORY_GB >= 16 ? 3.0 : _DEVICE_MEMORY_GB >= 8 ? 2.0 : _DEVICE_MEMORY_GB >= 4 ? 1.0 : 0.5)
        * 1024 * 1024 * 1024,
    renderers: new Set(),
};

class RadianceWebGLRenderer extends RadianceRenderer {
    constructor(canvas) {
        super(canvas);
        this.gl = null;
        this.sourceGamut = 0;
        this.programs = {};
        this.textures = {};
        this.framebuffers = {};
        this._uniformCache = new Map();
        this._attribCache = new Map();
        this._uniformValueCache = new Map();
        this._nextProgId = 0;
        this.lutTexture = null;
        this.depthTexture = null;
        this.lutSize = 33;

        // WebGL-only state
        this.linearFalseColor = true;
        this.curveData = new Float32Array(256 * 4);
        this._scopePixels = new Uint8Array(512 * 512 * 4);
        this.curveLutTexture = null;
        this.secondaryCurveLutTexture = null;

        for (let i = 0; i < 256; i++) {
            const v = i / 255;
            this.curveData[i * 4 + 0] = v;
            this.curveData[i * 4 + 1] = v;
            this.curveData[i * 4 + 2] = v;
            this.curveData[i * 4 + 3] = 1.0;
        }

        this.lensDistortionK2 = 0.0;
        this.anamorphicStreaks = 0.0;
        this.streakThreshold = 0.85;
        this.streakLength = 0.08;

        this.bilateralSigmaD = 3.0;
        this.bilateralSigmaR = 0.10;
        this.bilateralHalfRes = true;
        this._bilateralFBO = null;
        this._bilateralProgram = null;

        this._frameCache = new Map();
        // AUDIT-FIX (2026-08): count-based eviction alone is resolution-blind.
        // 24 cached frames is 400 MB at 1080p fp16 but 6.8 GB at 8K fp16 --
        // OOM long before the count limit is reached. Evict by BYTES as well.
        // Both limits apply; whichever is hit first evicts. The limits are the
        // page's, shared by every renderer (see _SHARED_FRAME_CACHE).
        this._frameCacheBytes = 0;
        _SHARED_FRAME_CACHE.renderers.add(this);

        this.init();
    }

    /** This renderer's share of the page-wide frame texture count. */
    get _frameCacheMaxSize() {
        return Math.max(2, Math.floor(_SHARED_FRAME_CACHE.maxFrames / _SHARED_FRAME_CACHE.renderers.size));
    }

    /** This renderer's share of the page-wide frame texture bytes. */
    get _frameCacheByteBudget() {
        return Math.floor(_SHARED_FRAME_CACHE.maxBytes / Math.max(1, _SHARED_FRAME_CACHE.renderers.size));
    }


    setWipe(pos, enabled = true) {
        this.wipe = pos;
        this.wipeEnabled = enabled;
    }

    setWipeRef(enabled) {
        this.wipeRefEnabled = enabled;
    }

    /**
     * 3.5.0: what the whole frame shows when comparing. 0 = A (with the wipe
     * if on), 1 = B only (the B side of blink, or "show B"), 2 = |A - B| times
     * `gain`, on display values. B is the reference texture (the compare
     * input's frame, or a pinned still). Only difference and B need it; with
     * no reference both show A.
     */
    setCompareShow(show, gain = 4) {
        this.compareShow = show | 0;
        this.diffGain = Number.isFinite(gain) ? gain : 4;
    }

    setDisplayLutStrength(v) {
        this.displayLutStrength = v;
    }

    setBokehPhysics(bias, soap, vig) {
        this.bokehHighlightBias = bias;
        this.bokehSoapBubble = soap;
        this.bokehOpticalVig = vig;
    }

    setGridMode(mode) {
        this.gridMode = mode;
    }

    setQualifier(data) {
        if (!data) return;
        this.qualifierEnabled = data.enabled;
        this.qualifierShowMask = data.showMask;
        if (data.h !== undefined) this.qualifier.h = data.h;
        if (data.hW !== undefined) this.qualifier.hW = data.hW;
        if (data.hS !== undefined) this.qualifier.hS = data.hS;
        if (data.s !== undefined) this.qualifier.s = data.s;
        if (data.sW !== undefined) this.qualifier.sW = data.sW;
        if (data.sS !== undefined) this.qualifier.sS = data.sS;
        if (data.l !== undefined) this.qualifier.l = data.l;
        if (data.lW !== undefined) this.qualifier.lW = data.lW;
        if (data.lS !== undefined) this.qualifier.lS = data.lS;
    }

    setMask(data) {
        // BLACK-VIEWER FIX: this wrote to this.mask.type etc., and no
        // constructor in the hierarchy ever created this.mask. The base
        // class keeps the mask as flat fields (maskType, maskCenter, ...), so
        // the very first render() after a frame loaded threw
        // "Cannot set properties of undefined (setting 'type')" inside
        // setMask, before the draw call, and the canvas stayed black on the
        // WebGL backend. The uniform upload read the same phantom object.
        // Both now go through the base class fields.
        super.setMask(data);
    }

    setLift(r, g, b) { this.lift = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setGamma(r, g, b) { this.gradingGamma = (g === undefined) ? [r, r, r] : [r, g, b]; } // Maps to gradingGamma
    setGain(r, g, b) { this.gain = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setTemperature(v) { this.temperature = v; }
    setTint(v) { this.tint = v; }
    setContrast(v) { this.contrast = v; }
    setPivot(v) { this.pivot = v; }
    setSaturation(v) { this.saturation = v; }
    // v3.0 Ops
    setColorBoost(v) { this.colorBoost = v; }
    setShadows(v) { this.shadows = v; }
    setHighlights(v) { this.highlights = v; }
    setMidDetail(v) { this.midDetail = v; }
    setHueShift(v) { this.hueShift = v; }
    setLumaMix(v) { this.lumaMix = v; }
    setColorScience(v) { this.colorScience = v; }

    // v3.3: Log Wheels
    setLogShadow(r, g, b) { this.logShadow = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setLogMidtone(r, g, b) { this.logMidtone = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setLogHighlight(r, g, b) { this.logHighlight = (g === undefined) ? [r, r, r] : [r, g, b]; }

    // v2.5: High-speed GPU Scope Rendering
    renderScope(mode, targetCanvas, sourceTexture, isLinear, parade = false) {
        if (!this.gl || !this.programs[mode]) return;
        const gl = this.gl;
        const program = this.programs[mode];

        // Setup specialized viewport for scope (Square 512x512 internally)
        const size = 512;
        if (!this.scopeFBO) {
            // v4.1: Scope FBO precision tracks pipelinePrecision so HDR waveforms
            // and vectorscopes accumulate in float rather than clamping to 8-bit.
            const precFmt = this._glPrecFmt();
            this.scopeFBO = gl.createFramebuffer();
            this.scopeTex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.scopeTex);
            gl.texImage2D(gl.TEXTURE_2D, 0, precFmt.internalFmt, size, size, 0, gl.RGBA, precFmt.type, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.scopeFBO);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.scopeTex, 0);
            // Validate FBO — fall back to RGBA/UNSIGNED_BYTE if float FBO not supported
            const fboStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
            if (fboStatus !== gl.FRAMEBUFFER_COMPLETE) {
                console.warn(`[Radiance] Scope FBO at ${precFmt.label} failed (${fboStatus}), falling back to RGBA8`);
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                gl.deleteTexture(this.scopeTex);
                gl.deleteFramebuffer(this.scopeFBO);
                // Recreate at safe RGBA8
                this.scopeFBO = gl.createFramebuffer();
                this.scopeTex = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, this.scopeTex);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.bindFramebuffer(gl.FRAMEBUFFER, this.scopeFBO);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.scopeTex, 0);
                this._scopeFBOType = gl.UNSIGNED_BYTE;
            } else {
                this._scopeFBOType = precFmt.type;
            }
        }

        // Render scope into offscreen FBO — never touch the display framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.scopeFBO);
        gl.viewport(0, 0, size, size);
        gl.clearColor(0.02, 0.02, 0.04, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Additive blending for density accumulation
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

        // v3.2: Initialize programs if they don't exist
        if (!this.programs.chromaticity) {
            this.programs.chromaticity = this.createProgram(
                this.getScopePointVertexShader('chromaticity'),
                this.getScopePointFragmentShader()
            );
        }

        gl.useProgram(program);
        gl.uniform1i(this.getUniform(program, 'u_image'), 0);
        gl.uniform1i(this.getUniform(program, 'u_isLinear'), isLinear ? 1 : 0);
        gl.uniform1f(this.getUniform(program, 'u_intensity'), mode === 'waveform' ? 0.04 : 0.02);

        // v3.1: Waveform Parade control
        if (mode === 'waveform') {
            gl.uniform1i(this.getUniform(program, 'u_parade'), parade ? 1 : 0);
        }

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sourceTexture);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.scopeBuffer);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.POINTS, 0, this.scopePointCount);
        gl.disable(gl.BLEND);

        // Read pixels — reuse pre-allocated buffers to avoid GC pressure (v3.1 PERF)
        //
        // The type must match the colour attachment. This was hard-coded to
        // UNSIGNED_BYTE while the scope FBO is created at pipelinePrecision,
        // which defaults to f32 -> RGBA32F. Per WebGL2, RGBA/UNSIGNED_BYTE is
        // only a valid ReadPixels pair for a NORMALIZED FIXED-POINT buffer, so
        // on every GPU with EXT_color_buffer_float (i.e. all desktop GPUs) the
        // call raised INVALID_OPERATION and left the buffer untouched --
        // waveform, vectorscope, histogram, parade and chromaticity all stayed
        // black or stale. The RGBA8 fallback above never fired because a float
        // FBO *is* FRAMEBUFFER_COMPLETE.
        const pixels = this._scopePixels;
        const fboType = this._scopeFBOType || gl.UNSIGNED_BYTE;
        if (fboType === gl.UNSIGNED_BYTE) {
            gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        } else {
            if (!this._scopePixelsF32 || this._scopePixelsF32.length !== size * size * 4) {
                this._scopePixelsF32 = new Float32Array(size * size * 4);
            }
            const fpix = this._scopePixelsF32;
            gl.readPixels(0, 0, size, size, gl.RGBA, gl.FLOAT, fpix);
            // Scope points are additively blended in [0,1]; saturate to 8-bit
            // for the ImageData copy below.
            for (let i = 0; i < fpix.length; i++) {
                const v = fpix[i];
                pixels[i] = v <= 0 ? 0 : (v >= 1 ? 255 : (v * 255) | 0);
            }
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        // Copy to target 2D canvas
        const ctx = targetCanvas.getContext('2d');
        const tw = targetCanvas.width;
        const th = targetCanvas.height;
        ctx.clearRect(0, 0, tw, th);

        const imgData = new ImageData(size, size);
        for (let y = 0; y < size; y++) {
            const srcRow = (size - 1 - y) * size * 4;
            const dstRow = y * size * 4;
            imgData.data.set(pixels.subarray(srcRow, srcRow + size * 4), dstRow);
        }

        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = size;
        tmpCanvas.height = size;
        tmpCanvas.getContext('2d').putImageData(imgData, 0, 0);
        ctx.drawImage(tmpCanvas, 0, 0, tw, th);

        // ── v3.2: HDR Graticules & Labels ────────────────────────────────────
        ctx.font = '10px "Inter", sans-serif';
        ctx.textAlign = 'right';

        if (mode === 'waveform') {
            // Draws PQ-based nit scale lines
            const nits = [100, 400, 1000, 4000];
            const colors = ['#666', '#555', '#bb4444', '#ccaa44'];

            nits.forEach((n, i) => {
                // PQ formula approx for labels
                const L = n / 10000;
                const m1 = 2610 / 4096 * (1/4);
                const m2 = 2523 / 4096 * 128;
                const c1 = 3424 / 4096;
                const c2 = 2413 / 4096 * 32;
                const c3 = 2392 / 4096 * 32;
                const y_pq = Math.pow((c1 + c2 * Math.pow(L, m1)) / (1 + c3 * Math.pow(L, m1)), m2);
                const py = th - (y_pq * th);

                ctx.strokeStyle = colors[i];
                ctx.globalAlpha = 0.5;
                ctx.beginPath();
                ctx.moveTo(0, py);
                ctx.lineTo(tw, py);
                ctx.stroke();

                ctx.fillStyle = colors[i];
                ctx.globalAlpha = 1.0;
                ctx.fillText(n + (n>=1000 ? ' nit' : ''), tw - 5, py - 2);
            });
        }
        ctx.globalAlpha = 1.0;
    }

    setOffset(r, g, b) { this.offset = (g === undefined) ? [r, r, r] : [r, g, b]; }
    setExposure(v) { this.exposure = v; }
    setGrain(v) { this.grainAmount = v; }
    setDenoise(v) { this.denoise = v; }
    setShowDepth(v) { this.showDepth = v; }
    setDoFEnabled(v) { this.dofEnabled = v; }
    setFocusDistance(v) { this.focusDistance = v; }
    setAperture(v) { this.aperture = v; }
    setApertureShape(blades, rotation, anamorphic) {
        this.apertureBlades = blades;
        this.apertureRotation = rotation;
        this.apertureAnamorphic = anamorphic;
    }
    setLensDistortion(k1, fringe) {
        this.lensDistortion = k1;
        this.lensFringe = fringe;
    }
    setVignette(intensity, falloff) {
        this.vignetteIntensity = intensity;
        this.vignetteFalloff = falloff;
    }

    // v3.1: Extended Effects
    setGrainSize(v) { this.grainSize = v; }
    setGrainColor(v) { this.grainColor = v; }
    setGrainAnimate(v) { this.grainAnimate = v; }
    setBloom(v) { this.bloom = v; }
    setHalation(v) { this.halation = v; }
    setDiffusion(v) { this.diffusion = v; }
    setFrame(v) { this.frame = v; }
    setTime(v) { this.time = v; }

    // ── v3.4: Printer Lights (per-channel exposure offset, -50..+50) ──────────
    /** Set RGB Printer Lights: integer offsets in -50..+50 range (like a film printer) */
    setPrinterLights(r, g, b) {
        this.printerLightsR = r || 0;
        this.printerLightsG = g || 0;
        this.printerLightsB = b || 0;
    }

    // ── v3.4: Soft Clip / Highlight Rolloff ──────────────────────────────────
    /** Smooth roll-off into highlights. 0 = hard clip, 0.5 = gentle shoulder */
    setSoftClip(v) { this.softClip = v; }

    // ── v3.2 Phase 11: Cinematic Optical ─────────────────────────────────────
    /** Brown-Conrady k2 quartic distortion coefficient (pair with k1) */
    setLensDistortionK2(v) { this.lensDistortionK2 = v; }

    /** Anamorphic highlight streaks strength (0 = off, 1 = full) */
    setAnamorphicStreaks(v) { this.anamorphicStreaks = v; }

    /** Luma threshold (0–1) above which streaks are triggered */
    setStreakThreshold(v) { this.streakThreshold = v; }

    /** Streak horizontal reach as a fraction of image width (e.g. 0.08) */
    setStreakLength(v) { this.streakLength = v; }

    // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────────────
    /**
     * Set bilateral filter parameters.
     * @param {number} sigmaD – Spatial sigma in pixels (1–8). Controls how far
     *                          neighbouring pixels are considered. Larger = wider blur.
     * @param {number} sigmaR – Range sigma (0.01–0.5). Controls how aggressively
     *                          colour-different neighbours are rejected. Smaller = sharper edges.
     */
    setBilateralSigma(sigmaD, sigmaR) {
        this.bilateralSigmaD = sigmaD;
        this.bilateralSigmaR = sigmaR;
    }

    /**
     * Toggle half-resolution bilateral pass for performance at 4K+.
     * true  = render bilateral at 0.5× then upscale (fast, imperceptible at distance)
     * false = full-resolution (maximum quality)
     */
    setBilateralHalfRes(enabled) {
        this.bilateralHalfRes = enabled;
        // Invalidate cached FBO if switching modes
        this._destroyBilateralFBO();
    }

    // ── v3.2 Fix 5: Half-res bilateral FBO helpers ───────────────────────────

    _ensureBilateralFBO(fullW, fullH) {
        const gl = this.gl;
        const halfW = Math.max(1, Math.floor(fullW / 2));
        const halfH = Math.max(1, Math.floor(fullH / 2));

        // Reuse if size unchanged
        if (this._bilateralFBO &&
            this._bilateralFBO.width === halfW &&
            this._bilateralFBO.height === halfH) {
            return this._bilateralFBO;
        }

        this._destroyBilateralFBO();

        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        // v4.0: RGBA32F for full 32-bit float pipeline (was RGBA16F)
        const internalFmt = this.isWebGL2 ? gl.RGBA32F : gl.RGBA;
        const dataType = this.isWebGL2 ? gl.FLOAT : gl.UNSIGNED_BYTE;
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, halfW, halfH, 0, gl.RGBA, dataType, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            console.warn(`[Radiance] Bilateral half-res FBO incomplete (${status}), falling back to full-res`);
            gl.deleteTexture(tex);
            gl.deleteFramebuffer(fbo);
            this.bilateralHalfRes = false;  // auto-disable
            return null;
        }

        this._bilateralFBO = { fbo, tex, width: halfW, height: halfH };
        console.log(`[Radiance] Bilateral half-res FBO: ${halfW}×${halfH}`);
        return this._bilateralFBO;
    }

    _destroyBilateralFBO() {
        if (!this._bilateralFBO) return;
        const gl = this.gl;
        gl.deleteTexture(this._bilateralFBO.tex);
        gl.deleteFramebuffer(this._bilateralFBO.fbo);
        this._bilateralFBO = null;
    }

    // ── v4.0: Multi-pass Kawase Bloom FBO Chain ──────────────────────────────
    // Industry-standard progressive downsample/upsample bloom (UE4/Flame/Resolve style).
    // 6 mip levels at [1/2, 1/4, 1/8, 1/16, 1/32, 1/64] — wide glow with minimal cost.
    // All FBOs use RGBA32F for full HDR precision throughout the chain.

    _initBloomFBOs(srcW, srcH) {
        const gl = this.gl;
        if (!this.isWebGL2) return;

        // Check if already allocated at this resolution
        if (this._bloomFBOs && this._bloomFBOs.length &&
            this._bloomSrcW === srcW && this._bloomSrcH === srcH) return;

        this._destroyBloomFBOs();

        const LEVELS = 6;
        this._bloomFBOs = [];
        this._bloomSrcW = srcW;
        this._bloomSrcH = srcH;

        let w = Math.max(1, Math.floor(srcW / 2));
        let h = Math.max(1, Math.floor(srcH / 2));

        for (let i = 0; i < LEVELS; i++) {
            const tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            const fbo = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

            const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
            if (status !== gl.FRAMEBUFFER_COMPLETE) {
                console.warn(`[Radiance Bloom] FBO level ${i} incomplete (${status}), bloom disabled`);
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                this._destroyBloomFBOs();
                return;
            }

            this._bloomFBOs.push({ fbo, tex, width: w, height: h });
            w = Math.max(1, Math.floor(w / 2));
            h = Math.max(1, Math.floor(h / 2));
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        console.log(`[Radiance Bloom] Allocated ${LEVELS}-level FBO chain from ${this._bloomFBOs[0].width}×${this._bloomFBOs[0].height}`);
    }

    _destroyBloomFBOs() {
        if (!this._bloomFBOs) return;
        const gl = this.gl;
        for (const level of this._bloomFBOs) {
            gl.deleteTexture(level.tex);
            gl.deleteFramebuffer(level.fbo);
        }
        this._bloomFBOs = null;
        this._bloomSrcW = 0;
        this._bloomSrcH = 0;
    }

    /**
     * Run the full bloom pipeline: threshold → progressive downsample → upsample.
     * Returns the final bloom texture (at half source resolution), or null if
     * bloom is disabled or not supported.
     *
     * Uses dual-filter Kawase approach:
     *   Downsample: 5-tap filter (center + 4 diagonal half-texel offsets)
     *   Upsample:   8-tap tent filter (3×3 minus corners, half-texel offsets)
     *
     * The first downsample pass also applies a brightness threshold with a
     * soft knee to isolate overbright pixels.
     */
    _renderBloomChain() {
        if (!this.isWebGL2 || this.bloom <= 0 || !this.textures.image) return null;

        const gl = this.gl;
        const imgW = this.imageWidth || this.canvas.width;
        const imgH = this.imageHeight || this.canvas.height;

        // Save the currently bound framebuffer so we can restore it after bloom passes
        const prevFBO = gl.getParameter(gl.FRAMEBUFFER_BINDING);

        // Ensure FBO chain is allocated
        this._initBloomFBOs(imgW, imgH);
        if (!this._bloomFBOs || !this._bloomFBOs.length) return null;

        const progDown = this.programs.bloomDown;
        const progUp = this.programs.bloomUp;
        if (!progDown || !progUp) return null;

        // ── Pass 1: Threshold + Downsample into level 0 ──────────────────────
        const lvl0 = this._bloomFBOs[0];
        gl.bindFramebuffer(gl.FRAMEBUFFER, lvl0.fbo);
        gl.viewport(0, 0, lvl0.width, lvl0.height);

        gl.useProgram(progDown);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(this.getUniform(progDown, 'u_src'), 0);
        gl.uniform2f(this.getUniform(progDown, 'u_srcTexelSize'), 1.0 / imgW, 1.0 / imgH);
        gl.uniform1i(this.getUniform(progDown, 'u_applyThreshold'), 1);
        // v4.3: Threshold accounts for current exposure offset so bloom fires on
        // the graded-equivalent luminance, not raw texture values.
        // Exposure shift: a +1EV grade doubles scene-linear values → halve threshold.
        const expScale = Math.pow(2.0, this.exposure || 0);
        const threshLo = (this.isLinearTexture ? 1.0 : 0.82) / expScale;
        const threshHi = (this.isLinearTexture ? 4.0 : 1.4)  / expScale;
        gl.uniform1f(this.getUniform(progDown, 'u_thresholdLo'), threshLo);
        gl.uniform1f(this.getUniform(progDown, 'u_thresholdHi'), threshHi);
        gl.uniform1f(this.getUniform(progDown, 'u_exposure'), this.exposure);
        gl.uniform1i(this.getUniform(progDown, 'u_isLinear'), this.isLinearTexture ? 1 : 0);
        this.drawQuad(progDown);

        // ── Pass 2..N: Progressive downsample (no threshold) ─────────────────
        for (let i = 1; i < this._bloomFBOs.length; i++) {
            const src = this._bloomFBOs[i - 1];
            const dst = this._bloomFBOs[i];
            gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
            gl.viewport(0, 0, dst.width, dst.height);

            gl.useProgram(progDown);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, src.tex);
            gl.uniform1i(this.getUniform(progDown, 'u_src'), 0);
            gl.uniform2f(this.getUniform(progDown, 'u_srcTexelSize'), 1.0 / src.width, 1.0 / src.height);
            gl.uniform1i(this.getUniform(progDown, 'u_applyThreshold'), 0);
            this.drawQuad(progDown);
        }

        // ── Pass N..1: Progressive upsample + accumulate ─────────────────────
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);  // additive

        for (let i = this._bloomFBOs.length - 2; i >= 0; i--) {
            const src = this._bloomFBOs[i + 1];
            const dst = this._bloomFBOs[i];
            gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
            gl.viewport(0, 0, dst.width, dst.height);

            gl.useProgram(progUp);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, src.tex);
            gl.uniform1i(this.getUniform(progUp, 'u_src'), 0);
            gl.uniform2f(this.getUniform(progUp, 'u_srcTexelSize'), 1.0 / src.width, 1.0 / src.height);
            this.drawQuad(progUp);
        }

        gl.disable(gl.BLEND);
        gl.bindFramebuffer(gl.FRAMEBUFFER, prevFBO);

        // Return the accumulated bloom texture at level 0 (half-res)
        return this._bloomFBOs[0].tex;
    }

    /**
     * Run the bilateral denoise pass at half resolution and return the result
     * texture.  The composite shader can then sample this upsampled texture
     * instead of running the expensive 7×7 kernel at full resolution.
     *
     * Returns the half-res texture (WebGL bilinear upsampling handles the rest).
     * Returns null if not needed (denoise=0 or half-res disabled).
     */
    _renderBilateralPass() {
        if (!this.bilateralHalfRes || this.denoise <= 0 || !this.textures.image) return null;

        const gl = this.gl;
        const prog = this.programs.composite;
        const fboInfo = this._ensureBilateralFBO(this.imageWidth, this.imageHeight);
        if (!fboInfo) return null;

        // Render bilateral at half-res into FBO
        gl.bindFramebuffer(gl.FRAMEBUFFER, fboInfo.fbo);
        gl.viewport(0, 0, fboInfo.width, fboInfo.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        // Re-use the composite program but override u_denoise = 1.0 so the
        // bilateral pass runs at full strength into the FBO.
        // The actual mix with the unfiltered image is done in the main pass.
        // Note: We'll integrate this more tightly in a future dedicated shader;
        // for now the FBO pre-warms bilinear interpolation at half resolution.

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        return fboInfo.tex;
    }

    // v2.4 Curves
    updateCurveLut(data) {
        // Create 256x1 RGBA float texture from curve data.
        // Accepts Float32Array(1024) [r,g,b,a × 256] for full precision,
        // or Uint8Array(1024) for legacy 8-bit fallback.
        const gl = this.gl;
        // Always activate the dedicated curve LUT unit (TEXTURE3) before binding
        gl.activeTexture(gl.TEXTURE3);
        if (!this.curveLutTexture) {
            this.curveLutTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
        gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        if (data instanceof Float32Array && this.isWebGL2) {
            // v4.1: Always use RGBA32F/FLOAT for curve LUT — it's only 256×1 (4KB),
            // so f16 savings are negligible.  Using HALF_FLOAT with a Float32Array
            // triggers WebGL INVALID_OPERATION ("type HALF_FLOAT but not Uint16Array").
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, 1, 0, gl.RGBA, gl.FLOAT, data);
        } else {
            // Legacy 8-bit fallback (Uint8Array or WebGL1)
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        }
    }

    updateSecondaryCurveLut(data) {
        const gl = this.gl;
        // Always activate the dedicated secondary curve LUT unit (TEXTURE4) before binding
        gl.activeTexture(gl.TEXTURE4);
        if (!this.secondaryCurveLutTexture) {
            this.secondaryCurveLutTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.secondaryCurveLutTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
        gl.bindTexture(gl.TEXTURE_2D, this.secondaryCurveLutTexture);
        if (data instanceof Float32Array && this.isWebGL2) {
            // v4.1: Always use RGBA32F/FLOAT for curve LUT — it's only 256×1 (4KB),
            // so f16 savings are negligible.  Using HALF_FLOAT with a Float32Array
            // triggers WebGL INVALID_OPERATION ("type HALF_FLOAT but not Uint16Array").
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, 1, 0, gl.RGBA, gl.FLOAT, data);
        } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        }
    }

    setCurveMix(v) { this.curveMix = v; }
    /** Enable secondary curve pass (HvH/HvS/HvL). Called automatically when user edits secondary curves. */
    setSecondaryCurveMix(v) { this.secondaryCurveMix = v; }
    /** FIX 5: Set slope of curve at highlight end for HDR extrapolation. Values are
     *  (lut[255]-lut[254])*255 per channel, clamped to [0,4] on the CPU side. */
    setCurveSlope(r, g, b) { this.curveSlope = [r, g, b]; }

    // ── v4.1: Pipeline Precision API ──────────────────────────────────────────

    /**
     * Returns the WebGL internal format, data type, and bytes-per-channel for
     * the current pipelinePrecision mode, with automatic capability fallback.
     * Used by every FBO and LUT texture allocation in the renderer.
     *
     * @returns {{ internalFmt, type, bytesPerChannel, label }}
     */
    _glPrecFmt() {
        const gl = this.gl;
        const mode = this.pipelinePrecision;

        // WebGL1 or no float extension: always 8-bit
        if (!this.isWebGL2 || !this.extColorBufferFloat) {
            return { internalFmt: gl.RGBA, type: gl.UNSIGNED_BYTE, bytesPerChannel: 1, label: 'INT 8-bit' };
        }

        if (mode === 'f16') {
            return { internalFmt: gl.RGBA16F, type: gl.HALF_FLOAT, bytesPerChannel: 2, label: 'FLOAT 16-bit' };
        }
        if (mode === 'f32') {
            return { internalFmt: gl.RGBA32F, type: gl.FLOAT, bytesPerChannel: 4, label: 'FLOAT 32-bit' };
        }
        // 'u8' fallback
        return { internalFmt: gl.RGBA, type: gl.UNSIGNED_BYTE, bytesPerChannel: 1, label: 'INT 8-bit' };
    }

    /**
     * Switch pipeline precision mode at runtime.
     * Destroys all precision-dependent GPU resources so they are recreated
     * at the new precision on the next render call.
     *
     * @param {'u8'|'f16'|'f32'} mode
     */
    setPipelinePrecision(mode) {
        if (mode === this.pipelinePrecision) return;
        if (!['u8', 'f16', 'f32'].includes(mode)) {
            console.warn(`[Radiance] Unknown precision mode '${mode}', ignoring.`);
            return;
        }

        const gl = this.gl;
        const prev = this.pipelinePrecision;
        this.pipelinePrecision = mode;

        // Destroy scope FBO — recreated at new precision by renderScope()
        if (this.scopeFBO) {
            gl.deleteFramebuffer(this.scopeFBO);
            gl.deleteTexture(this.scopeTex);
            this.scopeFBO = null;
            this.scopeTex = null;
        }

        // Destroy curve LUT textures — recreated by updateCurveLut() / updateSecondaryCurveLut()
        if (this.curveLutTexture) {
            gl.deleteTexture(this.curveLutTexture);
            this.curveLutTexture = null;
        }
        if (this.secondaryCurveLutTexture) {
            gl.deleteTexture(this.secondaryCurveLutTexture);
            this.secondaryCurveLutTexture = null;
        }

        // Destroy bilateral FBO — recreated at new precision by _getBilateralFBO()
        this._destroyBilateralFBO();

        // Destroy bloom FBOs — recreated at new precision by _initBloomFBOs()
        this._destroyBloomFBOs?.();

        const fmt = this._glPrecFmt();
        console.log(`[Radiance] Pipeline precision: ${prev} -> ${mode} (${fmt.label}, ${fmt.bytesPerChannel * 8}-bit per channel)`);
    }

    /**
     * Returns a structured description of the full pipeline precision chain.
     * Used by the viewer status bar to show Nuke/Flame style pipeline info.
     *
     * @returns {{ input: string, grading: string, display: string, label: string }}
     */
    getPipelineInfo() {
        const fmt = this._glPrecFmt();
        const inputLabel = this.isLinearTexture
            ? (this.textures.image ? 'Linear Float' : '—')
            : 'sRGB 8-bit';

        // Detect actual texture precision from upload path
        let inputPrecision = 'INT 8';
        if (this.isLinearTexture) {
            // Check if fp16 or fp32 was used for the image texture
            inputPrecision = (this.pipelinePrecision === 'f16') ? 'FP16' : 'FP32';
        }

        return {
            input: inputPrecision,
            grading: fmt.label,
            display: this.isWebGL2 ? 'sRGB 8-bit Canvas' : 'WebGL1 8-bit',
            label: fmt.label,
            mode: this.pipelinePrecision
        };
    }

    // ── v3.0 #9: Linear-space False Color & Zebra setter ─────────────────────
    /** When true, false color + zebra are evaluated in scene-linear space (pre-OETF),
     *  giving exposure-accurate stop thresholds. Default: true. */
    setLinearFalseColor(v) { this.linearFalseColor = v; }

    // ── v3.0 #6: GPU Histogram HUD ───────────────────────────────────────────
    /**
     * Render a 256-bin per-channel histogram to a target canvas.
     * Uses the existing scope-point GPU pipeline at full 512-px scatter density.
     * R/G/B histograms are drawn as additive colored lines on a dark background.
     *
     * @param {HTMLCanvasElement} targetCanvas  – destination 2D canvas (histogram HUD)
     * @param {boolean}           logScale      – if true, use log2 Y-axis for HDR content
     */
    renderHistogram(targetCanvas, logScale = false, sourceTexture = null, isLinear = null) {
        if (!this.textures.image) return;
        // Use the existing histogram scope-point program (scatter by luma/channel)
        this.renderScope('histogram', targetCanvas, sourceTexture || this.textures.image,
            isLinear ?? this.isLinearTexture);

        // Overlay colored channel lines on top of the luma scatter
        const ctx = targetCanvas.getContext('2d');
        const w = targetCanvas.width;
        const h = targetCanvas.height;

        // Draw grid lines (log or linear)
        ctx.globalAlpha = 0.25;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        const gridStops = logScale ? [0.01, 0.1, 0.5, 1.0, 2.0, 6.0] : [0.25, 0.5, 0.75, 1.0];
        gridStops.forEach(v => {
            const x = Math.min(w - 1, Math.round(Math.min(v, 1.0) * (w - 1)));
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        });

        // HDR indicator: dotted line at x=1.0
        ctx.globalAlpha = 0.5;
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#ffcc00';
        ctx.beginPath(); ctx.moveTo(w - 1, 0); ctx.lineTo(w - 1, h); ctx.stroke();
        ctx.setLineDash([]);

        // Labels
        ctx.globalAlpha = 0.7;
        ctx.font = '9px monospace';
        ctx.fillStyle = '#aaa';
        ctx.fillText('0', 2, h - 2);
        ctx.fillText('1.0', w - 22, h - 2);
        if (logScale) ctx.fillText('HDR', w - 30, 12);
        ctx.globalAlpha = 1.0;
    }

    // ── v3.0 #8: LRU GPU Frame Texture Cache ─────────────────────────────────
    /**
     * Evict LRU entries until both the count limit and the byte budget can
     * accommodate `incomingBytes`. O(evicted) via Map insertion-order.
     */
    _evictFrameCacheFor(incomingBytes) {
        while (this._frameCache.size > 0 &&
               (this._frameCache.size >= this._frameCacheMaxSize ||
                this._frameCacheBytes + incomingBytes > this._frameCacheByteBudget)) {
            const lruId = this._frameCache.keys().next().value;
            const lruEntry = this._frameCache.get(lruId);
            if (lruEntry) {
                if (this.gl && lruEntry.tex) this.gl.deleteTexture(lruEntry.tex);
                this._frameCacheBytes -= (lruEntry.bytes || 0);
            }
            this._frameCache.delete(lruId);
        }
        if (this._frameCacheBytes < 0) this._frameCacheBytes = 0;
    }

    /**
     * Upload a float16 frame and cache it by frameId.
     * LRU eviction is O(1) using Map insertion-order: the first key in the Map
     * is always the least-recently-used entry (delete-on-access + re-insert).
     * Eviction is both count- and byte-budget-based (see _evictFrameCacheFor).
     */
    _releaseImageTexture() {
        const t = this.textures.image;
        if (!t) return;
        for (const e of this._frameCache.values()) if (e.tex === t) return;
        this.gl.deleteTexture(t);
        this.textures.image = null;
    }

    /**
     * 3.5.0: compare input (the node's compare_image preview). WebGL never had
     * this method, so the base-class stub threw and neither compare_image nor
     * Pin Frame could reach the wipe. The preview PNG is display-referred (the
     * node bakes it through the same view the A side uses), which is exactly
     * what the wipe's B side samples. Kept in its own texture so it never
     * deletes a reference-shelf still.
     */
    loadCompareTexture(img) {
        const gl = this.gl;
        if (!img) return null;
        this._clearCompareSource();             // a raw still (a pin) is used as it is
        if (this._compareTex) gl.deleteTexture(this._compareTex);
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        this._compareTex = tex;
        this.textures.reference = tex;
        this.wipeRefEnabled = true;
        return tex;
    }

    // ── B for compare: the same view as A, placed by pixel size ───────────
    //
    // B used to be the node's 8-bit preview PNG, stretched over A, while A
    // was live float through the current view: Difference lit up on the
    // view transform and the resampling rather than on content. B is now
    // B's own float frame when the node sent one, drawn through this
    // renderer's pipeline (grade and view, no overlays) at its own size,
    // then placed on A's pixel grid: 1:1 and centred, or fitted. Without a
    // float frame the preview PNG is placed the same way.

    /**
     * src: null, { kind: 'float', hdr: {fp16data|data, width, height,
     * channels}, encoding }, or { kind: 'display', image, width, height }
     * where width/height are the source's pixel size (a proxy PNG can be
     * smaller than the frame it stands for).
     */
    setCompareSource(src) {
        const gl = this.gl;
        this._clearCompareSource();
        if (!src || !gl) return;
        let tex = null, w = src.width, h = src.height;
        if (src.kind === 'float' && src.hdr) {
            w = src.hdr.width; h = src.hdr.height;
            tex = this._uploadFloatAside(src.hdr);
        } else if (src.kind === 'display' && src.image) {
            tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src.image);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            w = w || src.image.width; h = h || src.image.height;
        }
        if (!tex || !w || !h) return;
        this._compareSrc = { kind: src.kind, tex, width: w, height: h, encoding: src.encoding };
        this.wipeRefEnabled = true;
    }

    /** 'pixel' (1:1, centred, the default) or 'fit' (scaled to fit inside A). */
    setCompareFit(mode) {
        this.compareFit = mode === 'fit' ? 'fit' : 'pixel';
    }

    _clearCompareSource() {
        const gl = this.gl;
        const src = this._compareSrc;
        if (!src) return;
        if (gl && src.tex) gl.deleteTexture(src.tex);
        this._compareSrc = null;
        if (this.textures.reference === this._compareRef?.tex) this.textures.reference = null;
    }

    /** Upload a float frame to a texture of its own, leaving the picture's texture alone. */
    _uploadFloatAside(hdr) {
        const kept = {
            image: this.textures.image, w: this.imageWidth, h: this.imageHeight,
            linear: this._texIsLinear, isFloat: this._imageIsFloat,
        };
        this.textures.image = null;              // so the upload releases nothing
        const ch = hdr.channels || hdr.shape?.[2] || 4;
        let tex = null;
        try {
            tex = hdr.fp16data
                ? this.loadFloat16Texture(hdr.fp16data, hdr.width, hdr.height, ch)
                : this.loadFloat32Texture(hdr.data, hdr.width, hdr.height, ch);
        } finally {
            this.textures.image = kept.image;
            this.imageWidth = kept.w;
            this.imageHeight = kept.h;
            this._texIsLinear = kept.linear;
            this._imageIsFloat = kept.isFloat;
        }
        return tex;
    }

    /** An RGBA8 render target of w x h, reused while the size holds. */
    _compareTarget(slot, w, h) {
        const gl = this.gl;
        let t = this[slot];
        if (t && t.width === w && t.height === h) return t;
        if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this[slot] = t = { tex, fbo, width: w, height: h };
        return t;
    }

    /**
     * Draw B for this frame into textures.reference, on A's pixel grid.
     * Called from render() while a compare mode shows B.
     */
    _renderCompareReference() {
        const gl = this.gl;
        const src = this._compareSrc;
        const A = { w: this.imageWidth, h: this.imageHeight };
        if (!src || !A.w || !A.h) return;
        let view = src.tex, viewIsRendered = false;
        if (src.kind === 'float') {
            // B through this pipeline, at B's size: same grade, same view,
            // none of the overlays (scopeSignal), no compare of its own.
            const target = this._compareTarget('_compareView', src.width, src.height);
            const kept = {
                image: this.textures.image, w: this.imageWidth, h: this.imageHeight,
                linear: this._texIsLinear, display: this.sourceDisplayEncoded,
                show: this.compareShow, wipe: this.wipeEnabled, ref: this.wipeRefEnabled,
                signal: this.scopeSignal, fbo: this._exportFBO,
            };
            this.textures.image = src.tex;
            this.imageWidth = src.width; this.imageHeight = src.height;
            this._texIsLinear = true;
            if (src.encoding) this.sourceDisplayEncoded = src.encoding === 'srgb';
            this.compareShow = 0; this.wipeEnabled = false; this.wipeRefEnabled = false;
            this.scopeSignal = true;
            this._exportFBO = { fbo: target.fbo, width: target.width, height: target.height };
            this._renderingCompare = true;
            try {
                this.render();
            } finally {
                this._renderingCompare = false;
                this.textures.image = kept.image;
                this.imageWidth = kept.w; this.imageHeight = kept.h;
                this._texIsLinear = kept.linear; this.sourceDisplayEncoded = kept.display;
                this.compareShow = kept.show; this.wipeEnabled = kept.wipe; this.wipeRefEnabled = kept.ref;
                this.scopeSignal = kept.signal; this._exportFBO = kept.fbo;
            }
            view = target.tex;
            viewIsRendered = true;
        }

        // Place it on A's grid. Coordinates are top-origin image space;
        // render targets hold their bottom row first (see createQuad).
        if (!this.programs.comparePlace) {
            this.programs.comparePlace = this.createProgram(this.getBasicVertexShader(), `#version 300 es
                precision highp float;
                in vec2 v_texcoord;
                out vec4 fragColor;
                uniform sampler2D u_src;
                uniform vec2 u_scale;
                uniform vec2 u_offset;
                uniform bool u_srcRendered;
                void main() {
                    vec2 a = vec2(v_texcoord.x, 1.0 - v_texcoord.y);
                    vec2 b = a * u_scale - u_offset;
                    if (any(lessThan(b, vec2(0.0))) || any(greaterThan(b, vec2(1.0)))) {
                        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }
                    if (u_srcRendered) b.y = 1.0 - b.y;
                    fragColor = vec4(texture(u_src, b).rgb, 1.0);
                }`);
        }
        const program = this.programs.comparePlace;
        if (!program) return;
        const B = { w: src.width, h: src.height };
        const s = this.compareFit === 'fit' ? Math.min(A.w / B.w, A.h / B.h) : 1;
        const scale = [A.w / (B.w * s), A.h / (B.h * s)];
        const offset = [(A.w - B.w * s) / (2 * B.w * s), (A.h - B.h * s) / (2 * B.h * s)];
        const ref = this._compareTarget('_compareRef', A.w, A.h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, ref.fbo);
        gl.viewport(0, 0, A.w, A.h);
        gl.useProgram(program);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, view);
        gl.uniform1i(this.getUniform(program, 'u_src'), 0);
        gl.uniform2f(this.getUniform(program, 'u_scale'), scale[0], scale[1]);
        gl.uniform2f(this.getUniform(program, 'u_offset'), offset[0], offset[1]);
        gl.uniform1i(this.getUniform(program, 'u_srcRendered'), viewIsRendered ? 1 : 0);
        this.drawQuad(program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this.textures.reference = ref.tex;
        this.wipeRefEnabled = true;
    }

    loadFloat16TextureCached(frameId, fp16data, width, height, channels) {
        if (this._frameCache.has(frameId)) {
            // Promote to MRU: delete then re-insert at tail
            const entry = this._frameCache.get(frameId);
            this._frameCache.delete(frameId);
            this._frameCache.set(frameId, entry);
            this.textures.image = entry.tex;
            this.imageWidth = width;
            this.imageHeight = height;
            this.isLinearTexture = true;
            return entry.tex;
        }

        const bytes = width * height * (channels || 4) * 2;
        this._evictFrameCacheFor(bytes);

        const tex = this.loadFloat16Texture(fp16data, width, height, channels);
        if (tex) {
            this._frameCache.set(frameId, { tex, bytes });
            this._frameCacheBytes += bytes;
        }
        return tex;
    }

    /**
     * Upload a float32 frame and cache it by frameId.
     * O(1) LRU eviction via Map insertion-order (same as float16 variant).
     */
    loadFloat32TextureCached(frameId, data, width, height, channels) {
        if (this._frameCache.has(frameId)) {
            const entry = this._frameCache.get(frameId);
            this._frameCache.delete(frameId);
            this._frameCache.set(frameId, entry);
            this.textures.image = entry.tex;
            this.imageWidth = width;
            this.imageHeight = height;
            this.isLinearTexture = true;
            return entry.tex;
        }

        const bytes = width * height * (channels || 4) * 4;
        this._evictFrameCacheFor(bytes);

        const tex = this.loadFloat32Texture(data, width, height, channels);
        if (tex) {
            this._frameCache.set(frameId, { tex, bytes });
            this._frameCacheBytes += bytes;
        }
        return tex;
    }

    // ── v3.0 #10: Display-P3 / ICC Detection ─────────────────────────────────
    /**
     * Detect display gamut via CSS media query and store result.
     * Call once after canvas creation (before init()).
     * B-7 FIX: Removed inert 2D context creation. Fixed ternary that returned
     * 'display-p3' for both branches. Result is now stored on canvas so init()
     * can pass { colorSpace } to getContext('webgl2').
     *
     * @returns {{ isP3: boolean, isHDR: boolean, canvasColorSpace: string }}
     */
    static initDisplayP3(canvas) {
        const isP3 = window.matchMedia('(color-gamut: p3)').matches;
        // 3.5.0: HDR capability is (dynamic-range: high); rec2020 is a gamut.
        const isHDR = window.matchMedia('(dynamic-range: high)').matches;
        let colorSpace = 'srgb';

        if (isHDR) {
            colorSpace = 'display-p3'; // Rec.2020 monitors still use P3 canvas gamut
            console.log('[Radiance v3.0] Wide-gamut monitor detected (Rec.2020+ HDR)');
        } else if (isP3) {
            colorSpace = 'display-p3';
            console.log('[Radiance v3.0] Display-P3 monitor detected');
        } else {
            console.log('[Radiance v3.0] sRGB display detected');
        }

        // Store on canvas element so init() can read it
        canvas._radianceColorSpace = colorSpace;

        return { isP3, isHDR, canvasColorSpace: colorSpace };
    }

    // ── v3.2 Phase 8: Public Waveform Render Helper ───────────────────────────
    /**
     * Render a Luma waveform or RGB Parade to any target canvas element.
     * Call this each frame in your UI update loop.
     *
     * @param {HTMLCanvasElement} targetCanvas  – destination 2D canvas (e.g. 256×256 HUD panel)
     * @param {boolean}           parade        – false = Luma waveform, true = RGB Parade
     */
    renderWaveform(targetCanvas, parade = false) {
        if (!this.textures.image) return;
        this.renderScope('waveform', targetCanvas, this.textures.image, this.isLinearTexture, parade);
    }

    // ── v3.2 Phase 10: Comparison Bridge ─────────────────────────────────────
    /**
     * Grab the current fully-composited frame from the GL canvas and store it
     * as a WebGL texture in the active reference shelf slot.
     * The texture is then immediately available for wipe comparison.
     *
     * @returns {number} The shelf index where the still was stored.
     */
    grabReferenceStill() {
        const gl = this.gl;

        // Read current framebuffer pixels
        const w = this.canvas.width;
        const h = this.canvas.height;
        const pixels = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

        // Release previous texture at this slot
        const slot = this.activeShelfIndex;
        if (this.referenceShelf[slot]) {
            gl.deleteTexture(this.referenceShelf[slot]);
        }

        // Upload as a new texture
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        // WebGL readPixels returns bottom-to-top; flip vertically via pixel re-pack
        const flipped = new Uint8Array(w * h * 4);
        for (let y = 0; y < h; y++) {
            const src = (h - 1 - y) * w * 4;
            const dst = y * w * 4;
            flipped.set(pixels.subarray(src, src + w * 4), dst);
        }
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, flipped);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.referenceShelf[slot] = tex;

        // Wire as active reference for wipe comparison
        this.textures.reference = tex;
        console.log(`[Radiance] Grabbed reference still → shelf[${slot}] (${w}×${h})`);
        return slot;
    }

    /**
     * Activate a previously grabbed shelf slot for wipe comparison.
     * Automatically enables wipe reference mode.
     *
     * @param {number} index – Shelf slot index (0–7)
     */
    swapReferenceShelf(index) {
        if (index < 0 || index >= this.referenceShelf.length) {
            console.warn(`[Radiance] Shelf index ${index} out of range`);
            return;
        }
        const tex = this.referenceShelf[index];
        if (!tex) {
            console.warn(`[Radiance] Shelf slot ${index} is empty — grab a still first`);
            return;
        }
        this.activeShelfIndex = index;
        this._clearCompareSource();              // a pinned still replaces the compare input
        this.textures.reference = tex;
        this.wipeRefEnabled = true;
        console.log(`[Radiance] Reference shelf → slot ${index} activated`);
    }

    /**
     * Release all GPU textures in the reference shelf.
     */
    clearReferenceShelf() {
        const gl = this.gl;
        this.referenceShelf.forEach((tex, i) => {
            if (tex) { gl.deleteTexture(tex); this.referenceShelf[i] = null; }
        });
        this.referenceShelf = [];
        this.textures.reference = null;
        this.wipeRefEnabled = false;
        console.log('[Radiance] Reference shelf cleared');
    }

    /**
     * Attach mouse/touch drag listeners to a canvas element so the user can
     * drag the wipe divider interactively.  Call once after the viewer mounts.
     *
     * @param {HTMLCanvasElement} canvas – The GL canvas (or an overlay div)
     * @param {function} [onChange]      – Optional callback(pos: 0–1) called on drag
     */
    initWipeDragging(canvas, onChange) {
        let dragging = false;

        const getPos = (e) => {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        };

        const onDown = (e) => {
            if (!this.wipeEnabled) return;
            const pos = getPos(e);
            // Only start drag if near the divider (±3% of width)
            if (Math.abs(pos - this.wipe) < 0.04) {
                dragging = true;
                e.preventDefault();
            }
        };
        const onMove = (e) => {
            if (!dragging) return;
            const pos = getPos(e);
            this.wipe = pos;
            if (onChange) onChange(pos);
            e.preventDefault();
        };
        const onUp = () => { dragging = false; };

        canvas.addEventListener('mousedown', onDown, { passive: false });
        canvas.addEventListener('mousemove', onMove, { passive: false });
        canvas.addEventListener('mouseup', onUp);
        canvas.addEventListener('touchstart', onDown, { passive: false });
        canvas.addEventListener('touchmove', onMove, { passive: false });
        canvas.addEventListener('touchend', onUp);

        // Visual cursor feedback
        canvas.addEventListener('mousemove', (e) => {
            if (!this.wipeEnabled) return;
            const pos = getPos(e);
            canvas.style.cursor = Math.abs(pos - this.wipe) < 0.04 ? 'col-resize' : 'default';
        });

        console.log('[Radiance] Wipe drag listener attached');
    }

    /**
     * 3.5.0: what the shader's output values mean to the browser. 'srgb' by
     * default; 'display-p3' when the view encodes for a Display P3 monitor
     * (OCIO's "Display P3 - Display"). Returns false where unsupported.
     */
    setDisplayColorSpace(cs) {
        const gl = this.gl;
        if (!gl || !('drawingBufferColorSpace' in gl)) return false;
        try { gl.drawingBufferColorSpace = cs; } catch { return false; }
        this.displayColorSpace = gl.drawingBufferColorSpace;
        return this.displayColorSpace === cs;
    }

    init() {
        // Once per renderer. The constructor runs it and the viewer called it
        // again, which compiled every shader twice (104 of 199 ms of building
        // a viewer), leaked the first set of programs and registered the
        // context-loss handlers twice.
        if (this._initResult !== undefined) return this._initResult;
        this._initResult = this._init();
        // A renderer without a context caches nothing, so it takes no share.
        if (!this._initResult) _SHARED_FRAME_CACHE.renderers.delete(this);
        return this._initResult;
    }

    _init() {
        // B-7 FIX: Use Display-P3 colorSpace if detected by initDisplayP3()
        const colorSpace = this.canvas._radianceColorSpace || 'srgb';
        const ctxAttrs = {
            alpha: false,
            antialias: false,
            preserveDrawingBuffer: true,
            premultipliedAlpha: false,
            powerPreference: 'high-performance', // Request dedicated GPU
            desynchronized: true // Reduce latency
        };
        // 3.5.0: 'colorSpace' is not a WebGL context attribute (it was passed
        // here and ignored). The drawing buffer's colour space is set with
        // gl.drawingBufferColorSpace, per view, by setDisplayColorSpace().
        void colorSpace;

        // WebGL 2.0 for float32 textures and advanced features
        this.gl = this.canvas.getContext('webgl2', ctxAttrs);

        if (!this.gl) {
            console.warn('[Radiance] WebGL 2 not available, falling back to WebGL 1');
            this.gl = this.canvas.getContext('webgl', {
                alpha: true,
                preserveDrawingBuffer: true,
                powerPreference: 'high-performance'
            });
            this.isWebGL2 = false;
        } else {
            this.isWebGL2 = true;
            console.log("[Radiance] WebGL 2.0 initialized");
        }

        if (!this.gl) {
            console.error('[Radiance] WebGL not supported');
            return false;
        }

        const gl = this.gl;

        // Enable float textures extension for WebGL 1 fallback
        if (!this.isWebGL2 && gl.getExtension) {
            gl.getExtension('OES_texture_float');
            gl.getExtension('EXT_color_buffer_float');
        }

        // v3.0 FIX: WebGL2 also needs EXT_color_buffer_float for rendering to float FBOs
        if (this.isWebGL2 && gl.getExtension) {
            gl.getExtension('EXT_color_buffer_float');
        }

        // Enable linear filtering for float textures
        if (gl.getExtension) {
            this.extColorFloatLinear = gl.getExtension('OES_texture_float_linear');
            // v3.0 #3: OES_texture_half_float is required for correct HALF_FLOAT upload path
            this.extHalfFloat = gl.getExtension('OES_texture_half_float') ||
                gl.getExtension('EXT_texture_norm16');  // WebGL2 fallback
            this.extColorHalfFloatLinear = gl.getExtension('OES_texture_half_float_linear');
            this.extColorBufferFloat = gl.getExtension('EXT_color_buffer_float');
            if (this.extHalfFloat) {
                console.log('[Radiance v2.4.2] OES_texture_half_float available — HALF_FLOAT path active');
            } else {
                console.warn('[Radiance v2.4.2] OES_texture_half_float unavailable — falling back to FLOAT');
            }
        } else {
            this.extColorFloatLinear = null;
            this.extHalfFloat = null;
            this.extColorHalfFloatLinear = null;
            this.extColorBufferFloat = null;
        }

        // v4.1: Auto-downgrade pipelinePrecision if GPU lacks float FBO support.
        // EXT_color_buffer_float is required for rendering into RGBA32F / RGBA16F FBOs.
        if (!this.isWebGL2 || !this.extColorBufferFloat) {
            if (this.pipelinePrecision !== 'u8') {
                console.warn(`[Radiance] EXT_color_buffer_float unavailable — pipeline precision downgraded from '${this.pipelinePrecision}' to 'u8' (INT 8-bit)`);
                this.pipelinePrecision = 'u8';
            }
        } else {
            console.log(`[Radiance] Pipeline precision: ${this.pipelinePrecision} (EXT_color_buffer_float OK)`);
        }

        // Create shader programs
        this.createPrograms();

        // Create fullscreen quad
        this.createQuad();

        // v2.5: High-res sampling buffers
        this.createScopeBuffers();

        // ── I-10 FIX: WebGL Context Loss / Recovery ─────────────────────
        // Under VRAM pressure or GPU driver hiccups, the context can be lost.
        // Without handlers, the entire renderer silently dies with no recovery.
        this._contextLost = false;
        this.canvas.addEventListener('webglcontextlost', (e) => {
            this._contextLost = true;
            // 3.5.0: destroy() releases the context on purpose (a removed
            // node, a graph loaded over this one). That is not a failure and
            // must not be kept restorable or logged as one.
            if (this._destroyed) return;
            e.preventDefault(); // Required to allow restoration
            console.error('[Radiance] WebGL context lost — renderer paused. Waiting for recovery...');
            // The viewer shows that the picture is gone, rather than a black frame.
            try { this.onContextLost?.(); } catch (err) { console.warn('[Radiance] onContextLost failed:', err); }
        }, false);

        this.canvas.addEventListener('webglcontextrestored', () => {
            if (this._destroyed) return;
            console.log('[Radiance] WebGL context restored — reinitializing...');
            this._contextLost = false;
            // Clear all caches that hold stale GL object references. Every
            // handle from the lost context is dead; none may be deleted or bound.
            this._uniformCache.clear();
            this._attribCache.clear();
            this._uniformValueCache.clear();
            this._frameCache.clear();
            this._frameCacheBytes = 0;
            this.programs = {};
            this.textures = {};
            this.framebuffers = {};
            this.scopeFBO = null; this.scopeTex = null;
            this.curveLutTexture = null; this.secondaryCurveLutTexture = null;
            this._compareTex = null;
            this._compareSrc = null; this._compareView = null; this._compareRef = null;
            this._bilateralFBO = null;
            this._bloomFBOs = null; this._bloomSrcW = 0; this._bloomSrcH = 0;
            this._exportFBO = null;
            this.referenceShelf = [];
            const ocio = this._ocioInfo;
            this._ocio = null;
            this.ocioEnabled = false;

            // Re-acquire extensions
            if (this.isWebGL2 && this.gl.getExtension) {
                this.gl.getExtension('EXT_color_buffer_float');
                this.extColorFloatLinear = this.gl.getExtension('OES_texture_float_linear');
                this.extHalfFloat = this.gl.getExtension('OES_texture_half_float');
                this.extColorHalfFloatLinear = this.gl.getExtension('OES_texture_half_float_linear');
                this.extColorBufferFloat = this.gl.getExtension('EXT_color_buffer_float');
            }
            this._ocioFloatLinear = undefined;

            // Recreate GPU resources
            this._nextProgId = 0;
            this.createPrograms();
            this.createQuad();
            this.createScopeBuffers();
            // What the renderer itself was given: the 3D LUT and the OCIO view.
            if (this._lutSource) this.loadLUT(this._lutSource.lutData, this._lutSource.size);
            if (ocio) this.setOCIODisplay(ocio);

            console.log('[Radiance] WebGL context recovery complete.');
            // The frame, compare and depth textures come from the viewer.
            try { this.onContextRestored?.(); } catch (err) { console.warn('[Radiance] onContextRestored failed:', err); }
        }, false);

        console.log("[Radiance] Renderer initialized");
        return true;
    }

    createPrograms() {
        // Basic image display with exposure/gamma and proper sRGB OETF
        this.programs.basic = this.createProgram(
            this.getBasicVertexShader(),
            this.getBasicFragmentShader()
        );

        // 3D LUT + DoF application (composite)
        this.programs.composite = this.createProgram(
            this.getBasicVertexShader(),
            this.getCompositeFragmentShader()
        );

        // RGB Parade waveform
        this.programs.parade = this.createProgram(
            this.getBasicVertexShader(),
            this.getParadeFragmentShader()
        );

        // v2.5: GPU-Accelerated Scopes (True 32-bit Analysis)
        this.programs.vectorscope = this.createProgram(
            this.getScopePointVertexShader('vectorscope'),
            this.getScopePointFragmentShader()
        );
        this.programs.histogram = this.createProgram(
            this.getScopePointVertexShader('histogram'),
            this.getScopePointFragmentShader()
        );
        this.programs.waveform = this.createProgram(
            this.getScopePointVertexShader('waveform'),
            this.getScopePointFragmentShader()
        );

        // v4.0: Multi-pass Kawase bloom programs
        if (this.isWebGL2) {
            this.programs.bloomDown = this.createProgram(
                this.getBasicVertexShader(),
                this._getBloomDownsampleShader()
            );
            this.programs.bloomUp = this.createProgram(
                this.getBasicVertexShader(),
                this._getBloomUpsampleShader()
            );
        }

        console.log("[Radiance] Shader programs compiled");
    }

    createProgram(vertexSource, fragmentSource) {
        const gl = this.gl;

        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexSource);
        gl.compileShader(vertexShader);

        if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(vertexShader);
            console.error('[Radiance] Vertex shader compilation failed:', log);
            // B-14 FIX: Removed blocking browser alert; professional tools log silently
            return null;
        }

        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentSource);
        gl.compileShader(fragmentShader);

        if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(fragmentShader);
            console.error('[Radiance] Fragment shader compilation failed:', log);
            console.error('[Radiance] Fragment shader source:', fragmentSource);
            // B-14 FIX: Removed blocking browser alert
            return null;
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const log = gl.getProgramInfoLog(program);
            console.error('[Radiance] Shader program linking failed:', log);
            // B-14 FIX: Removed blocking browser alert
            return null;
        }

        // B-11 FIX: Assign unique numeric ID for uniform value cache keys.
        // WebGLProgram objects all stringify to '[object WebGLProgram]', causing
        // cache collisions between programs. Numeric IDs are unique per program.
        program._rid = this._nextProgId++;
        return program;
    }

    // P1: Cached uniform location lookup — O(1) after first call per program+name
    getUniform(program, name) {
        let progMap = this._uniformCache.get(program);
        if (!progMap) { progMap = new Map(); this._uniformCache.set(program, progMap); }
        let loc = progMap.get(name);
        if (loc === undefined) { loc = this.gl.getUniformLocation(program, name); progMap.set(name, loc); }
        return loc;
    }

    // P1: Cached attribute location lookup
    getAttrib(program, name) {
        let progMap = this._attribCache.get(program);
        if (!progMap) { progMap = new Map(); this._attribCache.set(program, progMap); }
        let loc = progMap.get(name);
        if (loc === undefined) { loc = this.gl.getAttribLocation(program, name); progMap.set(name, loc); }
        return loc;
    }

    // ── v3.2 Fix: Dirty-flag uniform upload helpers ───────────────────────────
    // These replace raw gl.uniform* calls inside render().
    // Only issues the GL call when the value has changed since the last upload.
    // B-11 FIX: Use prog._rid (numeric ID) as cache key prefix instead of
    // prog.toString() which produces identical '[object WebGLProgram]' for all programs.

    _uf1(prog, name, v) {
        const key = prog._rid + ':' + name;
        if (this._uniformValueCache.get(key) === v) return;
        this._uniformValueCache.set(key, v);
        this.gl.uniform1f(this.getUniform(prog, name), v);
    }

    _ui1(prog, name, v) {
        const key = prog._rid + ':' + name;
        if (this._uniformValueCache.get(key) === v) return;
        this._uniformValueCache.set(key, v);
        this.gl.uniform1i(this.getUniform(prog, name), v);
    }

    _uf3(prog, name, x, y, z) {
        const key = prog._rid + ':' + name;
        const packed = `${x},${y},${z}`;
        if (this._uniformValueCache.get(key) === packed) return;
        this._uniformValueCache.set(key, packed);
        this.gl.uniform3f(this.getUniform(prog, name), x, y, z);
    }

    _uf2(prog, name, x, y) {
        const key = prog._rid + ':' + name;
        const packed = `${x},${y}`;
        if (this._uniformValueCache.get(key) === packed) return;
        this._uniformValueCache.set(key, packed);
        this.gl.uniform2f(this.getUniform(prog, name), x, y);
    }

    _uf4v(prog, name, arr) {
        const key = prog._rid + ':' + name;
        const packed = arr.join(',');
        if (this._uniformValueCache.get(key) === packed) return;
        this._uniformValueCache.set(key, packed);
        this.gl.uniform4fv(this.getUniform(prog, name), arr);
    }

    /**
     * Invalidate the uniform value cache for a specific program.
     * Call this after gl.useProgram() if switching between programs,
     * or after loadImageTexture() since texture bindings change.
     */
    invalidateUniformCache(prog) {
        if (!prog) {
            this._uniformValueCache.clear();
        } else {
            const prefix = prog._rid + ':';
            for (const key of this._uniformValueCache.keys()) {
                if (key.startsWith(prefix)) this._uniformValueCache.delete(key);
            }
        }
    }

    createQuad() {
        const gl = this.gl;

        // Fullscreen quad vertices
        // UVs flipped V (0->1, 1->0) to display upside-down WebGL textures correctly
        const vertices = new Float32Array([
            -1, -1, 0, 1,
            1, -1, 1, 1,
            -1, 1, 0, 0,
            1, 1, 1, 0
        ]);

        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    }

    // ── v4.0: Bloom Downsample Shader (Kawase dual-filter) ──────────────────
    // 5-tap filter: center (weight 4) + 4 diagonal half-texel offsets (weight 1 each).
    // First pass applies brightness threshold with soft knee; subsequent passes skip it.
    // Linearizes sRGB input on first pass if source is not scene-linear.
    _getBloomDownsampleShader() {
        return `#version 300 es
            precision highp float;
            in vec2 v_texcoord;
            out vec4 fragColor;

            uniform sampler2D u_src;
            uniform vec2  u_srcTexelSize;
            uniform int   u_applyThreshold;
            uniform float u_thresholdLo;
            uniform float u_thresholdHi;
            uniform float u_exposure;
            uniform int   u_isLinear;

            vec3 linearize(vec3 c) {
                bvec3 lo = lessThan(c, vec3(0.04045));
                return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, vec3(lo));
            }

            void main() {
                vec2 uv = v_texcoord;
                vec2 hs = u_srcTexelSize * 0.5; // half-texel

                // 5-tap Kawase downsample (center weight 4, corners weight 1)
                vec3 c  = texture(u_src, uv).rgb;
                vec3 tl = texture(u_src, uv + vec2(-hs.x,  hs.y)).rgb;
                vec3 tr = texture(u_src, uv + vec2( hs.x,  hs.y)).rgb;
                vec3 bl = texture(u_src, uv + vec2(-hs.x, -hs.y)).rgb;
                vec3 br = texture(u_src, uv + vec2( hs.x, -hs.y)).rgb;

                vec3 color = (c * 4.0 + tl + tr + bl + br) / 8.0;

                if (u_applyThreshold == 1) {
                    // Linearize if sRGB input
                    if (u_isLinear == 0) color = linearize(color);

                    // Apply exposure before threshold (so user exposure affects what blooms)
                    color *= pow(2.0, u_exposure);

                    // Soft knee threshold: smoothstep from lo→hi
                    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
                    float knee = smoothstep(u_thresholdLo, u_thresholdHi, lum);
                    color *= knee;

                    // v4.1: Reinhard soft-clamp to prevent extreme HDR values (fire,
                    // specular >>10.0) from propagating through the 6-level FBO chain.
                    // Without this, accumulated bloom values can reach 1000s, causing
                    // blocky blowout artefacts in the final composite.
                    color = color / (vec3(1.0) + color);
                }

                fragColor = vec4(color, 1.0);
            }
        `;
    }

    // ── v4.0: Bloom Upsample Shader (Kawase dual-filter) ─────────────────────
    // 9-tap tent filter (3×3 bilinear samples at half-texel offsets).
    // Weighted: corners=1, edges=2, center=4 → /16 total.
    // Used with additive blending to accumulate into the next-larger FBO.
    _getBloomUpsampleShader() {
        return `#version 300 es
            precision highp float;
            in vec2 v_texcoord;
            out vec4 fragColor;

            uniform sampler2D u_src;
            uniform vec2 u_srcTexelSize;

            void main() {
                vec2 uv = v_texcoord;
                vec2 hs = u_srcTexelSize * 0.5;

                // 3×3 tent filter (9 taps)
                vec3 sum = vec3(0.0);
                sum += texture(u_src, uv + vec2(-hs.x,  hs.y)).rgb;       // TL  ×1
                sum += texture(u_src, uv + vec2( 0.0,   hs.y)).rgb * 2.0; // T   ×2
                sum += texture(u_src, uv + vec2( hs.x,  hs.y)).rgb;       // TR  ×1
                sum += texture(u_src, uv + vec2(-hs.x,  0.0 )).rgb * 2.0; // L   ×2
                sum += texture(u_src, uv).rgb * 4.0;                       // C   ×4
                sum += texture(u_src, uv + vec2( hs.x,  0.0 )).rgb * 2.0; // R   ×2
                sum += texture(u_src, uv + vec2(-hs.x, -hs.y)).rgb;       // BL  ×1
                sum += texture(u_src, uv + vec2( 0.0,  -hs.y)).rgb * 2.0; // B   ×2
                sum += texture(u_src, uv + vec2( hs.x, -hs.y)).rgb;       // BR  ×1

                fragColor = vec4(sum / 16.0, 1.0);
            }
        `;
    }

    getBasicVertexShader() {
        const version = this.isWebGL2 ? '#version 300 es' : '';
        const inOut = this.isWebGL2 ? 'in' : 'attribute';
        const outVar = this.isWebGL2 ? 'out' : 'varying';

        return `${version}
            ${inOut} vec2 a_position;
            ${inOut} vec2 a_texcoord;
            ${outVar} vec2 v_texcoord;

            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texcoord = a_texcoord;
            }
        `;
    }

    // Basic fragment shader with sRGB OETF
    getBasicFragmentShader() {
        const version = this.isWebGL2 ? '#version 300 es' : '';
        const inVar = this.isWebGL2 ? 'in' : 'varying';
        const outColor = this.isWebGL2 ? 'out vec4 fragColor;' : '';
        const fragColor = this.isWebGL2 ? 'fragColor' : 'gl_FragColor';
        const texture2D = this.isWebGL2 ? 'texture' : 'texture2D';

        return `${version}
            precision highp float;
            ${inVar} vec2 v_texcoord;
            ${outColor}

            uniform sampler2D u_image;
            uniform float u_exposure;
            uniform float u_gamma;
            uniform float u_saturation;
            uniform bool u_isLinear;
${GRADE_GLSL}
            // sRGB OETF (linear → display)
            vec3 linearToSRGB(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(linear, vec3(1.0/2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            // sRGB EOTF (display → linear)
            vec3 sRGBToLinear(vec3 srgb) {
                bvec3 cutoff = lessThan(srgb, vec3(0.04045));
                vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
                vec3 lower = srgb / vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            void main() {
                vec3 color = ${texture2D}(u_image, v_texcoord).rgb;

                // Linearize sRGB PNG input; float textures are already linear
                if (!u_isLinear) {
                    color = sRGBToLinear(color);
                }

                // Exposure (linear space)
                color *= pow(2.0, u_exposure);

                // Saturation (linear space)
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                color = mix(vec3(luma), color, u_saturation);

                // Gamma (artistic control)
                if (u_gamma != 1.0) {
                    // Was pow(max(color,0), 1.0 / u_gamma) with no floor: at
                    // gamma 0 that is 1/0 = Infinity and the image splits into
                    // hard black and blown. radGamma carries the floor.
                    color = radGamma(color, vec3(u_gamma));
                }

                // Display transform (sRGB OETF)
                color = linearToSRGB(max(color, vec3(0.0)));

                ${fragColor} = vec4(color, 1.0);
            }
        `;
    }



    getCompositeFragmentShader() {
        return `#version 300 es
            precision highp float;
            precision highp int;
            precision highp sampler2D;
            precision highp sampler3D;

            in vec2 v_texcoord;
            out vec4 fragColor;

            uniform sampler2D u_image;
            uniform sampler2D u_depth;
            uniform sampler3D u_lut;

            uniform float u_lutSize;
            uniform float u_lutStrength;
            uniform bool u_lutEnabled;
            uniform bool u_lutIsDisplayTransform; // true = LUT is a full display transform (already has OETF baked in)

            uniform float u_exposure;
            uniform vec3 u_lift;
            uniform vec3 u_gamma;
            uniform vec3 u_gain;
            uniform vec3 u_offset; // v3.0

            uniform int u_colorScience; // 0=Linear/sRGB (default), 1=ACEScct

            uniform float u_temperature;
            uniform float u_tint;
            uniform float u_contrast;
            uniform float u_pivot;

            uniform float u_saturation;
            uniform float u_grainAmount;
            uniform int u_frame;
            uniform float u_time;

            // v3.1: Professional Masking (Power Windows)
            uniform int u_maskType; // 0=None, 1=Circle, 2=Box
            uniform vec2 u_maskCenter;
            uniform vec2 u_maskScale;
            uniform float u_maskFeather;
            uniform float u_maskRotation;
            uniform bool u_maskInvert;
            uniform bool u_maskShowOverlay;

            // v3.0: Resolve-style Uniforms
            uniform float u_colorBoost;
            uniform float u_shadows;
            uniform float u_highlights;
            uniform float u_midDetail;
            uniform float u_hueShift;
            uniform float u_lumaMix;

            // v3.3: Log Wheels
            uniform vec3 u_logShadow;
            uniform vec3 u_logMidtone;
            uniform vec3 u_logHighlight;

            uniform bool u_dofEnabled;
            uniform float u_focusDist;
            uniform float u_aperture;
            uniform vec2 u_texSize;

            uniform bool u_falseColor;
            uniform bool u_hdrHeatmap;
            uniform bool u_zebra;
            uniform float u_zebraThreshold;

            // v2.3 Analytics
            uniform bool u_gamutWarning;
            uniform bool u_clippingMonitor;

            // HDR pipeline: true when texture contains linear float data
            uniform bool u_isLinear;

            // v2.2: Channel isolation (0=RGB, 1=R, 2=G, 3=B, 4=Luma, 5=Alpha)
            uniform int u_channelMode;
            // v2.2: Focus peaking
            uniform bool u_focusPeaking;
            uniform float u_focusPeakThreshold;
            uniform int u_displayLutMode;
            uniform int u_inputLutMode;
            uniform int u_sourceGamut;
            uniform float u_displayLutStrength;

            // v2.2 Pro Comparison
            uniform bool u_wipeEnabled;
            uniform bool u_exportSceneLinear;  // 3.5.0: graded EXR = scene-linear, no view, no overlays
            uniform bool u_scopeSignal;        // 3.5.0: the displayed picture only, for the scopes
            uniform bool u_compareSignal;      // the same, for compare B: keeps the viewer f-stop
            uniform float u_viewExposure;      // 3.5.0: viewer-only f-stops (never rendered out)
            uniform float u_viewGamma;         // 3.5.0: viewer-only display gamma
            uniform bool u_dither;             // 3.5.0: +-1/2 LSB triangular dither on the 8-bit output
            uniform float u_wipe;
            uniform bool u_wipeRefEnabled;
            uniform sampler2D u_referenceImage;
            uniform int u_compareShow;          // 3.5.0: 0 A, 1 B, 2 |A-B| x gain
            uniform float u_diffGain;

            // v2.2 Pro Grids
            uniform int u_gridMode;
            uniform vec4 u_gridColor;

            // v2.3: Denoise & Depth Eval
            uniform float u_denoise;
            uniform bool u_showDepth;

            // v2.4: Custom Curves (1D LUT)
            // 256x1 texture where R=RedCurve, G=GreenCurve, B=BlueCurve
            // Alpha channel is unused (or could be Luma curve master)
            uniform sampler2D u_curveLut;
            uniform float u_curveMix; // 0.0 = disabled, 1.0 = full effect
            // FIX 5: Slope of the curve at the highlight end (lut[255]-lut[254])*255.
            // Used for physically correct HDR extrapolation above 1.0 instead of
            // a linear pass-through that bypasses the user's curve shape entirely.
            // u_curveSlope removed (v4.2): ratio-based HDR extrapolation via topVal
            // sampling (FIX 6) fully replaced slope-based extrapolation. Slope is
            // still computed on the JS side for backward compat but never uploaded.

            // v3.4: Secondary Curves (Hue vs X)
            uniform sampler2D u_secondaryCurveLut;
            uniform float u_secondaryCurveMix;


            // v2.5: Qualifiers (HSL)
            uniform bool u_qualifierEnabled;
            uniform bool u_qualifierShowMask;

            // Hue (0..1)
            uniform float u_qualifierHue;
            uniform float u_qualifierHueWidth;
            uniform float u_qualifierHueSoft;

            // Saturation (0..1)
            uniform float u_qualifierSat;
            uniform float u_qualifierSatWidth;
            uniform float u_qualifierSatSoft;

            // Luma (0..1)
            uniform float u_qualifierLuma;
            uniform float u_qualifierLumaWidth;
            uniform float u_qualifierLumaSoft;

            // v2.6: Lens & Optical Effects
            uniform int u_apertureBlades;
            uniform float u_apertureRotation;
            uniform float u_apertureAnamorphic;

            uniform float u_lensDistortion; // k1
            uniform float u_lensFringe; // Chromatic Aberration


            uniform float u_vignetteIntensity;
            uniform float u_vignetteFalloff;

            // v2.6.1: Realistic Bokeh Physics
            uniform float u_bokehHighlightBias;
            uniform float u_bokehSoapBubble;
            uniform float u_bokehOpticalVig;

            // v3.1: Extended Grain & Lens
            uniform float u_grainSize;
            uniform float u_grainColor;
            uniform float u_grainAnimate;  // 0=static, 1=animated (video)
            uniform float u_bloom;
            uniform float u_halation;
            uniform float u_diffusion;

            // v4.0: Pre-computed multi-pass bloom texture (Kawase chain result)
            uniform sampler2D u_bloomTex;
            uniform int u_bloomTexEnabled;

            // ── OpenColorIO ─────────────────────────────────────────────────
            // When a config is loaded, OpenColorIO 2.5 generates the GLSL below
            // from the show's config and it replaces this shader's display
            // transform entirely. The block declares its own helpers, its own
            // LUT samplers and OCIODisplay(). Nothing in Radiance
            // re-implements it -- that is the point: the config decides.
            // With no config loaded the entry point still has to exist or the
            // shader will not compile; u_ocioEnabled is false, so the stub is
            // never reached.
            uniform bool u_ocioEnabled;
${this._ocioShaderSource || '            vec4 OCIODisplay(vec4 inPixel) { return inPixel; }'}
            // ── end OpenColorIO ─────────────────────────────────────────────

            // ── v3.2 Phase 11: Anamorphic Streaks + k2 Distortion ────────────
            uniform float u_lensDistortionK2;   // Brown-Conrady quartic term
            uniform float u_anamorphicStreaks;   // 0=off, strength 0..1
            uniform float u_streakThreshold;     // Luma trigger level (0..1)
            uniform float u_streakLength;        // Horizontal reach in UV space

            // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────
            uniform float u_bilateralSigmaD;     // Spatial sigma (pixels)
            uniform float u_bilateralSigmaR;     // Range sigma (color diff)
            uniform bool  u_bilateralHalfRes;    // true = half-res FBO pre-computed

            // ── v3.4: Printer Lights ──────────────────────────────────────────
            // Per-channel additive exposure in log space. Range -50..+50 (like film printer).
            // Internally converted to a multiplier: 2^(offset/50) per channel.
            uniform float u_printerR;   // -50..+50
            uniform float u_printerG;
            uniform float u_printerB;

            // ── v3.4: Soft Clip ───────────────────────────────────────────────
            // Rolls highlights smoothly into a shoulder instead of hard-clipping.
            // 0 = disabled, 1 = full soft shoulder.
            uniform float u_softClip;

            // ----------------------------------------------------------------
            // Color Ops v3.0
            // ----------------------------------------------------------------

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            vec3 applyHueShift(vec3 color, float shift) {
                if (shift == 0.0) return color;
                vec3 hsv = rgb2hsv(color);
                hsv.x += shift / 360.0;
                if (hsv.x > 1.0) hsv.x -= 1.0;
                if (hsv.x < 0.0) hsv.x += 1.0;
                return hsv2rgb(hsv);
            }

            vec3 applyColorBoost(vec3 color, float boost) {
                if (boost == 0.0) return color;
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                float maxRGB = max(color.r, max(color.g, color.b));
                float sat = maxRGB - min(color.r, min(color.g, color.b)); // Simple saturation estimate

                // Vibrance logic: Boosts low-sat pixels more than high-sat
                float boostFactor = (1.0 - sat * 0.8) * boost;
                vec3 boosted = mix(vec3(luma), color, 1.0 + boostFactor);
                return max(boosted, 0.0);
            }


            // ----------------------------------------------------------------
            // Masking (Power Windows)
            // ----------------------------------------------------------------

            float calculateMask(vec2 uv) {
                if (u_maskType == 0) return 1.0;

                vec2 p = uv - u_maskCenter;

                // Aspect Ratio Compensation: If u_texSize is available, we use it to keep masks round
                float aspect = u_texSize.x / u_texSize.y;
                p.x *= aspect;

                // Apply rotation
                float s = sin(-u_maskRotation);
                float c = cos(-u_maskRotation);
                p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);

                // Apply scaling (Inverse of scale factor)
                p /= max(vec2(0.001), u_maskScale);

                float dist = 0.0;
                if (u_maskType == 1) { // Circle
                    dist = length(p);
                } else if (u_maskType == 2) { // Box
                    dist = max(abs(p.x), abs(p.y));
                }

                float mask = 1.0 - smoothstep(1.0 - u_maskFeather, 1.0, dist);
                return u_maskInvert ? 1.0 - mask : mask;
            }

            // ----------------------------------------------------------------
            // Curves
            // ----------------------------------------------------------------

            vec3 applyCurves(vec3 color) {
                if (u_curveMix <= 0.0) return color;

                // Sample only the 0..1 range from the LUT
                vec3 c = clamp(color, 0.0, 1.0);

                float r = texture(u_curveLut, vec2(c.r, 0.5)).r;
                float g = texture(u_curveLut, vec2(c.g, 0.5)).g;
                float b = texture(u_curveLut, vec2(c.b, 0.5)).b;
                vec3 curved = vec3(r, g, b);

                // FIX 6: Ratio-preserving HDR extrapolation for values > 1.0.
                // Old slope-based method failed on highlight roll-off curves:
                // when the curve flattens at the top (slope → 0), ALL HDR values
                // above 1.0 collapsed to the same output, destroying highlight
                // separation. Ratio-based is what Nuke/Resolve use internally —
                // the curve output at 1.0 becomes the effective "gain" for HDR:
                //   output = color * curve(1.0)
                // This is continuous at the boundary (1.0 * topVal == topVal)
                // and preserves relative differences between HDR values the same
                // way exposure (multiplicative) does.
                // Ratio-preserving HDR extrapolation (FIX 6 / v4.2):
                // A single texture fetch returns all 4 channels; use .rgb swizzle
                // instead of three separate fetches at the same coordinate.
                vec3 topVal = texture(u_curveLut, vec2(1.0, 0.5)).rgb;
                curved = mix(curved, color * max(topVal, vec3(0.0)), step(vec3(1.0), color));

                return mix(color, curved, u_curveMix);
            }

            // ----------------------------------------------------------------
            // Secondary Curves (Hue vs Hue / Sat)
            // ----------------------------------------------------------------
            vec3 applySecondaryCurves(vec3 color) {
                if (u_secondaryCurveMix <= 0.0) return color;

                vec3 hsv = rgb2hsv(color);

                // Texture lookup based on Hue (x coordinate)
                // R = HueVsHue, G = HueVsSat, B = HueVsLuma
                vec3 lookup = texture(u_secondaryCurveLut, vec2(hsv.x, 0.5)).rgb;

                // R: HueVsHue — 0.5 = no change. Range: +/-0.5 = +/-180 deg hue rotation.
                // fract() wraps the hue into [0.0, 1.0) and correctly handles
                // negative arguments in GLSL (fract(-0.1) = 0.9).
                // FIX 2: removed dead 'if (hsv.x < 0.0)' guard — fract() makes it unreachable.
                float hueShift = (lookup.r - 0.5);
                hsv.x = fract(hsv.x + hueShift);

                // G: HueVsSat — 0.5 = 1× sat. 0.0 = 0× sat. 1.0 = 2× sat.
                float satMult = lookup.g * 2.0;
                hsv.y = clamp(hsv.y * satMult, 0.0, 1.0);

                // B: HueVsLuma — 0.5 = no change. <0.5 darken, >0.5 lift.
                float lumaShift = (lookup.b - 0.5) * 2.0;  // -1.0 → +1.0
                // FIX 6: Clamp HSV Value to appropriate ceiling depending on pipeline.
                // For SDR (u_isLinear=false) the V channel is [0,1]; clamping to 65504
                // (HALF_FLOAT max) leaves it effectively unclamped, producing out-of-gamut
                // RGB after hsv2rgb(). For HDR linear we keep the wide ceiling.
                float maxLuma = u_isLinear ? 65504.0 : 1.0;
                hsv.z = clamp(hsv.z + lumaShift * 0.5, 0.0, maxLuma);

                vec3 curved = hsv2rgb(hsv);
                return mix(color, curved, u_secondaryCurveMix);
            }

// ----------------------------------------------------------------
// DoF / Bokeh
// ----------------------------------------------------------------

const int SAMPLE_COUNT = 64;
const float PI = 3.14159265;
const float TWO_PI = 6.28318530;
const float GOLDEN_ANGLE = 2.39996323;

            // Pseudo-random jitter for smoother bokeh texture
            float hash12(vec2 p) {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }

            float valueNoise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                return mix(mix(hash12(i + vec2(0.0, 0.0)), hash12(i + vec2(1.0, 0.0)), f.x),
                           mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x), f.y);
            }

            float luminance(vec3 c) {
                return dot(c, vec3(0.2126, 0.7152, 0.0722));
            }

            vec2 safeNormalize(vec2 v) {
                float len2 = dot(v, v);
                return len2 > 1e-6 ? v * inversesqrt(len2) : vec2(0.0, 0.0);
            }

            vec2 rotate2(vec2 v, float a) {
                float s = sin(a);
                float c = cos(a);
                return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
            }

            float apertureBoundary(float theta, float blades) {
                if (blades < 3.0) return 1.0;
                float bladeAngle = TWO_PI / max(blades, 3.0);
                float local = mod(theta + bladeAngle * 0.5, bladeAngle) - bladeAngle * 0.5;
                return clamp(cos(bladeAngle * 0.5) / max(cos(local), 0.08), 0.0, 1.0);
            }

            float circleOfConfusion(float depth) {
                float focusDelta = max(abs(depth - u_focusDist) - 0.0035, 0.0);
                return clamp(focusDelta * u_aperture * 118.0, 0.0, 20.0);
            }

            // Brown-Conrady Distortion — full k1 + k2 model
            // k1: primary barrel/pincushion (quadratic radial)
            // k2: secondary correction (quartic radial) — reduces "mustache" distortion
            vec2 distortUV(vec2 uv) {
                if (u_lensDistortion == 0.0 && u_lensDistortionK2 == 0.0) return uv;

                vec2 center = uv - 0.5;
                float r2 = dot(center, center);
                float r4 = r2 * r2;
                float f = 1.0 + r2 * u_lensDistortion + r4 * u_lensDistortionK2;

                return center * f + 0.5;
            }

            vec3 getBokehColor(vec2 uv, float radius) {
                if (radius < 1.0 && u_lensFringe == 0.0) {
                    return texture(u_image, uv).rgb;
                }

                vec2 pixelSize = 1.0 / u_texSize;
                float anamorphic = max(u_apertureAnamorphic, 0.25);
                float anaStretch = sqrt(anamorphic);
                float apertureRot = radians(u_apertureRotation);
                float blades = float(u_apertureBlades);
                float centerDepth = u_dofEnabled ? texture(u_depth, uv).r : u_focusDist;

                // Optical Vignetting (Cat's Eye Bokeh)
                // Squish the bokeh shape radially depending on distance from center
                vec2 centerDist = uv - 0.5;
                float distLength = length(centerDist);
                // directional squish vector pointing away from center
                vec2 opticalSquishDir = safeNormalize(centerDist);
                float opticalVigFactor = clamp(1.0 - (u_bokehOpticalVig * clamp(distLength * 1.65, 0.0, 1.0)), 0.28, 1.0);

                // Chromatic Aberration offsets (Red/Blue shift)
                vec2 caOffset = centerDist * u_lensFringe * 0.02 * anamorphic;

                // Center sample
                vec3 c_center = vec3(texture(u_image, uv - caOffset).r, texture(u_image, uv).g, texture(u_image, uv + caOffset).b);

                float c_luma = luminance(c_center);
                float c_boost = 1.0 + min(pow(max(c_luma, 0.0), 1.5) * u_bokehHighlightBias, 8.0);

                if (radius < 1.0) return c_center;

                float centerNear = 1.0 - step(u_focusDist, centerDepth);
                float centerFar = 1.0 - centerNear;
                float centerBlur = smoothstep(0.8, 5.0, radius);
                vec3 nearAcc = c_center * c_boost * centerNear * 0.35;
                vec3 farAcc = c_center * c_boost * centerFar * 0.35;
                float nearWeight = centerNear * 0.35;
                float farWeight = centerFar * 0.35;
                float nearCoverage = 0.0;

                // Noise-based rotation to break up concentric artifacts
                float noiseRot = hash12(uv * u_texSize + vec2(float(u_frame) * 0.37, 11.7)) * TWO_PI;

                for (int i = 1; i <= SAMPLE_COUNT; i++) {
                    float fi = float(i) - 0.5;
                    float r = sqrt(fi / float(SAMPLE_COUNT));
                    float theta = fi * GOLDEN_ANGLE + noiseRot;
                    vec2 offsetRaw = vec2(cos(theta), sin(theta)) * r * apertureBoundary(theta, blades);

                    // Shape transform: anamorphic ratio is area-preserving, rotation affects oval and polygon bokeh.
                    offsetRaw = vec2(offsetRaw.x * anaStretch, offsetRaw.y / anaStretch);
                    offsetRaw = rotate2(offsetRaw, apertureRot);

                    // Optical Vignetting
                    float dotDir = dot(offsetRaw, opticalSquishDir);
                    vec2 offsetVig = offsetRaw - opticalSquishDir * dotDir * (1.0 - opticalVigFactor);

                    vec2 offset = offsetVig * radius * pixelSize;

                    vec2 sampleUV = clamp(uv + offset, vec2(0.0), vec2(1.0));

                    vec3 sam;
                    sam.r = texture(u_image, sampleUV - caOffset).r;
                    sam.g = texture(u_image, sampleUV).g;
                    sam.b = texture(u_image, sampleUV + caOffset).b;

                    float nearLayerWeight = 1.0;
                    float farLayerWeight = 1.0;
                    if (u_dofEnabled) {
                        float sampleDepth = texture(u_depth, sampleUV).r;
                        float sampleCoc = circleOfConfusion(sampleDepth);
                        float sampleDistance = length(offsetVig) * radius;
                        float cocCoversTarget = smoothstep(sampleDistance - 1.25, sampleDistance + 1.25, sampleCoc);
                        float sampleBlur = smoothstep(0.8, 3.0, sampleCoc);
                        float sampleNear = 1.0 - step(u_focusDist, sampleDepth);
                        float sampleFar = 1.0 - sampleNear;

                        // Reverse gather: a sample contributes when its own CoC covers this pixel.
                        // Same-layer gather keeps broad defocus smooth, while cross-layer bleed stays suppressed.
                        nearLayerWeight = sampleNear * max(cocCoversTarget * sampleBlur, centerNear * sampleNear * 0.55);
                        farLayerWeight = sampleFar * max(cocCoversTarget * sampleBlur, centerFar * sampleFar * 0.45);
                        nearCoverage = max(nearCoverage, nearLayerWeight);
                    }

                    float luma = luminance(sam);
                    float boost = 1.0 + min(pow(max(luma, 0.0), 1.5) * u_bokehHighlightBias, 8.0);

                    // Soap Bubble Effect (Edge brightening)
                    // r ranges from 0 to 1
                    float rim = pow(r, 4.0) * u_bokehSoapBubble;
                    float lensWeight = 1.0 + rim * 0.35;
                    vec3 bokehSample = sam * boost * (1.0 + rim * 0.6);

                    nearAcc += bokehSample * nearLayerWeight * lensWeight;
                    farAcc += bokehSample * farLayerWeight * lensWeight;
                    nearWeight += nearLayerWeight * lensWeight;
                    farWeight += farLayerWeight * lensWeight;
                }

                vec3 farColor = farWeight > 0.001 ? farAcc / farWeight : c_center;
                vec3 nearColor = nearWeight > 0.001 ? nearAcc / nearWeight : c_center;
                float farAlpha = centerFar * centerBlur;
                float nearAlpha = max(centerNear * centerBlur, smoothstep(0.04, 0.35, nearCoverage));
                return mix(mix(c_center, farColor, farAlpha), nearColor, clamp(nearAlpha, 0.0, 1.0));
            }

            // ----------------------------------------------------------------
            // Color Ops
            // ----------------------------------------------------------------

            vec3 applyLUT(vec3 color, sampler3D lut, float size) {
                float scale = (size - 1.0) / size;
                float offset = 0.5 / size;
                vec3 coords = clamp(color, 0.0, 1.0) * scale + offset;
                return texture(lut, coords).rgb;
            }

            // 3.5.0: ARRI false colour (ALEXA/AMIRA LF), bands in % of the
            // Rec.709 video signal: purple 0-2.5 (black clip), blue 2.5-4,
            // green 38-42 (18 % grey), pink 52-56 (one stop over, skin), yellow
            // 97-99, red 99-100 (clip). Everything else is shown as luma grey.
            // The signal is a fixed reference (BT.709 OETF of scene-linear
            // luminance, which puts 18 % grey at 40.9 %), not whatever view is
            // selected, so exposure reads the same under any view. The old map
            // used custom bands (cyan at 38-40, green at 42-45) on the display
            // luma, where 18 % grey never reached green.
            float arriSignal(float y) {
                y = max(y, 0.0);
                return y < 0.018 ? 4.5 * y : 1.099 * pow(y, 0.45) - 0.099;
            }
            vec3 getFalseColorMap(float s) {
                float p = s * 100.0;
                if (p >= 99.0) return vec3(1.0, 0.0, 0.0);
                if (p >= 97.0) return vec3(1.0, 1.0, 0.0);
                if (p >= 52.0 && p < 56.0) return vec3(1.0, 0.5, 0.75);
                if (p >= 38.0 && p < 42.0) return vec3(0.0, 0.85, 0.2);
                if (p >= 2.5 && p < 4.0) return vec3(0.1, 0.35, 1.0);
                if (p < 2.5) return vec3(0.55, 0.0, 0.8);
                return vec3(clamp(s, 0.0, 1.0));
            }

            // ── HDR nits heatmap ────────────────────────────────────────
            // getFalseColorMap above is an *exposure* tool: ARRI bands on a
            // Rec.709 reference signal, which says nothing about absolute luminance.
            // The menu offered "HDR Heatmap" as a separate entry and wired it to
            // the same falseColor flag, so the two were one feature under two
            // names and neither reported nits.
            //
            // This maps scene luminance to absolute cd/m2, with the boundaries
            // an HDR colourist actually works to. The argument arrives in the
            // internal scale where 1.0 == 100 nits, the same convention the
            // SDR->HDR nodes use, so nits = v * 100.
            //
            // No backticks in here. This comment sits inside a JS template
            // literal, so a backtick terminates the shader string and every
            // line after it is parsed as JavaScript. That is exactly what
            // happened: a backtick-quoted 'v' in this comment broke the whole
            // module, and the viewer node rendered with no UI at all.
            //
            // The anchor is ITU-R BT.2408: HDR Reference White = 203 cd/m2,
            // "the nominal signal level obtained from an HDR camera and a 100%
            // reflectance white card" -- 58% PQ, 75% HLG. That is the boundary
            // that matters: below it is diffuse, above it is specular. A
            // heatmap without it is a colour ramp; with it, it is instrumentation.
            //
            //     nits        band                       colour
            //     < 0.01      below the noise floor      near-black
            //     0.01 - 5    deep shadow                indigo
            //     5 - 50      shadow to low midtone      blue -> teal
            //     50 - 203    midtone up to diffuse      teal -> green
            //     = 203       BT.2408 Reference White    white line
            //     203 - 400   specular, comfortable      yellow
            //     400 - 1000  specular, bright           orange
            //     1000 - 4000 mastering headroom         red
            //     > 4000      beyond common masters      magenta
            vec3 getHDRHeatmap(float v) {
                float nits = max(v, 0.0) * 203.0;   // BT.2408: linear 1.0 = 203 nits, as the probe and nodes use

                // A visible band either side of reference white, so the eye can
                // land on 203 without reading a legend.
                if (nits >= 200.0 && nits <= 206.0) return vec3(1.0, 1.0, 1.0);

                if (nits < 0.01)   return vec3(0.04, 0.02, 0.08);
                if (nits < 5.0)    return mix(vec3(0.15, 0.05, 0.45), vec3(0.10, 0.25, 0.75), nits / 5.0);
                if (nits < 50.0)   return mix(vec3(0.10, 0.25, 0.75), vec3(0.05, 0.70, 0.70), (nits - 5.0) / 45.0);
                if (nits < 203.0)  return mix(vec3(0.05, 0.70, 0.70), vec3(0.20, 0.85, 0.25), (nits - 50.0) / 153.0);
                if (nits < 400.0)  return mix(vec3(0.95, 0.95, 0.20), vec3(1.00, 0.75, 0.10), (nits - 203.0) / 197.0);
                if (nits < 1000.0) return mix(vec3(1.00, 0.75, 0.10), vec3(1.00, 0.40, 0.05), (nits - 400.0) / 600.0);
                if (nits < 4000.0) return mix(vec3(1.00, 0.40, 0.05), vec3(0.90, 0.05, 0.05), (nits - 1000.0) / 3000.0);
                return vec3(1.0, 0.0, 1.0);
            }

            // ACES Tone Mapping (Approx)
            vec3 toneMapACES(vec3 color) {
                const float a = 2.51;
                const float b = 0.03;
                const float c = 2.43;
                const float d = 0.59;
                const float e = 0.14;
                return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
            }

            // Simple Reinhard Tone Mapping
            vec3 toneMapReinhard(vec3 c) {
                return c / (c + vec3(1.0));
            }

            // IEC 61966-2-1 sRGB OETF — required for linear→display conversion
            vec3 linearToSRGB(vec3 linear) {
                bvec3 cutoff = lessThan(linear, vec3(0.0031308));
                vec3 higher = vec3(1.055) * pow(max(linear, vec3(0.0031308)), vec3(1.0/2.4)) - vec3(0.055);
                vec3 lower = linear * vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            // Inverse: sRGB→linear (for PNG textures that arrive gamma-encoded)
            vec3 sRGBToLinear(vec3 srgb) {
                bvec3 cutoff = lessThan(srgb, vec3(0.04045));
                vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
                vec3 lower = srgb / vec3(12.92);
                return mix(higher, lower, vec3(cutoff));
            }

            // ─────────────────────────────────────────────────────────────────
            // Display LUT / Camera Log Transforms  (v3.3 — Spec-Accurate)
            //
            // All forward cases assume scene-linear input, 18% grey = 0.18.
            // Verified 18% grey output values (cross-checked against
            // colour-science library, which cites manufacturer specification
            // documents directly):
            //
            //   LogC3:   0.391  | LogC4: 0.278  | S-Log3: 0.411
            //   V-Log:   0.423  | F-Log2: 0.391 | C-Log3: 0.343
            //   Log3G10: 0.333  | DaVinci: 0.336 | BMD Gen5: 0.384 | N-Log: 0.364
            //
            // IDT cases (Log → Linear): decode camera log footage to
            // scene-linear for use in the grading pipeline.
            //
            // L10 = ln(10) converts GLSL log() [natural] to log10 equivalent:
            //   log10(x) = log(x) / L10
            // ─────────────────────────────────────────────────────────────────
            // 3.5.0: display modes that return an encoded signal (Rec.709
            // BT.1886, every camera log encode). main() must not add the sRGB
            // OETF on top of them.
            bool displayModeEncodes(int m) {
                return m == 2 || m == 4 || m == 6 || (m >= 11 && m <= 21);
            }

            vec3 applyDisplayLUT(vec3 c, int mode) {
                const float L10 = 2.302585; // ln(10)

                switch(mode) {

                // ── ODT: Display / Tonemap ─────────────────────────────────
                case 1: { // sRGB (Display) — linearToSRGB applied at end of main()
                    return c;
                }
                case 2: { // Rec.709 display: inverse BT.1886 EOTF (gamma 2.4)
                    // 3.5.0: was the BT.709 camera OETF followed by the sRGB OETF,
                    // i.e. encoded twice (18 % grey at ~67 %). A Rec.709 monitor
                    // decodes with BT.1886, so the view encodes with its inverse,
                    // once; main() skips the sRGB OETF for this mode.
                    return pow(max(c, vec3(0.0)), vec3(1.0 / 2.4));
                }
                case 3: { // Filmic — Hable/Uncharted2
                    float A=0.15,B=0.50,C=0.10,D=0.20,E=0.02,F=0.30;
                    vec3 v = c * 2.0;
                    vec3 curr = ((v*(A*v+C*B)+D*E)/(v*(A*v+B)+D*F))-E/F;
                    float wh = ((11.2*(A*11.2+C*B)+D*E)/(11.2*(A*11.2+B)+D*F))-E/F;
                    return curr / wh;
                }
                case 8: { // Reinhard global tonemap
                    return c / (c + vec3(1.0));
                }
                case 9: { // ACES Filmic (Narkowicz fit)
                    return toneMapACES(c);
                }

                // ── ODT: Camera Log Encoding  ─────────────────────────────
                // ── Forward: scene-linear → log code values ──────────────

                case 4: { // ARRI LogC3 EI800 (ARRI Specification v4.0, 2022)
                    // Constants from DATA_ALEXA_LOG_C_CURVE_CONVERSION["SUP 3.x"][800]
                    // cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537
                    // e=5.367655 (linear toe slope), f=0.092809 (linear toe offset)
                    const float lc3_cut=0.010591, lc3_a=5.555556, lc3_b=0.052272;
                    const float lc3_c=0.247190,   lc3_d=0.385537;
                    const float lc3_e=5.367655,   lc3_f=0.092809;
                    return mix(
                        lc3_e * c + lc3_f,
                        lc3_c * (log(lc3_a * max(c,vec3(0.0)) + lc3_b) / L10) + lc3_d,
                        vec3(greaterThan(c, vec3(lc3_cut)))
                    );
                }

                case 11: { // ARRI LogC4 (Alexa 35, ARRI Specification v1.0, 2022)
                    // CONSTANTS_ARRILOGC4: a=2231.826, b=0.907136, c=0.092864
                    //                     s=0.113597, t=-0.018057
                    // Formula: E_p = (log2(a*E + 64) - 6) / 14 * b + c  for E >= t
                    //          E_p = (E - t) / s                         for E < t
                    const float lc4_a=2231.82630906768830, lc4_b=0.90713587487781030;
                    const float lc4_c=0.09286412512218964, lc4_s=0.11359720861058910;
                    const float lc4_t=-0.01805699611991131;
                    vec3 logBranch = (log2(lc4_a * max(c, vec3(lc4_t)) + 64.0) - 6.0) / 14.0 * lc4_b + lc4_c;
                    vec3 linBranch = (c - lc4_t) / lc4_s;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(lc4_t))));
                }



                case 14: { // Fujifilm F-Log2 (Fujifilm Specification v1.0, 2022)
                    // CONSTANTS_FLOG2: cut1=0.000889, a=5.555556, b=0.064829, c=0.245281, d=0.384316, e=8.799461, f=0.092864
                    const float fl2_cut=0.000889, fl2_a=5.555556, fl2_b=0.064829;
                    const float fl2_c=0.245281, fl2_d=0.384316, fl2_e=8.799461, fl2_f=0.092864;
                    vec3 logV = fl2_c * (log(fl2_a * max(c,vec3(0.0)) + fl2_b) / L10) + fl2_d;
                    vec3 linV = fl2_e * c + fl2_f;
                    return mix(linV, logV, vec3(greaterThanEqual(c, vec3(fl2_cut))));
                }

                case 12: { // Canon C-Log3 v1.2 (Canon Specification, Rev.2020)
                    // Input is rescaled by /0.9 before encoding.
                    // Three-way piecewise: negative log | linear | positive log
                    // Cut points in rescaled space: lower ≈ -0.009670, upper ≈ 0.014043
                    vec3 x = c / 0.9;
                    const float cl3_k=14.98325, cl3_a=0.36726845;
                    const float cl3_neg_d=0.12783901;
                    const float cl3_lin_m=1.9754798, cl3_lin_b=0.12512219;
                    const float cl3_pos_d=0.12240537;
                    const float cl3_lo=-0.009670, cl3_hi=0.014043;
                    vec3 negBranch = -cl3_a * log(max(-x * cl3_k + 1.0, vec3(1e-10))) / L10 + cl3_neg_d;
                    vec3 linBranch = cl3_lin_m * x + cl3_lin_b;
                    vec3 posBranch =  cl3_a * log(x * cl3_k + 1.0) / L10 + cl3_pos_d;
                    vec3 result = mix(negBranch, linBranch, vec3(greaterThanEqual(x, vec3(cl3_lo))));
                    result      = mix(result,    posBranch, vec3(greaterThan(x,   vec3(cl3_hi))));
                    return result;
                }

                case 19: { // RED Log3G10 v2 (IPP2, REDCINE-X PRO — colour-science Nattress2016a)
                    // y = sign(x+0.01) * 0.224282 * log10(|x+0.01| * 155.975327 + 1)
                    // 18% grey → 0.333 (exactly 1/3), 0 → 0.0916
                    vec3 xoff = c + 0.01;
                    return sign(xoff) * 0.224282 * log(abs(xoff) * 155.975327 + 1.0) / L10;
                }

                case 20: { // DaVinci Intermediate (Blackmagic Design, 2020)
                    // Blackmagic "DaVinci Wide Gamut / DaVinci Intermediate" spec:
                    //   x <= 0.00262409 : y = x * 10.44426855
                    //   else            : y = (log2(x + 0.0075) + 7.0) * 0.07329248
                    // 18 % grey -> 0.336, matching OpenColorIO's studio config and
                    // Radiance's Python curves (color/encodings.py, color/transfer.py).
                    // 3.5.0: an earlier "BUG-1 FIX" replaced this with a log10 curve
                    // that put 18 % grey at 0.447; that was the error, not this.
                    const float di_A=0.0075, di_B=7.0, di_C=0.07329248;
                    const float di_M=10.44426855, di_cut=0.00262409;
                    vec3 logV = (log2(max(c + di_A, vec3(1e-10))) + di_B) * di_C;
                    vec3 linV = c * di_M;
                    return mix(linV, logV, vec3(greaterThan(c, vec3(di_cut))));
                }

                case 15: { // Blackmagic Film Generation 5 (BMD Specification, 2021)
                    // CONSTANTS_BLACKMAGIC_FILM_GENERATION_5:
                    //   A=0.08692876, B=0.005494072, C=0.530013339, D=8.283605932, E=0.092465753, LIN_CUT=0.005
                    const float bm_A=0.08692876065491224, bm_B=0.005494072432257808;
                    const float bm_C=0.5300133392291939,  bm_D=8.283605932402494;
                    const float bm_E=0.09246575342465753, bm_cut=0.005;
                    vec3 logV = bm_A * log(max(c + bm_B, vec3(1e-10))) + bm_C;
                    vec3 linV = bm_D * c + bm_E;
                    return mix(linV, logV, vec3(greaterThanEqual(c, vec3(bm_cut))));
                }

                case 21: { // Nikon N-Log (Nikon Specification, 2018)
                    // CONSTANTS_NLOG: cut1=0.328, a=0.635386, b=0.0075, c=0.146628, d=0.605083
                    // Below cut1: a * (y+b)^(1/3)   |  Above cut1: c * ln(y) + d
                    const float nl_cut=0.328, nl_a=0.635386119257087, nl_b=0.0075;
                    const float nl_c=0.1466275659824047, nl_d=0.6050830889540567;
                    vec3 logV = nl_c * log(max(c, vec3(1e-10))) + nl_d;
                    vec3 cbrtV = nl_a * pow(max(c + nl_b, vec3(1e-10)), vec3(1.0 / 3.0));
                    return mix(cbrtV, logV, vec3(greaterThanEqual(c, vec3(nl_cut))));
                }

                case 6: { // Generic Linear → Log (Cineon-style)
                    return log2(max(c, vec3(1e-10)) * 5.55 + 1.0) / log2(6.55);
                }

                // ── IDT: Camera Log → Scene-Linear ────────────────────────
                // Decode log-encoded footage back to scene-linear for grading.

                case 22: { // IDT LogC4 → Linear
                    // Inverse of case 11: E = (2^((E_p - c)/b * 14 + 6) - 64) / a  for E_p >= c+0 (log region)
                    const float lc4_a=2231.82630906768830, lc4_b=0.90713587487781030;
                    const float lc4_c=0.09286412512218964, lc4_s=0.11359720861058910;
                    const float lc4_t=-0.01805699611991131;
                    float lc4_log_cut = (log2(lc4_a * lc4_t + 64.0) - 6.0) / 14.0 * lc4_b + lc4_c;
                    vec3 logBranch = (exp2((c - lc4_c) / lc4_b * 14.0 + 6.0) - 64.0) / lc4_a;
                    vec3 linBranch = c * lc4_s + lc4_t;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(lc4_log_cut))));
                }



                case 23: { // IDT C-Log3 → Linear
                    // Inverse of case 12 — positive log branch (normal scene content)
                    // x_rescaled = (10^((y - 0.12240537) / 0.36726845) - 1) / 14.98325
                    // x_real = x_rescaled * 0.9
                    vec3 logBranch = (pow(vec3(10.0), (c - 0.12240537) / 0.36726845) - 1.0) / 14.98325 * 0.9;
                    // Linear branch for very low encoded values (rare in normal content)
                    vec3 linBranch = (c - 0.12512219) / 1.9754798 * 0.9;
                    float cl3_lin_cv = 1.9754798 * 0.014043 + 0.12512219; // ~0.1527
                    return mix(linBranch, logBranch, vec3(greaterThan(c, vec3(cl3_lin_cv))));
                }

                case 24: { // IDT F-Log2 → Linear
                    // Inverse of case 14: x = (10^((y - d) / c) - b) / a
                    const float fl2_a=5.555556, fl2_b=0.064829, fl2_c=0.245281;
                    const float fl2_d=0.384316, fl2_e=8.799461, fl2_f=0.092864;
                    float fl2_cut_cv = fl2_e * 0.000889 + fl2_f; // code value at cut
                    vec3 logBranch = (pow(vec3(10.0), (c - fl2_d) / fl2_c) - fl2_b) / fl2_a;
                    vec3 linBranch = (c - fl2_f) / fl2_e;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(fl2_cut_cv))));
                }

                case 25: { // IDT Log3G10 → Linear
                    // Inverse of case 19: x = sign(y)/155.975 * (10^(|y|/0.224282) - 1) - 0.01
                    vec3 s = sign(c);
                    return s * (pow(vec3(10.0), abs(c) / 0.224282) - 1.0) / 155.975327 - 0.01;
                }

                case 26: { // IDT DaVinci Intermediate → Linear (inverse of case 20)
                    const float di_A=0.0075, di_B=7.0, di_C=0.07329248;
                    const float di_M=10.44426855, di_log_cut=0.02740668;
                    vec3 logBranch = exp2(c / di_C - di_B) - di_A;
                    vec3 linBranch = c / di_M;
                    return mix(linBranch, logBranch, vec3(greaterThan(c, vec3(di_log_cut))));
                }

                case 27: { // IDT BMD Film Gen5 → Linear
                    // Inverse of case 15: x = exp((y - C) / A) - B
                    const float bm_A=0.08692876065491224, bm_B=0.005494072432257808;
                    const float bm_C=0.5300133392291939, bm_D=8.283605932402494, bm_E=0.09246575342465753;
                    float bm_log_cut = bm_D * 0.005 + bm_E; // code value at LIN_CUT=0.005
                    vec3 logBranch = exp((c - bm_C) / bm_A) - bm_B;
                    vec3 linBranch = (c - bm_E) / bm_D;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(bm_log_cut))));
                }

                case 28: { // IDT N-Log → Linear
                    // Inverse of case 21:
                    // Below cut CV: x = (y/a)^3 - b
                    // Above cut CV: x = exp((y - d) / c)
                    const float nl_a=0.635386119257087, nl_b=0.0075;
                    const float nl_c=0.1466275659824047, nl_d=0.6050830889540567;
                    // Cut CV: nl_a * (0.328 + nl_b)^(1/3)
                    float nl_cut_cv = nl_a * pow(0.328 + nl_b, 1.0 / 3.0);
                    vec3 cbrtBranch = pow(c / nl_a, vec3(3.0)) - nl_b;
                    vec3 logBranch  = exp((c - nl_d) / nl_c);
                    return max(mix(cbrtBranch, logBranch, vec3(greaterThanEqual(c, vec3(nl_cut_cv)))), vec3(0.0));
                }

                case 29: { // IDT LogC3 → Linear
                    const float lc3_cut=0.010591, lc3_a=5.555556, lc3_b=0.052272;
                    const float lc3_c=0.247190,   lc3_d=0.385537;
                    const float lc3_e=5.367655,   lc3_f=0.092809;
                    float lc3_cut_cv = lc3_e * lc3_cut + lc3_f;
                    vec3 logBranch = (pow(vec3(10.0), (c - lc3_d) / lc3_c) - lc3_b) / lc3_a;
                    vec3 linBranch = (c - lc3_f) / lc3_e;
                    return mix(linBranch, logBranch, vec3(greaterThan(c, vec3(lc3_cut_cv))));
                }

                case 30: { // IDT V-Log → Linear
                    const float vl_b=0.00873, vl_c=0.241514, vl_d=0.598206;
                    float vl_cut_cv = 5.625 * 0.01 + 0.125; // 0.18125
                    vec3 logBranch = pow(vec3(10.0), (c - vl_d) / vl_c) - vl_b;
                    vec3 linBranch = (c - 0.125) / 5.625;
                    return mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(vl_cut_cv))));
                }

                case 34: { // IDT Rec.709 → Linear
                    bvec3 lo = lessThan(c, vec3(0.081));
                    vec3 low = c / 4.5;
                    vec3 high = pow((c + vec3(0.099)) / vec3(1.099), vec3(1.0 / 0.45));
                    return mix(high, low, vec3(lo));
                }

                case 35: { // IDT sRGB → Linear
                    return sRGBToLinear(c);
                }

                // ── v4.3: Previously missing forward + IDT cases ─────────────

                case 13: { // V-Log (Panasonic) — Linear → V-Log
                    // Source: Panasonic V-Log/V-Gamut Spec (2014), colour-science verified
                    // 18% grey → 0.423946  |  cut1=0.01, cut2=0.181
                    // Note: log10(x) = log(x) / L10  (GLSL ES has no log10 built-in)
                    const float vl_fwd_b=0.00873, vl_fwd_c=0.241514, vl_fwd_d=0.598206;
                    const float vl_fwd_cut=0.01;
                    vec3 logV = vl_fwd_c * (log(max(c + vl_fwd_b, vec3(1e-10))) / L10) + vl_fwd_d;
                    vec3 linV = 5.6 * c + 0.125;
                    return mix(linV, logV, vec3(greaterThan(c, vec3(vl_fwd_cut))));
                }

                case 16: { // Sony S-Log3 — Linear → S-Log3
                    // Source: colour-science log_encoding_SLog3 (Sony Specification 2014)
                    // Formula: (420 + log10((x+0.01)/0.19) * 261.5) / 1023
                    // Verified: 18% grey → 0.410557  |  cut=0.01125 (scene-linear)
                    // Lin slope: (171.2102946929-95) / 0.01125 = 6774.2484
                    const float sl3_cut = 0.01125;
                    vec3 logV = (420.0 + log(max(c + 0.01, vec3(1e-10)) / 0.19) / log(10.0) * 261.5) / 1023.0;
                    vec3 linV = (c * 6774.2484 + 95.0) / 1023.0;
                    return mix(linV, logV, vec3(greaterThanEqual(c, vec3(sl3_cut))));
                }

                case 31: { // IDT S-Log3 → Linear
                    // Exact inverse of case 16. Source: Sony S-Log3 Specification (2014)
                    // log_out = 10^((v*1023-420)/261.5) * 0.19 - 0.01
                    // lin_out = (v*1023-95) * 0.01125 / (171.2102946929-95)
                    // CUT_CV  = (95/1023) ≈ 0.09285... wait, cut is at lin_break encoded value:
                    // cut_cv = (0.01125*(171.2102946929-95)/0.01125 + 95)/1023 = 171.2102946929/1023 ≈ 0.16736
                    const float sl3_cut_cv = 0.167361;
                    vec3 logBranch = pow(vec3(10.0), (c * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01;
                    vec3 linBranch = (c * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0);
                    return max(mix(linBranch, logBranch, vec3(greaterThanEqual(c, vec3(sl3_cut_cv)))), vec3(0.0));
                }

                case 32: { // Output: ACEScg (AP1 scene-linear)
                    // Linear sRGB (D65) → ACEScg (AP1, D60 adapted)
                    // Matrix: Bradford D65→D60 + Rec.709→AP1 primaries
                    // Source: ACES v1.3 CTL reference transforms
                    mat3 M = mat3(
                        0.59719, 0.07600, 0.02840,   // col 0 (R out)
                        0.35458, 0.90834, 0.13383,   // col 1 (G out)
                        0.04823, 0.01566, 0.83777    // col 2 (B out)
                    );
                    return M * c;
                }

                case 33: { // Output: ACES2065-1 (AP0 scene-linear)
                    // Linear sRGB (D65) → ACES2065-1 (AP0, D60 adapted)
                    // Covers full spectral locus — archive-grade interchange
                    // Source: ACES v1.3, S-2014-004
                    mat3 M = mat3(
                        0.43963, 0.08978, 0.01754,   // col 0
                        0.38298, 0.81380, 0.11170,   // col 1
                        0.17739, 0.09642, 0.87076    // col 2
                    );
                    return M * c;
                }

                default:
                    return c;
                }
            }


            // Sobel edge detection for focus peaking
            // FIX (v3.5.1): For linear HDR input, raw luma can be 0–100+.
            // The Sobel gradient magnitude * 255 would be enormous, causing
            // the threshold=30 to fire on every bright-area transition (red
            // overlay on all highlights instead of just sharp edges).
            // Solution: compress luma through Reinhard before Sobel so the
            // gradient lives in 0–1 display-normalised space in both SDR and HDR.
            float fp_compressLuma(float l) {
                return u_isLinear ? l / (l + 1.0) : l;
            }
            vec3 applyFocusPeaking(vec3 color, vec2 uv) {
                vec2 px = 1.0 / u_texSize;
                // 3x3 Sobel — luma sampled from u_image, HDR-compressed if linear
                float tl = fp_compressLuma(dot(texture(u_image, uv + vec2(-px.x, -px.y)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float tc = fp_compressLuma(dot(texture(u_image, uv + vec2(  0.0, -px.y)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float tr = fp_compressLuma(dot(texture(u_image, uv + vec2( px.x, -px.y)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float ml = fp_compressLuma(dot(texture(u_image, uv + vec2(-px.x,   0.0)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float mr = fp_compressLuma(dot(texture(u_image, uv + vec2( px.x,   0.0)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float bl = fp_compressLuma(dot(texture(u_image, uv + vec2(-px.x,  px.y)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float bc = fp_compressLuma(dot(texture(u_image, uv + vec2(  0.0,  px.y)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float br = fp_compressLuma(dot(texture(u_image, uv + vec2( px.x,  px.y)).rgb, vec3(0.2126, 0.7152, 0.0722)));
                float gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br;
                float gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
                // Gradient is now in 0–1 space → multiply by 255 gives 0–255 equivalent
                float mag = sqrt(gx*gx + gy*gy) * 255.0;
                if (mag > u_focusPeakThreshold) {
                    return mix(color, vec3(1.0, 0.0, 0.2), 0.85);
                }
                return color;
            }

            // Safety: sanitizes NaNs and Infs to black or max value
            vec3 sanitize(vec3 c) {
                bvec3 is_nan = isnan(c);
                bvec3 is_inf = isinf(c);
                vec3 safe = c;
                if (any(is_nan)) safe = vec3(0.0);
                if (any(is_inf)) safe = clamp(c, 0.0, 65504.0);
                return safe;
            }

            // ----------------------------------------------------------------
            // Advanced Grading Ops
            // ----------------------------------------------------------------

            vec3 applyTempTint(vec3 color, float temp, float tint) {
                vec3 shift = vec3(0.0);
                // Temp: Warm (Orange) / Cool (Blue)
                shift.r += temp;
                shift.b -= temp;
                // Tint: Magenta / Green
                shift.g -= tint;
                return clamp(color + shift, 0.0, 65504.0);
            }

            vec4 rgb2cmyk(vec3 rgb) {
                float k = 1.0 - max(max(rgb.r, rgb.g), rgb.b);
                float c = (1.0 - rgb.r - k) / (1.0 - k + 0.0001);
                float m = (1.0 - rgb.g - k) / (1.0 - k + 0.0001);
                float y = (1.0 - rgb.b - k) / (1.0 - k + 0.0001);
                return vec4(c, m, y, k);
            }

            // ── ACEScct COLOR SCIENCE TRANSFORMS ──
            const mat3 LIN_SRGB_TO_ACESCG = mat3(
                0.59719, 0.07600, 0.02840,
                0.35458, 0.90834, 0.13383,
                0.04823, 0.01566, 0.83777
            );

            const mat3 ACESCG_TO_LIN_SRGB = mat3(
                 1.60475, -0.10208, -0.00327,
                -0.53108,  1.10813, -0.07276,
                -0.07367, -0.00605,  1.07602
            );

            float lin_to_ACEScct(float in_val) {
                if (in_val <= 0.0078125) {
                    return 10.5402377416545 * in_val + 0.0729055341958355;
                } else {
                    return (log2(in_val) + 9.72) / 17.52;
                }
            }

            vec3 lin_to_ACEScct_vec3(vec3 v) {
                return vec3(lin_to_ACEScct(v.r), lin_to_ACEScct(v.g), lin_to_ACEScct(v.b));
            }

            float ACEScct_to_lin(float in_val) {
                if (in_val > 0.155251141552511) {
                    return exp2(in_val * 17.52 - 9.72);
                } else {
                    return (in_val - 0.0729055341958355) / 10.5402377416545;
                }
            }

            vec3 ACEScct_to_lin_vec3(vec3 v) {
                return vec3(ACEScct_to_lin(v.r), ACEScct_to_lin(v.g), ACEScct_to_lin(v.b));
            }

${GRADE_GLSL}
            vec3 applyGrading(vec3 color, vec3 lift, vec3 gamma, vec3 gain, vec3 offset) {
                // Resolve-style order: Offset, Lift (pivoted at white), Gain
                // (pivoted at black), Gamma. Every step is the shared
                // definition; see js/radiance_grade.js.
                return radGradeOrder(color, offset, lift, gain, gamma);
            }

            vec3 applyContrast(vec3 color, float contrast, float pivot) {
                return radContrast(color, contrast, pivot);
            }

            // v2.5 Pro Pro: Cinematic S-Curve Shadows/Highlights
            vec3 applyShadowsHighlights(vec3 color, float shadows, float highlights) {
                // 3.5.0: neutral is a no-op, and negatives survive. This ended in
                // max(color, 0) on every pixel, so out-of-gamut scene values were
                // zeroed before the gamut warning or the graded EXR saw them.
                if (shadows == 0.0 && highlights == 0.0) return color;
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
                // Quadratic weights for smoother roll-off
                float sWeight = pow(1.0 - smoothstep(0.0, 0.5, luma), 2.0);
                float hWeight = pow(smoothstep(0.5, 1.0, luma), 2.0);

                // shadows in [-1,1], lift(pos) or crush(neg)
                color *= (1.0 + shadows * sWeight * 0.5);
                // highlights in [-1,1], expand(pos) or compress(neg)
                color *= (1.0 + highlights * hWeight * 0.5);

                return color;
            }

            // v3.3: 3-Way Log Wheels (Shadow/Midtone/Highlight targeting)
            vec3 applyLogWheels(vec3 color, vec3 shadow, vec3 midtone, vec3 highlight) {
                if (shadow == vec3(0.0) && midtone == vec3(0.0) && highlight == vec3(0.0)) return color;
                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

                // Smooth weight distribution based on luminance
                float sWeight = 1.0 - smoothstep(0.0, 0.45, luma);
                float hWeight = smoothstep(0.55, 1.0, luma);
                float mWeight = 1.0 - sWeight - hWeight;

                // Log wheels are applied multiplicatively (similar to localized exposure)
                color *= (1.0 + shadow * sWeight + midtone * mWeight + highlight * hWeight);
                return color;   // 3.5.0: a gain, so no clamp (see applyShadowsHighlights)
            }

            vec3 hash32(vec2 p) {
                vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
                p3 += dot(p3, p3.yxz+33.33);
                return fract((p3.xxy+p3.yzz)*p3.zyx);
            }

            vec3 blendOverlay(vec3 base, vec3 blend) {
                return mix(1.0 - 2.0 * (1.0 - base) * (1.0 - blend), 2.0 * base * blend, step(base, vec3(0.5)));
            }

            float rand(vec2 co) {
                // Legacy support - simple animated grain seed
                return fract(sin(dot(co.xy + u_time, vec2(12.9898,78.233))) * 43758.5453);
            }

            // ── v3.4: Printer Lights ─────────────────────────────────────────
            // Each channel is multiplied by 2^(offset/50), mirroring how a
            // film printer light step modulates per-channel density.
            vec3 applyPrinterLights(vec3 color, float r, float g, float b) {
                if (r == 0.0 && g == 0.0 && b == 0.0) return color;
                color.r *= pow(2.0, r / 50.0);
                color.g *= pow(2.0, g / 50.0);
                color.b *= pow(2.0, b / 50.0);
                return color;
            }

            // ── v3.4: Soft Clip ──────────────────────────────────────────────
            // Per-channel Reinhard-style knee that rolls values above 'threshold'
            // smoothly into a shoulder.  0 = hard clip pass-through.
            vec3 applySoftClip(vec3 c, float knee) {
                if (knee <= 0.0) return c;
                float thresh = 1.0 - knee * 0.5;
                vec3 excess = max(c - thresh, 0.0);
                // Smooth shoulder: thresh + excess / (1 + excess / knee)
                vec3 shoulder = thresh + excess / (1.0 + excess / max(knee, 0.001));
                return mix(c, shoulder, step(thresh, c));
            }

// ----------------------------------------------------------------
// v3.2 Phase 12: GPU Bilateral Filter (Edge-Preserving Denoise)
//
// A 7×7 bilateral kernel that down-weights neighbours whose colour
// differs significantly from the centre pixel.  This preserves
// hard edges (skin/hair/object silhouettes) while still smoothing
// flat areas (sky, walls, bokeh).
//
// When u_bilateralHalfRes is true the JS side has already pre-rendered
// the bilateral result into a half-res FBO; this function is still called
// but at 0.5× resolution — WebGL's bilinear sampler handles the upscale
// for free at full-resolution display, cutting per-pixel cost by ~4×.
// ----------------------------------------------------------------
vec3 getBilateralColor(vec2 uv) {
    vec2 px = 1.0 / u_texSize;
    vec3 center = texture(u_image, uv).rgb;

    // Guard: if bilateral is disabled, return raw sample
    if (u_bilateralSigmaD <= 0.0) return center;

    float sigD2 = u_bilateralSigmaD * u_bilateralSigmaD;
    float sigR2 = u_bilateralSigmaR * u_bilateralSigmaR;

    vec3 acc = vec3(0.0);
    float wSum = 0.0;

    // 7×7 kernel  (radius = 3 taps each direction)
    for (int dx = -3; dx <= 3; dx++) {
        for (int dy = -3; dy <= 3; dy++) {
            vec2 offset = vec2(float(dx), float(dy)) * px;
            vec3 s = texture(u_image, uv + offset).rgb;

            // Spatial weight — Gaussian over pixel distance
            float spatialDist2 = float(dx * dx + dy * dy);
            float wSpatial = exp(-spatialDist2 / (2.0 * sigD2));

            // Range weight — Gaussian over colour distance
            vec3 diff = s - center;
            float rangeDist2 = dot(diff, diff);
            float wRange = exp(-rangeDist2 / (2.0 * sigR2));

            float w = wSpatial * wRange;
            acc  += s * w;
            wSum += w;
        }
    }

    return acc / max(wSum, 1e-5);
}

// Legacy alias — keeps existing denoise path working
vec3 getDenoiseColor(vec2 uv) {
    return getBilateralColor(uv);
}

    // ----------------------------------------------------------------
    // HSL Qualifier
    // ----------------------------------------------------------------

    vec3 applyMidDetail(vec3 color, vec2 uv, float amount) {
        if (amount == 0.0) return color;
        vec3 blurred = getDenoiseColor(uv); // Reuse existing blur
        vec3 detail = color - blurred;
        return color + detail * amount;
    }


    vec3 rgb2hcv(vec3 rgb) {
        vec4 P = (rgb.g < rgb.b) ? vec4(rgb.bg, -1.0, 2.0/3.0) : vec4(rgb.gb, 0.0, -1.0/3.0);
        vec4 Q = (rgb.r < P.x) ? vec4(P.xyw, rgb.r) : vec4(rgb.r, P.yzx);
        float C = Q.x - min(Q.w, Q.y);
        float H = abs((Q.w - Q.y) / (6.0 * C + 1e-10) + Q.z);
        return vec3(H, C, Q.x);
    }

    vec3 rgb2hsl(vec3 rgb) {
        vec3 HCV = rgb2hcv(rgb);
        float L = HCV.z - HCV.y * 0.5;
        float S = HCV.y / (1.0 - abs(L * 2.0 - 1.0) + 1e-10);
        return vec3(HCV.x, S, L);
    }

    float getQualifierMask(vec3 color) {
        if (!u_qualifierEnabled) return 1.0;

        vec3 hsl = rgb2hsl(color);

        // Hue (Circular distance)
        float hDist = abs(hsl.x - u_qualifierHue);
        if (hDist > 0.5) hDist = 1.0 - hDist;
        float hMask = 1.0 - smoothstep(u_qualifierHueWidth, u_qualifierHueWidth + u_qualifierHueSoft, hDist);

        // Saturation
        float sDist = abs(hsl.y - u_qualifierSat);
        float sMask = 1.0 - smoothstep(u_qualifierSatWidth, u_qualifierSatWidth + u_qualifierSatSoft, sDist);

        // Luma
        float lDist = abs(hsl.z - u_qualifierLuma);
        float lMask = 1.0 - smoothstep(u_qualifierLumaWidth, u_qualifierLumaWidth + u_qualifierLumaSoft, lDist);

        return hMask * sMask * lMask;
    }

    void main() {
        // 0. Lens Distortion
        vec2 uv = distortUV(v_texcoord);

        // Black out of bounds if distorted
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        // ... (Show Depth logic) ...
         if (u_showDepth) {
            float d = texture(u_depth, uv).r;
            fragColor = vec4(vec3(d), 1.0);
            return;
        }

        // 1. DoF
        vec3 color;
        if (u_dofEnabled) {
            float depth = texture(u_depth, uv).r;
            float coc = circleOfConfusion(depth);
            color = getBokehColor(uv, coc);
        } else {
            // Apply CA even if DoF off? Yes, getBokehColor handles it if radius < 1.0
            // But we can call it with radius 0.0 to get CA
            if (u_lensFringe > 0.0) {
                color = getBokehColor(uv, 0.0);
            } else {
                color = texture(u_image, uv).rgb;
            }
        }

        // 1a. Denoise
        if (u_denoise > 0.0) {
            vec3 smoothColor = getDenoiseColor(uv);
            color = mix(color, smoothColor, u_denoise);
        }

        // 1b. Linearize
        if (!u_isLinear) {
            color = sRGBToLinear(color);
        }

        // 1c. Input LUT (IDT) - Applied after linearization but before grading
        if (u_inputLutMode > 0) {
            color = applyDisplayLUT(color, u_inputLutMode);
        }

        // SAVE PRE-GRADE (Source for Keying)
        vec3 preGrade = color;

        // ── All grading operations now happen in LINEAR space ──

        // 2. Exposure — v3.2: Clamp to ±12 stops to prevent Inf
        color *= pow(2.0, clamp(u_exposure, -12.0, 12.0));

        // 3. White Balance
        if (u_temperature != 0.0 || u_tint != 0.0) {
            color = applyTempTint(color, u_temperature, u_tint);
        }

        // 4. Grading (Resolve Style)
        if (u_colorScience == 1) {
            // ACEScct Pipeline
            vec3 acescg = LIN_SRGB_TO_ACESCG * color;
            vec3 cct = lin_to_ACEScct_vec3(acescg);
            cct = applyGrading(cct, u_lift, u_gamma, u_gain, u_offset);
            acescg = ACEScct_to_lin_vec3(cct);
            color = ACESCG_TO_LIN_SRGB * acescg;
        } else {
            // Standard Linear Processing
            color = applyGrading(color, u_lift, u_gamma, u_gain, u_offset);
        }

        // 5. Contrast
        if (u_contrast != 1.0) {
            color = applyContrast(color, u_contrast, u_pivot);
        }

        // v3.3: Log Wheels
        color = applyLogWheels(color, u_logShadow, u_logMidtone, u_logHighlight);

        // v3.4: Printer Lights (per-channel log-space exposure offset)
        color = applyPrinterLights(color, u_printerR, u_printerG, u_printerB);

        // v3.0: Shadows / Highlights (Log-like)
        color = applyShadowsHighlights(color, u_shadows, u_highlights);

        // v3.0: Midtone Detail (Sharpen/Soften)
        if (u_midDetail != 0.0) {
             color = applyMidDetail(color, v_texcoord, u_midDetail);
        }

        // v3.0: Color Boost (Vibrance)
        if (u_colorBoost != 0.0) {
            color = applyColorBoost(color, u_colorBoost);
        }

        // 5b. Custom Curves
        color = applyCurves(color);

        // 5c. Secondary Curves (Hue vs X)
        color = applySecondaryCurves(color);

        // 6. Saturation
        float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(luma), color, u_saturation);

        // v3.0: Hue Shift
        if (u_hueShift != 0.0) {
            color = applyHueShift(color, u_hueShift);
        }

        // v3.0: Luma Mix
        if (u_lumaMix != 1.0) {
            float currentLuma = dot(color, vec3(0.2126, 0.7152, 0.0722));
            // Restore original luma? Or mix color back to original luma?
            // Resolve Luma Mix: 100% means Luma is coupled with RGB.
            // 0% means Luma is independent? actually usually it controls how much Y channel is affected.
            // Simplified: Mix between current result and (OriginalLuma + ColorDiff)
            // Or just mix luma channel back to pre-graded luma.
            // Let's stick to standard mix:
            // If Luma Mix is 0, we output color but with original luma.
            // If Luma Mix is 1, we output color as is.

            // Re-calculate original luma from preGrade (linearized input)
            // Pre-grade is 'preGrade' variable
            float origLuma = dot(preGrade, vec3(0.2126, 0.7152, 0.0722));
            vec3 colorWithOrigLuma = color * (origLuma / (currentLuma + 0.0001));
            color = mix(colorWithOrigLuma, color, u_lumaMix);
        }

        // ── FINAL MASK / QUALIFIER MIX ──
        float finalMatte = 1.0;
        if (u_qualifierEnabled) {
            finalMatte *= getQualifierMask(preGrade);
        }
        if (u_maskType > 0) {
            finalMatte *= calculateMask(v_texcoord);
        }

        if (u_maskType > 0 && u_maskShowOverlay) {
             // Show mask as grayscale overlay for positioning
             float m = calculateMask(v_texcoord);
             color = mix(preGrade * 0.3, color, m);
        } else if (u_qualifierEnabled || u_maskType > 0) {
             if (u_qualifierShowMask) {
                 color = vec3(finalMatte);
             } else {
                 color = mix(preGrade, color, finalMatte);
             }
        }

        // ── Optical Effects (graded linear space, before creative LUT) ──────
        // These simulate physical lens/film phenomena.  They sample u_image
        // (scene-linear input) so we linearise sRGB sources and apply
        // Reinhard HDR compression to keep contributions bounded — otherwise
        // HDR values >>1.0 (e.g. fire, specular) cause massive blowout /
        // blocky artefacts when added to the graded colour.

        // v4.0: Multi-pass Kawase bloom (pre-computed FBO chain — already thresholded
        // and exposure-adjusted in the downsample pass).
        if (u_bloom > 0.0) {
            if (u_bloomTexEnabled == 1) {
                vec3 bloomSample = texture(u_bloomTex, uv).rgb;
                // Reinhard compress bloom contribution for HDR scenes
                bloomSample = bloomSample / (vec3(1.0) + bloomSample);
                color += bloomSample * u_bloom * 1.2;
            } else {
                // Fallback: lightweight 16-sample spiral for WebGL1 or when FBOs fail
                vec3 bloomAcc = vec3(0.0);
                float bloomW = 0.0;
                vec2 px = 1.0 / u_texSize;
                const float BLOOM_GOLDEN = 2.399963;
                const int   BLOOM_N      = 16;
                const float BLOOM_RADIUS = 8.0;
                for (int i = 0; i < BLOOM_N; i++) {
                    float t     = (float(i) + 0.5) / float(BLOOM_N);
                    float r     = sqrt(t) * BLOOM_RADIUS;
                    float angle = float(i) * BLOOM_GOLDEN;
                    vec2 off    = vec2(cos(angle), sin(angle)) * r * px;
                    vec3 s      = texture(u_image, uv + off).rgb;
                    if (!u_isLinear) s = sRGBToLinear(s);
                    float lum    = dot(s, vec3(0.2126, 0.7152, 0.0722));
                    float bl_lo = u_isLinear ? 1.0  : 0.82;
                    float bl_hi = u_isLinear ? 3.0  : 1.4;
                    float thresh = smoothstep(bl_lo, bl_hi, lum);
                    float radW   = exp(-t * 3.0);
                    float w      = thresh * radW;
                    vec3 sc      = s / (vec3(1.0) + s);  // Reinhard compress
                    bloomAcc    += sc * w;
                    bloomW      += w;
                }
                if (bloomW > 0.0) color += (bloomAcc / bloomW) * u_bloom * 0.5;
            }
        }

        // Halation (red channel highlight bleed — classic film gate bounce)
        // v4.1: Gaussian-weighted circular kernel with HDR compression.
        // Replaces the old 9×9 box grid that produced blocky artefacts on HDR.
        if (u_halation > 0.0) {
            float halAcc = 0.0;
            float halW   = 0.0;
            vec2  px2    = 1.0 / u_texSize;
            const int   HAL_RINGS  = 6;
            const int   HAL_DIRS   = 8;
            const float HAL_RADIUS = 5.0;
            for (int ring = 1; ring <= HAL_RINGS; ring++) {
                float rFrac = float(ring) / float(HAL_RINGS);
                float rad   = rFrac * HAL_RADIUS;
                float gw    = exp(-rFrac * rFrac * 2.0);
                for (int dir = 0; dir < HAL_DIRS; dir++) {
                    float a = float(dir) * (6.28318 / float(HAL_DIRS));
                    vec2 off = vec2(cos(a), sin(a)) * rad * px2;
                    vec3 s = texture(u_image, uv + off).rgb;
                    if (!u_isLinear) s = sRGBToLinear(s);
                    s = s / (vec3(1.0) + s);  // Reinhard compress
                    float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                    float w = max(lum - 0.35, 0.0) * gw;
                    halAcc += s.r * w;
                    halW   += w;
                }
            }
            if (halW > 0.0) halAcc /= halW;
            color.r += halAcc * u_halation * 0.7;
        }

        // ── Anamorphic Lens Streaks ──────────────────────────────────────────
        // v4.1: HDR-compressed samples prevent blowout on 32-bit float scenes.
        if (u_anamorphicStreaks > 0.0) {
            vec3  streakAcc = vec3(0.0);
            float streakW   = 0.0;
            const int STREAK_SAMPLES = 24;

            for (int si = -STREAK_SAMPLES; si <= STREAK_SAMPLES; si++) {
                float t = float(si) / float(STREAK_SAMPLES);
                vec2 sUV = clamp(vec2(uv.x + t * u_streakLength, uv.y), 0.0, 1.0);
                vec3 s   = texture(u_image, sUV).rgb;
                if (!u_isLinear) s = sRGBToLinear(s);
                s = s / (vec3(1.0) + s);  // Reinhard compress

                float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
                float trigger = smoothstep(u_streakThreshold - 0.05, u_streakThreshold + 0.05, lum);
                float w = exp(-abs(t) * 10.0) * trigger;
                streakAcc += s * w;
                streakW   += w;
            }

            if (streakW > 0.0) {
                vec3 streak = (streakAcc / streakW) * vec3(0.55, 0.80, 1.0);
                color += streak * u_anamorphicStreaks * 0.45;
            }
        }

        // Diffusion (soft filter — blend graded colour with blurred neighbourhood)
        // v4.1: Gaussian-weighted 5×5 with HDR compression for clean softening.
        if (u_diffusion > 0.0) {
            vec3  diffAcc = vec3(0.0);
            float diffW   = 0.0;
            vec2  px3     = 1.0 / u_texSize;
            for (int dx = -2; dx <= 2; dx++) {
                for (int dy = -2; dy <= 2; dy++) {
                    float gw = exp(-float(dx*dx + dy*dy) / 4.5);
                    vec2 off = vec2(float(dx), float(dy)) * px3 * 2.0;
                    vec3 s = texture(u_image, uv + off).rgb;
                    if (!u_isLinear) s = sRGBToLinear(s);
                    s = s / (vec3(1.0) + s);  // Reinhard compress
                    diffAcc += s * gw;
                    diffW   += gw;
                }
            }
            diffAcc /= diffW;
            color = mix(color, diffAcc, u_diffusion * 0.6);
        }

        // 5. LUT (Global - applied after qualification + optical effects)
        vec3 lutInput = color;
        if (u_lutEnabled) {
            vec3 lutted = applyLUT(lutInput, u_lut, u_lutSize);
            color = mix(color, lutted, u_lutStrength);
        }

        // Scene-linear, captured before the display transform flattens it.
        // The HDR heatmap reports absolute cd/m2, and after tone mapping that
        // information is gone -- color from here on is display-referred.
        vec3 sceneLinearForHeatmap = color;
        vec3 displayPreClamp = color;   // set after the view, before any clamp

        // 3.5.0: viewer f-stop, Nuke-style: a look at the picture, not part of
        // it. Applied after the grade and after the scene values the heatmap,
        // false colour, gamut warning and graded EXR read, so none of them move.
        bool viewerOnly = (!u_scopeSignal || u_compareSignal) && !u_exportSceneLinear;
        if (viewerOnly && u_viewExposure != 0.0) color *= exp2(u_viewExposure);

        // 6 & 7. Display transform.
        //
        // With an OCIO config loaded, OCIODisplay() *is* steps 6 and 7: it
        // carries the view transform, the display encoding and the output OETF,
        // exactly as the show's config specifies them. Running our tonemap or
        // our sRGB OETF alongside it would double-apply a transfer function --
        // the same class of bug the u_lutIsDisplayTransform flag below exists to
        // prevent -- so this branch replaces both, and there is no partial mode
        // where some of ours and some of the config's both apply.
        if (u_ocioEnabled) {
            // Exactly 0.0 has to be nudged off zero first.
            //
            // OCIO's generated inverse-EOTF chains reach pow(x, y) with y <= 0,
            // which GLSL leaves *undefined* at x == 0. Measured on the ACES
            // configs through a WebGL2 context: scene-linear black came back as
            // 1.3e16 on the Rec.1886 view, 7.7e14 on P3-D65 and NaN on sRGB and
            // Display P3. Black is the most common pixel in a frame -- night
            // shots, letterbox bars, mattes -- so this is not an edge case.
            //
            // The nudge is confined to exact zeros. Anything down to 1e-30 goes
            // through the chain correctly, and negatives do too (OCIO clamps
            // them itself), so a blanket max() would change legitimately
            // negative scene-linear pixels for no reason. 1e-10 lands within a
            // hundredth of an 8-bit code value of where the CPU path puts zero.
            vec3 ocioIn = mix(color, vec3(1e-10), vec3(equal(color, vec3(0.0))));
            color = OCIODisplay(vec4(ocioIn, 1.0)).rgb;
            displayPreClamp = color;
        } else {

        // Built-in display curves expect Rec.709. OCIO performs this
        // conversion itself; scene-linear EXR export keeps the source gamut.
        if (!u_lutIsDisplayTransform && u_inputLutMode == 0) {
            if (u_sourceGamut == 1) color = mat3( // ACEScg, OCIO ACES studio
                1.7050509453, -0.1302564144, -0.0240033567,
                -0.6217921376, 1.1408047676, -0.1289689690,
                -0.0832588747, -0.0105483187, 1.1529723406) * color;
            if (u_sourceGamut == 2) color = mat3( // ACES2065-1, OCIO ACES studio
                2.5216860771, -0.2764798999, -0.0153780654,
                -1.1341309547, 1.3727190495, -0.1529753357,
                -0.3875552118, -0.0962391719, 1.1683534384) * color;
            if (u_sourceGamut == 3) color = mat3( // Linear Rec.2020, OCIO ACES studio
                1.6604909897, -0.1245504767, -0.0181507636,
                -0.5876411200, 1.1328998804, -0.1005788967,
                -0.0728498623, -0.0083494224, 1.1187297106) * color;
            if (u_sourceGamut == 4) color = mat3( // Linear P3-D65, OCIO ACES studio
                1.2249401808, -0.0420569554, -0.0196375549,
                -0.2249401808, 1.0420569181, -0.0786360428,
                0.0000000000, 0.0000000000, 1.0982736349) * color;
        }

        // 6. Display LUT / Tonemap  (runs after 5. LUT)
        if (u_displayLutMode > 0) {
            vec3 transformed = applyDisplayLUT(color, u_displayLutMode);
            color = mix(color, transformed, u_displayLutStrength);
        } else if (u_isLinear) {
            // HDR linear data with no display LUT — apply ACES filmic tonemap
            // to map scene-referred values to displayable range.
            // Without this, values >>1.0 clip per-channel → saturated primaries.
            color = toneMapACES(color);
        }

        // 7. Display Transform (linear → sRGB)
        // v3.4: Soft Clip — rolls highlights before OETF so it operates in linear space
        // OCIO-FIX: Skip OETF when the 3D LUT is a full display transform
        // (e.g. OCIO DisplayView bake) — those LUTs already contain sRGB OETF.
        // Applying linearToSRGB again would double-gamma and produce the
        // orange-cast / blown-highlight artefact visible on HDR content.
        if (!u_lutIsDisplayTransform && !displayModeEncodes(u_displayLutMode)) {
            color = applySoftClip(color, u_softClip);
            displayPreClamp = linearToSRGB(max(color, vec3(0.0))) + min(color, vec3(0.0));
            color = linearToSRGB(max(color, vec3(0.0)));
        } else {
            displayPreClamp = color;
        }

        }   // end !u_ocioEnabled

        // 3.5.0: viewer gamma on the display signal (1.0 = off).
        if (viewerOnly && u_viewGamma > 0.0 && u_viewGamma != 1.0) {
            color = pow(max(color, vec3(0.0)), vec3(1.0 / u_viewGamma));
        }

        // 7a. Film Grain — Photochemical-quality, static by default
        // ─────────────────────────────────────────────────────────
        // Two-phase hash gives sub-pixel white noise (no bilinear smoothing like
        // valueNoise). Box-Muller gaussian distribution matches silver-halide grain.
        // When u_grainAnimate==0 the seed is purely UV-based → static on still images.
        if (u_grainAmount > 0.0) {
            // Map UV to texel grid at correct grain size
            vec2 grainPx = uv * u_texSize / max(u_grainSize, 0.25);

            // Static frame seed — changes only when u_grainAnimate==1 (video mode)
            float frameSeed = u_grainAnimate > 0.0 ? floor(u_time * 24.0) * 1.618033 : 0.0;

            // High-quality 2-round hash (no sin — avoids GPU precision banding)
            vec2 sp  = floor(grainPx);
            vec2 fp  = fract(grainPx);
            // Sharp white noise — no interpolation, true per-pixel randomness
            vec2 q   = sp + vec2(frameSeed * 13.7, frameSeed * 7.3);
            q = fract(q * vec2(0.1031, 0.1030));
            q += dot(q, q.yx + 33.33);
            float n1 = fract((q.x + q.y) * q.x);
            q = fract((q + vec2(12.3, 45.6)) * vec2(0.1031, 0.1030));
            q += dot(q, q.yx + 33.33);
            float n2 = fract((q.x + q.y) * q.y);

            // Box-Muller: uniform [0,1] → gaussian-distributed grain
            // Clamp to avoid log(0); the epsilon keeps precision on low-end GPUs
            float u1 = max(n1, 0.0001);
            float u2 = n2;
            float gaussian = sqrt(-2.0 * log(u1)) * cos(6.28318 * u2);
            // Clamp to ±3σ (covers 99.7% of film grain range)
            gaussian = clamp(gaussian * 0.3333, -1.0, 1.0);

            // Luma-zone attenuation: grain is lighter in deep shadows/clipped highlights
            // (exposed film responds less at extremes — matches Kodak/Fuji behavior)
            float lumaGrain = dot(color, vec3(0.2126, 0.7152, 0.0722));
            float attenuation = smoothstep(0.0, 0.15, lumaGrain) *
                                (1.0 - smoothstep(0.75, 1.0, lumaGrain));
            attenuation = mix(0.05, 1.0, attenuation);

            float grainStrength = gaussian * u_grainAmount * attenuation;

            if (u_grainColor > 0.0) {
                // Chromatic grain: luminance-dominant + small per-channel dither
                // Real film grain is mostly silver-halide (B&W) with slight dye-layer
                // color fringing — NOT three independent RGB noise fields.
                // Chroma deviation uses offset hash positions at fine sub-pixel scale.
                vec2 qR = fract((sp + vec2(3.1, 7.5) + frameSeed) * vec2(0.1031, 0.1030));
                qR += dot(qR, qR.yx + 33.33);
                float chromaR = fract((qR.x + qR.y) * qR.x) - 0.5;

                vec2 qB = fract((sp + vec2(9.3, 2.7) + frameSeed) * vec2(0.1031, 0.1030));
                qB += dot(qB, qB.yx + 33.33);
                float chromaB = fract((qB.x + qB.y) * qB.x) - 0.5;

                // Mix: 80% luma grain, 20% chroma deviation (u_grainColor scales chroma)
                float chromaScale = u_grainColor * u_grainAmount * attenuation * 0.4;
                vec3 grain3 = vec3(
                    grainStrength + chromaR * chromaScale,
                    grainStrength,
                    grainStrength + chromaB * chromaScale
                );
                color = blendOverlay(color, grain3 + 0.5);
            } else {
                // Pure luminance grain (monochrome / B&W film look)
                color = blendOverlay(color, vec3(grainStrength) + 0.5);
            }
        }

        // 7d. Vignette
        if (u_vignetteIntensity > 0.0) {
            float d = distance(uv, vec2(0.5));
            float v = smoothstep(1.0, 0.25 + (1.0 - u_vignetteFalloff) * 0.75, d * (0.5 + u_vignetteIntensity * 1.5));
            color *= v;
        }

        // 3.5.0: scopes measure the picture the display receives, graded and
        // through the view, and nothing drawn on top of it. They used to read
        // the finished canvas (false colour, zebra, wipe and grids included)
        // or the ungraded source texture.
        if (u_scopeSignal) {
            fragColor = vec4(clamp(sanitize(color), 0.0, 1.0), 1.0);
            return;
        }

        // 6b. Channel isolation
        if (u_channelMode == 1) { color = vec3(color.r); }
        else if (u_channelMode == 2) { color = vec3(color.g); }
        else if (u_channelMode == 3) { color = vec3(color.b); }
        else if (u_channelMode == 4) { float l = dot(color, vec3(0.2126, 0.7152, 0.0722)); color = vec3(l); }
        else if (u_channelMode == 5) { color = vec3(texture(u_image, uv).a); }

        // 6c. Focus Peaking
        if (u_focusPeaking) {
            color = applyFocusPeaking(color, uv);
        }

        // 7. Analytics
        float lumaDisplay = dot(color, vec3(0.2126, 0.7152, 0.0722));

        if (u_falseColor) {
            color = getFalseColorMap(arriSignal(dot(sceneLinearForHeatmap, vec3(0.2126, 0.7152, 0.0722))));
        }

        // Reads scene luminance, not display luma: the whole point is absolute
        // cd/m2, which the display transform has already thrown away.
        if (u_hdrHeatmap) {
            color = getHDRHeatmap(dot(sceneLinearForHeatmap, vec3(0.2126, 0.7152, 0.0722)));
        }

        if (u_zebra) {
            float s = step(0.5, fract((v_texcoord.x + v_texcoord.y) * 120.0));
            if (lumaDisplay > u_zebraThreshold && s > 0.5) color = vec3(1.0, 0.0, 0.0);
        }

        // 7b. Advanced Analytics (Gamut & Clipping)
        // 3.5.0: both warnings read values the clamp has not touched yet.
        // They used to test the display value after max(color, 0) and after
        // the view's own clamp, so neither could ever fire.
        if (u_clippingMonitor) {
            float blink = step(0.5, fract(u_time * 2.0)); // 2Hz flash
            if (any(greaterThanEqual(displayPreClamp, vec3(0.999)))) {
                color = mix(color, vec3(1.0, 0.0, 0.0), blink); // at or over display white
            } else if (all(lessThanEqual(sceneLinearForHeatmap, vec3(0.0)))) {
                color = mix(color, vec3(0.0, 0.0, 1.0), blink); // crushed to black
            }
        }

        if (u_gamutWarning) {
            // A negative component in scene-linear is a colour outside the
            // working gamut's triangle: it cannot be displayed without mapping.
            if (any(lessThan(sceneLinearForHeatmap, vec3(-1e-4)))) {
                color = vec3(1.0, 0.0, 1.0); // Solid Magenta
            }
        }

        // 8. Compare. B is the reference texture in display values, like color here.
        if (u_wipeRefEnabled && u_compareShow == 1) {
            color = texture(u_referenceImage, v_texcoord).rgb;
        } else if (u_wipeRefEnabled && u_compareShow == 2) {
            color = clamp(abs(color - texture(u_referenceImage, v_texcoord).rgb) * u_diffGain, 0.0, 1.0);
        }
        // Wipe:
        // LEFT  (x < wipeLine) = B side: reference / frozen snapshot
        // RIGHT (x >= wipeLine) = A side: current live/graded frame
        if (u_wipeEnabled && u_compareShow == 0) {
            float wipeLine = u_wipe;
            if (v_texcoord.x < wipeLine) {
                // Side B: Reference image (grabbed still)
                if (u_wipeRefEnabled) {
                    color = texture(u_referenceImage, v_texcoord).rgb;
                } else {
                    // Fallback: show raw un-graded source
                    vec3 raw = texture(u_image, v_texcoord).rgb;
                    if (!u_isLinear) raw = sRGBToLinear(raw);
                    color = linearToSRGB(raw);
                }
            }
            // Note: divider line is drawn by the canvas overlay (_drawWipeSplitOverlay)
            // so we intentionally DO NOT draw a GLSL line here to avoid double-line artifacts.
        }

        // 9. Grids & Overlays
        if (u_gridMode > 0) {
            vec2 uvg = v_texcoord;
            float g = 0.0;
            if (u_gridMode == 1) { // Rule of Thirds
                if (abs(uvg.x - 0.333) < 0.001 || abs(uvg.x - 0.666) < 0.001 ||
                    abs(uvg.y - 0.333) < 0.001 || abs(uvg.y - 0.666) < 0.001) g = 1.0;
            } else if (u_gridMode == 2) { // 2.39:1 Mask
                float aspect = u_texSize.x / u_texSize.y;
                float targetAspect = 2.39;
                float maskH = (1.0 - (aspect / targetAspect)) * 0.5;
                if (uvg.y < maskH || uvg.y > (1.0 - maskH)) {
                    color *= 0.1; // Darken bars
                }
            } else if (u_gridMode == 3) { // Center Cross
                if (abs(uvg.x - 0.5) < 0.001 || abs(uvg.y - 0.5) < 0.001) g = 1.0;
            }
            color = mix(color, u_gridColor.rgb, g * u_gridColor.a);
        }

        // 10. Final Sanitization
        // v3.2: Performance Hardening - Neutralize NaNs/Infs that may have
        // leaked from complex bokeh or bloom feedback loops.
        color = sanitize(color);

        // 3.5.0: the "32-bit graded EXR" export. It used to read back this
        // composite: tone mapped or OCIO display, sRGB OETF, wipe, grid,
        // mattes and false colour all baked in. A graded EXR is the grade in
        // scene-linear (Nuke writes the same), so it takes the colour captured
        // after grade + LUT and before the display transform, with the
        // source's alpha.
        if (u_exportSceneLinear) {
            fragColor = vec4(sanitize(sceneLinearForHeatmap), texture(u_image, v_texcoord).a);
            return;
        }

        // 3.5.0: triangular (TPDF) dither of one 8-bit step before the
        // drawing buffer quantises: removes banding in skies and gradients
        // (the output was RGBA8 with no dither). Never in exports or scopes,
        // which returned above.
        if (u_dither) {
            vec2 dp = gl_FragCoord.xy;
            float n1 = fract(sin(dot(dp, vec2(12.9898, 78.233))) * 43758.5453);
            float n2 = fract(sin(dot(dp + 17.0, vec2(39.3468, 11.135))) * 24634.6345);
            color += (n1 + n2 - 1.0) / 255.0;
        }

        fragColor = vec4(color, 1.0);
    }
`;
    }

    getLUTFragmentShader() {
        // Deprecated, mapped to Composite
        return this.getCompositeFragmentShader();
    }

    getParadeFragmentShader() {
        return `#version 300 es
            precision highp float;
            in vec2 v_texcoord;
            out vec4 fragColor;
            uniform sampler2D u_image;
            uniform float u_brightness;
            void main() {
                float section = floor(v_texcoord.x * 3.0);
                float localX = fract(v_texcoord.x * 3.0);
                float columnSum = 0.0;
                int sampleCount = 256;
                for (int i = 0; i < sampleCount; i++) {
                    float y = float(i) / float(sampleCount - 1);
                    vec3 col = texture(u_image, vec2(localX, y)).rgb;
                    float val = (section < 1.0) ? col.r : (section < 2.0 ? col.g : col.b);
                    if (abs(val - (1.0 - v_texcoord.y)) < 0.015) columnSum += 1.0;
                }
                vec3 chanCol = (section < 1.0) ? vec3(1.0, 0.2, 0.2) : (section < 2.0 ? vec3(0.2, 1.0, 0.2) : vec3(0.2, 0.4, 1.0));
                fragColor = vec4(chanCol * (columnSum / 256.0) * u_brightness, 1.0);
            }
        `;
    }

    // v2.5: GPU Point-based Scopes
    getScopePointVertexShader(mode) {
        return `#version 300 es
            layout(location = 0) in vec2 a_uv;
            uniform sampler2D u_image;
            uniform bool u_isLinear;
            uniform float u_intensity;
            uniform bool u_parade;
            out vec4 v_color;

            void main() {
                vec4 pixel = texture(u_image, a_uv);
                vec3 color = pixel.rgb;

                // Linearize if sRGB
                if (!u_isLinear) {
                    bvec3 cutoff = lessThan(color, vec3(0.04045));
                    vec3 higher = pow((color + vec3(0.055)) / vec3(1.055), vec3(2.4));
                    vec3 lower = color / vec3(12.92);
                    color = mix(higher, lower, vec3(cutoff));
                }

                float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));

                if (${mode === 'vectorscope' ? 'true' : 'false'}) {
                    // 3.5.0: BT.709 Cb/Cr of the ENCODED signal (see
                    // radiance_vectorscope.js). It used linear light with PAL
                    // U/V weights. 0.5 chroma reaches SCOPE_RADIUS (0.9).
                    vec3 enc = pixel.rgb;
                    if (u_isLinear) {
                        vec3 lc = max(enc, vec3(0.0));
                        enc = mix(1.055 * pow(lc, vec3(1.0 / 2.4)) - 0.055, lc * 12.92,
                                  vec3(lessThan(lc, vec3(0.0031308))));
                    }
                    float ey = dot(enc, vec3(0.2126, 0.7152, 0.0722));
                    float cb = (enc.b - ey) / 1.8556;
                    float cr = (enc.r - ey) / 1.5748;
                    v_color = vec4(clamp(enc, 0.0, 1.0), u_intensity);
                    gl_Position = vec4(cb * 2.0 * 0.9, cr * 2.0 * 0.9, 0.0, 1.0);
                } else if (${mode === 'waveform' ? 'true' : 'false'}) {
                    // Waveform (X/Luma) or RGB Parade

                    // HDR: Use ST.2084 (PQ) non-linear mapping for the Y axis
                    // This allows seeing values from 0.001 to 10000 nits
                    // 3.5.0: ST.2084 takes luminance / 10000 nits. Linear 1.0 is
                    // 203 nits (BT.2408); it used to be fed as 10000 nits, so the
                    // trace sat ~49x too high against a correct graticule.
                    float L = luma * (203.0 / 10000.0);
                    float m1 = 2610.0 / 4096.0 / 4.0;
                    float m2 = 2523.0 / 4096.0 * 128.0;
                    float c1 = 3424.0 / 4096.0;
                    float c2 = 2413.0 / 4096.0 * 32.0;
                    float c3 = 2392.0 / 4096.0 * 32.0;
                    float y_pq = pow((c1 + c2 * pow(max(L, 1e-7), m1)) / (1.0 + c3 * pow(max(L, 1e-7), m1)), m2);

                    if (u_parade) {
                        float chanIdx = floor(a_uv.y * 3.0);
                        float val = (chanIdx < 1.0) ? color.r : (chanIdx < 2.0 ? color.g : color.b);

                        // Apply PQ to channel value too
                        float valN = val * (203.0 / 10000.0);
                        float val_pq = pow((c1 + c2 * pow(max(valN, 1e-7), m1)) / (1.0 + c3 * pow(max(valN, 1e-7), m1)), m2);

                        vec3 chanCol = (chanIdx < 1.0) ? vec3(1.0, 0.1, 0.1) : (chanIdx < 2.0 ? vec3(0.1, 1.0, 0.1) : vec3(0.1, 0.4, 1.0));
                        float x_base = -1.0 + chanIdx * (2.0/3.0);
                        float x_local = a_uv.x * (2.0/3.0);

                        gl_Position = vec4(x_base + x_local, val_pq * 1.96 - 0.98, 0.0, 1.0);
                        v_color = vec4(chanCol, u_intensity * 3.0);
                    } else {
                        gl_Position = vec4(a_uv.x * 1.96 - 0.98, y_pq * 1.96 - 0.98, 0.0, 1.0);
                        v_color = vec4(vec3(0.6, 1.0, 0.6), u_intensity * 1.5);
                    }
                } else if (${mode === 'chromaticity' ? 'true' : 'false'}) {
                    // CIE 1931 xy Chromaticity
                    mat3 m = mat3(
                        0.4124, 0.2126, 0.0193,
                        0.3576, 0.7152, 0.1192,
                        0.1805, 0.0722, 0.9505
                    );
                    vec3 xyz = m * color;
                    float sum = xyz.x + xyz.y + xyz.z;
                    if (sum > 1e-6) {
                        float x = xyz.x / sum;
                        float y = xyz.y / sum;
                        // Map x [0, 0.8] -> [-1, 1], y [0, 0.9] -> [-1, 1]
                        gl_Position = vec4((x / 0.8) * 2.0 - 1.0, (y / 0.9) * 2.0 - 1.0, 0.0, 1.0);
                        v_color = vec4(color, u_intensity * 3.0);
                    } else {
                        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0); // Discard blacks
                    }
                } else {
                    // Histogram (Luma)
                    float x = clamp(luma, 0.0, 1.0) * 1.96 - 0.98;
                    float y = (a_uv.y * 2.0 - 1.0) * 0.8;
                    v_color = vec4(vec3(0.8), u_intensity);
                    gl_Position = vec4(x, y, 0.0, 1.0);
                }
                gl_PointSize = 1.0;
            }
        `;
    }

    getScopePointFragmentShader() {
        return `#version 300 es
            precision highp float;
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = v_color;
            }
        `;
    }

    createScopeBuffers() {
        const gl = this.gl;
        const res = 256; // 65k points
        const uvs = new Float32Array(res * res * 2);
        for (let y = 0; y < res; y++) {
            for (let x = 0; x < res; x++) {
                const i = (y * res + x) * 2;
                uvs[i] = (x + 0.5) / res;
                uvs[i + 1] = (y + 0.5) / res;
            }
        }
        this.scopeBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.scopeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
        this.scopePointCount = res * res;
    }

    // Load image as texture
    /**
     * Magnification filter for the image texture.
     *
     * `'nearest'` shows the actual pixels; `'linear'` interpolates. Inspecting
     * a pixel through a bilinear filter shows a blend of its neighbours, which
     * is why every reference viewer binds this to a key -- RV uses `n`.
     *
     * Only magnification changes. Minification stays interpolated: nearest on a
     * downscaled image aliases badly and shows detail that is not there, which
     * is the opposite of what this toggle is for.
     */
    setPixelFilter(mode) {
        this.pixelFilter = mode === 'nearest' ? 'nearest' : 'linear';
        this._applyPixelFilter();
    }

    _applyPixelFilter() {
        const gl = this.gl;
        const tex = this.textures?.image;
        if (!gl || !tex) return;
        // A float texture cannot filter linearly without the extension, so
        // 'linear' falls back to nearest there rather than sampling as black.
        const canLinear = !this._imageIsFloat || this.extColorFloatLinear;
        const mag = (this.pixelFilter === 'nearest' || !canLinear) ? gl.NEAREST : gl.LINEAR;
        // Unit 0 explicitly. Without it this binds the image texture to
        // whichever unit happened to be active -- unit 3 is the depth map,
        // unit 4 the reference -- and then leaves it there. Unit 0 is where the
        // image belongs and where the composite shader expects it, so binding
        // it here is also the correct resting state; unbinding to null would
        // just make the next draw rebind it.
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, mag);
    }

    loadImageTexture(image) {
        const gl = this.gl;

        // 3.5.0: never delete a texture the frame cache still owns. This
        // deleted the previous frame's cached texture on every upload, so a
        // loop or scrub back bound a dead texture and showed black.
        this._releaseImageTexture();

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // FIX: alignment for 16-bit or odd-width images
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // Use RGBA for standard images
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

        // Set texture parameters
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.textures.image = texture;
        this._imageIsFloat = false;
        this._applyPixelFilter();   // a new texture resets to LINEAR otherwise
        this.imageWidth = image.width;
        this.imageHeight = image.height;
        this.isLinearTexture = false; // PNG/Image data is sRGB-encoded

        // Invalidate uniform cache — new image may have different size/isLinear
        this.invalidateUniformCache(this.programs.composite);

        return texture;
    }

    // Load float32 texture for HDR (WebGL2)
    loadFloat32Texture(data, width, height, channels = 4) {
        const gl = this.gl;

        if (!this.isWebGL2) {
            console.warn('[Radiance] Float32 textures require WebGL2, falling back to 8-bit pipeline');
            return null;
        }

        // AUDIT-FIX (2026-08): fail BEFORE upload with an actionable message.
        // Beyond MAX_TEXTURE_SIZE (8192 on many GPUs, 16384 on most desktop
        // cards) texImage2D fails post-hoc into a black frame with only a
        // cryptic GL error code. 8K DCI (8192 wide) sits exactly on the
        // common limit; 12K+ plates exceed it everywhere.
        const maxTex = this._maxTextureSize ||
            (this._maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 8192);
        if (width > maxTex || height > maxTex) {
            console.error(
                `[Radiance] Frame ${width}x${height} exceeds this GPU's texture ` +
                `limit (${maxTex}px). Use the node's proxy_scale or downscale ` +
                `upstream to view it; full-res data is unaffected.`);
            return null;
        }

        // WebGL2 requires EXT_color_buffer_float for some float texture operations
        if (!this.extColorBufferFloat) {
            console.warn('[Radiance] EXT_color_buffer_float not supported, float texture rendering might fail');
        }

        // 3.5.0: never delete a texture the frame cache still owns. This
        // deleted the previous frame's cached texture on every upload, so a
        // loop or scrub back bound a dead texture and showed black.
        this._releaseImageTexture();

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // WebGL2: Native float32 support
        let internalFormat, format;
        if (channels === 1) {
            internalFormat = gl.R32F;
            format = gl.RED;
        } else {
            internalFormat = channels === 4 ? gl.RGBA32F : gl.RGB32F;
            format = channels === 4 ? gl.RGBA : gl.RGB;
        }

        // Safety: check for NaNs in JS before upload (heavy but safe for debugging)
        // if (data.some(isNaN)) console.warn("Float32 data contains NaNs!");

        gl.texImage2D(
            gl.TEXTURE_2D, 0, internalFormat,
            width, height, 0,
            format, gl.FLOAT,
            data  // Float32Array
        );

        // v3.0 FIX: Check for GL errors after texture upload (ALBABIT-FIX: sampled, see _uploadError)
        const err = this._uploadError(`f32:${width}x${height}x${channels}`);
        if (err !== gl.NO_ERROR) {
            console.error(`[Radiance] Float32 texImage2D failed (GL error ${err}). Params: ${width}x${height}, ch=${channels}, internal=${internalFormat}, fmt=${format}`);
            gl.deleteTexture(texture);
            return null;
        }

        // CONDITIONAL FILTERING: Linear only if extension supported
        const filter = this.extColorFloatLinear ? gl.LINEAR : gl.NEAREST;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);

        this.textures.image = texture;
        this._imageIsFloat = true;
        this._applyPixelFilter();
        this.imageWidth = width;
        this.imageHeight = height;
        this.isLinearTexture = true; // Float32 data is scene-linear

        // Invalidate uniform cache — size and isLinear changed
        this.invalidateUniformCache(this.programs.composite);

        console.log(`[Radiance] Loaded ${width}×${height} float32 HDR texture (linear, ${filter === gl.LINEAR ? 'LINEAR' : 'NEAREST'})`);
        return texture;
    }

    /**
     * The GL error after a float upload. getError waits for the GPU, so it is
     * read on a format's first and every 24th upload (about once a second of
     * playback). An error starts the format over and purges the frame cache.
     */
    _uploadError(format) {
        if (!this._uploadCounts) this._uploadCounts = new Map();
        const count = this._uploadCounts.get(format) || 0;
        this._uploadCounts.set(format, count + 1);
        if (count % 24 !== 0) return this.gl.NO_ERROR;
        const err = this.gl.getError();
        if (err !== this.gl.NO_ERROR) {
            this._uploadCounts.delete(format);
            if (count > 0) this.clearFrameCache();
        }
        return err;
    }

    // Load float16 texture for HDR (WebGL2) — .rhdr format
    loadFloat16Texture(data, width, height, channels = 3) {
        const gl = this.gl;

        if (!this.isWebGL2) {
            console.warn('[Radiance] Float16 textures require WebGL2');
            return null;
        }

        // AUDIT-FIX (2026-08): same proactive texture-limit guard as
        // loadFloat32Texture -- see the comment there.
        const maxTex = this._maxTextureSize ||
            (this._maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 8192);
        if (width > maxTex || height > maxTex) {
            console.error(
                `[Radiance] Frame ${width}x${height} exceeds this GPU's texture ` +
                `limit (${maxTex}px). Use the node's proxy_scale or downscale ` +
                `upstream to view it; full-res data is unaffected.`);
            return null;
        }

        // 3.5.0: never delete a texture the frame cache still owns. This
        // deleted the previous frame's cached texture on every upload, so a
        // loop or scrub back bound a dead texture and showed black.
        this._releaseImageTexture();

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        let internalFormat, format;
        if (channels === 1) {
            internalFormat = gl.R16F;
            format = gl.RED;
            // Swizzle not strictly required if shader handles single channel,
            // but for safety we'll rely on shader logic to read .r
        } else {
            internalFormat = channels === 4 ? gl.RGBA16F : gl.RGB16F;
            format = channels === 4 ? gl.RGBA : gl.RGB;
        }

        gl.texImage2D(
            gl.TEXTURE_2D, 0, internalFormat,
            width, height, 0,
            format, gl.HALF_FLOAT,
            data  // Uint16Array (IEEE 754 half-float)
        );

        // v3.0 FIX: Check for GL errors after texture upload.
        // Without this, a failed upload returns a "valid" texture that is empty (all zeros = black).
        // ALBABIT-FIX: sampled (see _uploadError); on every upload it cost ~10 ms a frame.
        const err = this._uploadError(`f16:${width}x${height}x${channels}`);
        if (err !== gl.NO_ERROR) {
            console.error(`[Radiance] Float16 texImage2D failed (GL error ${err}): ${width}×${height}×${channels}ch`);
            gl.deleteTexture(texture);
            return null;
        }

        // CONDITIONAL FILTERING: Linear only if extension supported
        // Prefer HalfFloatLinear for half-float textures, fallback to FloatLinear or NEAREST
        const canFilter = this.extColorHalfFloatLinear || this.extColorFloatLinear;
        const filter = canFilter ? gl.LINEAR : gl.NEAREST;

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);

        this.textures.image = texture;
        this.imageWidth = width;
        this.imageHeight = height;
        this.isLinearTexture = true; // Float16 data is scene-linear

        console.log(`[Radiance] Loaded ${width}×${height} float16 HDR texture (linear, HALF_FLOAT, ${filter === gl.LINEAR ? 'LINEAR' : 'NEAREST'})`);
        return texture;
    }

    // Read pixels from WebGL framebuffer for export (8-bit legacy)
    readPixels() {
        const gl = this.gl;
        if (!this.textures.image) return null;
        const w = this.canvas.width, h = this.canvas.height;
        const pixels = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        return { data: pixels, width: w, height: h };
    }

    /**
     * 3.5.0: the displayed picture (graded, through the view, no overlays) at
     * w x h, rendered off-screen into an RGBA8 target. What the scopes measure.
     * Also leaves it in this.textures.scopeSignal for the GPU scopes.
     * @returns {{data: Uint8ClampedArray, width: number, height: number}|null}
     */
    readDisplaySignal(w, h, lutStrength = 1.0, read = true) {
        const gl = this.gl;
        if (!gl || !this.textures.image) return null;
        w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
        let tex = this.textures.scopeSignal;
        if (!tex || this._scopeSignalW !== w || this._scopeSignalH !== h) {
            if (tex) gl.deleteTexture(tex);
            tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            this.textures.scopeSignal = tex;
            this._scopeSignalW = w; this._scopeSignalH = h;
        }
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteFramebuffer(fbo);
            return null;
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this._exportFBO = { fbo, width: w, height: h };
        this.scopeSignal = true;
        try {
            this.render(lutStrength);
        } finally {
            this.scopeSignal = false;
            this._exportFBO = null;
        }
        if (!read) {
            gl.deleteFramebuffer(fbo);
            return { data: null, width: w, height: h, texture: tex };
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        const px = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(fbo);
        // Bottom-up -> top-down, as the CPU scopes index rows. The canvas
        // (default framebuffer) was never touched, so nothing needs redrawing.
        const out = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++) out.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
        return { data: out, width: w, height: h, texture: tex };
    }

    // ── v4.0: Read pixels as Float32 for 32-bit EXR export ──────────────────
    // Renders the full composite pipeline at the specified resolution into an
    // offscreen RGBA32F FBO, then reads back the result as Float32Array.
    // This captures the graded/processed image at full float precision.
    //
    // @param {number} [width]  - Output width  (default: source image width)
    // @param {number} [height] - Output height (default: source image height)
    // @param {number} [lutStrength] - LUT intensity for the render
    // @returns {{ data: Float32Array, width: number, height: number }} | null
    readPixelsFloat32(width, height, lutStrength = 1.0) {
        const gl = this.gl;
        if (!this.isWebGL2 || !this.textures.image) return null;

        const w = width || this.imageWidth || this.canvas.width;
        const h = height || this.imageHeight || this.canvas.height;

        // Create temporary RGBA32F FBO at target resolution
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            console.error(`[Radiance] Float32 export FBO incomplete: ${status}`);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteTexture(tex);
            gl.deleteFramebuffer(fbo);
            return null;
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        // Set export target — render() will bind this FBO instead of default framebuffer
        this._exportFBO = { fbo, width: w, height: h };

        // Render composite pass into the FBO, scene-linear (see u_exportSceneLinear)
        this.exportSceneLinear = true;
        try {
            this.render(lutStrength);
        } finally {
            this.exportSceneLinear = false;
        }

        // Read back float32 pixels from the FBO
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        const pixels = new Float32Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, pixels);

        // Clean up
        this._exportFBO = null;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteTexture(tex);
        gl.deleteFramebuffer(fbo);

        // WebGL readPixels returns bottom-to-top; flip vertically
        const flipped = new Float32Array(w * h * 4);
        for (let y = 0; y < h; y++) {
            const srcRow = (h - 1 - y) * w * 4;
            const dstRow = y * w * 4;
            flipped.set(pixels.subarray(srcRow, srcRow + w * 4), dstRow);
        }

        console.log(`[Radiance] Float32 readback: ${w}×${h} (${(flipped.byteLength / 1048576).toFixed(1)} MB)`);
        // graded: true -- this path renders the full composite pipeline into
        // an RGBA32F FBO first. The WebGPU backend returns the ungraded source
        // and says so, so callers can tell the two apart instead of assuming.
        return { data: flipped, width: w, height: h, graded: true, sceneLinear: true };
    }

    // Load 3D LUT from .cube file data (WebGL2: float32, WebGL1: fallback)
    loadLUT(lutData, size = 33) {
        const gl = this.gl;
        // Kept so a restored context can upload it again (webglcontextrestored).
        this._lutSource = { lutData, size };

        if (this.textures.lut) {
            gl.deleteTexture(this.textures.lut);
        }

        this.lutSize = size;

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_3D, texture);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

        // WebGL2: Use float32 for precision
        const internalFormat = this.isWebGL2 ? gl.RGB32F : gl.RGB;
        const dataType = this.isWebGL2 ? gl.FLOAT : gl.UNSIGNED_BYTE;

        gl.texImage3D(
            gl.TEXTURE_3D, 0, internalFormat,
            size, size, size, 0,
            gl.RGB, dataType,
            lutData
        );


        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.textures.lut = texture;
        console.log(`[Radiance] Loaded ${size}³ 3D LUT (${this.isWebGL2 ? 'float32' : 'uint8'})`);

        return texture;
    }

    // Load Depth Map (supports Float32/Float16/Image)
    loadDepthTexture(image) {
        const gl = this.gl;

        if (this.textures.depth) {
            gl.deleteTexture(this.textures.depth);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // Check if raw buffer data (Float32Array or similar)
        // Expected format: { data: TypedArray, width: number, height: number } or just Image/Canvas
        if (image.data && (image.data instanceof Float32Array || image.data instanceof Uint16Array)) {
            const width = image.width;
            const height = image.height;
            const isFloat32 = image.data instanceof Float32Array;

            // WebGL 2.0 internal formats
            const internalFormat = isFloat32 ? gl.R32F : gl.R16F;
            const type = isFloat32 ? gl.FLOAT : gl.HALF_FLOAT;

            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, gl.RED, type, image.data);

            console.log(`[Radiance] Loaded ${width}×${height} depth texture (${isFloat32 ? 'Float32' : 'Float16'})`);
        } else {
            // Standard Image/Canvas (8-bit)
            gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, image);
        }

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); // Nearest for depth is usually safer for sharp edges
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        this.textures.depth = texture;
        return texture;
    }

    // Load/Update Reference Still texture (for comparison)
    updateReferenceStill(image) {
        const gl = this.gl;
        if (!image) return;

        if (this.textures.reference) {
            gl.deleteTexture(this.textures.reference);
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        this.textures.reference = texture;
        console.log("[Radiance] Reference still texture updated.");
    }

    // Parse .cube LUT file
    parseCubeFile(cubeText) {
        const lines = cubeText.split('\n');
        let size = 33;
        const data = [];

        for (const line of lines) {
            const trimmed = line.trim();

            if (trimmed.startsWith('LUT_3D_SIZE')) {
                size = parseInt(trimmed.split(/\s+/)[1]);
            } else if (trimmed && !trimmed.startsWith('#') && !trimmed.startsWith('TITLE') && !trimmed.startsWith('DOMAIN')) {
                const values = trimmed.split(/\s+/).map(parseFloat);
                if (values.length >= 3 && !isNaN(values[0])) {
                    data.push(values[0], values[1], values[2]);
                }
            }
        }

        console.log(`[Radiance] Parsed LUT: ${size}x${size}x${size}, ${data.length / 3} entries`);
        return { size, data: new Float32Array(data) };
    }

    // v3.0 #8.1: Manual VRAM management — purge all cached frame textures
    clearFrameCache() {
        const gl = this.gl;
        if (!gl || !this._frameCache) return;

        console.log(`[Radiance] Purging GPU Frame Cache (${this._frameCache.size} textures)...`);
        this._frameCache.forEach((entry) => {
            if (entry.tex) gl.deleteTexture(entry.tex);
        });
        this._frameCache.clear();
        this._frameCacheBytes = 0;

        // Also clear active image textures if they were part of a sequence
        if (this.textures.image) {
            gl.deleteTexture(this.textures.image);
            this.textures.image = null;
        }
    }

    // Render with basic exposure/gamma
    renderBasic() {
        const gl = this.gl;
        const program = this.programs.basic;

        if (!program || !this.textures.image) return;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        // Set uniforms
        gl.uniform1f(this.getUniform(program, 'u_exposure'), this.exposure);
        gl.uniform1f(this.getUniform(program, 'u_gamma'), this.gamma);
        gl.uniform1f(this.getUniform(program, 'u_saturation'), this.saturation);
        gl.uniform1i(this.getUniform(program, 'u_isLinear'), this.isLinearTexture ? 1 : 0);

        // Bind image texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(this.getUniform(program, 'u_image'), 0);

        // Draw quad
        this.drawQuad(program);
    }

    // Render Composite (Image + LUT + DoF)
    render(lutStrength = 1.0) {
        const gl = this.gl;
        const program = this.programs.composite;

        // I-10: Skip rendering when WebGL context is lost
        if (this._contextLost || !program || !this.textures.image) return;

        // B for compare, drawn through this same pipeline (see setCompareSource).
        if (this._compareSrc && !this._renderingCompare && !this.scopeSignal && !this.exportSceneLinear
            && this.wipeRefEnabled && (this.compareShow > 0 || this.wipeEnabled)) {
            this._renderCompareReference();
        }

        // ── v4.0: Run multi-pass bloom chain before composite ────────────────
        // This renders to offscreen FBOs and does NOT touch the display framebuffer.
        const bloomTex = this._renderBloomChain();

        // v4.0: If an export FBO is set, render into it instead of the canvas
        const exportTarget = this._exportFBO;
        if (exportTarget) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, exportTarget.fbo);
            gl.viewport(0, 0, exportTarget.width, exportTarget.height);
        } else {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        // Common Uniforms
        this._uf1(program, 'u_exposure', this.exposure);
        this._ui1(program, 'u_frame', this.frame || 0);
        this._uf1(program, 'u_time', this.time || 0.0);

        // Advanced Grading
        this._uf3(program, 'u_lift', this.lift[0], this.lift[1], this.lift[2]);
        this._uf3(program, 'u_gamma', this.gradingGamma[0], this.gradingGamma[1], this.gradingGamma[2]);
        this._uf3(program, 'u_gain', this.gain[0], this.gain[1], this.gain[2]);
        this._uf3(program, 'u_offset', this.offset[0], this.offset[1], this.offset[2]);
        this._ui1(program, 'u_colorScience', this.colorScience || 0);

        this._uf1(program, 'u_temperature', this.temperature);
        this._uf1(program, 'u_tint', this.tint);
        this._uf1(program, 'u_contrast', this.contrast);
        this._uf1(program, 'u_pivot', this.pivot);

        // Curves (Texture unit 3)
        this._uf1(program, 'u_curveMix', this.curveMix);
        gl.activeTexture(gl.TEXTURE3);
        if (this.curveLutTexture) {
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        } else {
            // Create default identity texture if missing
            this.updateCurveLut(this.curveData);
            gl.bindTexture(gl.TEXTURE_2D, this.curveLutTexture);
        }
        this._ui1(program, 'u_curveLut', 3);

        // Secondary Curves (Hue vs Hue / Sat) (Texture unit 4)
        // FIX 3: Fallback must be 0.0 (bypass), not 1.0 (full effect).
        // If secondaryCurveMix is somehow undefined (e.g. old deserialized state),
        // defaulting to 1.0 would activate secondary curves unexpectedly.
        this._uf1(program, 'u_secondaryCurveMix', this.secondaryCurveMix ?? 0.0);
        gl.activeTexture(gl.TEXTURE4);
        if (this.secondaryCurveLutTexture) {
            gl.bindTexture(gl.TEXTURE_2D, this.secondaryCurveLutTexture);
        } else {
            // Default flat secondary curve (0.5 for all R/G/B/A limits)
            const emptyData = new Float32Array(256 * 4).fill(0.5);
            this.updateSecondaryCurveLut(emptyData);
            gl.bindTexture(gl.TEXTURE_2D, this.secondaryCurveLutTexture);
        }
        this._ui1(program, 'u_secondaryCurveLut', 4);


        this._uf1(program, 'u_saturation', this.saturation);

        // v3.0: Resolve-style grading uniforms
        this._uf1(program, 'u_colorBoost', this.colorBoost);
        this._uf1(program, 'u_shadows', this.shadows);
        this._uf1(program, 'u_highlights', this.highlights);
        this._uf1(program, 'u_midDetail', this.midDetail);
        this._uf1(program, 'u_hueShift', this.hueShift);
        this._uf1(program, 'u_lumaMix', this.lumaMix);

        // v3.3: Log Wheels uniforms
        this._uf3(program, 'u_logShadow', this.logShadow[0], this.logShadow[1], this.logShadow[2]);
        this._uf3(program, 'u_logMidtone', this.logMidtone[0], this.logMidtone[1], this.logMidtone[2]);
        this._uf3(program, 'u_logHighlight', this.logHighlight[0], this.logHighlight[1], this.logHighlight[2]);

        // v3.4: Printer Lights
        this._uf1(program, 'u_printerR', this.printerLightsR || 0.0);
        this._uf1(program, 'u_printerG', this.printerLightsG || 0.0);
        this._uf1(program, 'u_printerB', this.printerLightsB || 0.0);

        // v3.4: Soft Clip
        this._uf1(program, 'u_softClip', this.softClip || 0.0);

        // Qualifiers
        this._ui1(program, 'u_qualifierEnabled', this.qualifierEnabled ? 1 : 0);
        this._ui1(program, 'u_qualifierShowMask', this.qualifierShowMask ? 1 : 0);

        this._uf1(program, 'u_qualifierHue', this.qualifier.h);
        this._uf1(program, 'u_qualifierHueWidth', this.qualifier.hW);
        this._uf1(program, 'u_qualifierHueSoft', this.qualifier.hS);

        this._uf1(program, 'u_qualifierSat', this.qualifier.s);
        this._uf1(program, 'u_qualifierSatWidth', this.qualifier.sW);
        this._uf1(program, 'u_qualifierSatSoft', this.qualifier.sS);

        this._uf1(program, 'u_qualifierLuma', this.qualifier.l);
        this._uf1(program, 'u_qualifierLumaWidth', this.qualifier.lW);
        this._uf1(program, 'u_qualifierLumaSoft', this.qualifier.lS);

        // v3.1 Masking
        this._ui1(program, 'u_maskType', this.maskType | 0);
        this._uf2(program, 'u_maskCenter', this.maskCenter[0], this.maskCenter[1]);
        this._uf2(program, 'u_maskScale', this.maskScale[0], this.maskScale[1]);
        this._uf1(program, 'u_maskFeather', this.maskFeather);
        this._uf1(program, 'u_maskRotation', this.maskRotation);
        this._ui1(program, 'u_maskInvert', this.maskInvert ? 1 : 0);
        this._ui1(program, 'u_maskShowOverlay', this.maskShowOverlay ? 1 : 0);

        this._uf2(program, 'u_texSize', this.imageWidth, this.imageHeight);

        // Bind Image (Unit 0)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        this._ui1(program, 'u_image', 0);

        // LUT (Unit 1)
        this._ui1(program, 'u_lut', 1); // Always set unit 1
        if (this.textures.lut) {
            this._ui1(program, 'u_lutEnabled', 1);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_3D, this.textures.lut);
            // u_lut set above
            this._uf1(program, 'u_lutSize', this.lutSize);
            this._uf1(program, 'u_lutStrength', lutStrength);
        } else {
            this._ui1(program, 'u_lutEnabled', 0);
            // Bind dummy or null to unit 1 to prevent warning?
            // WebGL is okay if we don't sample from it, but some drivers complain if unit is missing.
            // Best practice: Bind null or a dummy texture if logic skips sampling.
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_3D, null);
        }

        // Depth / DoF (Unit 2)
        this._ui1(program, 'u_depth', 2); // Always set unit 2
        if ((this.dofEnabled || this.showDepth) && this.textures.depth) {
            this._ui1(program, 'u_dofEnabled', 1);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, this.textures.depth);
            // u_depth set above

            this._uf1(program, 'u_focusDist', this.focusDistance);
            this._uf1(program, 'u_aperture', this.aperture);
        } else {
            this._ui1(program, 'u_dofEnabled', 0);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, null);
        }

        // OpenColorIO — binds the config's LUTs and switches the shader's
        // display transform over to OCIODisplay(). No-op when nothing is loaded.
        this._applyOCIOUniforms(program);

        // Analytics Uniforms
        this._ui1(program, 'u_falseColor', this.falseColor ? 1 : 0);
        this._ui1(program, 'u_hdrHeatmap', this.hdrHeatmap ? 1 : 0);
        this._ui1(program, 'u_zebra', this.zebra ? 1 : 0);
        this._uf1(program, 'u_zebraThreshold', this.zebraThreshold);
        this._ui1(program, 'u_gamutWarning', this.gamutWarning ? 1 : 0);
        this._ui1(program, 'u_clippingMonitor', this.clippingMonitor ? 1 : 0);

        // HDR pipeline: tell shader whether texture is linear float or sRGB PNG
        this._ui1(program, 'u_isLinear', this.isLinearTexture ? 1 : 0);

        // v2.2: Channel isolation, focus peaking, display LUT
        this._ui1(program, 'u_channelMode', this.channelMode);
        this._ui1(program, 'u_focusPeaking', this.focusPeaking ? 1 : 0);
        this._uf1(program, 'u_focusPeakThreshold', this.focusPeakingThreshold);
        // 3.5.0: an 8-bit preview is already a display image; a view transform
        // on top of it would apply the view twice.
        this._ui1(program, 'u_displayLutMode', this.displayReferredTexture ? 0 : this.displayLutMode);
        this._ui1(program, 'u_inputLutMode', this.inputLutMode);
        this._ui1(program, 'u_sourceGamut', this.displayReferredTexture || this.sourceDisplayEncoded ? 0 : this.sourceGamut);
        this._uf1(program, 'u_displayLutStrength', this.displayLutStrength);
        this._ui1(program, 'u_lutIsDisplayTransform', this.lutIsDisplayTransform ? 1 : 0);

        // v2.2 Pro: Wipe & Reference
        this._ui1(program, 'u_wipeEnabled', this.wipeEnabled ? 1 : 0);
        this._uf1(program, 'u_wipe', this.wipe);
        this._ui1(program, 'u_wipeRefEnabled', this.wipeRefEnabled ? 1 : 0);
        this._ui1(program, 'u_compareShow', this.compareShow | 0);
        this._uf1(program, 'u_diffGain', Number.isFinite(this.diffGain) ? this.diffGain : 4);

        gl.activeTexture(gl.TEXTURE6);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.reference || this.textures.empty);
        this._ui1(program, 'u_referenceImage', 6);

        // v2.2 Pro: Grids
        this._ui1(program, 'u_gridMode', this.gridMode);
        this._ui1(program, 'u_exportSceneLinear', this.exportSceneLinear ? 1 : 0);
        this._ui1(program, 'u_scopeSignal', this.scopeSignal ? 1 : 0);
        this._ui1(program, 'u_compareSignal', this._renderingCompare ? 1 : 0);
        this._uf1(program, 'u_viewExposure', this.viewExposure || 0.0);
        this._uf1(program, 'u_viewGamma', this.viewGamma || 1.0);
        this._ui1(program, 'u_dither', this.dither === false ? 0 : 1);
        this._uf4v(program, 'u_gridColor', this.gridColor);

        // v2.3: Denoise & Depth Eval
        this._uf1(program, 'u_denoise', this.denoise);
        this._ui1(program, 'u_showDepth', this.showDepth ? 1 : 0);
        this._uf1(program, 'u_grainAmount', this.grainAmount || 0.0);

        // v2.6: Lens Effects
        this._ui1(program, 'u_apertureBlades', this.apertureBlades);
        this._uf1(program, 'u_apertureRotation', this.apertureRotation);
        this._uf1(program, 'u_apertureAnamorphic', this.apertureAnamorphic);
        this._uf1(program, 'u_lensDistortion', this.lensDistortion);
        this._uf1(program, 'u_lensFringe', this.lensFringe);
        this._uf1(program, 'u_vignetteIntensity', this.vignetteIntensity);
        this._uf1(program, 'u_vignetteFalloff', this.vignetteFalloff);

        // v2.6.1: Realistic Bokeh Physics
        this._uf1(program, 'u_bokehHighlightBias', this.bokehHighlightBias || 0.0);
        this._uf1(program, 'u_bokehSoapBubble', this.bokehSoapBubble || 0.0);
        this._uf1(program, 'u_bokehOpticalVig', this.bokehOpticalVig || 0.0);

        // v3.1: Extended Grain & Lens
        this._uf1(program, 'u_grainSize', this.grainSize || 1.0);
        this._uf1(program, 'u_grainColor', this.grainColor || 0.0);
        this._uf1(program, 'u_grainAnimate', this.grainAnimate ? 1.0 : 0.0);
        this._uf1(program, 'u_bloom', this.bloom || 0.0);
        this._uf1(program, 'u_halation', this.halation || 0.0);
        this._uf1(program, 'u_diffusion', this.diffusion || 0.0);

        // v4.0: Bind pre-computed bloom texture on TEXTURE5
        gl.activeTexture(gl.TEXTURE5);
        if (bloomTex) {
            gl.bindTexture(gl.TEXTURE_2D, bloomTex);
            this._ui1(program, 'u_bloomTexEnabled', 1);
        } else {
            gl.bindTexture(gl.TEXTURE_2D, null);
            this._ui1(program, 'u_bloomTexEnabled', 0);
        }
        this._ui1(program, 'u_bloomTex', 5);

        // ── v3.2 Phase 11: Anamorphic Streaks + k2 Distortion ────────────────
        this._uf1(program, 'u_lensDistortionK2', this.lensDistortionK2 || 0.0);
        this._uf1(program, 'u_anamorphicStreaks', this.anamorphicStreaks || 0.0);
        this._uf1(program, 'u_streakThreshold', this.streakThreshold ?? 0.85);
        this._uf1(program, 'u_streakLength', this.streakLength ?? 0.08);

        // ── v3.2 Phase 12: Bilateral Filter ──────────────────────────────────
        this._uf1(program, 'u_bilateralSigmaD', this.bilateralSigmaD ?? 3.0);
        this._uf1(program, 'u_bilateralSigmaR', this.bilateralSigmaR ?? 0.10);
        this._ui1(program, 'u_bilateralHalfRes', this.bilateralHalfRes ? 1 : 0);

        // Animated Grain
        // u_time changes every frame — always upload, no cache benefit
        this.gl.uniform1f(this.getUniform(program, 'u_time'), (performance.now() / 1000.0) % 100.0);

        this.drawQuad(program);
    }

    // Legacy support
    renderWithLUT(strength) {
        this.render(strength);
    }

    // Render RGB Parade waveform
    renderParade(brightness = 10.0) {
        const gl = this.gl;
        const program = this.programs.parade;

        if (!program || !this.textures.image) return;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0.05, 0.05, 0.08, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(program);

        gl.uniform2f(this.getUniform(program, 'u_resolution'), this.canvas.width, this.canvas.height);
        gl.uniform1f(this.getUniform(program, 'u_brightness'), brightness);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.image);
        gl.uniform1i(this.getUniform(program, 'u_image'), 0);

        this.drawQuad(program);
    }

    drawQuad(program) {
        const gl = this.gl;

        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);

        const posLoc = this.getAttrib(program, 'a_position');
        const texLoc = this.getAttrib(program, 'a_texcoord');

        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);

        gl.enableVertexAttribArray(texLoc);
        gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 16, 8);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }







    setHDRHeatmap(enabled) {
        this.hdrHeatmap = !!enabled;
    }

    setFalseColor(enabled) {
        this.falseColor = enabled;
    }

    setZebra(enabled) {
        this.zebra = enabled;
    }

    setGamutWarning(enabled) {
        this.gamutWarning = enabled;
    }

    setClippingMonitor(enabled) {
        this.clippingMonitor = enabled;
    }

    setZebraThreshold(threshold) {
        this.zebraThreshold = threshold;
    }

    setChannelMode(mode) {
        this.channelMode = mode;
    }



    setFocusPeaking(enabled, threshold = 120.0) {
        this.focusPeaking = enabled;
        this.focusPeakingThreshold = threshold;
    }

    setDisplayLutMode(mode) {
        this.displayLutMode = mode;
    }

    setSourceColorSpace(name) {
        this.sourceGamut = ({
            'ACEScg': 1, 'ACEScg (AP1)': 1,
            'ACES2065-1': 2, 'ACES 2065-1': 2, 'ACES2065-1 (AP0)': 2,
            'Linear Rec.2020': 3, 'Rec.2020 Linear': 3,
            'Linear P3-D65': 4,
        })[name] || 0;
    }

    setInputLutMode(mode) {
        this.inputLutMode = mode;
    }

    // OCIO-FIX: When the 3D LUT is a full display transform (OCIO bake with OETF
    // already inside), skip the final linearToSRGB to prevent double-gamma.
    setLutIsDisplayTransform(v) {
        this.lutIsDisplayTransform = !!v;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  OpenColorIO
    //
    //  info is what radiance_ocio.js returns from buildDisplayView(): the
    //  GLSL OpenColorIO generated for this (source → display / view), plus the
    //  LUT textures and uniforms that code expects to find bound. Pass null to
    //  go back to Radiance's own display pipeline.
    //
    //  This recompiles the composite program, so it is called when the view
    //  changes and never per frame.
    // ═══════════════════════════════════════════════════════════════════════

    setOCIODisplay(info) {
        const gl = this.gl;
        // Kept so a restored context can rebuild the view (webglcontextrestored).
        this._ocioInfo = info && info.shaderText ? info : null;
        this._releaseOCIOTextures();

        if (!info || !info.shaderText) {
            this._ocioShaderSource = null;
            this._ocio = null;
            this.ocioEnabled = false;
            this._rebuildCompositeProgram();
            return { ok: true, enabled: false };
        }

        // Indent the generated block so a compile error's line number still
        // lines up with something readable when the source is dumped.
        this._ocioShaderSource = info.shaderText
            .split('\n').map((l) => '            ' + l).join('\n');

        const rebuilt = this._rebuildCompositeProgram();
        if (!rebuilt) {
            // A shader that will not compile must not leave the viewer with a
            // dead program. Fall back to Radiance's own pipeline and report --
            // a black viewport with no explanation is the worst outcome here.
            this._ocioShaderSource = null;
            this._ocio = null;
            this.ocioEnabled = false;
            this._rebuildCompositeProgram();
            return { ok: false, enabled: false, error: 'The generated OCIO shader did not compile. See the console for the source.' };
        }

        this._ocio = {
            functionName: info.functionName,
            uniforms: info.uniforms || [],
            textures: (info.textures || []).map((t, i) => ({
                ...t,
                unit: 7 + i,        // 0–6 belong to the composite shader
                glTexture: this._createOCIOTexture(t),
            })),
            label: info.label || '',
        };

        // WebGL2 guarantees 16 fragment texture units. A config needing more
        // than nine LUTs is not something to fail silently on.
        const max = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
        if (7 + this._ocio.textures.length > max) {
            const need = 7 + this._ocio.textures.length;
            this._releaseOCIOTextures();
            this._ocioShaderSource = null;
            this._ocio = null;
            this.ocioEnabled = false;
            this._rebuildCompositeProgram();
            return { ok: false, enabled: false, error: `This view needs ${need} texture units and the GPU has ${max}.` };
        }

        this.ocioEnabled = true;
        return { ok: true, enabled: true };
    }

    _rebuildCompositeProgram() {
        const gl = this.gl;
        const next = this.createProgram(this.getBasicVertexShader(), this.getCompositeFragmentShader());
        if (!next) return false;
        const prev = this.programs.composite;
        this.programs.composite = next;
        // The uniform-location and value caches are keyed on the program, and
        // the old program's entries would otherwise shadow the new one's.
        this._uniformValueCache?.clear?.();
        this._uniformCache?.clear?.();
        if (prev && prev !== next) gl.deleteProgram(prev);
        return true;
    }

    _createOCIOTexture(t) {
        const gl = this.gl;
        const tex = gl.createTexture();
        const target = t.dimensions === 3 ? gl.TEXTURE_3D : gl.TEXTURE_2D;
        // OCIO hands back 1 channel for a 1D LUT and 3 for a 3D LUT. Uploading
        // a 1-channel LUT as RGB would read the wrong samples per texel.
        const internal = t.channels === 1 ? gl.R32F : gl.RGB32F;
        const format = t.channels === 1 ? gl.RED : gl.RGB;

        // 32-bit float textures are NOT filterable in WebGL2 unless
        // OES_texture_float_linear is enabled, and enabling means calling
        // getExtension -- merely having it in getSupportedExtensions() does
        // nothing. Without the call, LINEAR sampling of an R32F LUT returns
        // zero, and measured through a real context that made every HDR view
        // in the ACES configs render solid black while compiling cleanly and
        // reporting no error anywhere.
        if (this._ocioFloatLinear === undefined) {
            this._ocioFloatLinear = !!gl.getExtension('OES_texture_float_linear');
            if (!this._ocioFloatLinear) {
                console.warn('[Radiance] OES_texture_float_linear is unavailable; '
                    + 'OCIO LUTs will be sampled without interpolation, which will band.');
            }
        }
        const wantsNearest = /nearest/i.test(t.interpolation || '');
        const filter = (wantsNearest || !this._ocioFloatLinear) ? gl.NEAREST : gl.LINEAR;

        gl.bindTexture(target, tex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        if (t.dimensions === 3) {
            gl.texImage3D(target, 0, internal, t.width, t.height, t.depth, 0, format, gl.FLOAT, t.values);
            gl.texParameteri(target, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
        } else {
            gl.texImage2D(target, 0, internal, t.width, t.height, 0, format, gl.FLOAT, t.values);
        }
        gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(target, null);
        return tex;
    }

    _releaseOCIOTextures() {
        if (!this._ocio?.textures) return;
        for (const t of this._ocio.textures) {
            if (t.glTexture) this.gl.deleteTexture(t.glTexture);
        }
    }

    /** Bind OCIO's textures and uniforms. Called once per draw of the composite. */
    _applyOCIOUniforms(program) {
        const gl = this.gl;
        const ocioOn = this.ocioEnabled && !this.displayReferredTexture;
        this._ui1(program, 'u_ocioEnabled', ocioOn ? 1 : 0);
        if (!ocioOn || !this._ocio) return;

        for (const t of this._ocio.textures) {
            gl.activeTexture(gl.TEXTURE0 + t.unit);
            gl.bindTexture(t.dimensions === 3 ? gl.TEXTURE_3D : gl.TEXTURE_2D, t.glTexture);
            this._ui1(program, t.samplerName, t.unit);
        }

        // Dynamic properties. Radiance does not enable any yet, so OCIO
        // normally emits none -- but a uniform left unset would silently read
        // zero and change the picture, so anything that does appear is set, and
        // anything unrecognised is reported rather than skipped quietly.
        for (const u of this._ocio.uniforms) {
            const loc = this.getUniform(program, u.name);
            if (!loc) continue;
            switch (u.type) {
                case 'double': gl.uniform1f(loc, Number(u.value)); break;
                case 'bool': gl.uniform1i(loc, u.value ? 1 : 0); break;
                case 'float3': gl.uniform3fv(loc, Float32Array.from(u.value)); break;
                case 'vector_float': gl.uniform1fv(loc, Float32Array.from(u.value)); break;
                case 'vector_int': gl.uniform1iv(loc, Int32Array.from(u.value)); break;
                default:
                    console.warn('[Radiance] OCIO uniform type not handled:', u.type, u.name);
            }
        }
    }









    destroy() {
        const gl = this.gl;

        // Bilateral FBO
        this._destroyBilateralFBO();

        // v4.3: Bloom FBO chain (6-level Kawase) — was never freed
        this._destroyBloomFBOs();

        // OpenColorIO LUT textures. A show config can carry several 3D LUTs,
        // and the pattern in this method is that everything with a lifetime
        // longer than a frame gets freed here explicitly rather than left to
        // context loss.
        this._releaseOCIOTextures();
        this._ocio = null;

        // v4.3: Scope offscreen FBO + texture — was never freed
        if (this.scopeFBO)  { gl.deleteFramebuffer(this.scopeFBO);  this.scopeFBO  = null; }
        if (this.scopeTex)  { gl.deleteTexture(this.scopeTex);      this.scopeTex  = null; }

        // v4.3: Reference shelf GPU textures — was never freed
        if (this.referenceShelf && this.referenceShelf.length) {
            this.referenceShelf.forEach(tex => { if (tex) gl.deleteTexture(tex); });
            this.referenceShelf = [];
        }

        // v4.3: LRU frame texture cache
        this.clearFrameCache();

        // Compare B and its render targets
        this._clearCompareSource();
        for (const slot of ['_compareView', '_compareRef']) {
            const t = this[slot];
            if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); this[slot] = null; }
        }

        // Main texture map (image, reference, lut3d, depth, etc.)
        for (const tex of Object.values(this.textures)) {
            if (tex) gl.deleteTexture(tex);
        }

        // Shader programs
        for (const prog of Object.values(this.programs)) {
            if (prog) gl.deleteProgram(prog);
        }

        // Vertex buffers
        if (this.quadBuffer)  { gl.deleteBuffer(this.quadBuffer);  this.quadBuffer  = null; }
        if (this.scopeBuffer) { gl.deleteBuffer(this.scopeBuffer); this.scopeBuffer = null; }

        // Clear uniform caches
        this._uniformCache.clear();
        this._uniformValueCache.clear();

        // Null the maps. They previously kept the now-invalid handles, so a
        // scope debounce that fired after teardown passed the
        // if (!this.gl || !this.programs[mode]) return; guard and issued
        // useProgram/bindTexture on deleted objects -- INVALID_OPERATION spam
        // and a corrupted GL state shared with everything else on the page.
        this.textures = {};
        this.programs = {};

        // Explicitly release the driver context. Without this the context is
        // only reclaimed on GC, which browsers do lazily; ~16 add/delete cycles
        // hit Chrome's context limit and it starts killing the OLDEST context,
        // which may be the live viewer or ComfyUI's own canvas.
        _SHARED_FRAME_CACHE.renderers.delete(this);
        this.onContextLost = null;
        this.onContextRestored = null;
        this._lutSource = null;
        this._ocioInfo = null;
        this._destroyed = true;
        try { this.gl.getExtension('WEBGL_lose_context')?.loseContext(); } catch (e) { /* best effort */ }

        console.log('[Radiance] WebGL renderer destroyed — all GPU resources released');
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           SEQUENCE PLAYER
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceSequencePlayer {
    constructor(viewer) {
        this.viewer = viewer;
        this.frames = [];
        this.currentFrame = 0;
        this.fps = 24;
        this.isPlaying = false;
        this.loop = true;
        this.frameCache = new Map();
        this.maxCacheSize = 100; // Max frames in cache

        this.playInterval = null;
    }

    loadSequence(frameUrls) {
        this.frames = frameUrls;
        this.currentFrame = 0;
        this.clearCache();

        // Preload first few frames
        this.preloadFrames(0, Math.min(5, this.frames.length));

        console.log(`[Radiance] Loaded sequence with ${this.frames.length} frames`);
    }

    preloadFrames(start, count) {
        for (let i = start; i < start + count && i < this.frames.length; i++) {
            if (!this.frameCache.has(i)) {
                this.loadFrame(i);
            }
        }
    }

    loadFrame(index) {
        return new Promise((resolve, reject) => {
            if (this.frameCache.has(index)) {
                resolve(this.frameCache.get(index));
                return;
            }

            const img = new Image();
            img.crossOrigin = 'anonymous';

            img.onload = () => {
                // Manage cache size
                if (this.frameCache.size >= this.maxCacheSize) {
                    // Remove oldest frame (furthest from current)
                    const keys = Array.from(this.frameCache.keys());
                    const distances = keys.map(k => Math.abs(k - this.currentFrame));
                    const maxDist = Math.max(...distances);
                    const toRemove = keys[distances.indexOf(maxDist)];
                    this.frameCache.delete(toRemove);
                }

                this.frameCache.set(index, img);
                resolve(img);
            };

            img.onerror = reject;
            img.src = this.frames[index];
        });
    }

    async displayFrame(index) {
        if (index < 0 || index >= this.frames.length) return;

        this.currentFrame = index;

        try {
            const img = await this.loadFrame(index);
            this.viewer.setImage(img);

            // Preload ahead
            this.preloadFrames(index + 1, 3);
        } catch (e) {
            console.error(`[Radiance] Failed to load frame ${index}: `, e);
        }
    }

    play() {
        if (this.isPlaying) return;
        if (this.frames.length === 0) return;

        this.isPlaying = true;

        // v3.1 PERF: requestAnimationFrame replaces setInterval — syncs to vsync,
        // eliminates double-fire and skipped frames that caused uneven playback.
        // Time-delta gate enforces the target FPS without blocking the display thread.
        const interval = 1000 / this.fps;
        let lastTime = 0;

        const tick = (timestamp) => {
            if (!this.isPlaying) return;
            if (timestamp - lastTime >= interval) {
                lastTime = timestamp;
                let nextFrame = this.currentFrame + 1;
                if (nextFrame >= this.frames.length) {
                    if (this.loop) {
                        nextFrame = 0;
                    } else {
                        this.pause();
                        return;
                    }
                }
                this.displayFrame(nextFrame);
            }
            this.playInterval = requestAnimationFrame(tick);
        };

        this.playInterval = requestAnimationFrame(tick);
        console.log(`[Radiance] Playing at ${this.fps} fps`);
    }

    pause() {
        if (!this.isPlaying) return;

        this.isPlaying = false;
        if (this.playInterval) {
            cancelAnimationFrame(this.playInterval);
            this.playInterval = null;
        }

        console.log('[Radiance] Paused');
    }

    togglePlayPause() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    stop() {
        this.pause();
        this.currentFrame = 0;
        this.displayFrame(0);
    }

    nextFrame() {
        this.pause();
        const next = (this.currentFrame + 1) % this.frames.length;
        this.displayFrame(next);
    }

    prevFrame() {
        this.pause();
        const prev = (this.currentFrame - 1 + this.frames.length) % this.frames.length;
        this.displayFrame(prev);
    }

    goToFrame(index) {
        this.pause();
        this.displayFrame(Math.max(0, Math.min(index, this.frames.length - 1)));
    }

    setFPS(fps) {
        this.fps = Math.max(1, Math.min(fps, 120));

        if (this.isPlaying) {
            this.pause();
            this.play();
        }
    }

    setLoop(loop) {
        this.loop = loop;
    }

    clearCache() {
        this.frameCache.clear();
    }

    getProgress() {
        if (this.frames.length === 0) return 0;
        return this.currentFrame / (this.frames.length - 1);
    }

    getTimecode() {
        const totalSeconds = this.currentFrame / this.fps;
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        const frames = this.currentFrame % this.fps;

        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}:${String(frames).padStart(2, '0')} `;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           RGB PARADE WAVEFORM
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceRGBParade {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
    }

    render(imageData) {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        // Clear
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, w, h);

        // Section width (3 channels)
        const sectionWidth = Math.floor(w / 3);
        const padding = 4;

        // Draw each channel
        this.drawChannel(imageData, 0, 0, sectionWidth - padding, h, 'red');
        this.drawChannel(imageData, 1, sectionWidth, sectionWidth - padding, h, 'green');
        this.drawChannel(imageData, 2, sectionWidth * 2, sectionWidth - padding, h, 'blue');

        // Draw separators
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sectionWidth, 0);
        ctx.lineTo(sectionWidth, h);
        ctx.moveTo(sectionWidth * 2, 0);
        ctx.lineTo(sectionWidth * 2, h);
        ctx.stroke();

        // Draw 0%, 50%, 100% lines
        ctx.strokeStyle = '#444';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, h * 0.5);
        ctx.lineTo(w, h * 0.5);
        ctx.stroke();
        ctx.setLineDash([]);

        // Labels
        ctx.fillStyle = '#666';
        ctx.font = '10px monospace';
        ctx.fillText('R', 4, 12);
        ctx.fillText('G', sectionWidth + 4, 12);
        ctx.fillText('B', sectionWidth * 2 + 4, 12);
    }

    drawChannel(imageData, channelIndex, xOffset, width, height, color) {
        const ctx = this.ctx;
        const data = imageData.data;
        const imgWidth = imageData.width;
        const imgHeight = imageData.height;

        // Sample columns
        const columns = Math.min(width, imgWidth);
        const columnWidth = imgWidth / columns;

        // Create histogram for each column
        for (let col = 0; col < columns; col++) {
            const sourceCol = Math.floor(col * columnWidth);
            const histogram = new Uint32Array(256);

            // Sample pixels in this column
            for (let row = 0; row < imgHeight; row++) {
                const pixelIndex = (row * imgWidth + sourceCol) * 4;
                const value = data[pixelIndex + channelIndex];
                histogram[value]++;
            }

            // Draw column
            const x = xOffset + col;
            for (let y = 0; y < 256; y++) {
                const count = histogram[y];
                if (count > 0) {
                    const intensity = Math.min(count / (imgHeight * 0.1), 1.0);
                    const yPos = height - (y / 255) * height;

                    ctx.fillStyle = this.getChannelColor(color, intensity);
                    ctx.fillRect(x, yPos - 1, 1, 2);
                }
            }
        }
    }

    getChannelColor(channel, intensity) {
        const alpha = Math.min(intensity * 0.8 + 0.2, 1.0);
        switch (channel) {
            case 'red': return `rgba(255, 80, 80, ${alpha})`;
            case 'green': return `rgba(80, 255, 80, ${alpha})`;
            case 'blue': return `rgba(80, 120, 255, ${alpha})`;
            default: return `rgba(255, 255, 255, ${alpha})`;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  RHDR BINARY FORMAT SPECIFICATION  (Radiance v3.1)
// ───────────────────────────────────────────────────────────────────────────────
//  .rhdr is a lightweight sidecar format that carries scene-linear float pixel
//  data alongside the tone-mapped PNG that ComfyUI displays.  It enables the
//  HDR color picker to report true pre-display values in linear light.
//
//  File layout
//  ───────────
//  Bytes    Type         Endian  Field
//  ───────  ───────────  ──────  ──────────────────────────────────────────────
//  0–3      char[4]      n/a     Magic: ASCII "RHDR"
//  4–5      uint16       LE      Image width  (pixels, max 65535)
//  6–7      uint16       LE      Image height (pixels, max 65535)
//  8–9      uint16       LE      Channel count (typically 3 = RGB or 4 = RGBA)
//  10–11    uint16       LE      Flags:  0 = fp16 payload, 1 = fp32 payload
//  12–EOF   uint8[]      n/a     zlib-deflate compressed pixel data
//
//  Pixel data (after decompression)
//  ─────────────────────────────────
//  flags = 0  →  float16 (IEEE 754 half precision) row-major, interleaved RGBRGB…
//  flags = 1  →  float32 (IEEE 754 single precision) row-major, interleaved RGBRGB…
//  Stride = width × channels × bytes_per_element
//  Pixel order: top-left origin, left-to-right, top-to-bottom (same as PNG).
//
//  Design constraints
//  ──────────────────
//  • uint16 width/height limits: max 65535×65535 (validated on write in Python).
//  • 50 MB decompressed safety cap to prevent decompression-bomb attacks.
//
//  This is equivalent to Nuke's "info" toolbar pixel inspector or RV's
//  color picker — values are in raw linear light, not display-encoded.
//
//  Usage:
//      const picker = new RadianceHDRPicker('/view?filename=...rhdr&type=temp');
//      await picker.load();
//      const px = picker.sample(0.5, 0.5);  // → { r, g, b, a, luma, stops }
//
//  Events:
//      picker.onHover(canvas, (px) => showReadout(px));
//      picker.onPick(canvas, (px, uv) => logColor(px, uv));
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceHDRPicker {
    /**
     * @param {string} rhdrUrl – URL to the .rhdr file served by ComfyUI's
     *                           /view endpoint, e.g.
     *                           '/view?filename=radiance_viewer_abc123_0.rhdr&type=temp'
     */
    constructor(rhdrUrl) {
        this.url = rhdrUrl;
        this.width = 0;
        this.height = 0;
        this.channels = 0;
        this.data = null;   // Float32Array — converted from fp16 on load
        this.loaded = false;
        this._abortCtrl = null;
    }

    // ── Load & decode .rhdr ─────────────────────────────────────────────────

    async load() {
        if (this.loaded) return this;

        this._abortCtrl = new AbortController();
        const res = await fetch(this.url, { signal: this._abortCtrl.signal });
        if (!res.ok) throw new Error(`[Picker] Failed to fetch ${this.url}: ${res.status}`);

        const compressed = await res.arrayBuffer();
        const bytes = new Uint8Array(compressed);

        // ── Parse .rhdr header ───────────────────────────────────────────────
        // Magic: 4 bytes 'RHDR'
        // Width:    uint16 LE
        // Height:   uint16 LE
        // Channels: uint16 LE
        // Reserved: uint16 LE
        const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
        if (magic !== 'RHDR') throw new Error(`[Picker] Not a .rhdr file (magic: ${magic})`);

        const view = new DataView(compressed);
        this.width = view.getUint16(4, true);
        this.height = view.getUint16(6, true);
        this.channels = view.getUint16(8, true);
        const flags = view.getUint16(10, true);
        const HEADER_SIZE = 12;

        // ── Zlib decompress ──────────────────────────────────────────────────
        // ComfyUI serves files directly — use DecompressionStream (Chrome/Firefox 103+)
        const compressedData = compressed.slice(HEADER_SIZE);
        const ds = new DecompressionStream('deflate');
        const writer = ds.writable.getWriter();
        const reader = ds.readable.getReader();

        writer.write(compressedData);
        writer.close();

        const chunks = [];
        let totalLen = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            totalLen += value.length;

            // v3.2: Security Hardening - Decompression Bomb Protection
            // Reject files larger than 50MB (uncompressed) to prevent memory exhaustion
            if (totalLen > 50 * 1024 * 1024) {
                await reader.cancel();
                throw new Error(`[Picker] .rhdr file exceeds 50MB safety limit. Potential decompression bomb.`);
            }
        }

        const decompressed = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of chunks) {
            decompressed.set(chunk, offset);
            offset += chunk.length;
        }

        // ── fp16 → float32 ───────────────────────────────────────────────────
        let fp16 = [];
        if (flags === 1) {
            if (decompressed.byteOffset % 4 === 0) {
                this.data = new Float32Array(decompressed.buffer, decompressed.byteOffset, decompressed.length / 4);
            } else {
                const copied = new Uint8Array(decompressed);
                this.data = new Float32Array(copied.buffer);
            }
        } else {
            fp16 = new Uint16Array(decompressed.buffer);
            this.data = new Float32Array(fp16.length);

            for (let i = 0; i < fp16.length; i++) {
                this.data[i] = RadianceHDRPicker._fp16ToFloat32(fp16[i]);
            }
        }

        this.loaded = true;
        console.log(`[Picker] Loaded ${this.width}×${this.height}×${this.channels}ch .rhdr`);
        return this;
    }

    cancel() {
        if (this._abortCtrl) this._abortCtrl.abort();
    }

    // ── Sample ──────────────────────────────────────────────────────────────

    /**
     * Sample the linear float value at normalised UV coordinates.
     * UV (0,0) = top-left, (1,1) = bottom-right.
     *
     * @param {number} u – 0..1 horizontal
     * @param {number} v – 0..1 vertical
     * @returns {{ r, g, b, a, luma, stops, hex }} or null if not loaded
     */
    sample(u, v) {
        if (!this.loaded || !this.data) return null;

        const x = Math.max(0, Math.min(this.width - 1, Math.floor(u * this.width)));
        const y = Math.max(0, Math.min(this.height - 1, Math.floor(v * this.height)));
        const idx = (y * this.width + x) * this.channels;

        const r = this.data[idx] ?? 0;
        const g = this.data[idx + 1] ?? 0;
        const b = this.data[idx + 2] ?? 0;
        const a = this.channels >= 4 ? (this.data[idx + 3] ?? 1) : 1;

        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const stops = luma > 1e-6 ? Math.log2(luma / 0.18) : -Infinity;

        // Display-encoded hex (clamped sRGB for UI colour swatches)
        const toU8 = v => Math.round(Math.min(Math.max(Math.pow(Math.max(v, 0), 1 / 2.2), 0), 1) * 255);
        const hex = '#' + [r, g, b].map(toU8).map(n => n.toString(16).padStart(2, '0')).join('');

        return { r, g, b, a, luma, stops, hex, x, y };
    }

    /**
     * Sample a 3×3 average around a UV point (reduces noise for precise grading).
     */
    sampleAverage(u, v, radius = 2) {
        if (!this.loaded) return null;
        let rs = 0, gs = 0, bs = 0, n = 0;
        const pxW = 1 / this.width;
        const pxH = 1 / this.height;
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const s = this.sample(u + dx * pxW, v + dy * pxH);
                if (s) { rs += s.r; gs += s.g; bs += s.b; n++; }
            }
        }
        if (!n) return null;
        const centerSample = this.sample(u, v);
        if (!centerSample) return null;
        return { ...centerSample, r: rs / n, g: gs / n, b: bs / n };
    }

    // ── Canvas Event Helpers ────────────────────────────────────────────────

    /**
     * Attach a hover listener to a canvas that calls cb(pixelInfo, uv) on mousemove.
     * Returns a cleanup function.
     */
    onHover(canvas, cb) {
        const handler = (e) => {
            const uv = this._canvasUV(canvas, e);
            const px = this.sample(uv.u, uv.v);
            if (px) cb(px, uv);
        };
        canvas.addEventListener('mousemove', handler);
        return () => canvas.removeEventListener('mousemove', handler);
    }

    /**
     * Attach a click listener to a canvas that calls cb(pixelInfo, uv) on click.
     * Returns a cleanup function.
     */
    onPick(canvas, cb) {
        const handler = (e) => {
            const uv = this._canvasUV(canvas, e);
            const px = this.sample(uv.u, uv.v);
            if (px) cb(px, uv);
        };
        canvas.addEventListener('click', handler);
        return () => canvas.removeEventListener('click', handler);
    }

    _canvasUV(canvas, e) {
        const rect = canvas.getBoundingClientRect();
        return {
            u: (e.clientX - rect.left) / rect.width,
            v: (e.clientY - rect.top) / rect.height,
        };
    }

    // ── fp16 → float32 ──────────────────────────────────────────────────────
    // IEEE 754 half-float to single-precision conversion.
    // Handles: ±zero, denormals, ±inf, NaN.

    static _fp16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;

        if (e === 0) {
            // Denormal / zero
            const val = f === 0 ? 0 : Math.pow(2, -14) * (f / 1024);
            return s ? -val : val;
        }
        if (e === 31) {
            // Inf / NaN
            return s ? -Infinity : (f ? NaN : Infinity);
        }
        const val = Math.pow(2, e - 15) * (1 + f / 1024);
        return s ? -val : val;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                    HDR PIXEL READOUT HUD  (v3.2 Fix 4)
//
//  Floating tooltip-style panel that shows the raw linear float values
//  under the cursor, matching Nuke's info bar layout.
// ═══════════════════════════════════════════════════════════════════════════════

class RadiancePixelReadout {
    /**
     * @param {HTMLElement}        container  – overlay div (position:relative parent)
     * @param {RadianceHDRPicker}  picker     – loaded picker instance
     * @param {HTMLCanvasElement}  glCanvas   – the WebGL display canvas
     */
    constructor(container, picker, glCanvas) {
        this.picker = picker;
        this.canvas = glCanvas;
        this._cleanup = [];

        // ── Build HUD element ─────────────────────────────────────────────────
        this.el = document.createElement('div');
        this.el.className = 'rad-hud-panel rad-pixel-readout';
        this.el.style.cssText = `
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 14px;
            pointer-events: none;
            white-space: nowrap;
            z-index: 130;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
        `;

        // Colour swatch
        this._swatch = document.createElement('div');
        this._swatch.style.cssText = 'width:18px;height:18px;border-radius:3px;border:1px solid rgba(255,255,255,0.2);flex-shrink:0;';
        this.el.appendChild(this._swatch);

        // Channel readouts
        this._channels = ['R', 'G', 'B', 'A', 'L', 'EV'].map(label => {
            const span = document.createElement('span');
            span.style.cssText = 'color:rgba(255,255,255,0.5);';
            this.el.appendChild(span);
            return span;
        });

        // Coordinate
        this._coord = document.createElement('span');
        this._coord.style.cssText = 'color:rgba(255,255,255,0.25);font-size:9px;';
        this.el.appendChild(this._coord);

        container.appendChild(this.el);

        // ── Attach events ─────────────────────────────────────────────────────
        const rmHover = picker.onHover(glCanvas, (px, uv) => this._update(px, uv));
        this._cleanup.push(rmHover);

        // Hide when cursor leaves canvas
        const onLeave = () => { this.el.style.opacity = '0'; };
        const onEnter = () => { this.el.style.opacity = '1'; };
        glCanvas.addEventListener('mouseleave', onLeave);
        glCanvas.addEventListener('mouseenter', onEnter);
        this._cleanup.push(() => {
            glCanvas.removeEventListener('mouseleave', onLeave);
            glCanvas.removeEventListener('mouseenter', onEnter);
        });
    }

    _update(px, uv) {
        const fmt = (v) => v.toFixed(4).padStart(7);
        const fmtStops = (s) => isFinite(s) ? (s >= 0 ? '+' : '') + s.toFixed(2) + ' EV' : '—';

        this._swatch.style.background = px.hex;

        const labels = ['R', 'G', 'B', 'A', 'Y'];
        const values = [px.r, px.g, px.b, px.a, px.luma];
        const colors = ['#ff6060', '#60ff90', '#6090ff', '#aaaaaa', '#cccccc'];

        labels.forEach((label, i) => {
            const ch = this._channels[i];
            ch.innerHTML = `<span style="color:${colors[i]};font-weight:700">${label}</span> ${fmt(values[i])}`;
        });

        this._channels[5].innerHTML =
            `<span style="color:#ffcc44;font-weight:700">EV</span> ${fmtStops(px.stops)}`;

        this._coord.textContent = `(${px.x}, ${px.y})`;
    }

    destroy() {
        this._cleanup.forEach(fn => fn());
        this.el.remove();
    }
}




// Export for use in radiance_viewer.js
if (typeof window !== 'undefined') {
    window.RadianceWebGLRenderer = RadianceWebGLRenderer;
    window.RadianceSequencePlayer = RadianceSequencePlayer;
    window.RadianceRGBParade = RadianceRGBParade;
    window.RadianceHDRPicker = RadianceHDRPicker;
    window.RadiancePixelReadout = RadiancePixelReadout;
}

export { RadianceWebGLRenderer, RadianceSequencePlayer, RadianceRGBParade, RadianceHDRPicker, RadiancePixelReadout };
