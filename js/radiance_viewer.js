import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { RadianceWebGLRenderer } from "./radiance_webgl.js?v=2.3.2";
import { RadianceNeuralMonitor } from "./radiance_neural.js";



class RadianceViewer {
    static singletonHUD = null;
    static activeInstance = null;
    static allInstances = new Set();

    /**
     * Security: Escape HTML special characters to prevent XSS.
     * Use this before inserting any backend-supplied or user-supplied
     * string into innerHTML.
     */
    static escapeHtml(str) {
        if (typeof str !== 'string') str = String(str);
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    constructor(node, container) {
        this.node = node;
        this.container = container;
        RadianceViewer.allInstances.add(this);
        RadianceViewer.activeInstance = this;
        this._lastProgress = 0;

        // Neural Monitor integration
        this.neuralMonitor = null;
        this._setupComfyListeners();

        // v2.3.3: Cinema Scope Fonts
        // B-13 FIX: Removed external Google Fonts dependency (fails in air-gapped studios).
        // Uses a system font stack that provides equivalent aesthetics on all platforms.
        // If you bundle Inter/JetBrains Mono WOFF2 locally, add @font-face rules here.
        if (!document.getElementById('radiance-fonts')) {
            const style = document.createElement('style');
            style.id = 'radiance-fonts';
            style.textContent = `
                :root {
                    --radiance-font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    --radiance-font-mono: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', 'Liberation Mono', monospace;
                }
            `;
            document.head.appendChild(style);
        }

        // v2.4: Global HUD Styles
        if (!document.getElementById('radiance-hud-styles')) {
            const style = document.createElement('style');
            style.id = 'radiance-hud-styles';
            style.innerHTML = `
                .radiance-glass-dock {
                    position: fixed;
                    z-index: 10000;
                    background: rgba(16, 16, 24, 0.75);
                    backdrop-filter: blur(14px) saturate(180%);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255,255,255,0.05);
                    display: flex;
                    flex-direction: column;
                    transition: opacity 0.25s ease;
                    overflow: visible;
                }
                /* Panel-embedded mode: overrides fixed positioning */
                .radiance-glass-dock.radiance-panel-embedded {
                    position: relative !important;
                    z-index: 1 !important;
                    border-radius: 0 !important;
                    border: none !important;
                    box-shadow: none !important;
                    width: 100% !important;
                    height: 100% !important;
                    flex: 1 !important;
                    overflow: hidden !important;
                    background: transparent !important;
                    backdrop-filter: none !important;
                }
                .radiance-right-control-panel {
                    flex: 0 0 var(--rcp-width, 580px);
                    display: flex;
                    flex-direction: column;
                    background: rgba(14, 14, 22, 0.97);
                    border-left: 1px solid rgba(60, 70, 100, 0.3);
                    overflow: hidden;
                    position: relative;
                    min-width: 320px;
                    max-width: 900px;
                }
                .radiance-right-control-panel .rcp-resize-handle {
                    position: absolute;
                    left: 0; top: 0; width: 4px; height: 100%;
                    cursor: ew-resize;
                    background: transparent;
                    transition: background 0.2s;
                    z-index: 10;
                }
                .radiance-right-control-panel .rcp-resize-handle:hover {
                    background: #00a8ff;
                }
                .radiance-glass-dock input[type="range"] { accent-color: #00a8ff; }
                
                /* Help Overlay Styles */
                .radiance-help-overlay {
                    position: absolute;
                    top: 0; left: 0; width: 100%; height: 100%;
                    background: rgba(10, 10, 15, 0.9);
                    backdrop-filter: blur(20px);
                    z-index: 20000;
                    display: none;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 40px;
                    color: #fff;
                    font-family: 'Inter', sans-serif;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                }
                .radiance-help-content {
                    max-width: 800px;
                    width: 100%;
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 16px;
                    padding: 30px;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px 40px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
                }
                .help-group h3 {
                    font-size: 11px; color: #00a8ff; text-transform: uppercase;
                    letter-spacing: 2px; margin-bottom: 12px; border-bottom: 1px solid rgba(0, 168, 255, 0.2);
                    padding-bottom: 4px;
                }
                .help-item {
                    display: flex; justify-content: space-between; font-size: 13px; margin: 6px 0;
                }
                .help-key {
                    background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;
                    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #6af;
                    min-width: 20px; text-align: center;
                }
                .help-desc { color: #aaa; }
            `;
            document.head.appendChild(style);
        }

        this.theme = {
            bg: '#0a0a0f',
            panel: 'rgba(16, 16, 24, 0.95)',
            panelBorder: 'rgba(60, 70, 100, 0.25)',
            accent: '#00a8ff',
            text: '#e8e8f0',
            textDim: '#707088',
            font: "'Inter', system-ui, -apple-system, sans-serif",
            mono: "'JetBrains Mono', monospace"
        };

        // State
        this.image = null;
        this.compareImage = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.exposure = 0.0;
        this.channel = 'rgb';

        // DoF / Lens Settings
        this.dofEnabled = false;
        this.focusDistance = 0.5;
        this.aperture = 0.0;
        this.apertureBlades = 0;
        this.apertureRotation = 0.0;
        this.apertureAnamorphic = 1.0;
        this.activeLensSignature = null; // tracks active lens signature preset
        this.lensDistortion = 0.0;
        this.lensFringe = 0.0;
        this.vignetteIntensity = 0.0;
        this.vignetteFalloff = 0.5;
        this.bokehHighlightBias = 0.0;
        this.bokehSoapBubble = 0.0;
        this.bokehOpticalVig = 0.0;

        this.saturation = 1.0;
        this.zebraThreshold = 0.95;
        this.lutIntensity = 1.0;

        this.falseColor = false;
        this.zebra = false;
        this.gamutWarning = false;
        this.clippingMonitor = false;
        this.colorspace = 'sRGB';

        // Batch Navigation
        this.currentFrame = 0;
        this.totalFrames = 1;
        this.frameImages = [];
        this.frameCompareImages = [];
        this.frameZdepthImages = [];  // Z-Depth frames
        this.zdepthImage = null;       // Current zdepth image
        this.showZdepth = false;       // Toggle for zdepth display

        // ── Real-time Video Playback ──────────────────────────────────────────
        this.videoMode = false;         // true when a video file is loaded
        this.videoEl = null;            // HTMLVideoElement source
        this._videoRAF = null;          // requestAnimationFrame handle
        this._videoCanvas = null;       // offscreen canvas for frame capture
        this._videoCtx = null;          // 2D context of _videoCanvas
        this.playbackSpeed = 1.0;       // 0.25 / 0.5 / 1 / 2 / 4
        this.playbackFps = 24;          // target fps for frame-sequence playback
        this.isPlaying = false;
        this.loop = true;
        this.lastFrameTime = 0;
        // ─────────────────────────────────────────────────────────────────────

        // Safe Area Guides
        this.safeAreaMode = 'none'; // none, action, title, both

        // Focus Peaking
        this.focusPeaking = false;
        this.focusPeakingColor = '#ff0000';
        this.focusPeakingThreshold = 120;

        // Pixel Loupe
        this.showLoupe = true;
        this.loupeSize = 80;
        this.loupeMagnification = 8;

        // Comparison
        this.compareMode = 'none';
        this.wipePosition = 0.5;
        this.isDraggingWipe = false;

        // Scopes
        this.showHistogram = false;
        this.showWaveform = false;
        this.showVectorscope = false;
        this.scopeOverlay = false;
        this.waveformParadeMode = true; // true = RGB parade
        this.scopeMode = localStorage.getItem('radiance_scope_mode') || 'parade'; // parade|waveform|histogram|vectorscope|falsecolor
        this.generationID = 0; // v3.1: Unique ID per execution to cancel stale async loads

        // Grid & Safe Areas

        // Grid & Safe Areas
        this.showGrid = false;
        this.gridMode = 0; // 0=off, 1=thirds, 2=safe areas, 3=center

        // Fullscreen

        // Fullscreen
        this.isFullscreen = false;

        // Pixel data
        this.imageData = null;
        this.lastPixelColor = null;

        this.initialized = false; // Track if we've set initial size

        // Run & Prompt
        this.showPromptPanel = false;
        this.promptPanel = null;
        this.isQueueing = false;

        // Progress
        this.progressBar = null;
        this.progressText = null;
        this.progressStart = 0;

        this.progressHistory = [];

        // Color Space / LUT
        this.displayLut = localStorage.getItem('radiance_hud_display_lut') || 'None';
        this.inputSpace = localStorage.getItem('radiance_hud_input_space') || 'None';
        this.lutOptions = [
            "None",
            "sRGB (Display)",
            "Rec.709 (Broadcast)",
            "Filmic (Cinematic)",
            "Reinhard Tonemap",
            "ACES Filmic",
            "LogC3 (ARRI EI800)",
            "LogC4 (ARRI Alexa 35)",
            "F-Log2 (Fujifilm)",
            "C-Log3 (Canon)",
            "Log3G10 (RED IPP2)",
            "DaVinci Intermediate",
            "BMD Film Gen5",
            "V-Log (Panasonic)",
            "RED Log3G10",
            "—",
            "N-Log (Nikon)",
            "Linear to Log (Generic)",
            "IDT: LogC3 → Linear",
            "IDT: LogC4 → Linear",
            "IDT: V-Log → Linear",
            "IDT: Log3G10 → Linear",
            "IDT: DaVinci → Linear",
            "IDT: BMD Gen5 → Linear",
            "IDT: N-Log → Linear",
            "False Color (Exposure)"
        ];

        this.inputSpaceOptions = [
            "None",
            "IDT: LogC3 → Linear",
            "IDT: LogC4 → Linear",
            "IDT: V-Log → Linear",
            "IDT: Log3G10 → Linear",
            "IDT: DaVinci → Linear",
            "IDT: BMD Gen5 → Linear",
            "IDT: N-Log → Linear",
            "IDT: F-Log2 → Linear",
            "IDT: C-Log3 → Linear"
        ];
        this.denoise = 0.0;
        this.grain = 0.0;
        this.grainSize = 1.0;
        this.grainColor = 0.0;
        this.grainAnimate = false;
        this.bloom = 0.0;
        this.bloomThreshold = 1.0;
        this.halation = 0.0;
        this.halationRadius = 1.0;
        this.halationThreshold = 0.35;
        this.diffusion = 0.0;
        this.anamorphicStreaks = 0.0;
        this.streakThreshold = 0.85;
        this.streakLength = 0.08;

        // Grading State
        this.exposure = 0.0;
        this.temperature = 0.0;
        this.tint = 0.0;
        this.contrast = 1.0;
        this.pivot = 0.5;
        this.saturation = 1.0;
        this.lift = [0, 0, 0];
        this.gamma = [1, 1, 1];
        this.gain = [1, 1, 1];
        this.offset = [0, 0, 0];

        // v2.2 Pro features
        this.displayLutMode = 0;
        this.displayLutStrength = 1.0;
        this.lumaMix = 1.0;
        this.wipe = 0.5;
        this.wipeEnabled = false;
        this.wipeRefEnabled = false;
        this.gridMode = 0;
        this.gridColor = [0.0, 0.7, 1.0, 0.5]; // Light blue guide color

        // Help Screen
        this.showHelp = false;
        this.helpPanel = null;

        // A/B Grading Bypass
        this._gradingBypassed = false;
        this._savedGrading = null;

        // Undo / Redo Stack
        this._undoStack = [];
        this._redoStack = [];
        this._undoMaxSize = 50;

        // HUD / Controls Panel
        this.showControls = true;
        this.controlsPanel = null;

        // HUD Panel Sizing and Position (persisted)
        const savedHudWidth = localStorage.getItem('radiance_hud_width');
        this.hudPanelWidth = savedHudWidth ? parseInt(savedHudWidth) : 580;
        this.hudPanelMinWidth = 360;
        this.hudPanelMaxWidth = 1200;
        const savedHudHeight = localStorage.getItem('radiance_hud_height2');
        this.hudPanelHeight = savedHudHeight ? parseInt(savedHudHeight) : null; // null = auto

        const savedHudX = localStorage.getItem('radiance_hud_x');
        const savedHudY = localStorage.getItem('radiance_hud_y');
        this.hudX = savedHudX !== null ? parseFloat(savedHudX) : null; // null means default (bottom center)
        this.hudY = savedHudY !== null ? parseFloat(savedHudY) : null;
        this.hudMinimized = localStorage.getItem('radiance_hud_minimized') === 'true';

        // Scope update debouncing (for performance)
        this.scopeUpdateTimer = null;
        this.scopeDebounceMs = 150; // Reduced for snappier feedback (industry standard: 100-200ms)

        // Scope Panel Sizing
        // Load saved width from localStorage, default to 280px
        const savedWidth = localStorage.getItem('radiance_scope_width');
        this.scopePanelWidth = savedWidth ? parseInt(savedWidth) : 280;
        this.scopePanelMinWidth = 200;
        this.scopePanelMaxWidth = 600;
        this.isResizingScopePanel = false;

        // v2.5: Qualifier State Initialization
        this.qualifierState = {
            enabled: false,
            showMask: false,
            h: 0.0, hW: 0.1, hS: 0.05,
            s: 0.5, sW: 0.5, sS: 0.1,
            l: 0.5, lW: 0.5, lS: 0.1
        };

        // v3.1: Masking (Power Windows) state
        this.maskState = {
            type: 0, // 0=None, 1=Circle, 2=Box
            center: [0.5, 0.5],
            scale: [0.3, 0.3],
            feather: 0.2,
            rotation: 0.0,
            invert: false,
            showOverlay: false
        };

        // v3.5: Rendering Throttle & Debounce State
        this._renderRequested = false;
        this._scopeUpdateRequested = false;
        this._scopeUpdateTimer = null;

        this.init();
    }

    init() {
        this.createUI();

        this.setupProgressUI();
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        requestAnimationFrame(() => this.resize());

        // Continuous grain ticker — only fires when grain is active
        this._startGrainTicker();

        // Wire ComfyUI events → terminal
        this._termWireEvents();

        // v3.1: OCIO auto-discovery (async, non-blocking)
        this.ocioInit();
    }

    // ── v3.4: Continuous Grain Ticker ────────────────────────────────────────
    // Drives a ~24fps RAF loop that re-renders only when grain is active,
    // so the noise pattern animates even when the user isn't touching any control.
    _startGrainTicker() {
        let lastT = 0;
        const FPS = 24;
        const INTERVAL = 1000 / FPS;

        const tick = (t) => {
            this._grainRAF = requestAnimationFrame(tick);
            // Only animate if grain is on AND animate mode is enabled
            if ((this.grain || 0) <= 0.0) return;
            if (!this.grainAnimate) return;
            if (t - lastT < INTERVAL) return;
            lastT = t;
            if (!this.renderer || !this.renderer.textures.image) return;
            this.renderer.setTime(t / 1000.0);
            this.render();
        };
        this._grainRAF = requestAnimationFrame(tick);
    }

    _stopGrainTicker() {
        if (this._grainRAF) { cancelAnimationFrame(this._grainRAF); this._grainRAF = null; }
    }

    // ── v3.4: Eyedropper White Balance ───────────────────────────────────────
    // Clicking a neutral/grey pixel auto-computes the temperature+tint deviation
    // from D65 and sets the WB controls to correct it in one click.
    _toggleWBPicker(btn) {
        this._wbPickerActive = !this._wbPickerActive;
        if (btn) {
            btn.style.borderColor = this._wbPickerActive ? '#6af' : 'rgba(255,255,255,0.15)';
            btn.style.background = this._wbPickerActive ? 'rgba(80,160,255,0.2)' : 'rgba(255,255,255,0.05)';
            btn.style.color = this._wbPickerActive ? '#6af' : '#aaa';
        }

        // v3.5: OSD Feedback
        if (this._wbPickerActive) {
            this.canvas.style.cursor = 'crosshair';
            if (this._analysisLabel) {
                this._analysisLabel.textContent = 'PICK NEUTRAL POINT...';
                this._analysisLabel.style.display = 'block';
                this._analysisLabel.style.background = '#00a8ff';
                this._analysisLabel.style.color = '#fff';
            }

            this._wbPickHandler = (e) => {
                const rect = this.canvas.getBoundingClientRect();
                // v3.5: Correct for pan/zoom to find the exact image pixel
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                // Convert canvas pos -> image UV
                // (mouseX - panX) / zoom = imageX
                const imgU = (mouseX - this.panX) / (this.imageWidth * this.zoom);
                const imgV = (mouseY - this.panY) / (this.imageHeight * this.zoom);

                if (imgU < 0 || imgU > 1 || imgV < 0 || imgV > 1) {
                    console.warn('[WB] Picked outside image boundaries');
                    return;
                }

                // Sample from raw HDR data if available (32-bit float accuracy)
                let r, g, b;
                const hdr = this.hdrData || (this.frameHDRData && this.frameHDRData[this.currentFrame]);

                if (hdr && hdr.data) {
                    const ix = Math.floor(imgU * (hdr.width - 1));
                    const iy = Math.floor(imgV * (hdr.height - 1));
                    const idx = (iy * hdr.width + ix) * (hdr.channels || 3);
                    r = hdr.data[idx];
                    g = hdr.data[idx + 1];
                    b = hdr.data[idx + 2];
                } else {
                    // Fallback to display image (8-bit SDR)
                    const imgX = Math.round(imgU * (this.imageWidth - 1));
                    const imgY = Math.round(imgV * (this.imageHeight - 1));
                    const tmp = document.createElement('canvas');
                    tmp.width = 1; tmp.height = 1;
                    const tctx = tmp.getContext('2d');
                    tctx.drawImage(this.image, -imgX, -imgY);
                    const d = tctx.getImageData(0, 0, 1, 1).data;
                    r = d[0] / 255; g = d[1] / 255; b = d[2] / 255;
                }

                // Avoid picking near black or pure white (unreliable neutral)
                const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                if (luma < 0.01 || luma > 10.0) { // Wider range for HDR
                    console.warn('[WB] Picked pixel is invalid — choose a neutral grey');
                    return;
                }

                // Normalize to equal energy
                const avg = (r + g + b) / 3.0;
                if (avg < 1e-6) return;

                const devR = r / avg - 1.0;
                const devB = b / avg - 1.0;
                const devG = g / avg - 1.0;

                // Map RGB deviation -> temperature/tint correction
                const newTemp = -(devR - devB) * 0.8; // Increased gain for better correction
                const newTint = -(devR + devB - 2 * devG) * 0.4;

                this.temperature = Math.max(-2, Math.min(2, this.temperature + newTemp));
                this.tint = Math.max(-2, Math.min(2, this.tint + newTint));

                if (this.renderer) {
                    this.renderer.setTemperature(this.temperature);
                    this.renderer.setTint(this.tint);
                }
                this.requestRender();
                this.requestScopeUpdate();

                console.log(`[WB] Corrected temp=${this.temperature.toFixed(3)} tint=${this.tint.toFixed(3)} from luma=${luma.toFixed(3)}`);

                // Auto-deactivate after pick
                this._wbPickerActive = false;
                if (btn) {
                    btn.style.borderColor = 'rgba(255,255,255,0.15)';
                    btn.style.background = 'rgba(255,255,255,0.05)';
                    btn.style.color = '#aaa';
                }
                if (this._analysisLabel) this._analysisLabel.style.display = 'none';

                this.canvas.removeEventListener('mousedown', this._wbPickHandler);
                this.canvas.style.cursor = 'crosshair';
                if (this._lastRenderContent) this._lastRenderContent();
            };
            // v3.5: Use mousedown for better reliability in ComfyUI
            this.canvas.addEventListener('mousedown', this._wbPickHandler, { once: true });
        } else {
            if (this._wbPickHandler) this.canvas.removeEventListener('mousedown', this._wbPickHandler);
            if (this._analysisLabel) this._analysisLabel.style.display = 'none';
            this.canvas.style.cursor = 'crosshair';
        }
    }

    // ── v4.1: Pipeline Precision Control ─────────────────────────────────────

    /**
     * Cycle through pipeline precision modes: u8 → f16 → f32 → u8 …
     * Called by status bar badge click and Alt+B keyboard shortcut.
     */
    _cyclePipelinePrecision() {
        const modes = ['u8', 'f16', 'f32'];
        const current = this.renderer ? this.renderer.pipelinePrecision : 'f32';
        const next = modes[(modes.indexOf(current) + 1) % modes.length];
        this._setPipelinePrecision(next);
    }

    /**
     * Set pipeline precision to a specific mode and update all UI + renderer state.
     * @param {'u8'|'f16'|'f32'} mode
     */
    _setPipelinePrecision(mode) {
        if (this.renderer) {
            this.renderer.setPipelinePrecision(mode);
            // Re-upload curve LUTs at new precision so they take effect immediately
            if (this.curveEditor) {
                this.curveEditor.notifyChange();
            }
            this.render();
        }
        this._updateBitDepthBadge();
        // Persist to localStorage so the choice survives page reload
        localStorage.setItem('radiance_pipeline_precision', mode);
        this._termLog?.('info', `[Pipeline] Precision set to ${mode.toUpperCase()} — ${this._precisionLabel(mode)}`);
    }

    /** Human-readable label for a precision mode. */
    _precisionLabel(mode) {
        return mode === 'f32' ? 'FLOAT 32-bit (IEEE 754)' :
            mode === 'f16' ? 'FLOAT 16-bit (Half Float)' :
                'INT 8-bit (SDR)';
    }

    /**
     * Update the status bar bit-depth badge with full pipeline chain info.
     * Format: "FP32 · RGBA32F" (matches Nuke / Flame / Baselight style)
     */
    _updateBitDepthBadge() {
        if (!this.bitDepthInfo) return;

        // Input precision derived from image type
        let inputLabel = 'INT8';
        let inputColor = this.theme.textDim;

        if (this.hdrData) {
            if (this.hdrData.format === 'rhdr') {
                inputLabel = 'FP16';
                inputColor = '#60a5fa'; // blue — half-float
            } else if (this.hdrData.format === 'rhdr_f32') {
                inputLabel = 'FP32';
                inputColor = '#4ade80'; // green — full float
            } else {
                inputLabel = 'FP32';
                inputColor = '#4ade80'; // green — full float (EXR, npy, etc.)
            }
        } else if (this.image) {
            inputLabel = 'INT8';
            inputColor = this.theme.textDim;
        }

        // Pipeline precision from renderer
        let pipeLabel = '·  RGBA32F';
        let pipeColor = '#4ade80';
        if (this.renderer) {
            const mode = this.renderer.pipelinePrecision;
            if (mode === 'f32') {
                pipeLabel = '·  RGBA32F'; pipeColor = '#4ade80';  // green
            } else if (mode === 'f16') {
                pipeLabel = '·  RGBA16F'; pipeColor = '#60a5fa';  // blue
            } else {
                pipeLabel = '·  RGBA8'; pipeColor = this.theme.textDim;
            }
        }

        // Badge: "FP32 · RGBA32F"
        const dominantColor = (inputLabel === 'FP32' || (this.renderer && this.renderer.pipelinePrecision === 'f32'))
            ? '#4ade80' : (inputLabel === 'FP16' || (this.renderer && this.renderer.pipelinePrecision === 'f16'))
                ? '#60a5fa' : this.theme.textDim;

        this.bitDepthInfo.textContent = `${inputLabel}  ${pipeLabel}`;
        this.bitDepthInfo.style.color = dominantColor;
        this.bitDepthInfo.style.background = `${dominantColor}15`;
        this.bitDepthInfo.style.border = `1px solid ${dominantColor}35`;
        this.bitDepthInfo.title = `Input: ${inputLabel} | Pipeline: ${this.renderer ? this._precisionLabel(this.renderer.pipelinePrecision) : '—'}\nClick to cycle precision (INT 8 / FLOAT 16 / FLOAT 32)\nShortcut: Alt+B`;
    }

    showUI(visible) {
        const opacity = visible ? '1' : '0';
        const pointerEvents = visible ? 'all' : 'none';

        if (this.rightControlPanel) {
            // Panel-embedded: fade the whole panel
            this.rightControlPanel.style.opacity = opacity;
            this.rightControlPanel.style.pointerEvents = pointerEvents;
        } else if (this.controlsPanel) {
            this.controlsPanel.style.opacity = opacity;
            this.controlsPanel.style.pointerEvents = pointerEvents;
        }
        if (this.toolbar) {
            this.toolbar.style.opacity = opacity;
            this.toolbar.style.pointerEvents = pointerEvents;
            this.toolbar.style.transition = 'opacity 0.4s';
        }
        if (this.bottomInfoBar) {
            this.bottomInfoBar.style.display = visible ? 'flex' : 'none';
        }
        if (this.transportPanel) {
            this.transportPanel.style.opacity = opacity;
            this.transportPanel.style.pointerEvents = pointerEvents;
            this.transportPanel.style.transform = visible ? 'translateX(-50%)' : 'translate(-50%, 10px)';
        }
    }

    // --- Multi-Layer Grading Setup ---
    createGradeLayer() {
        return {
            exposure: 0.0,
            temperature: 0.0,
            tint: 0.0,
            contrast: 1.0,
            pivot: 0.5,
            saturation: 1.0,
            lift: [0, 0, 0],
            gamma: [1, 1, 1],
            gain: [1, 1, 1],
            offset: [0, 0, 0],
            colorScience: 0,
            inputSpace: 'None',
            denoise: 0.0,
            grain: 0.0,
            maskState: {
                type: 0, center: [0.5, 0.5], scale: [0.3, 0.3],
                feather: 0.2, rotation: 0.0, invert: false, showOverlay: false
            },
            qualifierState: {
                enabled: false, showMask: false,
                h: 0.0, hW: 0.1, hS: 0.05,
                s: 0.5, sW: 0.5, sS: 0.1,
                l: 0.5, lW: 0.5, lS: 0.1
            }
        };
    }

    addGradeLayer() {
        this.grades.push(this.createGradeLayer());
        this.activeGradeIndex = this.grades.length - 1;
        this.applyGradeLayer(this.activeGradeIndex);
    }

    applyGradeLayer(index) {
        if (!this.grades || !this.grades[index]) return;
        const g = this.grades[index];
        this.exposure = g.exposure;
        this.temperature = g.temperature;
        this.tint = g.tint;
        this.contrast = g.contrast;
        this.pivot = g.pivot;
        this.saturation = g.saturation;
        this.lift = [...g.lift];
        this.gamma = [...g.gamma];
        this.gain = [...g.gain];
        this.offset = [...g.offset];
        this.colorScience = g.colorScience;
        this.inputSpace = g.inputSpace || 'None';
        this.denoise = g.denoise;
        this.grain = g.grain;
        this.maskState = JSON.parse(JSON.stringify(g.maskState));
        this.qualifierState = JSON.parse(JSON.stringify(g.qualifierState));
        this.render();
    }

    syncActiveGradeLayer() {
        if (!this.grades || !this.grades[this.activeGradeIndex]) return;
        const g = this.grades[this.activeGradeIndex];
        g.exposure = this.exposure;
        g.temperature = this.temperature;
        g.tint = this.tint;
        g.contrast = this.contrast;
        g.pivot = this.pivot;
        g.saturation = this.saturation;
        g.lift = [...this.lift];
        g.gamma = [...this.gamma];
        g.gain = [...this.gain];
        g.offset = [...this.offset];
        g.colorScience = this.colorScience;
        g.inputSpace = this.inputSpace || 'None';
        g.denoise = this.denoise;
        g.grain = this.grain;
        g.maskState = JSON.parse(JSON.stringify(this.maskState));
        g.qualifierState = JSON.parse(JSON.stringify(this.qualifierState));
    }

    createUI() {
        const t = this.theme;

        this.container.style.cssText = `
            position: relative;
            width: 100%;
            height: 100%;
            min-height: 300px;
            background: linear-gradient(180deg, #0f0f14 0%, #08080c 100%);
            border-radius: 8px;
            overflow: hidden;
            user-select: none;
            border: 1px solid ${t.panelBorder};
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            flex-direction: column;
        `;

        // Toolbar - v3.5: Moved to Left HUD (analysisHUD)
        this.toolbar = document.createElement('div');
        this.toolbar.style.display = 'none'; // Hide horizontal top bar
        this.container.appendChild(this.toolbar);
        // this.createToolbar(); // Will be called inside createMainLeftHUD

        // Main area
        this.mainArea = document.createElement('div');
        this.mainArea.style.cssText = `flex: 1; display: flex; position: relative; overflow: hidden;`;
        this.container.appendChild(this.mainArea);

        // Canvas wrapper
        this.canvasWrapper = document.createElement('div');
        this.canvasWrapper.style.cssText = `flex: 1; position: relative; overflow: hidden;`;
        this.mainArea.appendChild(this.canvasWrapper);

        // ── Right Control Panel (HUD host) ────────────────────────────────────
        const rcpWidth = parseInt(localStorage.getItem('radiance_rcp_width') || '580');
        this.rightControlPanel = document.createElement('div');
        this.rightControlPanel.className = 'radiance-right-control-panel';
        this.rightControlPanel.style.setProperty('--rcp-width', rcpWidth + 'px');
        this.rightControlPanel.style.flex = `0 0 ${rcpWidth}px`;
        // Resize handle on left edge
        const rcpHandle = document.createElement('div');
        rcpHandle.className = 'rcp-resize-handle';
        this.rightControlPanel.appendChild(rcpHandle);
        let _rcpResizing = false;
        rcpHandle.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            _rcpResizing = true;
            const startX = e.clientX;
            const startW = this.rightControlPanel.offsetWidth;
            const onMove = (me) => {
                const newW = Math.min(900, Math.max(320, startW - (me.clientX - startX)));
                this.rightControlPanel.style.flex = `0 0 ${newW}px`;
                localStorage.setItem('radiance_rcp_width', newW);
            };
            const onUp = () => {
                _rcpResizing = false;
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
        this.mainArea.appendChild(this.rightControlPanel);

        // Create HUDs
        this.createHUD();
        this.createMainLeftHUD();

        // Main canvas (2D fallback)
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair;`;
        this.canvas.tabIndex = 0;
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
        this.canvasWrapper.appendChild(this.canvas);

        // WebGL Canvas (Primary renderer - GPU accelerated)
        this.glCanvas = document.createElement('canvas');
        this.glCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;`;
        this.canvasWrapper.appendChild(this.glCanvas); // Insert after 2D canvas so it overlays

        // Initialize WebGL Renderer
        this.useWebGL = true;
        try {
            if (typeof RadianceWebGLRenderer !== 'undefined') {
                this.renderer = new RadianceWebGLRenderer(this.glCanvas);
                if (this.renderer.init()) {
                    console.log("[Radiance] WebGL Renderer Initialized");
                    // v4.1: Restore pipeline precision from localStorage
                    // (renderer.init() sets extColorBufferFloat; check happens in setPipelinePrecision)
                    const savedPrec = localStorage.getItem('radiance_pipeline_precision') || 'f32';
                    if (savedPrec !== 'f32') {
                        // f32 is already the constructor default; only call if different
                        this.renderer.setPipelinePrecision(savedPrec);
                    }
                }
            } else {
                console.warn("[Radiance] WebGL Renderer class not found, falling back to 2D.");
                this.useWebGL = false;
            }
        } catch (e) {
            console.warn("[Radiance] WebGL init failed:", e);
            this.useWebGL = false;
        }

        // v3.0 #10: Detect Display-P3 / HDR monitor and configure canvas
        try {
            const p3info = RadianceWebGLRenderer.initDisplayP3(this.glCanvas);
            this._displayP3 = p3info;
            if (p3info.isP3) {
                console.log('[Radiance v3.0] Display-P3 monitor — canvas color management active');
            }
        } catch (e) {
            this._displayP3 = { isP3: false, isHDR: false };
        }

        // If WebGL failed, hide the canvas and use 2D fallback
        if (!this.renderer || !this.useWebGL) {
            this.glCanvas.style.display = 'none';
            console.warn("[Radiance] Falling back to 2D Canvas rendering");
        }
        // v3.0 FIX: Do NOT hide the 2D canvas. Architecture:
        //   glCanvas = off-screen WebGL render buffer (hidden in render())
        //   canvas   = display surface (2D context draws glCanvas with pan/zoom)
        // The old code hid the 2D canvas, making BOTH canvases invisible → black.



        // Overlay canvas
        this.overlayCanvas = document.createElement('canvas');
        this.overlayCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;`;
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.canvasWrapper.appendChild(this.overlayCanvas);

        // Scope panel
        this.scopePanel = document.createElement('div');
        this.scopePanel.style.cssText = `
            display: none;
            flex: 0 0 ${this.scopePanelWidth}px;
            flex-direction: column;
            background: rgba(0,0,0,0.9);
            border-left: 1px solid ${t.panelBorder};
            padding: 4px;
            overflow-y: auto;
        `;
        this.mainArea.appendChild(this.scopePanel);
        this.createScopes();

        // Resize handle for scope panel
        this.scopeResizeHandle = document.createElement('div');
        this.scopeResizeHandle.style.cssText = `
            position: absolute;
            left: 0;
            top: 0;
            width: 4px;
            height: 100%;
            cursor: ew-resize;
            background: transparent;
            transition: background 0.2s;
            z-index: 10;
        `;
        this.scopeResizeHandle.onmouseenter = () => {
            this.scopeResizeHandle.style.background = this.theme.accent;
        };
        this.scopeResizeHandle.onmouseleave = () => {
            if (!this.isResizingScopePanel) {
                this.scopeResizeHandle.style.background = 'transparent';
            }
        };
        this.scopePanel.appendChild(this.scopeResizeHandle);
        this.setupScopePanelResize();



        // Bottom Info Bar (HUD)
        this.bottomInfoBar = document.createElement('div');
        this.bottomInfoBar.style.cssText = `
            flex: 0 0 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            background: rgba(10, 10, 14, 0.95);
            border-top: 1px solid ${t.panelBorder};
            font-size: 10.5px;
            font-family: monospace;
            color: #aaa;
            white-space: nowrap;
            overflow: hidden;
            z-index: 1000;
        `;

        this.infoLeft = document.createElement('div');
        this.infoLeft.style.cssText = 'display: flex; gap: 16px; align-items: center;';
        this.bottomInfoBar.appendChild(this.infoLeft);

        this.infoRight = document.createElement('div');
        this.infoRight.style.cssText = 'display: flex; gap: 16px; align-items: center;';
        this.bottomInfoBar.appendChild(this.infoRight);

        // False Color Legend (Overlay within Bottom Bar)
        this.fcLegend = document.createElement('div');
        this.fcLegend.style.cssText = `
            position: absolute; left: 50%; transform: translateX(-50%);
            display: none; align-items: center; gap: 4px; height: 100%;
        `;
        const fcMap = [
            { s: 'CLIP B', c: '#9900cc' }, // Purple
            { s: 'SHAD', c: '#00ffff' },   // Cyan
            { s: 'MID', c: '#00cc33' },    // Green
            { s: 'SKIN', c: '#ff80cc' },   // Pink
            { s: 'NEAR W', c: '#ffff00' }, // Yellow
            { s: 'CLIP W', c: '#ff0000' }  // Red
        ];
        fcMap.forEach(item => {
            const block = document.createElement('div');
            block.style.cssText = `display: flex; align-items: center; gap: 3px;`;
            block.innerHTML = `<div style="width:8px; height:8px; background:${item.c}; border:1px solid rgba(255,255,255,0.2)"></div><span style="font-size:8.5px; color:#aaa">${item.s}</span>`;
            this.fcLegend.appendChild(block);
        });
        this.bottomInfoBar.appendChild(this.fcLegend);

        this.container.appendChild(this.bottomInfoBar);

        // Metadata Info Panel
        this.metadataPanel = document.createElement('div');
        this.metadataPanel.style.cssText = `
            position: absolute;
            top: 45px;
            right: 8px;
            width: 220px;
            background: rgba(0,0,0,0.92);
            border: 1px solid ${t.panelBorder};
            border-radius: 4px;
            padding: 8px;
            font-size: 9px;
            color: ${t.text};
            z-index: 90;
            display: none;
        `;

        const metaTitle = document.createElement('div');
        metaTitle.style.cssText = `font-weight: 600; font-size: 10px; margin-bottom: 8px; color: ${t.accent}; padding-bottom: 4px; border-bottom: 1px solid ${t.panelBorder};`;
        metaTitle.textContent = '◎ Image Info';
        this.metadataPanel.appendChild(metaTitle);

        this.metadataContent = document.createElement('div');
        this.metadataContent.style.cssText = 'line-height: 1.5; font-family: monospace;';
        this.metadataPanel.appendChild(this.metadataContent);

        this.canvasWrapper.appendChild(this.metadataPanel);

        // Status bar
        this.statusBar = document.createElement('div');
        this.statusBar.style.cssText = `
            flex: 0 0 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 8px;
            background: ${t.panel};
            border-top: 1px solid ${t.panelBorder};
            font-size: 9px;
            color: ${t.textDim};
        `;
        this.container.appendChild(this.statusBar);

        this.cursorInfo = document.createElement('span');
        this.cursorInfo.textContent = 'X: — Y: —';
        this.statusBar.appendChild(this.cursorInfo);

        this.colorInfo = document.createElement('span');
        this.colorInfo.textContent = 'RGB: — — —';
        this.colorInfo.style.cursor = 'pointer';
        this.colorInfo.title = 'Click to copy';
        this.colorInfo.onclick = () => this.copyColor();
        this.statusBar.appendChild(this.colorInfo);

        this.colorspaceInfo = document.createElement('span');
        this.colorspaceInfo.textContent = 'sRGB / HDR';
        this.colorspaceInfo.style.color = t.accent;
        this.statusBar.appendChild(this.colorspaceInfo);

        this.dimensionInfo = document.createElement('span');
        this.dimensionInfo.textContent = '—×—';
        this.statusBar.appendChild(this.dimensionInfo);

        this.zoomInfo = document.createElement('span');
        this.zoomInfo.textContent = '100%';
        this.zoomInfo.style.color = t.accent;
        this.statusBar.appendChild(this.zoomInfo);

        // v4.1: Pipeline precision indicator — shows full INPUT → PIPELINE chain
        // like Nuke's viewer bit-depth badge: "FP32  ·  RGBA32F"
        this.bitDepthInfo = document.createElement('span');
        this.bitDepthInfo.textContent = '—';
        this.bitDepthInfo.style.cssText = `
            padding: 1px 8px;
            border-radius: 3px;
            font-weight: 700;
            font-size: 9px;
            letter-spacing: 0.6px;
            cursor: pointer;
            user-select: none;
        `;
        this.bitDepthInfo.title = 'Pipeline bit depth — click to cycle precision (INT 8 / FLOAT 16 / FLOAT 32)';
        this.bitDepthInfo.addEventListener('click', () => this._cyclePipelinePrecision());
        this.statusBar.appendChild(this.bitDepthInfo);

        // Create metadata overlay (once, not per-render)
        this.createMetadataOverlay();

        // ── v3.4: Professional Terminal ──────────────────────────────────────
        this.createTerminal();
    }

    createTerminal() {
        // v2.3.3: Professional Terminal
        const t = this.theme;

        // ── Resize handle ────────────────────────────────────────────────────
        const resizeHandle = document.createElement('div');
        resizeHandle.style.cssText = `
            flex: 0 0 4px;
            width: 100%;
            background: transparent;
            cursor: ns-resize;
            transition: background 0.15s;
            border-top: 1px solid ${t.panelBorder};
        `;
        resizeHandle.onmouseenter = () => resizeHandle.style.background = t.accent + '55';
        resizeHandle.onmouseleave = () => { if (!this._termResizing) resizeHandle.style.background = 'transparent'; };

        // ── Terminal container ───────────────────────────────────────────────
        const termContainer = document.createElement('div');
        this._termContainer = termContainer;
        const savedH = parseInt(localStorage.getItem('radiance_term_height') || '180');
        termContainer.style.cssText = `
            flex: 0 0 ${savedH}px;
            display: flex;
            flex-direction: column;
            background: rgba(10, 15, 20, 0.95);
            backdrop-filter: blur(20px) saturate(160%);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
            font-family: ${t.mono};
            font-size: 11px;
            position: relative;
        `;

        // Scanline/Glow Effect
        const scanline = document.createElement('div');
        scanline.style.cssText = `
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.15) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
            background-size: 100% 2.5px, 2px 100%;
            pointer-events: none;
            z-index: 10;
            opacity: 0.1;
        `;
        termContainer.appendChild(scanline);

        // ── Header bar ───────────────────────────────────────────────────────
        const header = document.createElement('div');
        header.style.cssText = `
            flex: 0 0 22px;
            display: flex;
            align-items: center;
            padding: 0 8px;
            gap: 8px;
            background: #0d1117;
            border-bottom: 1px solid ${t.panelBorder};
            cursor: ns-resize;
            user-select: none;
        `;

        const tabsContainer = document.createElement('div');
        tabsContainer.style.cssText = `display: flex; gap: 8px; margin-right: auto;`;

        const createTab = (name, active) => {
            const btn = document.createElement('div');
            btn.textContent = name;
            btn.style.cssText = `
                font-size: 9px; font-weight: 600; letter-spacing: 0.08em;
                color: ${active ? '#00a8ff' : '#555'}; text-transform: uppercase;
                cursor: pointer; padding: 4px 8px;
                border-bottom: 2px solid ${active ? '#00a8ff' : 'transparent'};
                transition: 0.2s;
            `;
            return btn;
        };

        const tabTerm = createTab('◎ TERMINAL', true);
        const tabScript = createTab('◎ SCRIPTS', false);
        const tabDeliver = createTab('◎ DELIVER', false);

        tabsContainer.appendChild(tabTerm);
        tabsContainer.appendChild(tabScript);
        tabsContainer.appendChild(tabDeliver);
        header.appendChild(tabsContainer);

        // Session counter badge
        const badge = document.createElement('span');
        badge.style.cssText = `font-size: 8px; color: #555;`;
        badge.textContent = 'v2.3.3 · FXTD STUDIOS';
        header.appendChild(badge);

        // Clear button
        const clearBtn = document.createElement('span');
        clearBtn.textContent = 'CLEAR';
        clearBtn.style.cssText = `font-size: 8px; color: #555; cursor: pointer; padding: 1px 5px; border: 1px solid #222; border-radius: 2px; letter-spacing: 0.06em; transition: color 0.15s;`;
        clearBtn.onmouseenter = () => clearBtn.style.color = '#aaa';
        clearBtn.onmouseleave = () => clearBtn.style.color = '#555';
        clearBtn.onclick = () => { outputEl.innerHTML = ''; this._termLog('system', 'Terminal cleared.'); };
        header.appendChild(clearBtn);

        // Collapse toggle
        this._termCollapsed = false;
        const collapseBtn = document.createElement('span');
        collapseBtn.textContent = '▾';
        collapseBtn.style.cssText = `font-size: 11px; color: #555; cursor: pointer; line-height: 1; transition: color 0.15s;`;
        collapseBtn.onmouseenter = () => collapseBtn.style.color = '#aaa';
        collapseBtn.onmouseleave = () => collapseBtn.style.color = '#555';
        collapseBtn.onclick = () => this.toggleTerminal();

        this._termCollapseBtn = collapseBtn;
        this._termContainer = termContainer;
        this._termSavedH = savedH;

        header.appendChild(collapseBtn);
        termContainer.appendChild(header);

        // ── Output area ──────────────────────────────────────────────────────
        const outputEl = document.createElement('div');
        outputEl.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 6px 10px;
            line-height: 1.55;
            color: #c8d3e0;
            scroll-behavior: smooth;
            word-break: break-all;
        `;
        // Scrollbar styling
        outputEl.style.scrollbarWidth = 'thin';
        outputEl.style.scrollbarColor = '#222 transparent';
        termContainer.appendChild(outputEl);
        this._termOutput = outputEl;
        this._termOutputEl = outputEl;

        // ── Script Editor Area (Hidden by default) ───────────────────────────
        const scContainer = document.createElement('div');
        scContainer.style.cssText = `flex: 1; display: none; flex-direction: column; background: #080b0f;`;

        const scToolbar = document.createElement('div');
        scToolbar.style.cssText = `flex: 0 0 26px; display: flex; align-items: center; padding: 0 8px; gap: 8px; border-bottom: 1px solid #111; background: #0a0d12;`;

        const scModeSelect = document.createElement('select');
        scModeSelect.style.cssText = `background: #111; color: #aaa; border: 1px solid #222; font-size: 10px; padding: 2px; border-radius: 3px; outline: none;`;
        ['Batch Prompts', 'JavaScript'].forEach(m => {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m;
            scModeSelect.appendChild(opt);
        });

        const scTextarea = document.createElement('textarea');
        scTextarea.placeholder = "// Enter prompts (one per line) or JS code here...";
        scTextarea.style.cssText = `
            flex: 1; background: transparent; border: none; outline: none;
            color: #d8dee8; font-family: ${t.mono}; font-size: 11px;
            padding: 10px; resize: none; line-height: 1.4;
            scrollbar-width: thin; scrollbar-color: #333 transparent;
        `;

        const scSaveBtn = document.createElement('button');
        scSaveBtn.textContent = '💾 SAVE';
        scSaveBtn.style.cssText = `background: rgba(100, 100, 100, 0.15); color: #aaa; border: 1px solid rgba(100, 100, 100, 0.4); font-size: 9px; font-weight: bold; border-radius: 3px; padding: 3px 10px; cursor: pointer; margin-left: auto;`;
        scSaveBtn.onclick = () => {
            const data = scTextarea.value || "";
            if (!data) return;
            const blob = new Blob([data], { type: "text/plain" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `radiance_script_${Date.now()}.rds`;
            a.click();
            URL.revokeObjectURL(url);
        };

        const scLoadBtn = document.createElement('button');
        scLoadBtn.textContent = '📂 LOAD';
        scLoadBtn.style.cssText = `background: rgba(100, 100, 100, 0.15); color: #aaa; border: 1px solid rgba(100, 100, 100, 0.4); font-size: 9px; font-weight: bold; border-radius: 3px; padding: 3px 10px; cursor: pointer;`;
        scLoadBtn.onclick = () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.rds,.txt,.js,.json';
            input.onchange = e => {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = e => scTextarea.value = e.target.result;
                reader.readAsText(file);
            };
            input.click();
        };

        const scPathLabel = document.createElement('span');
        scPathLabel.textContent = 'SAVE PATH:';
        scPathLabel.style.cssText = `color: #e2a; font-size: 9px; font-weight: bold; margin-right: 5px; letter-spacing: 0.05em; display: none;`;

        const scPathInput = document.createElement('input');
        scPathInput.type = 'text';
        scPathInput.placeholder = 'Batch Save Path (e.g. D:/Exports/)';
        scPathInput.value = localStorage.getItem('radiance_batch_path') || '';
        scPathInput.style.cssText = `flex: 1; background: #1a1e24; color: #aaa; border: 1px solid #333; padding: 2px 5px; font-size: 10px; border-radius: 2px; margin-right: 5px; display: none;`;
        scPathInput.oninput = () => localStorage.setItem('radiance_batch_path', scPathInput.value);

        // Show/Hide path UI based on mode
        scModeSelect.onchange = () => {
            const isBatch = scModeSelect.value === 'Batch Prompts';
            scPathInput.style.display = isBatch ? 'block' : 'none';
            scPathLabel.style.display = isBatch ? 'block' : 'none';
        };
        const isInitialBatch = scModeSelect.value === 'Batch Prompts';
        scPathInput.style.display = isInitialBatch ? 'block' : 'none';
        scPathLabel.style.display = isInitialBatch ? 'block' : 'none';

        const scRunBtn = document.createElement('button');
        scRunBtn.textContent = '▶ RUN AUTOMATION';
        scRunBtn.style.cssText = `background: rgba(0, 168, 255, 0.1); color: #00a8ff; border: 1px solid rgba(0, 168, 255, 0.35); font-size: 9px; font-weight: bold; border-radius: 3px; padding: 3px 10px; cursor: pointer; white-space: nowrap;`;

        scToolbar.appendChild(scModeSelect);
        scToolbar.appendChild(scSaveBtn);
        scToolbar.appendChild(scLoadBtn);
        scToolbar.appendChild(scPathLabel);
        scToolbar.appendChild(scPathInput);
        scToolbar.appendChild(scRunBtn);

        scContainer.appendChild(scToolbar);
        scContainer.appendChild(scTextarea);
        termContainer.appendChild(scContainer);

        // ── VFX Delivery Area (Phase 5) ──────────────────────────────────
        const dvContainer = document.createElement('div');
        dvContainer.style.cssText = `flex: 1; display: none; flex-direction: column; background: #0b0e14; padding: 12px; overflow-y: auto;`;

        const dvTitle = document.createElement('div');
        dvTitle.textContent = '◎ RENDER SETTINGS — EXPORT GRADED MASTER';
        dvTitle.style.cssText = `font-size: 10px; font-weight: 800; color: #777; letter-spacing: 0.15em; margin-bottom: 12px; border-bottom: 1px solid #222; padding-bottom: 5px;`;
        dvContainer.appendChild(dvTitle);

        const dvForm = document.createElement('div');
        dvForm.style.cssText = `display: grid; grid-template-columns: 100px 1fr 100px 1fr; gap: 10px; align-items: center;`;

        const addRow = (label, el, span = 1) => {
            const lbl = document.createElement('label');
            lbl.textContent = label;
            lbl.style.cssText = `font-size: 9px; color: #555; text-transform: uppercase; font-weight: bold;`;
            dvForm.appendChild(lbl);
            if (span > 1) el.style.gridColumn = `span ${span}`;
            dvForm.appendChild(el);
        };

        const dvFilename = document.createElement('input');
        dvFilename.value = 'Radiance_Export';
        dvFilename.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 3px 6px; font-size: 10px; border-radius: 2px;`;
        addRow('Filename', dvFilename);

        const dvPath = document.createElement('input');
        dvPath.placeholder = 'Default Output Folder';
        dvPath.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 3px 6px; font-size: 10px; border-radius: 2px;`;
        addRow('Location', dvPath);

        const dvFormat = document.createElement('select');
        dvFormat.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 2px; font-size: 10px; border-radius: 2px;`;
        // IMPORTANT: These strings MUST exactly match OUTPUT_FORMATS in nodes_io.py
        [
            'Video — MP4 (H.264)',
            'Video — MP4 (H.265 10-bit)',
            'Video — MOV (ProRes 422 HQ)',
            'Video — MOV (ProRes 4444)',
            'Video — MOV (ProRes 4444 XQ)',
            'Video — MOV (ProRes 4444 HDR Log)',
            'Image Sequence — PNG (8-bit)',
            'Image Sequence — PNG (16-bit)',
            'Image Sequence — EXR (32-bit)',
            'Image Sequence — JPEG',
            'GIF (Animated)',
            'WEBP (Animated)',
        ].forEach(f => {
            const opt = document.createElement('option'); opt.value = f; opt.textContent = f; dvFormat.appendChild(opt);
        });
        addRow('Format', dvFormat);

        const dvColorSpace = document.createElement('select');
        dvColorSpace.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 2px; font-size: 10px; border-radius: 2px;`;
        ['Linear (sRGB)', 'sRGB (Standard)', 'ARRI LogC3', 'ARRI LogC4', 'Sony S-Log3', 'Panasonic V-Log', 'Canon Log 3', 'RED Log3G10', 'ACEScct', 'DaVinci Intermediate'].forEach(f => {
            const opt = document.createElement('option'); opt.value = f; opt.textContent = f; dvColorSpace.appendChild(opt);
        });
        addRow('Color Space', dvColorSpace);

        const dvRangeIn = document.createElement('input');
        dvRangeIn.type = 'number'; dvRangeIn.value = '1';
        dvRangeIn.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 3px 6px; font-size: 10px; border-radius: 2px; width: 50px;`;
        addRow('Range In', dvRangeIn);

        const dvRangeOut = document.createElement('input');
        dvRangeOut.type = 'number'; dvRangeOut.value = '0';
        dvRangeOut.title = '0 means exact end of sequence';
        dvRangeOut.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 3px 6px; font-size: 10px; border-radius: 2px; width: 50px;`;
        addRow('Range Out', dvRangeOut);

        const dvAspect = document.createElement('select');
        dvAspect.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 2px; font-size: 10px; border-radius: 2px;`;
        ['None', '2.35:1 (Scope)', '1.85:1 (Flat)', '1:1 (Square)', '9:16 (Vertical)'].forEach(f => {
            const opt = document.createElement('option'); opt.value = f; opt.textContent = f; dvAspect.appendChild(opt);
        });
        addRow('Letterbox', dvAspect);

        const dvFPS = document.createElement('input');
        dvFPS.type = 'number'; dvFPS.value = '24';
        dvFPS.style.cssText = `background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 3px 6px; font-size: 10px; border-radius: 2px; width: 50px;`;
        addRow('Frame Rate', dvFPS);

        const dvQuality = document.createElement('input');
        dvQuality.type = 'range'; dvQuality.min = '0'; dvQuality.max = '51'; dvQuality.value = '18';
        dvQuality.style.cssText = `width: 100%; height: 4px; accent-color: #00a8ff;`;
        addRow('Quality (CRF)', dvQuality);

        // Apex Options Section
        const dvApexSection = document.createElement('div');
        dvApexSection.style.cssText = `grid-column: span 4; margin-top: 15px; border-top: 1px solid #222; padding-top: 10px; display: flex; flex-wrap: wrap; gap: 15px;`;

        const createCheck = (label, id, def = false) => {
            const wrap = document.createElement('label');
            wrap.style.cssText = `display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 9px; color: #aaa; font-weight: 600; text-transform: uppercase;`;
            const cb = document.createElement('input');
            cb.type = 'checkbox'; cb.checked = def;
            cb.style.cssText = `accent-color: #00a8ff;`;
            wrap.appendChild(cb);
            wrap.appendChild(document.createTextNode(label));
            return { wrap, cb };
        };

        const { wrap: wUpscale, cb: dvUpscale } = createCheck('◎ AI Upscale (2x)', 'dvUpscale', false);
        const { wrap: wSmartVer, cb: dvSmartVer } = createCheck('◎ Smart Versioning', 'dvSmartVer', true);
        const { wrap: wSlate, cb: dvSlate } = createCheck('◎ Include Slate', 'dvSlate', false);
        const { wrap: wBurnTC, cb: dvBurnTC } = createCheck('◎ Burn Timecode', 'dvBurnTC', false);
        const { wrap: wBurnVer, cb: dvBurnVer } = createCheck('◎ Burn Version', 'dvBurnVer', false);
        const { wrap: wBurnFrame, cb: dvBurnFrame } = createCheck('◎ Burn Frame#', 'dvBurnFrame', false);
        const { wrap: wBurnLUT, cb: dvBurnLUT } = createCheck('◎ Burn LUT', 'dvBurnLUT', false);
        const { wrap: wExportCDL, cb: dvExportCDL } = createCheck('◎ Export .CDL', 'dvExportCDL', false);
        const { wrap: wRevealTarget, cb: dvRevealTarget } = createCheck('◎ Launch Folder', 'dvRevealTarget', true);
        const { wrap: wSoftClip, cb: dvSoftClip } = createCheck('◎ Soft Clip Highlights', 'dvSoftClip', true);

        dvApexSection.appendChild(wUpscale);
        dvApexSection.appendChild(wSmartVer);
        dvApexSection.appendChild(wExportCDL);
        dvApexSection.appendChild(wRevealTarget);
        dvApexSection.appendChild(wSoftClip);
        dvApexSection.appendChild(wSlate);
        dvApexSection.appendChild(wBurnTC);
        dvApexSection.appendChild(wBurnVer);
        dvApexSection.appendChild(wBurnFrame);
        dvApexSection.appendChild(wBurnLUT);
        dvForm.appendChild(dvApexSection);

        const customTextWrap = document.createElement('div');
        customTextWrap.style.cssText = `grid-column: span 4; display: flex; align-items: center; gap: 10px; margin-top: 5px;`;
        const customTextLbl = document.createElement('div');
        customTextLbl.textContent = "CUSTOM BURN-IN TEXT:";
        customTextLbl.style.cssText = `font-size: 9px; color: #666; font-weight: bold;`;
        const dvCustomText = document.createElement('input');
        dvCustomText.type = 'text'; dvCustomText.placeholder = 'e.g. VFX REVIEW [DO NOT DISTRIBUTE]';
        dvCustomText.style.cssText = `flex: 1; background: #1a1e24; color: #ccc; border: 1px solid #333; padding: 3px 6px; font-size: 10px; border-radius: 2px;`;
        customTextWrap.appendChild(customTextLbl);
        customTextWrap.appendChild(dvCustomText);
        dvForm.appendChild(customTextWrap);

        dvContainer.appendChild(dvForm);

        const dvSubmitRow = document.createElement('div');
        dvSubmitRow.style.cssText = `margin-top: 20px; display: flex; gap: 10px; border-top: 1px solid #222; padding-top: 15px;`;

        const dvRenderBtn = document.createElement('button');
        dvRenderBtn.textContent = '◎ ADD TO RENDER QUEUE & START EXPORT';
        dvRenderBtn.style.cssText = `background: #00a8ff; color: #fff; border: none; font-size: 10px; font-weight: 900; letter-spacing: 0.1em; border-radius: 3px; padding: 8px 20px; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px rgba(0, 168, 255, 0.3);`;
        dvRenderBtn.onmouseenter = () => dvRenderBtn.style.background = '#33beff';
        dvRenderBtn.onmouseleave = () => dvRenderBtn.style.background = '#00a8ff';

        dvSubmitRow.appendChild(dvRenderBtn);

        // Progress Tracking UI
        const dvProgressContainer = document.createElement('div');
        dvProgressContainer.style.cssText = `margin-top: 15px; display: none; flex-direction: column; gap: 5px;`;

        const dvProgressLabel = document.createElement('div');
        dvProgressLabel.style.cssText = `font-size: 9px; color: #00a8ff; font-weight: bold; text-transform: uppercase;`;
        dvProgressLabel.textContent = 'READY TO EXPORT';

        const dvProgressTrack = document.createElement('div');
        dvProgressTrack.style.cssText = `width: 100%; height: 2px; background: #1a1e24; border-radius: 1px; overflow: hidden;`;

        const dvProgressBar = document.createElement('div');
        dvProgressBar.style.cssText = `width: 0%; height: 100%; background: #00a8ff; transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 10px rgba(0, 168, 255, 0.5);`;

        dvProgressTrack.appendChild(dvProgressBar);
        dvProgressContainer.appendChild(dvProgressLabel);
        dvProgressContainer.appendChild(dvProgressTrack);
        dvContainer.appendChild(dvProgressContainer);

        // Render History UI
        const dvHistoryTitle = document.createElement('div');
        dvHistoryTitle.textContent = '◎ RECENT EXPORTS';
        dvHistoryTitle.style.cssText = `font-size: 9px; font-weight: 800; color: #444; margin-top: 25px; margin-bottom: 10px; border-bottom: 1px solid #1a1e24; padding-bottom: 3px;`;
        dvContainer.appendChild(dvHistoryTitle);

        const dvHistoryList = document.createElement('div');
        dvHistoryList.style.cssText = `display: flex; flex-direction: column; gap: 5px; max-height: 150px; overflow-y: auto;`;
        dvContainer.appendChild(dvHistoryList);

        const addHistoryItem = (path, name, qc) => {
            const item = document.createElement('div');
            item.style.cssText = `background: #111; border: 1px solid #222; padding: 6px 10px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; font-size: 10px;`;

            const info = document.createElement('div');
            info.innerHTML = `
                <div style="color: #eee; font-weight: bold;">${name}</div>
                <div style="color: #555; font-size: 9px; margin-top: 2px;">${path}</div>
                <div style="color: ${qc.includes('[QC PASS]') ? '#00ff88' : '#ff4444'}; font-size: 8px; margin-top: 2px;">${qc}</div>
            `;

            const openBtn = document.createElement('button');
            openBtn.textContent = '◎';
            openBtn.title = 'Open Path';
            openBtn.style.cssText = `background: transparent; border: none; cursor: pointer; filter: grayscale(1) opacity(0.5); font-size: 12px;`;
            openBtn.onclick = () => {
                // Try to open folder if supported by backend bridge, otherwise just log
                this._termLog('system', `Opening: ${path}`);
            };

            item.appendChild(info);
            item.appendChild(openBtn);

            if (dvHistoryList.firstChild) dvHistoryList.insertBefore(item, dvHistoryList.firstChild);
            else dvHistoryList.appendChild(item);
        };

        dvContainer.appendChild(dvSubmitRow);
        termContainer.appendChild(dvContainer);

        // Tab Switching Logic
        let activeTab = 'term';
        tabTerm.onclick = () => {
            activeTab = 'term';
            tabTerm.style.color = '#00a8ff'; tabTerm.style.borderColor = '#00a8ff';
            tabScript.style.color = '#555'; tabScript.style.borderColor = 'transparent';
            outputEl.style.display = 'block';
            this._termInputRow.style.display = 'flex';
            scContainer.style.display = 'none';
            this._termInput.focus();
        };
        tabScript.onclick = () => {
            activeTab = 'script';
            tabScript.style.color = '#7a92b0'; tabScript.style.borderColor = '#7a92b0';
            tabTerm.style.color = '#555'; tabTerm.style.borderColor = 'transparent';
            tabDeliver.style.color = '#555'; tabDeliver.style.borderColor = 'transparent';
            outputEl.style.display = 'none';
            this._termInputRow.style.display = 'none';
            scContainer.style.display = 'flex';
            dvContainer.style.display = 'none';
            setTimeout(() => scTextarea.focus(), 50);
        };
        tabDeliver.onclick = () => {
            activeTab = 'deliver';
            tabDeliver.style.color = '#00a8ff'; tabDeliver.style.borderColor = '#00a8ff';
            tabTerm.style.color = '#555'; tabTerm.style.borderColor = 'transparent';
            tabScript.style.color = '#555'; tabScript.style.borderColor = 'transparent';
            outputEl.style.display = 'none';
            this._termInputRow.style.display = 'none';
            scContainer.style.display = 'none';
            dvContainer.style.display = 'flex';
        };

        // Tab Switching Logic (cont)
        const oldTabTermClick = tabTerm.onclick;
        tabTerm.onclick = () => {
            oldTabTermClick();
            tabDeliver.style.color = '#555'; tabDeliver.style.borderColor = 'transparent';
            dvContainer.style.display = 'none';
        };

        // VFX Delivery Logic (Phase 5)
        dvRenderBtn.onclick = async () => {
            const nodeId = this.node.id;
            if (!nodeId) {
                alert("Viewer not initialized properly (Missing instance ID). Please queue a prompt first.");
                return;
            }

            dvRenderBtn.disabled = true;
            dvRenderBtn.textContent = '◎ PROCESSING VFX RENDER...';
            dvRenderBtn.style.opacity = '0.5';
            dvProgressContainer.style.display = 'flex';
            dvProgressBar.style.width = '0%';
            dvProgressLabel.textContent = 'Initializing Export Pipeline...';

            let progressInterval = setInterval(async () => {
                try {
                    const progRes = await fetch(`/radiance/progress?id=${this.instanceId || nodeId}`);
                    const prog = await progRes.json();
                    const percent = (prog.current / prog.total) * 100;
                    dvProgressBar.style.width = `${percent}%`;
                    dvProgressLabel.textContent = `${prog.message} [${percent.toFixed(0)}%]`;
                    if (prog.status === 'done' || prog.status === 'error') clearInterval(progressInterval);
                } catch (e) {
                    clearInterval(progressInterval);
                }
            }, 800);

            try {
                const response = await api.fetchApi('/radiance/deliver', {
                    method: 'POST',
                    body: JSON.stringify({
                        instance_id: this.instanceId || nodeId,
                        grading: {
                            exposure: this.exposure,
                            // Full per-channel arrays — NOT [0] only. Avoids silently
                            // dropping teal/orange and other per-channel grade decisions.
                            gamma: Array.isArray(this.gamma) ? [...this.gamma] : [this.gamma || 1.0, this.gamma || 1.0, this.gamma || 1.0],
                            gain: Array.isArray(this.gain) ? [...this.gain] : [this.gain || 1.0, this.gain || 1.0, this.gain || 1.0],
                            lift: Array.isArray(this.lift) ? [...this.lift] : [this.lift || 0.0, this.lift || 0.0, this.lift || 0.0],
                            offset: Array.isArray(this.offset) ? [...this.offset] : [this.offset || 0.0, this.offset || 0.0, this.offset || 0.0],
                            contrast: this.contrast,
                            pivot: this.pivot,
                            saturation: this.saturation,
                            temperature: this.temperature,
                            tint: this.tint,
                            colorScience: this.colorScience || 0,
                            // FX params — must match viewer for what-you-see = what-you-export
                            grain: this.grain || 0.0,
                            bloom: this.bloom || 0.0,
                            halation: this.halation || 0.0,
                            diffusion: this.diffusion || 0.0,
                            denoise: this.denoise || 0.0,
                        },
                        settings: {
                            filename: dvFilename.value,
                            path: dvPath.value,
                            format: dvFormat.value,
                            colorSpace: dvColorSpace.value,
                            aspect_ratio: dvAspect.value,
                            range_in: parseInt(dvRangeIn.value) || 1,
                            range_out: parseInt(dvRangeOut.value) || 0,
                            fps: dvFPS.value,
                            quality: dvQuality.value,
                            upscale_2x: dvUpscale.checked,
                            smart_versioning: dvSmartVer.checked,
                            export_cdl: dvExportCDL.checked,
                            reveal_folder: dvRevealTarget.checked,
                            include_slate: dvSlate.checked,
                            burn_in_tc: dvBurnTC.checked,
                            burn_in_ver: dvBurnVer.checked,
                            burn_in_frame: dvBurnFrame.checked,
                            burn_in_lut: dvBurnLUT.checked,
                            burn_custom_text: dvCustomText.value,
                            soft_clip: dvSoftClip.checked
                        }
                    })
                });

                const result = await response.json();
                clearInterval(progressInterval);

                if (result.status === 'success') {
                    dvProgressBar.style.width = '100%';
                    dvProgressLabel.textContent = '◎ EXPORT COMPLETE';
                    this._termLog('success', `[DELIVERY] ${result.message}`);
                    if (result.qc) {
                        const qcColor = result.qc.includes('[QC PASS]') ? '#00ff88' : '#ff4444';
                        this._termLog('system', `<span style="color: ${qcColor}">${result.qc}</span>`);
                    }
                    this._termLog('system', ` Saved to: ${result.path}`);

                    addHistoryItem(result.path, dvFilename.value, result.qc || 'QC N/A');

                    setTimeout(() => {
                        dvProgressContainer.style.display = 'none';
                    }, 3000);

                    tabTerm.onclick(); // Show success in terminal
                } else {
                    this._termLog('error', `[DELIVERY FAILED] ${result.error}`);
                    dvProgressLabel.textContent = '◎ EXPORT FAILED';
                    dvProgressBar.style.background = '#ff4444';
                    tabTerm.onclick();
                }
            } catch (e) {
                clearInterval(progressInterval);
                this._termLog('error', `[API ERROR] ${e.message}`);
                tabTerm.onclick();
            } finally {
                dvRenderBtn.disabled = false;
                dvRenderBtn.textContent = '◎ ADD TO RENDER QUEUE & START EXPORT';
                dvRenderBtn.style.opacity = '1';
            }
        };

        // Script Runner Logic
        scRunBtn.onclick = async () => {
            const mode = scModeSelect.value;
            const code = scTextarea.value.trim();
            if (!code) return;

            // Switch to terminal view to show logs
            tabTerm.onclick();
            this._termLog('event', `[Script] Starting ${mode} automation...`);

            if (mode === 'JavaScript') {
                try {
                    // Provide app and api in scope
                    const ctxFunc = new Function('app', 'api', 'logger', `
                        try {
                            ${code}
                        } catch(e) {
                            logger('error', '[JS Eval Error] ' + e.message);
                        }
                    `);
                    ctxFunc(app, api, (type, msg) => this._termLog(type, msg));
                } catch (e) {
                    this._termLog('error', `[Eval Error] ${e.message}`);
                }
            } else if (mode === 'Batch Prompts') {
                // Split by period or newline to allow paragraph-style or list-style entries, filter out empties
                const lines = code.split(/[\n\.]/).map(l => l.trim()).filter(l => l && l.length > 2);

                // Find CLIPTextEncode or Radiance Prompt node
                const nodes = app.graph._nodes;
                const clipNode = nodes.find(n =>
                    (n.type === 'CLIPTextEncode' && n.title !== 'Negative Prompt') ||
                    n.type === 'RadianceCinematicPromptEncoder' ||
                    n.type === 'FXTDCinematicPromptEncoder'
                );

                if (!clipNode) {
                    this._termLog('error', '[Batch] Could not find a primary CLIPTextEncode or Radiance Prompt node in graph!');
                    return;
                }
                const textWidget = clipNode.widgets.find(w => w.name === 'text' || w.name === 'base_prompt');
                if (!textWidget) {
                    this._termLog('error', '[Batch] Prompt node has no text or base_prompt widget.');
                    return;
                }

                scRunBtn.textContent = '⏳ RUNNING...';
                scRunBtn.style.opacity = '0.5';
                scRunBtn.disabled = true;

                // Find all potential Save nodes to hijack
                const saveNodes = nodes.filter(n =>
                    n.type.includes('SaveImage') ||
                    n.type.includes('RadianceSave') ||
                    n.type.includes('RadianceWrite') ||
                    n.type.includes('ImageSave')
                );

                if (saveNodes.length === 0) {
                    this._termLog('warn', '[Batch] No compatible Save nodes (Radiance/Standard) found in graph.');
                }

                let customPath = scPathInput.value.trim();
                let isAbsolute = customPath.match(/^[a-zA-Z]:[\\\/]/) || customPath.startsWith('/');
                let finalSubfolder = '';

                if (isAbsolute) {
                    finalSubfolder = customPath;
                } else {
                    finalSubfolder = customPath || 'radiance_automation/';
                    if (finalSubfolder.toLowerCase().startsWith('output/')) finalSubfolder = finalSubfolder.substring(7);
                    if (finalSubfolder.toLowerCase().startsWith('output\\')) finalSubfolder = finalSubfolder.substring(7);
                }

                saveNodes.forEach(node => {
                    this._termLog('info', `[Batch] Hijacking node: ${node.title || node.type} (${node.id})`);

                    const widgets = node.widgets || [];
                    const extWidget = widgets.find(w => w.name.toLowerCase().includes('extension') || w.name.toLowerCase().includes('format'));
                    const pfxWidget = widgets.find(w => w.name === 'filename_prefix' || w.name === 'filename');
                    const subWidget = widgets.find(w => w.name === 'subfolder' || w.name === 'output_path');
                    const depthWidget = widgets.find(w => w.name === 'bit_depth');

                    // 1. Force EXR Format
                    if (extWidget) {
                        if (node.type.includes('Radiance')) {
                            extWidget.value = 'EXR';
                            if (depthWidget) depthWidget.value = '32-bit Float';
                        } else {
                            extWidget.value = 'exr (32-bit)';
                            if (extWidget.options?.values?.includes('exr')) extWidget.value = 'exr';
                        }
                    }

                    // 2. Set Path
                    if (subWidget) {
                        subWidget.value = finalSubfolder;
                    } else if (isAbsolute && pfxWidget) {
                        // For vanilla SaveImage with absolute path: we have to put it in the prefix
                        let fullPfx = finalSubfolder;
                        if (!fullPfx.endsWith('/') && !fullPfx.endsWith('\\')) fullPfx += '/';
                        // Note: basePfx will be updated per-loop below
                    }
                });

                this._termLog('info', `[Batch] Target Path: ${isAbsolute ? '' : 'output/'}${finalSubfolder}`);

                this._termLog('system', `[Batch] Queuing ${lines.length} prompts (split by newline/period)`);
                for (let i = 0; i < lines.length; i++) {
                    const prompt = lines[i];
                    textWidget.value = `[${i + 1}/${lines.length}] ${prompt}`;

                    // Update prefixes for all save nodes
                    saveNodes.forEach(node => {
                        const widgets = node.widgets || [];
                        const pfxWidget = widgets.find(w => w.name === 'filename_prefix' || w.name === 'filename');
                        const subWidget = widgets.find(w => w.name === 'subfolder' || w.name === 'output_path');

                        let basePfx = 'batch_';
                        if (isAbsolute && !subWidget) {
                            basePfx = finalSubfolder;
                            if (!basePfx.endsWith('/') && !basePfx.endsWith('\\')) basePfx += '/';
                        }

                        if (pfxWidget) {
                            pfxWidget.value = `${basePfx}${String(i + 1).padStart(3, '0')}_`;
                        }
                    });

                    this._termLog('cmd', `[Batch Queue ${i + 1}/${lines.length}] > ${prompt.substring(0, 40)}...`);
                    await app.queuePrompt(0);
                    await new Promise(r => setTimeout(r, 100));
                }

                this._termLog('success', `[Batch] Successfully queued ${lines.length} images.`);
                scRunBtn.textContent = '▶ RUN AUTOMATION';
                scRunBtn.style.opacity = '1';
                scRunBtn.disabled = false;
            }
        };

        // ── Input row ────────────────────────────────────────────────────────
        const inputRow = document.createElement('div');
        inputRow.style.cssText = `
            flex: 0 0 26px;
            display: flex;
            align-items: center;
            border-top: 1px solid #111;
            background: #080b0f;
        `;

        const promptLabel = document.createElement('span');
        promptLabel.textContent = 'FXTD STUDIOS ❯';
        promptLabel.style.cssText = `color: #3d8; font-size: 10px; padding: 0 8px; flex-shrink: 0; letter-spacing: 0.05em;`;
        inputRow.appendChild(promptLabel);

        const input = document.createElement('textarea');
        input.placeholder = 'enter command (try: help, status, lut list, grade reset)';
        input.rows = 1;
        input.style.cssText = `
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: #c8d3e0;
            font-family: ${t.mono};
            font-size: 10.5px;
            caret-color: #3d8;
            padding: 6px 6px 6px 0;
            resize: none;
            overflow-y: hidden;
            min-height: 26px;
            max-height: 200px;
        `;

        // Auto-resize textarea
        input.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        inputRow.appendChild(input);
        termContainer.appendChild(inputRow);
        this._termInput = input;
        this._termInputRow = inputRow;

        // ── Append to container ──────────────────────────────────────────────
        this.container.appendChild(resizeHandle);
        this.container.appendChild(termContainer);

        // ── Resize drag ──────────────────────────────────────────────────────
        let startY = 0, startH = 0;
        const startDrag = (e) => {
            this._termResizing = true;
            startY = e.clientY;
            startH = termContainer.getBoundingClientRect().height;
            document.addEventListener('mousemove', onDrag);
            document.addEventListener('mouseup', endDrag);
        };
        const onDrag = (e) => {
            const delta = startY - e.clientY;
            const newH = Math.max(80, Math.min(600, startH + delta));
            termContainer.style.flex = `0 0 ${newH}px`;
            localStorage.setItem('radiance_term_height', newH);
        };
        const endDrag = () => {
            this._termResizing = false;
            resizeHandle.style.background = 'transparent';
            document.removeEventListener('mousemove', onDrag);
            document.removeEventListener('mouseup', endDrag);
        };
        resizeHandle.addEventListener('mousedown', startDrag);
        header.addEventListener('mousedown', (e) => {
            // Allow dragging from header background or tabs container (excluding buttons if any)
            if (e.target === header || tabsContainer.contains(e.target) || badge.contains(e.target)) {
                startDrag(e);
            }
        });

        // ── Command history & Tab Autocomplete ───────────────────────────────
        this._termHistory = [];
        this._termHistoryIdx = -1;
        this._tabCompleteMatches = [];
        this._tabCompleteIdx = -1;
        this._tabCompletePrefix = '';
        this._termIsRecording = false;
        this._termMacroBuffer = [];

        // IMPORTANT: Use capture phase + stopImmediatePropagation to prevent
        // ComfyUI's global keyboard handlers from intercepting Enter/arrows/etc.
        input.addEventListener('keydown', (e) => {
            e.stopPropagation();
            e.stopImmediatePropagation();

            if (e.key === 'Tab') {
                e.preventDefault(); // keep focus in input
                if (!input.value) return;

                // If continuing a tab session
                if (this._tabCompleteMatches.length > 0 && input.value.startsWith(this._tabCompletePrefix)) {
                    this._tabCompleteIdx = (this._tabCompleteIdx + 1) % this._tabCompleteMatches.length;
                    input.value = this._tabCompleteMatches[this._tabCompleteIdx] + ' ';
                    return;
                }

                // Start new tab session
                const prefix = input.value.split(/\s+/)[0].toLowerCase();
                this._tabCompletePrefix = prefix;

                const builtins = [
                    'help', 'status', 'export', 'frame', 'compare', 'ab', 'metadata',
                    'alias', 'grade', 'lut', 'exposure', 'sat', 'gamma', 'zoom',
                    'channel', 'scope', 'printer', 'softclip', 'wb', 'info',
                    'run', 'queue', 'clear', 'eval', 'bypass', 'tracker', 'shotgrid', 'ftrack'
                ];

                let aliases = [];
                try { aliases = Object.keys(JSON.parse(localStorage.getItem('radiance_term_aliases') || '{}')); } catch { }

                const allCmds = [...new Set([...builtins, ...aliases])].sort();
                this._tabCompleteMatches = allCmds.filter(c => c.startsWith(prefix));

                if (this._tabCompleteMatches.length === 1) {
                    input.value = this._tabCompleteMatches[0] + ' ';
                    this._tabCompleteMatches = [];
                } else if (this._tabCompleteMatches.length > 1) {
                    this._tabCompleteIdx = 0;
                    input.value = this._tabCompleteMatches[0] + ' ';
                    // Show options if more than one
                    this._termLog('info', `  ${this._tabCompleteMatches.join('  ')}`);
                }
            } else if (e.key === 'ArrowUp') {
                this._tabCompleteMatches = []; // reset autocomplete
                this._termHistoryIdx = Math.min(this._termHistoryIdx + 1, this._termHistory.length - 1);
                input.value = this._termHistory[this._termHistoryIdx] || '';
                e.preventDefault();
            } else if (e.key === 'ArrowDown') {
                this._tabCompleteMatches = [];
                this._termHistoryIdx = Math.max(this._termHistoryIdx - 1, -1);
                input.value = this._termHistoryIdx >= 0 ? this._termHistory[this._termHistoryIdx] : '';
                e.preventDefault();
            } else if (e.key === 'Enter') {
                if (e.shiftKey) {
                    // Let default shift+enter behavior happen (newline)
                    // The auto-resize input listener will handle the height jump
                    return;
                }

                e.preventDefault();
                this._tabCompleteMatches = [];
                const cmd = input.value.trim();
                if (!cmd) return;

                // For history, replace literal newlines with \n strings to keep it one line in history
                const histCmd = cmd.replace(/\n/g, '\\n');
                this._termHistory.unshift(histCmd);
                this._termHistoryIdx = -1;

                input.value = '';
                input.style.height = 'auto'; // reset height

                // Log multi-line commands cleanly
                const displayCmd = cmd.split('\n').map((line, i) => i === 0 ? `❯ ${line}` : `  ${line}`).join('\n');
                this._termLog('cmd', displayCmd);

                this._termExec(cmd);
            } else if (e.key === 'Escape') {
                this._tabCompleteMatches = [];
                input.value = '';
                input.blur();
            } else {
                // Any other typing resets autocomplete
                this._tabCompleteMatches = [];
            }
        }, true);


        // ── Boot message ─────────────────────────────────────────────────────
        this._termLog('system', '=============================================');
        this._termLog('system', '  FXTD STUDIOS RADIANCE TERMINAL · v2.3.3');
        this._termLog('system', '  Type "help" for available commands');
        this._termLog('system', '=============================================');

        // ── Run Startup Script ────────────────────────────────────────────────
        setTimeout(() => {
            try {
                const aliases = JSON.parse(localStorage.getItem('radiance_term_aliases') || '{}');
                if (aliases['startup']) {
                    this._termLog('system', 'Executing startup macro...');
                    this._termExec('startup');
                }
            } catch (e) { }
        }, 500);
    }

    // ── Log a line to the terminal ───────────────────────────────────────────
    _termLog(type, msg) {
        if (!this._termOutput) return;
        const line = document.createElement('div');

        const colors = {
            system: '#3d6060',
            info: '#00a8ff',
            warn: '#ff9f43',
            error: '#ff4757',
            cmd: '#2ed573',
            success: '#7bed9f',
            grade: '#a29bfe',
            event: '#eccc68',
            result: '#70a1ff',
        };
        const c = colors[type] || '#c8d3e0';

        line.style.cssText = `
            padding: 2px 8px;
            margin: 1px 0;
            white-space: pre-wrap;
            border-left: 2px solid ${type === 'cmd' ? c : 'transparent'};
            background: ${type === 'cmd' ? 'rgba(46, 213, 115, 0.05)' : 'transparent'};
            display: flex;
            align-items: flex-start;
            gap: 8px;
            font-size: 10.5px;
            transition: background 0.2s;
        `;

        const ts = new Date().toTimeString().slice(0, 8);
        const tsSpan = `<span style="color:rgba(255,255,255,0.15);user-select:none;font-size:9px;min-width:45px;">${ts}</span>`;

        // Sanitize backend-supplied messages against XSS
        const escaped = RadianceViewer.escapeHtml(msg);
        const highlighted = escaped.replace(/(\[[\w\s·\.]+\])/g, `<span style="color:${c};font-weight:600">$1</span>`);

        line.innerHTML = tsSpan + `<span style="color:${c};flex:1">${highlighted}</span>`;

        // Add indicator for errors/warnings
        if (type === 'error' || type === 'warn') {
            line.style.borderLeftColor = c;
            line.style.background = `linear-gradient(90deg, ${c}11 0%, transparent 100%)`;
        }

        this._termOutput.appendChild(line);

        // Keep max 500 lines
        while (this._termOutput.children.length > 500) {
            this._termOutput.removeChild(this._termOutput.firstChild);
        }
        this._termOutput.scrollTop = this._termOutput.scrollHeight;
    }

    // ── Command executor ─────────────────────────────────────────────────────
    _termExec(cmd) {
        // Alias expansion
        if (!this._termAliases) {
            try { this._termAliases = JSON.parse(localStorage.getItem('radiance_term_aliases') || '{}'); }
            catch { this._termAliases = {}; }
        }

        const rawParts = cmd.trim().split(/\s+/);
        let firstWord = rawParts[0].toLowerCase();

        // Expand alias if it exists and we're not defining/deleting one
        if (firstWord !== 'alias' && this._termAliases[firstWord]) {
            const expanded = this._termAliases[firstWord] + ' ' + rawParts.slice(1).join(' ');
            cmd = expanded.trim();
        }

        const parts = cmd.trim().split(/\s+/);
        const verb = parts[0].toLowerCase();
        const args = parts.slice(1);

        // Macro Recorder Intercept
        if (this._termIsRecording && verb !== 'record') {
            this._termMacroBuffer.push(cmd);
        }

        switch (verb) {
            case 'help': {
                const cmds = [
                    ['help', 'Show this help'],
                    ['<python>', 'Arbitrary python commands sent to backend!'],
                    ['status', 'Show viewer + renderer state'],
                    ['export [name]', 'Export current frame as PNG'],
                    ['export exr32 [name]', 'Export graded frame as 32-bit EXR'],
                    ['frame <next|prev|#>', 'Navigate image sequence'],
                    ['compare <save|toggle>', 'A/B split between grades (or: ab)'],
                    ['metadata', 'Show parsed EXR/image metadata'],
                    ['bypass <id|title>', 'Toggle node bypass state'],
                    ['tracker <sync|status>', 'Push to ShotGrid/ftrack (stub)'],
                    ['alias <nm> "<cmd>"', 'Create shortcut (e.g. alias sq "frame next")'],
                    ['grade', 'Show current grade values'],
                    ['grade reset', 'Reset all grading to defaults'],
                    ['grade save <name>', 'Save current grade to slot'],
                    ['lut list', 'List available display LUTs'],
                    ['lut <name>', 'Set display LUT (e.g. lut sRGB)'],
                    ['exposure <val>', 'Set exposure (e.g. exposure 0.5)'],
                    ['sat <val>', 'Set saturation (e.g. sat 1.2)'],
                    ['gamma <val>', 'Set master gamma (e.g. gamma 1.1)'],
                    ['zoom <pct>', 'Set zoom (e.g. zoom 100)'],
                    ['zoom fit', 'Fit image to view'],
                    ['channel <r|g|b|rgb|luma>', 'Isolate channel'],
                    ['scope <mode>', 'Set scope: parade|waveform|histogram|vector|falsecolor'],
                    ['scope log', 'Toggle log view on scopes'],
                    ['printer r|g|b <v>', 'Set printer light offset (-50..+50)'],
                    ['softclip <v>', 'Set soft clip (0–1)'],
                    ['wb pick', 'Activate white balance eyedropper'],
                    ['wb reset', 'Reset temperature and tint to 0'],
                    ['info', 'Show image info'],
                    ['run', 'Queue ComfyUI workflow'],
                    ['queue', 'Show ComfyUI queue status'],
                    ['clear', 'Clear terminal'],
                    ['eval <js>', 'Evaluate JS (advanced)'],
                    ['record <start|stop name>', 'Record a series of commands to an alias'],
                    ['ls <nodes|selected>', 'List graph nodes (e.g. ls nodes)'],
                    ['find <type|title>', 'Find and highlight a node'],
                    ['doc <python_module>', 'Fetch python documentation'],
                ];
                cmds.forEach(([c, d]) => this._termLog('result', `  ${c.padEnd(26)} ${d}`));
                break;
            }

            case 'record': {
                if (args[0] === 'start') {
                    if (this._termIsRecording) {
                        this._termLog('warn', '[Record] Already recording! Type `record stop [name]` to save.');
                        break;
                    }
                    this._termIsRecording = true;
                    this._termMacroBuffer = [];
                    this._termLog('event', '[Record] 🔴 Recording started. Type commands, then `record stop <name>`');
                } else if (args[0] === 'stop') {
                    if (!this._termIsRecording) {
                        this._termLog('warn', '[Record] Not currently recording.');
                        break;
                    }
                    this._termIsRecording = false;
                    const name = args[1];
                    if (!name) {
                        this._termLog('warn', '[Record] Stopped. Discarded macro (no name provided).');
                        this._termMacroBuffer = [];
                        break;
                    }
                    if (this._termMacroBuffer.length === 0) {
                        this._termLog('warn', `[Record] Stopped. Discarded macro (empty).`);
                        break;
                    }
                    // Compile commands with semicolons
                    const compiled = this._termMacroBuffer.join('; ');
                    this._termAliases[name] = compiled;
                    localStorage.setItem('radiance_term_aliases', JSON.stringify(this._termAliases));
                    this._termLog('success', `[Record] ⏹ Stopped. Macro saved as alias: ${name}`);
                    this._termMacroBuffer = [];
                } else {
                    this._termLog('warn', 'usage: record start | record stop <name>');
                }
                break;
            }

            case 'ls': {
                if (!app.graph || !app.graph._nodes) {
                    this._termLog('error', '[LS] Graph not available.');
                    break;
                }
                if (args[0] === 'nodes') {
                    const count = app.graph._nodes.length;
                    this._termLog('info', `[LS] Graph contains ${count} nodes:`);
                    app.graph._nodes.forEach(n => {
                        this._termLog('result', `  [${n.id}] ${n.type} ${n.title ? `"${n.title}"` : ''}`);
                    });
                } else if (args[0] === 'selected') {
                    const sel = Object.values(app.canvas.selected_nodes || {});
                    if (!sel.length) {
                        this._termLog('result', '[LS] No nodes selected.');
                        break;
                    }
                    this._termLog('info', `[LS] ${sel.length} nodes selected:`);
                    sel.forEach(n => {
                        this._termLog('result', `  [${n.id}] ${n.type} ${n.title ? `"${n.title}"` : ''}`);
                    });
                } else {
                    this._termLog('warn', 'usage: ls nodes | ls selected');
                }
                break;
            }

            case 'find': {
                if (!args[0]) {
                    this._termLog('warn', 'usage: find <type or title string>');
                    break;
                }
                const query = args.join(' ').toLowerCase();
                const nodes = app.graph._nodes || [];
                const matches = nodes.filter(n =>
                    String(n.id) === query ||
                    (n.title && n.title.toLowerCase().includes(query)) ||
                    (n.type && n.type.toLowerCase().includes(query))
                );

                if (!matches.length) {
                    this._termLog('warn', `[Find] No nodes found matching "${query}"`);
                    break;
                }

                app.canvas.deselectAllNodes();
                matches.forEach(m => app.canvas.selectNode(m, true));
                app.canvas.centerOnNode(matches[0]);
                app.canvas.setDirty(true, true);

                this._termLog('success', `[Find] Found and selected ${matches.length} node(s).`);
                break;
            }

            case 'doc': {
                if (!args[0]) {
                    this._termLog('warn', 'usage: doc <python_module_or_object>');
                    break;
                }
                const target = args[0];
                const pyCmd = `import inspect; print(f"\\n--- Docstring for {target} ---\\n{inspect.getdoc(${target}) or 'No docstring available.'}\\n----------------------------")`;
                this._termLog('event', `[Doc] Fetching Python docstring for ${target}...`);
                this._termExec(pyCmd);
                break;
            }

            case 'gpu': {
                this._termLog('event', '[GPU] Querying hardware telemetry...');
                const pyCmd = `
import torch
from radiance.gpu_memory import get_gpu_memory_info
info = get_gpu_memory_info()
if info.get('available'):
    print(f"--- GPU DIAGNOSTICS ---\\nDevice   : {torch.cuda.get_device_name(0)}\\nAllocated: {info['allocated_mb']:.1f} MB\\nReserved : {info['reserved_mb']:.1f} MB\\nPeak     : {info['max_allocated_mb']:.1f} MB\\n----------------------")
else:
    print("GPU NOT AVAILABLE (CPU MODE)")
                `;
                this._termExec(pyCmd);
                break;
            }

            case 'inspect': {
                const target = args[0] || 'view';
                if (target === 'view') {
                    const stats = [
                        ['RESOLUTION', `${this.imageWidth} × ${this.imageHeight}`],
                        ['CHANNELS', this.channel === 'rgb' ? '3 (RGB)' : '4 (RGBA)'],
                        ['PIXEL DATA', this.imageData ? 'ALLOCATED' : 'NULL'],
                        ['GPU TEXTURE', this.useWebGL ? 'ACTIVE' : 'FALLBACK'],
                        ['MEMORY', `${((this.imageWidth * this.imageHeight * 4 * 4) / 1024 / 1024).toFixed(2)} MB (Est. 32-bit)`]
                    ];

                    this._termLog('system', '--- TENSOR INSPECTION [IMAGE] ---');
                    stats.forEach(([k, v]) => {
                        this._termLog('result', `${k.padEnd(12)} : ${v}`);
                    });
                    this._termLog('system', '-------------------------------');
                } else {
                    this._termLog('event', `[Inspect] Relaying to Python backend for ${target}...`);
                    this._termExec(`import torch; print(f"--- Tensor Info: {${target}} ---\\nType: {type(${target})}\\nShape: {${target}.shape if hasattr(${target}, 'shape') else 'N/A'}\\nDevice: {${target}.device if hasattr(${target}, 'device') else 'N/A'}\\nDtype: {${target}.dtype if hasattr(${target}, 'dtype') else 'N/A'}\\n-------------------")`);
                }
                break;
            }

            case 'status': {
                const r = this.renderer;
                this._termLog('result', `Viewer  : ${this.imageWidth || '—'}×${this.imageHeight || '—'}  zoom:${((this.zoom || 1) * 100).toFixed(0)}%`);
                this._termLog('result', `WebGL   : ${this.useWebGL ? 'ON' : 'OFF'}  Channel: ${this.channel || 'rgb'}`);
                this._termLog('result', `Display : LUT=${this.displayLut || 'None'}  CS=${this.colorScience === 1 ? 'ACEScct' : 'Linear'}`);
                this._termLog('result', `Grain   : ${(this.grain || 0).toFixed(2)}  Denoise: ${(this.denoise || 0).toFixed(2)}`);
                this._termLog('result', `Bloom   : ${(this.bloom || 0).toFixed(2)}  Halation: ${(this.halation || 0).toFixed(2)}`);
                break;
            }

            case 'grade': {
                if (args[0] === 'reset') {
                    // Replicate the same reset logic as RESET ALL button
                    this._pushUndo();
                    this.exposure = 0.0; this.lift = [0, 0, 0]; this.gamma = [1, 1, 1]; this.gain = [1, 1, 1];
                    this.temperature = 0.0; this.tint = 0.0; this.contrast = 1.0; this.pivot = 0.5; this.saturation = 1.0;
                    this.grain = 0.0; this.denoise = 0.0;
                    this.printerR = 0; this.printerG = 0; this.printerB = 0; this.softClip = 0.0;
                    this.bloom = 0.0; this.bloomThreshold = 1.0; this.halation = 0.0; this.halationRadius = 1.0; this.halationThreshold = 0.35; this.diffusion = 0.0;
                    this.grainSize = 1.0; this.grainColor = 0.0; this.grainAnimate = false;
                    this.bokehHighlightBias = 0.0; this.bokehSoapBubble = 0.0; this.bokehOpticalVig = 0.0;
                    this.apertureBlades = 0; this.apertureRotation = 0.0; this.apertureAnamorphic = 1.0;
                    this.anamorphicStreaks = 0.0; this.streakThreshold = 0.85; this.streakLength = 0.08; this.lensDistortion = 0.0; this.lensFringe = 0.0;
                    this.vignetteIntensity = 0.0; this.vignetteFalloff = 0.5;
                    if (this.curveEditor) this.curveEditor.resetAllChannels?.();
                    if (this.renderer) {
                        this.renderer.setExposure(0); this.renderer.setLift(0, 0, 0); this.renderer.setGamma(1, 1, 1); this.renderer.setGain(1, 1, 1);
                        this.renderer.setTemperature(0); this.renderer.setTint(0); this.renderer.setContrast(1); this.renderer.setPivot(0.5);
                        this.renderer.setSaturation(1); this.renderer.setGrain(0); this.renderer.setGrainSize(1.0); this.renderer.setGrainColor(0.0); this.renderer.setGrainAnimate(false);
                        this.renderer.setDenoise(0); this.renderer.setBloom(0); this.renderer.setHalation(0); this.renderer.setDiffusion(0);
                        this.renderer.setLensDistortion(0, 0); this.renderer.setVignette(0, 0.5);
                        this.renderer.setBokehPhysics(0, 0, 0); this.renderer.setApertureShape(0, 0, 1.0);
                        if (this.renderer.setAnamorphicStreaks) this.renderer.setAnamorphicStreaks(0);
                        this.renderer.setPrinterLights(0, 0, 0); this.renderer.setSoftClip(0);
                    }
                    this.render();
                    if (this._lastRenderContent) this._lastRenderContent();
                    this._termLog('success', '[Grade] Reset to defaults.');
                } else if (args[0] === 'save') {
                    const name = args[1] || `grade_${Date.now()}`;
                    const state = this._captureGradingState();
                    if (!this._savedGrades) this._savedGrades = {};
                    this._savedGrades[name] = state;
                    this._termLog('success', `[Grade] Saved as "${name}"`);
                } else if (args[0] === 'load') {
                    const name = args[1];
                    if (this._savedGrades && this._savedGrades[name]) {
                        this._restoreGradingState(this._savedGrades[name]);
                        if (this._lastRenderContent) this._lastRenderContent();
                        this._termLog('success', `[Grade] Loaded "${name}"`);
                    } else {
                        this._termLog('warn', `[Grade] No saved grade named "${name}"`);
                    }
                } else {
                    const g = this._captureGradingState();
                    this._termLog('grade', `  Exposure   : ${(g.exposure || 0).toFixed(3)}`);
                    this._termLog('grade', `  Temp/Tint  : ${(g.temperature || 0).toFixed(3)} / ${(g.tint || 0).toFixed(3)}`);
                    this._termLog('grade', `  Contrast   : ${(g.contrast || 1).toFixed(3)}  Pivot: ${(g.pivot || 0.5).toFixed(3)}`);
                    this._termLog('grade', `  Saturation : ${(g.saturation || 1).toFixed(3)}`);
                    this._termLog('grade', `  Lift       : ${(g.lift || [0, 0, 0]).map(v => v.toFixed(3)).join('  ')}`);
                    this._termLog('grade', `  Gamma      : ${(g.gamma || [1, 1, 1]).map(v => v.toFixed(3)).join('  ')}`);
                    this._termLog('grade', `  Gain       : ${(g.gain || [1, 1, 1]).map(v => v.toFixed(3)).join('  ')}`);
                    this._termLog('grade', `  Printer    : R${g.printerR || 0}  G${g.printerG || 0}  B${g.printerB || 0}`);
                    this._termLog('grade', `  SoftClip   : ${(g.softClip || 0).toFixed(3)}`);
                }
                break;
            }

            case 'lut': {
                if (args[0] === 'list') {
                    (this.lutOptions || []).forEach(l => {
                        const active = l === this.displayLut ? ' ◀ active' : '';
                        this._termLog('result', `  ${l}${active}`);
                    });
                } else {
                    const name = args.join(' ');
                    const found = (this.lutOptions || []).find(l => l.toLowerCase().includes(name.toLowerCase()));
                    if (found) {
                        this.displayLut = found;
                        if (this.renderer) this.renderer.setDisplayLutMode(({
                            'None': 0, 'sRGB (Display)': 1, 'Rec.709 (Broadcast)': 2, 'Filmic (Cinematic)': 3,
                            'Log C3 (ARRI)': 4, 'Log C4 (ARRI)': 11, 'ACES Filmic': 9, 'Reinhard Tonemap': 8
                        })[found] || 0);
                        this.render();
                        this._termLog('success', `[LUT] Set to "${found}"`);
                    } else {
                        this._termLog('warn', `[LUT] Not found: "${name}". Use  lut list  to see options.`);
                    }
                }
                break;
            }

            case 'exposure': {
                const v = parseFloat(args[0]);
                if (isNaN(v)) { this._termLog('warn', `usage: exposure <number>`); break; }
                this.exposure = v;
                if (this.renderer) this.renderer.setExposure(v);
                this.render();
                if (this._lastRenderContent) this._lastRenderContent();
                this._termLog('success', `[EXP] → ${v.toFixed(3)}`);
                break;
            }

            case 'sat': {
                const v = parseFloat(args[0]);
                if (isNaN(v)) { this._termLog('warn', `usage: sat <number>`); break; }
                this.saturation = v;
                if (this.renderer) this.renderer.setSaturation(v);
                this.render();
                if (this._lastRenderContent) this._lastRenderContent();
                this._termLog('success', `[SAT] → ${v.toFixed(3)}`);
                break;
            }

            case 'gamma': {
                const v = parseFloat(args[0]);
                if (isNaN(v) || v <= 0) { this._termLog('warn', `usage: gamma <positive number>`); break; }
                this.gamma = [v, v, v];
                if (this.renderer) this.renderer.setGamma(v, v, v);
                this.render();
                if (this._lastRenderContent) this._lastRenderContent();
                this._termLog('success', `[GAMMA] → ${v.toFixed(3)}`);
                break;
            }

            case 'zoom': {
                if (args[0] === 'fit') {
                    this.fitToView();
                    this._termLog('success', '[ZOOM] Fit to view');
                } else {
                    const v = parseFloat(args[0]);
                    if (isNaN(v)) { this._termLog('warn', `usage: zoom <percent>  or  zoom fit`); break; }
                    this.setZoom(v / 100);
                    this._termLog('success', `[ZOOM] → ${v.toFixed(0)}%`);
                }
                break;
            }

            case 'channel': {
                const ch = args[0]?.toLowerCase();
                const validChannels = { r: 'r', g: 'g', b: 'b', rgb: 'rgb', luma: 'luma', l: 'luma', a: 'a' };
                if (!validChannels[ch]) { this._termLog('warn', `usage: channel <r|g|b|rgb|luma>`); break; }
                this.channel = validChannels[ch];
                this.showZdepth = false; // channel switch always exits depth view
                this.render();
                this._termLog('success', `[CH] → ${this.channel}`);
                break;
            }

            case 'scope': {
                if (args[0] === 'log') {
                    this.scopeLogView = !this.scopeLogView;
                    localStorage.setItem('radiance_scope_log', this.scopeLogView ? '1' : '0');
                    if (this.activeTab === 'scopes' && this._lastRenderContent) this._lastRenderContent();
                    this._termLog('success', `[SCOPE] Log view: ${this.scopeLogView ? 'ON' : 'OFF'}`);
                } else {
                    const modeMap = { parade: 'parade', waveform: 'waveform', histogram: 'histogram', vector: 'vectorscope', vectorscope: 'vectorscope', falsecolor: 'falsecolor', 'false': 'falsecolor' };
                    const m = modeMap[args[0]?.toLowerCase()];
                    if (!m) { this._termLog('warn', `usage: scope <parade|waveform|histogram|vector|falsecolor>`); break; }
                    this.scopeMode = m;
                    localStorage.setItem('radiance_scope_mode', m);
                    if (this._lastRenderContent) this._lastRenderContent();
                    this._termLog('success', `[SCOPE] → ${m}`);
                }
                break;
            }

            case 'printer': {
                const ch = args[0]?.toLowerCase();
                const v = parseInt(args[1]);
                if (!['r', 'g', 'b'].includes(ch) || isNaN(v)) { this._termLog('warn', `usage: printer <r|g|b> <-50..+50>`); break; }
                const clamped = Math.max(-50, Math.min(50, v));
                if (ch === 'r') this.printerR = clamped;
                else if (ch === 'g') this.printerG = clamped;
                else this.printerB = clamped;
                if (this.renderer) this.renderer.setPrinterLights(this.printerR || 0, this.printerG || 0, this.printerB || 0);
                this.render();
                if (this._lastRenderContent) this._lastRenderContent();
                this._termLog('success', `[PRINTER] ${ch.toUpperCase()} → ${clamped}`);
                break;
            }

            case 'softclip': {
                const v = parseFloat(args[0]);
                if (isNaN(v)) { this._termLog('warn', `usage: softclip <0..1>`); break; }
                this.softClip = Math.max(0, Math.min(1, v));
                if (this.renderer) this.renderer.setSoftClip(this.softClip);
                this.render();
                if (this._lastRenderContent) this._lastRenderContent();
                this._termLog('success', `[SOFTCLIP] → ${this.softClip.toFixed(3)}`);
                break;
            }

            case 'wb': {
                if (args[0] === 'pick') {
                    this._toggleWBPicker(null);
                    this._termLog('success', '[WB] Click a neutral pixel on the canvas.');
                } else if (args[0] === 'reset') {
                    this.temperature = 0; this.tint = 0;
                    if (this.renderer) { this.renderer.setTemperature(0); this.renderer.setTint(0); }
                    this.render();
                    if (this._lastRenderContent) this._lastRenderContent();
                    this._termLog('success', '[WB] Reset to neutral.');
                } else {
                    // Show current WB values with Kelvin equivalent
                    const kelvin = Math.round(6500.0 + (this.temperature || 0) * 3500.0);
                    this._termLog('result',
                        `[WB] Temperature: ${(this.temperature || 0).toFixed(3)} internal  ≈ ${kelvin}K  |  Tint: ${(this.tint || 0).toFixed(3)}`);
                    this._termLog('info', 'Use: wb pick  |  wb reset');
                }
                break;
            }

            case 'info': {
                if (!this.image) { this._termLog('warn', '[Info] No image loaded.'); break; }
                this._termLog('result', `  Dimensions  : ${this.imageWidth}×${this.imageHeight}`);
                this._termLog('result', `  Format      : ${this.hdrData ? (
                    this.hdrData.format === 'rhdr' ? 'RHDR fp16' :
                        this.hdrData.format === 'rhdr_f32' ? 'RHDR fp32' :
                            this.hdrData.format === 'exr' ? 'OpenEXR' :
                                this.hdrData.format === 'rgbe' ? 'Radiance RGBE (.hdr)' :
                                    this.hdrData.format === 'tiff_f32' ? 'TIFF float32' :
                                        this.hdrData.format === 'tiff_u16' ? 'TIFF 16-bit' :
                                            this.hdrData.format === 'tiff_u8' ? 'TIFF 8-bit' :
                                                this.hdrData.format === 'rf32' ? 'RF32 float32' :
                                                    'Float32 HDR'
                ) : 'PNG 8-bit'}`);
                this._termLog('result', `  Frame       : ${(this.currentFrame || 0) + 1} / ${(this.frameImages || []).length || 1}`);
                this._termLog('result', `  Zoom        : ${((this.zoom || 1) * 100).toFixed(0)}%`);
                this._termLog('result', `  WebGL       : ${this.useWebGL ? '✓ Active' : '✗ Fallback (2D)'}`);
                break;
            }

            case 'bypass': {
                if (!args[0]) { this._termLog('warn', 'usage: bypass <node_id | node_title>'); break; }
                const query = args.join(' ').toLowerCase();
                const nodes = app.graph._nodes || [];
                const target = nodes.find(n => String(n.id) === query || (n.title && n.title.toLowerCase() === query) || (n.type && n.type.toLowerCase() === query));
                if (!target) {
                    this._termLog('warn', `[Bypass] Node not found: "${query}"`);
                    break;
                }
                const NEVER = 2; // LiteGraph.NEVER
                const ALWAYS = 0; // LiteGraph.ALWAYS

                if (target.mode === NEVER) {
                    target.mode = ALWAYS;
                    this._termLog('success', `[Bypass] Unmuted node: ${target.title || target.type} (id:${target.id})`);
                } else {
                    target.mode = NEVER;
                    this._termLog('success', `[Bypass] Muted node: ${target.title || target.type} (id:${target.id})`);
                }
                app.graph.setDirtyCanvas(true, true);
                break;
            }

            case 'shotgrid':
            case 'ftrack':
            case 'tracker': {
                if (args[0] === 'sync') {
                    if (!this.image) { this._termLog('warn', '[Tracker] No image to sync.'); break; }
                    this._termLog('info', `[Tracker] Assembling payload...`);
                    const meta = this.hdrData?.metadata || {};
                    const shot = meta['shot'] || meta['Shot'] || 'sc01_sh010';
                    const task = meta['task'] || 'comp_v01';

                    this._termLog('info', `[Tracker] Target Shot : ${shot}`);
                    this._termLog('info', `[Tracker] Task Name   : ${task}`);
                    this._termLog('event', `[Tracker] Uploading artifact preview... [STUB]`);

                    // Stub timeout for fake network delay
                    setTimeout(() => {
                        this._termLog('success', `[Tracker] ✅ Sync complete. Version published to pipeline.`);
                    }, 600);
                } else if (args[0] === 'status') {
                    this._termLog('result', '[Tracker] Status: connected (Stub Mode)');
                    this._termLog('result', '[Tracker] User  : fxtd_radiance');
                    this._termLog('result', '[Tracker] URL   : https://studio.shotgrid.local');
                } else {
                    this._termLog('warn', `usage: tracker <sync|status>`);
                }
                break;
            }

            case 'run': {
                this._termLog('event', '[Workflow] Queuing prompt…');
                this.runWorkflow();
                break;
            }

            case 'queue': {
                api.fetchApi('/queue').then(r => r.json()).then(data => {
                    const running = (data.queue_running || []).length;
                    const pending = (data.queue_pending || []).length;
                    this._termLog('result', `[Queue] Running: ${running}  Pending: ${pending}`);
                }).catch(e => this._termLog('error', `[Queue] Fetch failed: ${e.message}`));
                break;
            }

            case 'clear': {
                this._termOutput.innerHTML = '';
                this._termLog('system', 'Terminal cleared.');
                break;
            }

            case 'eval': {
                const code = args.join(' ');
                try {
                    // eslint-disable-next-line no-new-func
                    const result = new Function('viewer', 'app', 'api', `return (${code})`)(this, app, api);
                    if (result !== undefined) {
                        const str = typeof result === 'object' ? JSON.stringify(result, null, 2) : String(result);
                        this._termLog('result', `  ← ${str}`);
                    } else {
                        this._termLog('result', '  ← (undefined)');
                    }
                } catch (e) {
                    this._termLog('error', `  ✗ ${e.message}`);
                }
                break;
            }

            case 'alias': {
                if (!args[0]) {
                    const keys = Object.keys(this._termAliases || {});
                    if (!keys.length) { this._termLog('result', '  No aliases defined.'); break; }
                    keys.forEach(k => this._termLog('result', `  alias ${k} = "${this._termAliases[k]}"`));
                } else if (args[0] === 'delete' || args[0] === 'remove') {
                    if (args[1]) {
                        delete this._termAliases[args[1]];
                        localStorage.setItem('radiance_term_aliases', JSON.stringify(this._termAliases));
                        this._termLog('success', `[Alias] Removed ${args[1]}`);
                    }
                } else {
                    const name = args[0];
                    // Re-join args intelligently to handle quotes
                    const cmdStr = cmd.substring(cmd.indexOf(name) + name.length).trim().replace(/^['"]|['"]$/g, '');
                    if (!cmdStr) { this._termLog('warn', `usage: alias <name> "<command>"`); break; }
                    this._termAliases[name] = cmdStr;
                    localStorage.setItem('radiance_term_aliases', JSON.stringify(this._termAliases));
                    this._termLog('success', `[Alias] ${name} → ${cmdStr}`);

                    if (name === 'startup') {
                        this._termLog('info', '[Alias] ' + name + ' command will now execute automatically on boot.');
                    }
                }
                break;
            }

            case 'export': {
                if (!this.image) { this._termLog('warn', '[Export] No image loaded.'); break; }

                // v4.0: `export exr32 [name]` — 32-bit graded EXR
                if (args[0] === 'exr32') {
                    this.exportSnapshot('exr32');
                    break;
                }

                const canvas = this.useWebGL && this.renderer ? this.renderer.canvas : this.canvas2d;
                if (!canvas) { this._termLog('warn', '[Export] Canvas not available.'); break; }

                const fn = args[0] || `radiance_export_${Date.now()}`;
                const dataURL = canvas.toDataURL('image/png');
                const link = document.createElement('a');
                link.download = fn.endsWith('.png') ? fn : `${fn}.png`;
                link.href = dataURL;
                link.click();
                this._termLog('success', `[Export] Saved locally as ${link.download}`);
                break;
            }

            case 'frame': {
                if (!this.frameImages || this.frameImages.length < 2) {
                    this._termLog('warn', '[Frame] No image sequence loaded.');
                    break;
                }
                const max = this.frameImages.length - 1;
                let target = this.currentFrame;
                if (args[0] === 'next') target++;
                else if (args[0] === 'prev') target--;
                else target = parseInt(args[0]);

                if (isNaN(target)) { this._termLog('warn', `usage: frame <next|prev|#number>`); break; }

                target = Math.max(0, Math.min(max, target));
                if (target !== this.currentFrame) {
                    this.currentFrame = target;
                    this.render(true);
                    this._termLog('success', `[Frame] → ${target + 1} / ${max + 1}`);
                } else {
                    this._termLog('result', `[Frame] Already at ${target + 1}`);
                }
                break;
            }

            case 'metadata': {
                if (!this.image) { this._termLog('warn', '[Metadata] No image loaded.'); break; }
                const meta = this.hdrData?.metadata;
                if (!meta || Object.keys(meta).length === 0) {
                    this._termLog('result', '  No EXR metadata found.');
                    break;
                }
                this._termLog('result', '  EXR Metadata:');
                for (const [k, v] of Object.entries(meta)) {
                    this._termLog('result', `    ${k.padEnd(20)} : ${v}`);
                }
                break;
            }

            case 'compare':
            case 'ab': {
                if (args[0] === 'save') {
                    this._compareState = this._captureGradingState();
                    this._termLog('success', '[Compare] Grade state saved to memory.');
                } else {
                    // Default behavior is toggle if nothing specified
                    if (!this._compareState) {
                        this._termLog('warn', '[Compare] No state saved. Auto-saving original state first.');
                        this._compareState = this._captureGradingState();
                        break;
                    }
                    const current = this._captureGradingState();
                    this._restoreGradingState(this._compareState);
                    this._compareState = current; // Swap them
                    if (this._lastRenderContent) this._lastRenderContent();
                    this._termLog('success', '[Compare] Toggled grade state.');
                }
                break;
            }

            default: {
                this._termLog('event', '[Terminal] Executing Python on backend...');
                api.fetchApi('/radiance/terminal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: cmd })
                }).then(r => r.json()).then(data => {
                    if (data.status === 'success') {
                        if (data.output && data.output.trim() !== "") {
                            this._termLog('result', data.output.trim());
                        }
                    } else {
                        if (data.output) this._termLog('error', data.output.trim());
                        else this._termLog('error', 'Execution failed.');
                    }
                }).catch(e => {
                    this._termLog('error', `[Terminal API] Network error: ${e.message}`);
                });
                break;
            }
        }
    }

    toggleTerminal() {
        this._termCollapsed = !this._termCollapsed;
        if (this._termOutputEl) this._termOutputEl.style.display = this._termCollapsed ? 'none' : 'block';
        if (this._termInputRow) this._termInputRow.style.display = this._termCollapsed ? 'none' : 'flex';
        if (this._termCollapseBtn) this._termCollapseBtn.textContent = this._termCollapsed ? '▸' : '▾';
        if (this._termContainer) this._termContainer.style.flex = this._termCollapsed ? '0 0 22px' : `0 0 ${this._termSavedH}px`;

        if (!this._termCollapsed && this._termInput) {
            setTimeout(() => this._termInput.focus(), 50);
        }
    }

    // ── Wire ComfyUI events into the terminal (called from init) ────────────
    _termWireEvents() {
        // Internal state for this generation session
        const S = {
            startTime: 0,
            nodeStart: 0,
            stepStart: 0,
            lastStep: 0,
            promptId: null,
            nodeMap: {},   // id → title
            nodeOrder: [],   // execution order
            totalSteps: 0,
            doneSteps: 0,
            currentNode: null,
            stepTimes: [],   // ms per step (rolling avg)
            nodeStats: {},   // id → { start, steps, ms }
        };

        // Helper: resolve node title from app.graph
        const nodeTitle = (id) => {
            if (S.nodeMap[id]) return S.nodeMap[id];
            try {
                const n = app.graph._nodes.find(n => String(n.id) === String(id));
                if (n) { S.nodeMap[id] = n.title || n.type || `Node ${id}`; return S.nodeMap[id]; }
            } catch { }
            return `Node ${id}`;
        };

        // Helper: format ms duration
        const fmtMs = (ms) => ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(2)}s`;

        // Helper: ETA progress bar  ████████░░░░  50%
        const progressBar = (pct, w = 20) => {
            const filled = Math.round(pct * w / 100);
            return '█'.repeat(filled) + '░'.repeat(w - filled) + `  ${pct.toFixed(0)}%`;
        };

        // Helper: steps/sec
        const stepsPerSec = () => {
            if (S.stepTimes.length < 2) return null;
            const recent = S.stepTimes.slice(-8);
            const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
            return (1000 / avg).toFixed(2);
        };

        // ── execution_cached ──────────────────────────────────────────────
        api.addEventListener('execution_cached', ({ detail }) => {
            if (!detail?.nodes) return;
            detail.nodes.forEach(id => {
                this._termLog('info', `  ⚡ cached      ${nodeTitle(id)}`);
            });
        });

        // ── execution_start ───────────────────────────────────────────────
        api.addEventListener('execution_start', ({ detail }) => {
            S.startTime = performance.now();
            S.nodeStart = S.startTime;
            S.promptId = detail?.prompt_id || '—';
            S.nodeMap = {};
            S.nodeOrder = [];
            S.totalSteps = 0;
            S.doneSteps = 0;
            S.currentNode = null;
            S.stepTimes = [];
            S.nodeStats = {};

            this._termLog('system', '─'.repeat(52));
            this._termLog('event', `▶  GENERATION STARTED`);
            this._termLog('info', `   prompt_id  : ${S.promptId}`);
            try {
                const pending = app.graph._nodes?.length || '?';
                this._termLog('info', `   graph nodes: ${pending}`);
            } catch { }
            this._termLog('system', '─'.repeat(52));
        });

        // ── executing (node starts) ───────────────────────────────────────
        api.addEventListener('executing', ({ detail }) => {
            const id = detail;
            if (!id) return;
            const title = nodeTitle(id);
            S.currentNode = id;
            S.nodeStart = performance.now();
            S.lastStep = 0;

            if (!S.nodeStats[id]) S.nodeStats[id] = { start: S.nodeStart, steps: 0, ms: 0 };
            else S.nodeStats[id].start = S.nodeStart;

            S.nodeOrder.push(id);
            this._termLog('event', `  ▷ executing   ${title}  (id:${id})`);
        });

        // ── progress (diffusion steps) ────────────────────────────────────
        api.addEventListener('progress', ({ detail }) => {
            const { value, max, node, prompt_id } = detail;
            S.totalSteps = max;
            S.doneSteps = value;

            const now = performance.now();

            // Track per-step timing
            if (S.stepStart > 0) {
                S.stepTimes.push(now - S.stepStart);
                if (S.stepTimes.length > 20) S.stepTimes.shift();
            }
            S.stepStart = now;

            // Per-node step count
            const nid = node || S.currentNode;
            if (nid && S.nodeStats[nid]) S.nodeStats[nid].steps++;

            const pct = (value / max) * 100;
            const bar = progressBar(pct, 24);
            const sps = stepsPerSec();
            const elapsed = ((now - S.startTime) / 1000).toFixed(1);

            // ETA
            let etaStr = '';
            if (sps && value < max) {
                const remaining = (max - value) / parseFloat(sps);
                etaStr = `  ETA ${remaining < 60 ? remaining.toFixed(1) + 's' : Math.floor(remaining / 60) + 'm' + Math.floor(remaining % 60) + 's'}`;
            }

            const title = nid ? nodeTitle(nid) : '';
            const spsStr = sps ? `  ${sps} it/s` : '';

            // Log every step if ≤20 total, else every 5% + first + last
            const logInterval = max <= 20 ? 1 : Math.max(1, Math.floor(max / 20));
            if (value === 1 || value === max || value % logInterval === 0) {
                this._termLog('event',
                    `     ${bar}  step ${String(value).padStart(3)}/${max}` +
                    `${spsStr}${etaStr}  +${elapsed}s`
                );
            }

            // Update terminal header live badge
            if (this._termContainer) {
                const badge = this._termContainer.querySelector('span[data-gen-badge]');
                if (badge) badge.textContent = `${pct.toFixed(0)}%  ${spsStr}`;
            }
        });

        // ── execution_error ───────────────────────────────────────────────
        api.addEventListener('execution_error', ({ detail }) => {
            this._termLog('error', `❌ EXECUTION ERROR`);
            if (detail?.exception_type) this._termLog('error', `   Type : ${detail.exception_type}`);
            if (detail?.exception_message) this._termLog('error', `   Msg  : ${detail.exception_message}`);
            if (detail?.node_id) {
                const title = nodeTitle(detail.node_id);
                this._termLog('error', `   Node : ${title} (${detail.node_id})`);
            }
            this._termLog('system', '─'.repeat(52));
        });

        // ── execution_interrupted ─────────────────────────────────────────
        api.addEventListener('execution_interrupted', ({ detail }) => {
            this._termLog('warn', `◎  EXECUTION INTERRUPTED`);
            if (detail?.node_id) {
                const title = nodeTitle(detail.node_id);
                this._termLog('warn', `   Node: ${title}`);
            }
            this._termLog('system', '─'.repeat(52));
        });

        // ── executed (node finished) ──────────────────────────────────────
        api.addEventListener('executed', ({ detail }) => {
            if (!detail) return;
            const id = detail.node || detail.node_id;
            const title = id ? nodeTitle(id) : '?';
            const now = performance.now();

            let duration = '';
            if (id && S.nodeStats[id] && S.nodeStats[id].start) {
                const ms = now - S.nodeStats[id].start;
                S.nodeStats[id].ms = ms;
                duration = `  ${fmtMs(ms)}`;
            }

            // Detect output images
            const output = detail.output;
            if (output) {
                const imgCount = output.images?.length || 0;
                const vidCount = output.videos?.length || 0;
                const textCount = output.text?.length || 0;
                const latCount = output.latent?.length || 0;

                let outSummary = '';
                if (imgCount) outSummary += `  ◎  ${imgCount} image${imgCount > 1 ? 's' : ''}`;
                if (vidCount) outSummary += `  ◎  ${vidCount} video${vidCount > 1 ? 's' : ''}`;
                if (textCount) outSummary += `  ◎  text`;
                if (latCount) outSummary += `  ◎  latent`;

                if (outSummary) {
                    this._termLog('success', `  ✓ done        ${title}${duration}${outSummary}`);

                    // Show image filename/details
                    if (imgCount && output.images) {
                        output.images.forEach((img, i) => {
                            const fn = img.filename || img.name || `output_${i}`;
                            const sub = img.subfolder ? `${img.subfolder}/` : '';
                            this._termLog('result', `     image[${i}]   ${sub}${fn}  (${img.type || 'output'})`);
                        });
                    }
                } else {
                    this._termLog('success', `  ✓ done        ${title}${duration}`);
                }
            } else {
                this._termLog('success', `  ✓ done        ${title}${duration}`);
            }
        });

        // (Duplicate execution_error listener removed — see line 1644)

        // ── status (queue updates) ────────────────────────────────────────
        api.addEventListener('status', ({ detail }) => {
            const q = detail?.exec_info?.queue_remaining ?? null;
            if (q === 0 && S.startTime > 0) {
                const total = ((performance.now() - S.startTime) / 1000).toFixed(2);

                this._termLog('system', '─'.repeat(52));
                this._termLog('success', `✓  GENERATION COMPLETE  — ${total}s total`);

                // Per-node timing summary
                const done = Object.entries(S.nodeStats)
                    .filter(([, v]) => v.ms > 0)
                    .sort((a, b) => b[1].ms - a[1].ms)
                    .slice(0, 8);

                if (done.length) {
                    this._termLog('result', `   Node timing (slowest first):`);
                    done.forEach(([id, v]) => {
                        const name = nodeTitle(id).slice(0, 32).padEnd(32);
                        const bar = '▪'.repeat(Math.round(v.ms / done[0][1].ms * 12));
                        const steps = v.steps > 0 ? `  ${v.steps} steps` : '';
                        this._termLog('result', `   ${name}  ${fmtMs(v.ms).padStart(7)}  ${bar}${steps}`);
                    });
                }
                this._termLog('system', '─'.repeat(52));

                S.startTime = 0;  // reset so next run shows fresh
            } else if (q !== null && q > 0) {
                this._termLog('info', `   queue remaining: ${q}`);
            }
        });
    }

    createScopes() {
        // Histogram
        this.histogramCanvas = document.createElement('canvas');
        this.histogramCanvas.width = 256;
        this.histogramCanvas.height = 120; // Increased from 50px for better distribution visibility
        this.histogramCanvas.style.cssText = 'width: 100%; height: 120px; display: none;';
        this.histogramCtx = this.histogramCanvas.getContext('2d');
        this.histogramLabel = this.createLabel('Histogram', true);
        this.scopePanel.appendChild(this.histogramLabel);
        this.scopePanel.appendChild(this.histogramCanvas);

        // Waveform
        const waveformHeader = document.createElement('div');
        waveformHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-top: 5px;';

        this.waveformLabel = this.createLabel('Waveform', true);

        this.paradeToggle = document.createElement('div');
        this.paradeToggle.textContent = 'PARADE';
        this.paradeToggle.style.cssText = `
            font-size: 8px;
            padding: 1px 4px;
            border: 1px solid ${this.theme.accent};
            color: ${this.theme.accent};
            cursor: pointer;
            border-radius: 2px;
            display: none;
            opacity: ${this.waveformParadeMode ? 1 : 0.4};
        `;
        this.paradeToggle.onclick = () => {
            this.toggleParadeMode();
            this.paradeToggle.style.opacity = this.waveformParadeMode ? 1 : 0.4;
        };

        waveformHeader.appendChild(this.waveformLabel);
        waveformHeader.appendChild(this.paradeToggle);
        this.scopePanel.appendChild(waveformHeader);

        this.waveformCanvas = document.createElement('canvas');
        this.waveformCanvas.width = 256;
        this.waveformCanvas.height = 150; // Increased from 50px to broadcast standard
        this.waveformCanvas.style.cssText = 'width: 100%; height: 150px; display: none;';
        this.waveformCtx = this.waveformCanvas.getContext('2d');
        this.scopePanel.appendChild(this.waveformCanvas);

        // Vectorscope
        this.vectorscopeCanvas = document.createElement('canvas');
        this.vectorscopeCanvas.width = 256; // Increased from 140px
        this.vectorscopeCanvas.height = 256; // Increased from 140px
        this.vectorscopeCanvas.style.cssText = 'width: 100%; aspect-ratio: 1; display: none;';
        this.vectorscopeCtx = this.vectorscopeCanvas.getContext('2d');
        this.vectorscopeLabel = this.createLabel('Vectorscope', true);
        this.scopePanel.appendChild(this.vectorscopeLabel);
        this.scopePanel.appendChild(this.vectorscopeCanvas);
    }

    setupScopePanelResize() {
        let startX, startWidth;

        this.scopeResizeHandle.addEventListener('mousedown', (e) => {
            this.isResizingScopePanel = true;
            startX = e.clientX;
            startWidth = this.scopePanelWidth;
            this.scopeResizeHandle.style.background = this.theme.accent;
            e.preventDefault();
            e.stopPropagation();
        });

        document.addEventListener('mousemove', (e) => {
            if (!this.isResizingScopePanel) return;

            const delta = startX - e.clientX; // Reversed: dragging left makes wider
            const newWidth = Math.max(
                this.scopePanelMinWidth,
                Math.min(this.scopePanelMaxWidth, startWidth + delta)
            );

            this.scopePanelWidth = newWidth;
            this.scopePanel.style.flex = `0 0 ${newWidth}px`;
        });

        document.addEventListener('mouseup', () => {
            if (this.isResizingScopePanel) {
                this.isResizingScopePanel = false;
                this.scopeResizeHandle.style.background = 'transparent';
                // Save width to localStorage
                localStorage.setItem('radiance_scope_width', this.scopePanelWidth);
            }
        });
    }

    // v2.2: Full cleanup — prevents memory leaks on node deletion
    destroy() {
        // Remove global event listeners
        if (this._docMoveHandler) document.removeEventListener('mousemove', this._docMoveHandler);
        if (this._docUpHandler) document.removeEventListener('mouseup', this._docUpHandler);
        if (this._docKeyHandler) document.removeEventListener('keydown', this._docKeyHandler);
        if (this._winUpHandler) window.removeEventListener('mouseup', this._winUpHandler);
        if (this._hudResizeListener) window.removeEventListener('resize', this._hudResizeListener);
        // BUG FIX: _undoKeyListener was never removed, causing stale handler accumulation
        if (this._undoKeyListener) { document.removeEventListener('keydown', this._undoKeyListener); this._undoKeyListener = null; }

        // Video cleanup
        this.unloadVideo();
        if (this._transportSpaceHandler) {
            document.removeEventListener('keydown', this._transportSpaceHandler);
            this._transportSpaceHandler = null;
        }

        // BUG FIX: Was only checking document.body; now removes from any parent (e.g. rightControlPanel)
        if (this.controlsPanel && this.controlsPanel.parentNode) {
            this.controlsPanel.parentNode.removeChild(this.controlsPanel);
        }
        // Remove rightControlPanel from DOM
        if (this.rightControlPanel && this.rightControlPanel.parentNode) {
            this.rightControlPanel.parentNode.removeChild(this.rightControlPanel);
        }

        // Disconnect ResizeObserver
        if (this.resizeObserver) this.resizeObserver.disconnect();

        // Destroy WebGL renderer
        if (this.renderer) this.renderer.destroy();

        // Remove DOM
        if (this.container) this.container.innerHTML = '';

        console.log('[Radiance] Viewer destroyed');
    }

    createLabel(text, hidden = false) {
        const label = document.createElement('div');
        label.textContent = text;
        label.style.cssText = `
            color: ${this.theme.textDim};
            font-size: 11px;
            letter-spacing: 0.06em;
            font-family: ${this.theme.font};
            min-width: 50px;
            font-weight: 500;
            display: ${hidden ? 'none' : 'block'};
        `;
        return label;
    }

    // createToolbar moved to createMainLeftHUD logic

    createMainLeftHUD() {
        const t = this.theme;
        this.analysisHUD = document.createElement('div');
        this.analysisHUD.style.cssText = `
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 12px;
            width: 58px;
            z-index: 100;
            padding: 16px 8px;
            box-sizing: border-box;
            background: rgba(13, 13, 18, 0.85);
            backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6), inset 0 0 0 1px rgba(255, 255, 255, 0.05);
            pointer-events: auto;
            max-height: 90vh;
            align-items: center;
            overflow-y: auto;
            scrollbar-width: none;
        `;
        this.analysisHUD.style.msOverflowStyle = 'none';

        const hud = this.analysisHUD;

        // ── GROUP: File & System ──────────────────────────────────────────────
        this.addGrpLabel('FILE', hud);
        this.addButton('⛶', () => this.toggleFullscreen(), 'Fullscreen (F11)', hud);
        this.addButton('💾', (e) => this.showExportMenu(e), 'Export Frame', hud);
        this.addButton('↺', () => this.resetControls(), 'Reset All Grades', hud);
        this.addButton('?', () => this.toggleHelp(), 'Keyboard Shortcuts (?)', hud);
        this.addSep(hud);

        // ── GROUP: Grading ───────────────────────────────────────────────────
        this.addGrpLabel('GRADE', hud);
        this.controlsToggle = this.addButton('◎', () => this.toggleControls(), 'Toggle Grading Controls (H)', hud);
        this.controlsToggle.style.color = this.theme.accent;
        this.addSep(hud);

        // ── GROUP: View & LUT ────────────────────────────────────────────────
        this.addGrpLabel('VIEW', hud);
        this.addButton('Fit', () => this.fitToView(), 'Fit to view (F)', hud);
        this.addButton('1:1', () => this.setZoom(1.0), 'Actual pixels (1)', hud);

        // LUT Select (Vertical adapted)
        const lutWrap = document.createElement('div');
        lutWrap.style.cssText = 'display:flex; flex-direction:column; gap:2px; align-items:center;';
        const lutLbl = document.createElement('span');
        lutLbl.textContent = 'LUT';
        lutLbl.style.cssText = 'font-size:7px; color:rgba(255,255,255,0.2); font-weight:bold;';
        lutWrap.appendChild(lutLbl);

        const lutSel = document.createElement('select');
        lutSel.className = 'radiance-ocio-select';
        lutSel.style.cssText = `
            background: #121218;
            color: ${this.theme.accent};
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            padding: 2px;
            font-size: 8px;
            font-weight: bold;
            outline: none;
            cursor: pointer;
            width: 44px;
            text-align: center;
        `;
        this.lutOptions.forEach(opt => {
            const el = document.createElement('option');
            el.value = opt; el.textContent = opt;
            lutSel.appendChild(el);
        });
        lutSel.value = this.displayLut;
        lutSel.onchange = (e) => { 
            this.displayLut = e.target.value; 
            localStorage.setItem('radiance_hud_display_lut', this.displayLut);
            this.render(); 
        };
        lutWrap.appendChild(lutSel);
        hud.appendChild(lutWrap);

        // Input Space Select
        const inWrap = document.createElement('div');
        inWrap.style.cssText = 'display:flex; flex-direction:column; gap:2px; align-items:center;';
        const inLbl = document.createElement('span');
        inLbl.textContent = 'IN';
        inLbl.style.cssText = 'font-size:7px; color:rgba(255,255,255,0.2); font-weight:bold;';
        inWrap.appendChild(inLbl);

        const inSel = document.createElement('select');
        inSel.className = 'radiance-ocio-select';
        inSel.style.cssText = lutSel.style.cssText;
        this.inputSpaceOptions.forEach(opt => {
            const el = document.createElement('option');
            el.value = opt; el.textContent = opt;
            inSel.appendChild(el);
        });
        inSel.value = this.inputSpace;
        inSel.onchange = (e) => { 
            this.inputSpace = e.target.value; 
            localStorage.setItem('radiance_hud_input_space', this.inputSpace);
            this.render(); 
        };
        inWrap.appendChild(inSel);
        hud.appendChild(inWrap);

        this.addSep(hud);

        // ── GROUP: Channels ──────────────────────────────────────────────────
        this.addGrpLabel('CH', hud);
        const chGrid = document.createElement('div');
        chGrid.style.cssText = 'display:grid; grid-template-columns: 1fr 1fr; gap:3px; width: 100%;';
        ['RGB', 'R', 'G', 'B', 'A', 'L'].forEach(ch => {
            this.addButton(ch, () => {
                this.channel = ch.toLowerCase() === 'l' ? 'luma' : ch.toLowerCase();
                this.showZdepth = false;
                this.render();
            }, ch === 'A' ? 'Alpha Channel' : '', chGrid);
        });
        hud.appendChild(chGrid);
        this.addSep(hud);

        // ── GROUP: Navigation ────────────────────────────────────────────────
        this.addGrpLabel('NAV', hud);
        this.prevFrameBtn = this.addButton('◀', () => this.prevFrame(), 'Previous Frame (←)', hud);
        this.frameDisplay = document.createElement('span');
        this.frameDisplay.textContent = '1/1';
        this.frameDisplay.style.cssText = `color: ${this.theme.text}; font-size: 8px; text-align: center; font-weight: bold;`;
        hud.appendChild(this.frameDisplay);
        this.nextFrameBtn = this.addButton('▶', () => this.nextFrame(), 'Next Frame (→)', hud);

        this.playBtn = this.addButton('◎', () => this.togglePlayback(), 'Play/Pause Sequence (Space)', hud);
        this.addSep(hud);

        // ── GROUP: Compare ──────────────────────────────────────────────────
        this.addGrpLabel('COMP', hud);
        this._compareBtnEl = this.addButton('A|B', () => this.cycleCompareMode(), 'A/B Split Compare — click to grab reference & enable wipe', hud);
        this.addSep(hud);

        // ── GROUP: Analysis ──────────────────────────────────────────────────
        this.addGrpLabel('ANLYS', hud);
        const fcBtn = this.addButton('FC', () => {
            this.falseColor = !this.falseColor;
            if (this.falseColor) {
                this.zebra = this.focusPeaking = this.showZdepth = this.gamutWarning = this.clippingMonitor = false;
            }
            syncAll(); this.render();
        }, 'False Color (E)', hud);

        const gwBtn = this.addButton('GW', () => {
            this.gamutWarning = !this.gamutWarning;
            if (this.gamutWarning) { this.clippingMonitor = this.falseColor = false; }
            syncAll(); this.render();
        }, 'Gamut Warning', hud);

        const clpBtn = this.addButton('CLP', () => {
            this.clippingMonitor = !this.clippingMonitor;
            if (this.clippingMonitor) { this.gamutWarning = this.falseColor = false; }
            syncAll(); this.render();
        }, 'Clipping Monitor', hud);

        // Sprint 4: Live Gamut Compression toggle (ACES RGC-style knee)
        const gcBtn = this.addButton('GC', () => {
            this.gamutCompression = !this.gamutCompression;
            if (this.renderer && this.renderer.setGamutCompression) {
                this.renderer.setGamutCompression(this.gamutCompression);
            } else if (this.gamutCompression) {
                this._termLog?.('warn', '[GC] Gamut compression requires WebGL renderer. Applied to delivery only.');
            }
            syncAll(); this.render();
        }, 'Gamut Compression (ACES RGC knee) — compresses out-of-gamut pixels', hud);

        const zBtn = this.addButton('Z', () => {
            this.toggleZdepth();
            syncAll();
        }, 'Z-Depth / Zebra (Z)', hud);

        this.focusPeakingBtn = this.addButton('FP', () => {
            this.focusPeaking = !this.focusPeaking;
            if (this.focusPeaking) { this.falseColor = this.zebra = this.showZdepth = false; }
            syncAll(); this.render();
        }, 'Focus Peaking (K)', hud);

        this.gridBtn = this.addButton('▦', () => {
            this.cycleGridMode();
            syncAll();
        }, 'Grid / Safe Areas (G)', hud);

        const saBtn = this.addButton('◎', () => {
            this.cycleSafeAreas();
            syncAll();
        }, 'Safe Areas (S)', hud);

        this.loupeBtn = this.addButton('🔍', () => {
            this.showLoupe = !this.showLoupe;
            syncAll();
            this.renderOverlay();
        }, 'Pixel Loupe (Q)', hud);

        const updateBtn = (btn, active) => {
            if (!btn) return;
            btn.classList.toggle('active', active);
            btn.style.color = active ? this.theme.accent : this.theme.textDim;
            btn.style.background = active ? 'rgba(0, 168, 255, 0.15)' : 'rgba(255, 255, 255, 0.03)';
            btn.style.borderColor = active ? 'rgba(0, 168, 255, 0.4)' : 'rgba(255, 255, 255, 0.08)';
            btn.style.boxShadow = active ? '0 0 10px rgba(0, 168, 255, 0.2)' : 'none';
        };

        const syncAll = () => {
            updateBtn(fcBtn, this.falseColor);
            updateBtn(gwBtn, this.gamutWarning);
            updateBtn(clpBtn, this.clippingMonitor);
            updateBtn(gcBtn, this.gamutCompression);
            updateBtn(zBtn, this.showZdepth);
            updateBtn(this.focusPeakingBtn, this.focusPeaking);
            updateBtn(this.gridBtn, this.showGrid);
            updateBtn(saBtn, this.safeAreaMode !== 'none');
            updateBtn(this.loupeBtn, this.showLoupe);
        };

        // Initial sync
        setTimeout(() => syncAll(), 50);

        this.canvasWrapper.appendChild(this.analysisHUD);
    }

    addLbl(text, container = this.toolbar) {
        const lbl = document.createElement('span');
        lbl.textContent = text;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 9px;`;
        container.appendChild(lbl);
    }

    addGrpLabel(text, container = this.toolbar) {
        const grp = document.createElement('div');
        grp.style.cssText = `display:flex;flex-direction:column;align-items:center;padding:0 4px;gap:1px;flex-shrink:0;`;
        const lbl = document.createElement('span');
        lbl.textContent = text;
        lbl.style.cssText = `color:rgba(255,255,255,0.18);font-size:7.5px;letter-spacing:0.12em;font-weight:700;white-space:nowrap;`;
        const line = document.createElement('div');
        line.style.cssText = `width:100%;height:1px;background:rgba(255,255,255,0.06);`;
        grp.appendChild(lbl);
        grp.appendChild(line);
        container.appendChild(grp);
    }

    addButton(text, onClick, title = '', container = this.toolbar) {
        const btn = document.createElement('button');
        btn.innerHTML = text;
        btn.title = title;
        btn.style.cssText = `
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 24px;
            min-width: 24px;
            width: ${container === this.analysisHUD ? '100%' : 'auto'};
            padding: 0 6px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 6px;
            color: ${this.theme.textDim};
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            line-height: 1;
            box-sizing: border-box;
        `;
        btn.onmouseenter = () => { if (!btn.classList.contains('active')) { btn.style.color = '#fff'; btn.style.background = 'rgba(255,255,255,0.1)'; } };
        btn.onmouseleave = () => { if (!btn.classList.contains('active')) { btn.style.color = this.theme.textDim; btn.style.background = 'rgba(255,255,255,0.03)'; } };
        btn.onclick = onClick;
        container.appendChild(btn);
        return btn;
    }


    // v2.2: Removed dead createCustomSlider() — replaced by createControlRow() HUD sliders

    addSep(container = this.toolbar) {
        const s = document.createElement('div');
        // v3.5: Support vertical separation
        const isVertical = container !== this.toolbar;
        if (isVertical) {
            s.style.cssText = `width: 100%; height: 1px; background: rgba(255,255,255,0.05); margin: 4px 0;`;
        } else {
            s.style.cssText = `width: 1px; height: 12px; background: ${this.theme.panelBorder}; margin: 0 2px;`;
        }
        container.appendChild(s);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RESET & COPY
    // ═══════════════════════════════════════════════════════════════════════════

    resetControls() {
        // v2.2: Complete reset — all grading + analysis state
        this.exposure = 0.0;
        this.lift = [0, 0, 0];
        this.gamma = [1, 1, 1];
        this.gain = [1, 1, 1];
        this.saturation = 1.0;
        this.channel = 'rgb';
        this.falseColor = false;
        this.zebra = false;
        this.showZdepth = false;
        this.focusPeaking = false;
        this.displayLut = 'None';

        // Reset HUD slider inputs (v2.2: replaces crashed evSlider/gammaSlider refs)
        const resetSlider = (controlRow, value) => {
            if (!controlRow) return;
            const input = controlRow.querySelector('input[type="range"]');
            if (input) { input.value = value; input.dispatchEvent(new Event('input')); }
        };
        resetSlider(this.evControl, 0);
        resetSlider(this.gammaControl, 1.0);
        resetSlider(this.satControl, 1.0);

        if (this.renderer) {
            this.renderer.setExposure(0);
            this.renderer.setLift(0, 0, 0);
            this.renderer.setGamma(1, 1, 1);
            this.renderer.setGain(1, 1, 1);
            this.renderer.setSaturation(1.0);
            this.renderer.setChannelMode(0);
            this.renderer.setFocusPeaking(false);
            this.renderer.setDisplayLutMode(0);
        }
        this.render();
    }

    copyColor() {
        if (this.lastPixelColor) {
            const { r, g, b } = this.lastPixelColor;
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
            navigator.clipboard.writeText(hex).then(() => {
                this.colorInfo.textContent = `Copied: ${hex}`;
                setTimeout(() => {
                    if (this.lastPixelColor) {
                        this.colorInfo.textContent = `RGB: ${this.lastPixelColor.r} ${this.lastPixelColor.g} ${this.lastPixelColor.b}`;
                    }
                }, 1000);
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════════



    // ═══════════════════════════════════════════════════════════════════════════
    //                          BATCH NAVIGATION
    // ═══════════════════════════════════════════════════════════════════════════

    prevFrame() {
        if (this.totalFrames <= 1) return;
        this.currentFrame = (this.currentFrame - 1 + this.totalFrames) % this.totalFrames;
        this.loadCurrentFrame();
    }

    nextFrame() {
        if (this.totalFrames <= 1) return;
        this.currentFrame = (this.currentFrame + 1) % this.totalFrames;
        this.loadCurrentFrame();
    }

    togglePlayback() {
        if (this.totalFrames <= 1) return;

        this.isPlaying = !this.isPlaying;

        if (this.isPlaying) {
            this.playBtn.textContent = '⏸';
            this.playBtn.classList.add('active');
            this.playBtn.style.color = '#fff';
            this.playBtn.style.background = 'rgba(0, 168, 255, 0.15)';
            this.playBtn.style.borderColor = 'rgba(0, 168, 255, 0.5)';
            this.playBtn.style.boxShadow = '0 0 10px rgba(0, 168, 255, 0.2)';

            // Loop at ~24fps (41ms)
            this.playbackInterval = setInterval(() => {
                this.nextFrame();
            }, 41);
        } else {
            this.playBtn.textContent = '◎';
            this.playBtn.classList.remove('active');
            this.playBtn.style.color = this.theme.textDim;
            this.playBtn.style.background = 'rgba(255,255,255,0.03)';
            this.playBtn.style.borderColor = 'rgba(255,255,255,0.08)';
            this.playBtn.style.boxShadow = 'none';

            if (this.playbackInterval) {
                clearInterval(this.playbackInterval);
                this.playbackInterval = null;
            }
        }
    }

    loadCurrentFrame() {
        // v3.2: VRAM Pressure Monitor — purge LRU cache if batch is very large (>64 frames)
        // Helps prevent "Out of Memory" crashes on lower-end GPUs when scrubbing long sequences.
        if (this.totalFrames > 64 && this.renderer?._frameCache?.size > 32) {
            const currentTime = performance.now();
            // Only log once every 5 seconds to avoid spamming the terminal
            if (!this._lastVramWarn || (currentTime - this._lastVramWarn > 5000)) {
                this._termLog('warn', `[VRAM] High pressure: ${this.totalFrames} frames. Purging GPU cache.`);
                this._lastVramWarn = currentTime;
            }
            this.renderer.clearFrameCache();
        }

        if (this.frameImages[this.currentFrame]) {
            this.image = this.frameImages[this.currentFrame];
            this.imageWidth = this.image.width;
            this.imageHeight = this.image.height;

            // Update imageData for CPU processing
            if (!this._frameDataCanvas) {
                this._frameDataCanvas = document.createElement('canvas');
            }
            this._frameDataCanvas.width = this.image.width;
            this._frameDataCanvas.height = this.image.height;
            const ctx = this._frameDataCanvas.getContext('2d');
            ctx.drawImage(this.image, 0, 0);
            this.imageData = ctx.getImageData(0, 0, this.image.width, this.image.height).data;

            // Update Z-Depth
            if (this.frameZdepthImages && this.frameZdepthImages[this.currentFrame]) {
                this.zdepthImage = this.frameZdepthImages[this.currentFrame];
                if (this.renderer) this.renderer.loadDepthTexture(this.zdepthImage);
            } else {
                this.zdepthImage = null;
            }

            // Update compare image if batch has per-frame compare
            if (this.frameCompareImages && this.frameCompareImages[this.currentFrame]) {
                this.compareImage = this.frameCompareImages[this.currentFrame];
                this.diffCanvas = null; // Clear cached diff
            }

            // Update HDR Data
            if (this.frameHDRData && this.frameHDRData[this.currentFrame]) {
                this.hdrData = this.frameHDRData[this.currentFrame];
            } else {
                this.hdrData = null;
            }

            // Update WebGL Renderer Main Texture
            if (this.renderer) {
                let loadedFloat = false;
                if (this.hdrData && this.hdrData.data) {
                    // v2.2: Prefer fp16 upload for .rhdr format (half VRAM)
                    let tex;
                    if (this.hdrData.fp16data) {
                        tex = this.renderer.loadFloat16Texture(
                            this.hdrData.fp16data,
                            this.hdrData.width,
                            this.hdrData.height,
                            this.hdrData.channels
                        );
                    } else {
                        tex = this.renderer.loadFloat32Texture(
                            this.hdrData.data,
                            this.hdrData.width,
                            this.hdrData.height,
                            this.hdrData.channels
                        );
                    }
                    if (tex) {
                        loadedFloat = true;
                        // console.log("[Radiance] Loaded float HDR texture for frame", this.currentFrame);
                    }
                }

                if (!loadedFloat) {
                    this.renderer.loadImageTexture(this.image);
                }
            }

            // v2.2: Refit if resolution changed between frames
            if (this._lastFrameW !== this.imageWidth || this._lastFrameH !== this.imageHeight) {
                this.fitToView();
            }
            this._lastFrameW = this.imageWidth;
            this._lastFrameH = this.imageHeight;

            this.updateFrameDisplay();
            this.render();
            this.updateScopes();
        }
    }

    updateFrameDisplay() {
        if (this.frameDisplay) {
            this.frameDisplay.textContent = `${this.currentFrame + 1}/${this.totalFrames}`;
        }
    }

    // v2.2: Check if all frames in current batch have loaded
    _allFramesReady() {
        if (!this.totalFrames) return true;
        for (let i = 0; i < this.totalFrames; i++) {
            if (!this.frameImages[i] && !this.frameHDRData[i]) return false;
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          GRID & SAFE AREAS
    // ═══════════════════════════════════════════════════════════════════════════

    cycleGridMode() {
        // 0=off, 1=thirds, 2=safe areas, 3=center, 4=both
        this.gridMode = (this.gridMode + 1) % 5;
        this.showGrid = this.gridMode > 0;

        const labels = ['Off', 'Thirds', 'Safe', 'Center', 'All'];
        if (this.gridBtn) {
            this.gridBtn.title = `Grid: ${labels[this.gridMode]} (G)`;
        }
        this.renderOverlay();
    }

    cycleSafeAreas() {
        const modes = ['none', 'action', 'title', 'both'];
        const idx = modes.indexOf(this.safeAreaMode);
        this.safeAreaMode = modes[(idx + 1) % modes.length];
        this.renderOverlay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          HELP & SHORTCUTS
    // ═══════════════════════════════════════════════════════════════════════════

    toggleHelp() {
        if (!this.helpOverlay) {
            const overlay = document.createElement('div');
            overlay.className = 'radiance-help-overlay';
            overlay.innerHTML = `
                <div style="margin-bottom:20px; text-align:center">
                    <h2 style="margin:0; font-weight:600; font-size:24px; letter-spacing:-0.5px">Radiance Shortcuts</h2>
                    <p style="color:#777; margin:5px 0 0 0; font-size:14px">VFX Industry-Standard Viewport Controls</p>
                </div>
                <div class="radiance-help-content">
                    <div class="help-group">
                        <h3>Viewport</h3>
                        <div class="help-item"><span class="help-desc">Fit to View</span><span class="help-key">F</span></div>
                        <div class="help-item"><span class="help-desc">Zoom 100%</span><span class="help-key">1</span></div>
                        <div class="help-item"><span class="help-desc">Toggle Fullscreen</span><span class="help-key">Ctrl+F</span></div>
                        <div class="help-item"><span class="help-desc">Toggle Help</span><span class="help-key">?</span></div>
                    </div>
                    <div class="help-group">
                        <h3>Channels</h3>
                        <div class="help-item"><span class="help-desc">RGB Toggle</span><span class="help-key">C</span></div>
                        <div class="help-item"><span class="help-desc">Red / Green / Blue / Alpha</span><span class="help-key">R / G / B / Shift+A</span></div>
                        <div class="help-item"><span class="help-desc">Luminance</span><span class="help-key">L</span></div>
                        <div class="help-item"><span class="help-desc">False Color / Peaking</span><span class="help-key">E / K</span></div>
                        <div class="help-item"><span class="help-desc">Z-Depth Overlay</span><span class="help-key">Z</span></div>
                    </div>
                    <div class="help-group">
                        <h3>Scopes</h3>
                        <div class="help-item"><span class="help-desc">Histogram / Waveform</span><span class="help-key">H / W</span></div>
                        <div class="help-item"><span class="help-desc">Vectorscope</span><span class="help-key">V</span></div>
                        <div class="help-item"><span class="help-desc">RGB Parade Toggle</span><span class="help-key">M</span></div>
                    </div>
                    <div class="help-group">
                        <h3>Grading (Numpad)</h3>
                        <div class="help-item"><span class="help-desc">Printer Lights (RGB)</span><span class="help-key">7,9 / 4,6 / 1,3</span></div>
                        <div class="help-item"><span class="help-desc">Global Offset</span><span class="help-key">8,2</span></div>
                        <div class="help-item"><span class="help-desc">Exposure Adjustment</span><span class="help-key">+ / -</span></div>
                        <div class="help-item"><span class="help-desc">Reset All</span><span class="help-key">0</span></div>
                    </div>
                </div>
                <div style="margin-top:25px; color:#555; font-size:12px; font-family:'JetBrains Mono'">ESC to Dismiss</div>
            `;
            overlay.onclick = () => this.toggleHelp();
            this.container.appendChild(overlay);
            this.helpOverlay = overlay;
        }

        const isVisible = this.helpOverlay.style.display === 'flex';
        this.helpOverlay.style.display = isVisible ? 'none' : 'flex';
        this.showHelp = !isVisible;
        setTimeout(() => {
            if (this.helpOverlay) this.helpOverlay.style.opacity = isVisible ? '0' : '1';
        }, 10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          KEYBOARD SHORTCUTS
    // ═══════════════════════════════════════════════════════════════════════════

    setupKeyboardShortcuts() {
        // v2.5: Global document listener for standard pipeline reliability
        this._docKeyHandler = (e) => {
            // Ignore if in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            // Only handle if viewer is the active component or in fullscreen
            if (this.isFullscreen) {
                this.handleKey(e);
                return;
            }

            // Simple heuristic: if container is visible and not hidden behind other Comfy nodes
            if (this.container.style.display !== 'none') {
                this.handleKey(e);
            }
        };
        document.addEventListener('keydown', this._docKeyHandler);
    }

    handleKey(e) {
        // Printer Lights (Numpad) mapped to Offset for Scene-Linear manipulation
        if (e.code && e.code.startsWith('Numpad')) {
            const step = 0.01;
            const prevOffset = [...this.offset];

            switch (e.code) {
                // Red
                case 'Numpad7': this.offset[0] -= step; break;
                case 'Numpad9': this.offset[0] += step; break;
                // Green
                case 'Numpad4': this.offset[1] -= step; break;
                case 'Numpad6': this.offset[1] += step; break;
                // Blue
                case 'Numpad1': this.offset[2] -= step; break;
                case 'Numpad3': this.offset[2] += step; break;
                // Master (All Channels)
                case 'Numpad8': this.offset[0] += step; this.offset[1] += step; this.offset[2] += step; break;
                case 'Numpad2': this.offset[0] -= step; this.offset[1] -= step; this.offset[2] -= step; break;
                // Exposure
                case 'NumpadAdd': this.adjustEV(0.25); return;
                case 'NumpadSubtract': this.adjustEV(-0.25); return;
            }

            // Only update if changed
            if (this.offset[0] !== prevOffset[0] || this.offset[1] !== prevOffset[1] || this.offset[2] !== prevOffset[2]) {
                if (this.renderer) {
                    this.renderer.setOffset(this.offset[0], this.offset[1], this.offset[2]);
                    this._pushUndoDebounced();
                }
                if (this._lastRenderContent) this._lastRenderContent(); // Update HUD wheels
                this.render();
                return; // Handled
            }
        }

        const key = e.key.toLowerCase();
        // v4.1: Alt+B — cycle pipeline bit depth (INT 8 / FLOAT 16 / FLOAT 32)
        if (e.altKey && key === 'b') {
            e.preventDefault();
            this._cyclePipelinePrecision();
            return;
        }
        switch (key) {
            case '?': case '/': if (e.shiftKey) this.toggleHelp(); break;
            case 'f': this.fitToView(); break;
            case '1': this.setZoom(1.0); break;
            case 'r': this.channel = 'r'; this.showZdepth = false; this.render(); break;
            case 'g': if (e.shiftKey) { this.cycleGridMode(); } else if (!e.ctrlKey) { this.channel = 'g'; this.showZdepth = false; this.render(); } break;
            case 'b': this.channel = 'b'; this.showZdepth = false; this.render(); break;
            case 'l': this.channel = 'luma'; this.showZdepth = false; this.render(); break;
            case 'c': this.channel = 'rgb'; this.showZdepth = false; this.render(); break;
            case 'h': this.toggleHelp(); break;
            case 'w': this.toggleScope('waveform'); break;
            case 'm': this.toggleParadeMode(); break;
            case 'v': this.toggleScope('vectorscope'); break;
            case 'e': this.falseColor = !this.falseColor; this.zebra = false; this.focusPeaking = false; this.showZdepth = false; this.render(); break;
            case 'k': this.focusPeaking = !this.focusPeaking; this.falseColor = false; this.zebra = false; this.showZdepth = false; this.render(); break;
            case 'z': this.toggleZdepth(); break;
            case 'q':
                this.showLoupe = !this.showLoupe;
                if (this.loupeBtn) {
                    this.loupeBtn.classList.toggle('active', this.showLoupe);
                    this.loupeBtn.style.color = this.showLoupe ? this.theme.accent : this.theme.textDim;
                }
                this.renderOverlay();
                break;
            case 'a':
                if (e.shiftKey) { this.channel = 'a'; this.render(); }
                else { this.cycleCompareMode(); }
                break;
            case 's': if (!e.ctrlKey) this.cycleSafeAreas(); break;
            case 'arrowleft': this.prevFrame(); break;
            case 'arrowright': this.nextFrame(); break;
            case ' ':
                e.preventDefault(); // Stop default scroll
                this.togglePlayback();
                break;
            case 'escape':
                if (this.showHelp) { this.toggleHelp(); }
                else if (this.isFullscreen) { this.exitFullscreen(); }
                else if (this.showPromptPanel) { this.togglePromptPanel(); }
                break;
            case '=': case '+': this.adjustEV(0.5); break;
            case '-': this.adjustEV(-0.5); break;
            case '0': this.resetControls(); break;
            case 'p': if (!e.ctrlKey) this.togglePromptPanel(); break;
            case '`': case '~': this.toggleTerminal(); e.preventDefault(); break;
            case 'enter': if (e.shiftKey) this.runWorkflow(); break;
        }
    }

    resetControls() {
        this._pushUndo();
        this.exposure = 0.0;
        this.contrast = 1.0;
        this.saturation = 1.0;
        this.temperature = 0.0;
        this.tint = 0.0;
        this.offset = [0.0, 0.0, 0.0];
        this.gain = [1.0, 1.0, 1.0];
        this.gamma = [1.0, 1.0, 1.0];
        this.lift = [0.0, 0.0, 0.0];

        if (this.renderer) {
            this.renderer.setExposure(this.exposure);
            this.renderer.setContrast(this.contrast);
            this.renderer.setSaturation(this.saturation);
            this.renderer.setTemperature(this.temperature);
            this.renderer.setTint(this.tint);
            this.renderer.setOffset(...this.offset);
            this.renderer.setGain(...this.gain);
            this.renderer.setGamma(...this.gamma);
            this.renderer.setLift(...this.lift);
        }

        // Update HUD UI
        if (this._lastRenderContent) this._lastRenderContent();
        this.render();
    }

    adjustEV(delta) {
        this._pushUndoDebounced();
        this.exposure = Math.max(-12.0, Math.min(12.0, this.exposure + delta));
        // v2.2: Update HUD slider (replaces crashed evSlider.setValue)
        if (this.evControl) {
            const input = this.evControl.querySelector('input[type="range"]');
            if (input) { input.value = this.exposure; input.dispatchEvent(new Event('input')); }
        }
        if (this.renderer) this.renderer.setExposure(this.exposure);
        this.render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          FULLSCREEN
    // ═══════════════════════════════════════════════════════════════════════════

    toggleFullscreen() {
        if (!this.isFullscreen) {
            const elem = this.container;
            if (elem.requestFullscreen) elem.requestFullscreen();
            else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
            this.isFullscreen = true;

            const handler = () => {
                if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                    this.isFullscreen = false;
                    document.removeEventListener('fullscreenchange', handler);
                    document.removeEventListener('webkitfullscreenchange', handler);
                }
                requestAnimationFrame(() => { this.resize(); this.render(); });
            };
            document.addEventListener('fullscreenchange', handler);
            document.addEventListener('webkitfullscreenchange', handler);
        } else {
            this.exitFullscreen();
        }
    }

    exitFullscreen() {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
        this.isFullscreen = false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EXPORT
    // ═══════════════════════════════════════════════════════════════════════════

    showExportMenu(e) {
        if (this.exportMenu) this.exportMenu.remove();

        const menu = document.createElement('div');
        this.exportMenu = menu;
        menu.style.cssText = `
            position: absolute;
            background: rgba(15, 15, 20, 0.95);
            border: 1px solid rgba(100, 110, 150, 0.4);
            border-radius: 6px;
            padding: 4px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 2px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        `;

        const rect = e.target.getBoundingClientRect();
        menu.style.left = rect.left + 'px';
        menu.style.top = (rect.bottom + 5) + 'px';

        const addOption = (label, icon, onClick, disabled = false) => {
            const opt = document.createElement('div');
            opt.innerHTML = `<span style="margin-right: 8px;">${icon}</span> ${label}`;
            opt.style.cssText = `
                padding: 6px 12px;
                color: ${disabled ? '#555' : '#ccc'};
                font-size: 11px;
                cursor: ${disabled ? 'default' : 'pointer'};
                border-radius: 4px;
                white-space: nowrap;
                transition: 0.2s;
            `;
            if (!disabled) {
                opt.onmouseenter = () => opt.style.background = 'rgba(255,255,255,0.08)';
                opt.onmouseleave = () => opt.style.background = 'transparent';
                opt.onclick = () => {
                    onClick();
                    menu.remove();
                };
            }
            menu.appendChild(opt);
        };

        addOption('Save PNG (Result)', '◎', () => this.exportSnapshot('png'));

        addOption('Export CDL (Grade)', '◎', () => this._exportCDL());
        addOption('Import CDL (Grade)', '◎', () => this._importCDL());
        addOption('Export Grade as .CUBE LUT', '◎', () => this._exportGradeLUT());

        document.body.appendChild(menu);

        const closeMenu = (ev) => {
            if (!menu.contains(ev.target) && ev.target !== e.target) {
                menu.remove();
                document.removeEventListener('mousedown', closeMenu);
            }
        };
        setTimeout(() => document.addEventListener('mousedown', closeMenu), 10);
    }

    // v3.0 #7: ASC CDL Export — writes current grading state as .cdl XML
    _exportCDL() {
        // Gain (slope), Lift (offset), Power (gamma), Saturation
        const slope = this.gain || [1, 1, 1];
        const offset = this.lift || [0, 0, 0];
        // Power: inverse of gamma (CDL power = 1/gamma for gamma>0)
        const gamma = this.gamma && Array.isArray(this.gamma) ? this.gamma : [1, 1, 1];
        const power = gamma.map(g => g > 0 ? (1.0 / g).toFixed(6) : '1.000000');
        const sat = (this.saturation !== undefined ? this.saturation : 1.0).toFixed(6);

        const s = slope.map(v => v.toFixed(6)).join(' ');
        const o = offset.map(v => v.toFixed(6)).join(' ');
        const p = power.join(' ');

        const xml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<ColorDecisionList xmlns="urn:ASC:CDL:v1.2">',
            '  <ColorDecision>',
            '    <!-- Radiance Viewer v3.0 Grade Export -->',
            '    <ColorCorrection id="radiance_grade">',
            '      <SOPNode>',
            `        <Slope>${s}</Slope>`,
            `        <Offset>${o}</Offset>`,
            `        <Power>${p}</Power>`,
            '      </SOPNode>',
            '      <SatNode>',
            `        <Saturation>${sat}</Saturation>`,
            '      </SatNode>',
            '    </ColorCorrection>',
            '  </ColorDecision>',
            '</ColorDecisionList>',
        ].join('\n');

        const blob = new Blob([xml], { type: 'text/xml' });
        const link = document.createElement('a');
        link.download = `radiance_grade_${Date.now()}.cdl`;
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
        console.log('[Radiance v3.0] CDL exported');
    }

    // ── Sprint 3: .CUBE 3D LUT export from live grade ──────────────────────
    // Bakes the current LGG/Sat/Contrast grade into a 17³-point 3D LUT .cube
    // file that can be loaded into DaVinci Resolve, Nuke, Baselight, or SCRATCH.
    _exportGradeLUT() {
        const N = 17; // Grid size (17³ = 4913 points, standard for creative LUTs)
        const lines = [
            `# Radiance Viewer Grade LUT — exported ${new Date().toISOString()}`,
            `# Gain: ${(this.gain || [1, 1, 1]).map(v => v.toFixed(4)).join(' ')}`,
            `# Gamma: ${(this.gamma || [1, 1, 1]).map(v => v.toFixed(4)).join(' ')}`,
            `# Lift: ${(this.lift || [0, 0, 0]).map(v => v.toFixed(4)).join(' ')}`,
            `# Saturation: ${(this.saturation || 1).toFixed(4)}`,
            `# Contrast: ${(this.contrast || 1).toFixed(4)}  Pivot: ${(this.pivot || 0.18).toFixed(4)}`,
            'LUT_3D_SIZE 17',
            'DOMAIN_MIN 0.0 0.0 0.0',
            'DOMAIN_MAX 1.0 1.0 1.0',
            ''
        ];

        const gain = Array.isArray(this.gain) ? this.gain : [1, 1, 1];
        const gamma = Array.isArray(this.gamma) ? this.gamma : [1, 1, 1];
        const lift = Array.isArray(this.lift) ? this.lift : [0, 0, 0];
        const sat = this.saturation || 1.0;
        const con = this.contrast || 1.0;
        const piv = this.pivot || 0.18;

        // Apply grade inline (mirrors apply_grading Python logic)
        const applyGrade = (r, g, b) => {
            // Lift (luma-pivoted additive shadow shift)
            const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            const lumaPivot = Math.max(0, 1 - luma);
            r += lift[0] * lumaPivot;
            g += lift[1] * lumaPivot;
            b += lift[2] * lumaPivot;
            // Gain (multiplicative slope)
            r *= gain[0]; g *= gain[1]; b *= gain[2];
            // Gamma (power curve on positives)
            if (r > 0) r = Math.pow(r, 1.0 / Math.max(gamma[0], 0.01));
            if (g > 0) g = Math.pow(g, 1.0 / Math.max(gamma[1], 0.01));
            if (b > 0) b = Math.pow(b, 1.0 / Math.max(gamma[2], 0.01));
            // Contrast (around pivot)
            r = (r - piv) * con + piv;
            g = (g - piv) * con + piv;
            b = (b - piv) * con + piv;
            // Saturation
            const luma2 = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            r = luma2 + sat * (r - luma2);
            g = luma2 + sat * (g - luma2);
            b = luma2 + sat * (b - luma2);
            // Clamp to [0, 1] for LUT domain
            return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
        };

        // .CUBE Ordering: R varies fastest, then G, then B
        for (let bi = 0; bi < N; bi++) {
            for (let gi = 0; gi < N; gi++) {
                for (let ri = 0; ri < N; ri++) {
                    const r = ri / (N - 1), g = gi / (N - 1), bv = bi / (N - 1);
                    const [or, og, ob] = applyGrade(r, g, bv);
                    lines.push(`${or.toFixed(6)} ${og.toFixed(6)} ${ob.toFixed(6)}`);
                }
            }
        }

        const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
        const link = document.createElement('a');
        link.download = `radiance_grade_${Date.now()}.cube`;
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
        this._termLog?.('success', `[LUT] Exported 17³ .cube LUT from live grade`);
    }

    // v3.0 #7: ASC CDL Import — reads .cdl XML and applies to current grading state
    _importCDL() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.cdl,.xml';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(ev.target.result, 'text/xml');
                    const slope = doc.querySelector('Slope')?.textContent?.trim().split(/\s+/).map(Number);
                    const offset = doc.querySelector('Offset')?.textContent?.trim().split(/\s+/).map(Number);
                    const power = doc.querySelector('Power')?.textContent?.trim().split(/\s+/).map(Number);
                    const satEl = doc.querySelector('Saturation');
                    const sat = satEl ? parseFloat(satEl.textContent) : 1.0;

                    if (slope && slope.length === 3) {
                        this.gain = slope;
                        if (this.renderer) this.renderer.setGain(...slope);
                    }
                    if (offset && offset.length === 3) {
                        this.lift = offset;
                        if (this.renderer) this.renderer.setLift(...offset);
                    }
                    if (power && power.length === 3) {
                        // CDL Power → gamma: g = 1/power
                        const gamma = power.map(p => p > 0 ? 1.0 / p : 1.0);
                        this.gamma = gamma;
                        if (this.renderer) this.renderer.setGamma(...gamma);
                    }
                    this.saturation = sat;
                    if (this.renderer) this.renderer.setSaturation(sat);
                    this.render();
                    console.log('[Radiance v3.0] CDL imported:', { slope, offset, power, sat });
                } catch (err) {
                    console.error('[Radiance v3.0] CDL import failed:', err);
                }
            };
            reader.readAsText(file);
        };
        input.click();
    }


    exportSnapshot(format = 'png') {
        if (!this.image) return;

        if (format === 'exr') {
            const imgData = (this.lastResult || []).find(d => d.frame === this.currentFrame && !d.is_compare && !d.is_zdepth);
            if (imgData && imgData.exr_filename) {
                // Use dedicated EXR location metadata when available.
                // Falls back to the PNG thumbnail's subfolder/type so older
                // backend versions (pre-exr_subfolder) continue to work.
                const sub = imgData.exr_subfolder ?? imgData.subfolder ?? '';
                const type = imgData.exr_type ?? imgData.type ?? 'temp';
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(imgData.exr_filename)}`
                    + `&subfolder=${encodeURIComponent(sub)}`
                    + `&type=${encodeURIComponent(type)}`
                );
                const link = document.createElement('a');
                link.href = url;
                link.download = imgData.exr_filename;
                link.click();
                this._termLog?.('success', `[Export] Saved Source EXR: ${imgData.exr_filename}`);
            } else {
                this._termLog?.('warn', '[Export] Source EXR not available for this frame. Re-run the node to generate it.');
            }
            return;
        }

        // ── v4.0: 32-bit Graded EXR Export ───────────────────────────────────
        // Renders the full composite pipeline into an RGBA32F FBO, reads back
        // the float data, and encodes it as an OpenEXR file with FLOAT pixel
        // type (pixelType=2) and uncompressed scanlines.
        if (format === 'exr32') {
            if (!this.useWebGL || !this.renderer) {
                this._termLog?.('warn', '[Export] EXR 32-bit export requires WebGL renderer.');
                return;
            }
            const result = this.renderer.readPixelsFloat32(
                this.imageWidth, this.imageHeight, this.lutIntensity || 1.0
            );
            if (!result) {
                this._termLog?.('warn', '[Export] Float32 readback failed (WebGL2 required).');
                return;
            }

            const blob = this._encodeEXR32(result.data, result.width, result.height);
            if (!blob) {
                this._termLog?.('warn', '[Export] EXR encoding failed.');
                return;
            }

            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `radiance_graded_${Date.now()}.exr`;
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
            this._termLog?.('success', `[Export] Saved 32-bit graded EXR: ${result.width}×${result.height}`);
            return;
        }

        // Default PNG Export
        const exp = document.createElement('canvas');
        exp.width = this.imageWidth;
        exp.height = this.imageHeight;
        const ctx = exp.getContext('2d');

        if (this.useWebGL && this.renderer && this.renderer.textures.image) {
            const prevW = this.glCanvas.width, prevH = this.glCanvas.height;
            this.glCanvas.width = this.imageWidth;
            this.glCanvas.height = this.imageHeight;
            this.renderer.render(this.lutIntensity || 1.0);
            ctx.drawImage(this.glCanvas, 0, 0);
            this.glCanvas.width = prevW;
            this.glCanvas.height = prevH;
        } else {
            this.renderImage(ctx, this.image);
        }

        const link = document.createElement('a');
        link.download = `radiance_${Date.now()}.png`;
        link.href = exp.toDataURL('image/png');
        link.click();
    }

    // ── v4.0: OpenEXR 32-bit FLOAT Encoder ───────────────────────────────────
    // Encodes an RGBA Float32Array as a valid OpenEXR file with:
    //   - pixelType = 2 (FLOAT, 32-bit IEEE 754)
    //   - compression = 0 (NONE — lossless, maximum quality)
    //   - 4 channels: R, G, B, A (stored alphabetically: A, B, G, R per EXR spec)
    //   - Single-part scanline layout
    //
    // @param {Float32Array} pixels - RGBA interleaved float32 data (top-to-bottom)
    // @param {number} width
    // @param {number} height
    // @returns {Blob|null} - application/octet-stream blob of the .exr file
    _encodeEXR32(pixels, width, height) {
        if (!pixels || pixels.length < width * height * 4) return null;

        const nCh = 4;  // RGBA
        const bytesPerPixel = 4;  // float32
        const scanlineBytes = nCh * width * bytesPerPixel;

        // ── Helper: write null-terminated string ──────────────────────────────
        const encoder = new TextEncoder();
        const encStr = (s) => {
            const b = encoder.encode(s);
            const r = new Uint8Array(b.length + 1);
            r.set(b); r[b.length] = 0;
            return r;
        };

        // ── Build header ──────────────────────────────────────────────────────
        const headerParts = [];

        // channels attribute (alphabetical: A, B, G, R)
        // Each channel: name\0 + pixelType(4) + pLinear(1) + reserved(3) + xSampling(4) + ySampling(4)
        const channelNames = ['A', 'B', 'G', 'R'];
        const channelEntries = [];
        for (const ch of channelNames) {
            const nameBytes = encStr(ch);
            const entry = new Uint8Array(nameBytes.length + 16);
            entry.set(nameBytes, 0);
            const dv = new DataView(entry.buffer, entry.byteOffset);
            dv.setInt32(nameBytes.length, 2, true);      // pixelType=2 (FLOAT)
            dv.setUint8(nameBytes.length + 4, 0);        // pLinear=0
            // 3 bytes reserved (already 0)
            dv.setInt32(nameBytes.length + 8, 1, true);   // xSampling=1
            dv.setInt32(nameBytes.length + 12, 1, true);  // ySampling=1
            channelEntries.push(entry);
        }
        // channels value = all entries + null terminator byte
        const channelsValueLen = channelEntries.reduce((s, e) => s + e.length, 0) + 1;
        const channelsValue = new Uint8Array(channelsValueLen);
        let cp = 0;
        for (const e of channelEntries) { channelsValue.set(e, cp); cp += e.length; }
        channelsValue[cp] = 0; // null terminator for channel list

        // Write attribute: name\0 + type\0 + size(4) + value
        const writeAttr = (name, type, valueBytes) => {
            const n = encStr(name);
            const t = encStr(type);
            const sizeBytes = new Uint8Array(4);
            new DataView(sizeBytes.buffer).setInt32(0, valueBytes.length, true);
            headerParts.push(n, t, sizeBytes, valueBytes);
        };

        // channels
        writeAttr('channels', 'chlist', channelsValue);

        // compression = 0 (NONE)
        writeAttr('compression', 'compression', new Uint8Array([0]));

        // dataWindow
        const dwBytes = new Uint8Array(16);
        const dwView = new DataView(dwBytes.buffer);
        dwView.setInt32(0, 0, true);           // xMin
        dwView.setInt32(4, 0, true);           // yMin
        dwView.setInt32(8, width - 1, true);   // xMax
        dwView.setInt32(12, height - 1, true); // yMax
        writeAttr('dataWindow', 'box2i', dwBytes);

        // displayWindow (same as dataWindow)
        writeAttr('displayWindow', 'box2i', dwBytes);

        // lineOrder = 0 (increasing Y)
        writeAttr('lineOrder', 'lineOrder', new Uint8Array([0]));

        // pixelAspectRatio = 1.0
        const parBytes = new Uint8Array(4);
        new DataView(parBytes.buffer).setFloat32(0, 1.0, true);
        writeAttr('pixelAspectRatio', 'float', parBytes);

        // screenWindowCenter = (0, 0)
        const swcBytes = new Uint8Array(8);
        writeAttr('screenWindowCenter', 'v2f', swcBytes);

        // screenWindowWidth = 1.0
        const swwBytes = new Uint8Array(4);
        new DataView(swwBytes.buffer).setFloat32(0, 1.0, true);
        writeAttr('screenWindowWidth', 'float', swwBytes);

        // End of header (null byte)
        headerParts.push(new Uint8Array([0]));

        // ── Compute sizes ─────────────────────────────────────────────────────
        const headerSize = headerParts.reduce((s, p) => s + p.length, 0);
        const magicAndVersion = 8; // 4 bytes magic + 4 bytes version
        const offsetTableSize = height * 8; // one uint64 per scanline
        const headerTotalSize = magicAndVersion + headerSize;
        const dataStart = headerTotalSize + offsetTableSize;

        // Each scanline block: y_coord(4) + data_size(4) + pixel_data
        const scanlineBlockSize = 4 + 4 + scanlineBytes;
        const totalSize = dataStart + height * scanlineBlockSize;

        // ── Assemble file ─────────────────────────────────────────────────────
        const buffer = new ArrayBuffer(totalSize);
        const out = new Uint8Array(buffer);
        const view = new DataView(buffer);

        // Magic number: 20000630 (0x01312F76)
        view.setUint32(0, 20000630, true);
        // Version: 2 (single-part scanline, no flags)
        view.setUint32(4, 2, true);

        // Header attributes
        let wp = 8;
        for (const part of headerParts) {
            out.set(part, wp);
            wp += part.length;
        }

        // Offset table
        for (let y = 0; y < height; y++) {
            const offset = dataStart + y * scanlineBlockSize;
            // Write as two 32-bit values (low, high) since JS doesn't have native uint64
            view.setUint32(wp, offset, true);      // low 32 bits
            view.setUint32(wp + 4, 0, true);       // high 32 bits (0 for files < 4GB)
            wp += 8;
        }

        // ── Scanline data ─────────────────────────────────────────────────────
        // EXR stores channels in alphabetical order per scanline:
        // For each scanline: [all A pixels][all B pixels][all G pixels][all R pixels]
        // Channel order: A=0, B=1, G=2, R=3
        // Source RGBA order: R=0, G=1, B=2, A=3
        const chMap = [3, 2, 1, 0]; // A→src[3], B→src[2], G→src[1], R→src[0]

        for (let y = 0; y < height; y++) {
            const blockOff = dataStart + y * scanlineBlockSize;
            view.setInt32(blockOff, y, true);               // y coordinate
            view.setUint32(blockOff + 4, scanlineBytes, true); // data size

            const pixelOff = blockOff + 8;
            for (let ci = 0; ci < nCh; ci++) {
                const srcCh = chMap[ci];
                const chOff = pixelOff + ci * width * bytesPerPixel;
                for (let x = 0; x < width; x++) {
                    const srcIdx = (y * width + x) * 4 + srcCh;
                    view.setFloat32(chOff + x * 4, pixels[srcIdx], true);
                }
            }
        }

        console.log(`[Radiance EXR] Encoded ${width}×${height}×4ch FLOAT (${(totalSize / 1048576).toFixed(1)} MB)`);
        return new Blob([buffer], { type: 'application/octet-stream' });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          KEYBOARD HELP
    // ═══════════════════════════════════════════════════════════════════════════

    toggleHelp() {
        this.showHelp = !this.showHelp;
        if (this.showHelp) {
            this.showHelpScreen();
        } else {
            if (this.helpPanel) {
                this.helpPanel.remove();
                this.helpPanel = null;
            }
        }
    }

    showHelpScreen() {
        if (this.helpPanel) return;

        this.helpPanel = document.createElement('div');
        this.helpPanel.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(10, 10, 20, 0.98);
            border: 2px solid ${this.theme.accent};
            border-radius: 12px;
            padding: 24px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 10000;
            box-shadow: 0 8px 32px rgba(0,0,0,0.8);
        `;

        const shortcuts = [
            {
                category: 'Navigation', items: [
                    ['F', 'Fit to view'],
                    ['1', '1:1 pixel zoom'],
                    ['Mouse Wheel', 'Zoom in/out'],
                    ['Shift+Drag', 'Pan image'],
                    ['← →', 'Previous/Next frame']
                ]
            },
            {
                category: 'Display', items: [
                    ['R/G/B/L', 'View R/G/B/Luma channel'],
                    ['C', 'RGB (color) view'],
                    ['Shift+A', 'Alpha channel'],
                    ['+/−', 'Adjust exposure'],
                    ['0', 'Reset all controls'],
                    ['E', 'False color'],
                    ['K', 'Focus peaking (GPU)'],
                    ['Q', 'Pixel loupe'],
                    ['Z', 'Z-Depth / Zebra']
                ]
            },
            {
                category: 'Analysis', items: [
                    ['H', 'Histogram (HDR-aware)'],
                    ['W', 'Waveform'],
                    ['M', 'Toggle Parade mode'],
                    ['V', 'Vectorscope'],
                    ['Shift+G', 'Cycle grid modes'],
                    ['G', 'Green channel'],
                    ['S', 'Safe areas'],
                    ['A', 'A/B compare'],
                ]
            },
            {
                category: 'General', items: [
                    ['⛶', 'Fullscreen'],
                    ['Shift+Enter', 'Run workflow'],
                    ['P', 'Prompt editor'],
                    ['?', 'This help'],
                    ['Esc', 'Close help/Exit fullscreen']
                ]
            }
        ];

        let html = `<div style="color: ${this.theme.accent}; font-size: 18px; font-weight: bold; margin-bottom: 16px; text-align: center;">◎ Keyboard Shortcuts</div>`;

        shortcuts.forEach(section => {
            html += `<div style="margin-bottom: 16px;">`;
            html += `<div style="color: ${this.theme.accent}; font-size: 12px; font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid ${this.theme.panelBorder}; padding-bottom: 4px;">${section.category}</div>`;
            section.items.forEach(([key, desc]) => {
                html += `<div style="display: flex; justify-content: space-between; padding: 3px 0; font-size: 11px;">`;
                html += `<span style="background: #1a1a28; padding: 2px 8px; border-radius: 3px; font-family: monospace; color: ${this.theme.text};">${key}</span>`;
                html += `<span style="color: ${this.theme.textDim}; margin-left: 12px;">${desc}</span>`;
                html += `</div>`;
            });
            html += `</div>`;
        });

        html += `<div style="text-align: center; margin-top: 16px; padding-top: 12px; border-top: 1px solid ${this.theme.panelBorder}; color: ${this.theme.textDim}; font-size: 9px;">Press ? or Esc to close</div>`;

        this.helpPanel.innerHTML = html;
        this.canvasWrapper.appendChild(this.helpPanel);

        // Click to close
        this.helpPanel.onclick = () => this.toggleHelp();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          AUTO GRADE MATCHING
    // ═══════════════════════════════════════════════════════════════════════════

    matchGrade(shelfSlot) {
        if (!this.renderer || !this.renderer.gl || !this.renderer.referenceShelf[shelfSlot]) {
            console.warn("[Radiance] Cannot match: no reference at slot", shelfSlot);
            return;
        }

        const gl = this.renderer.gl;

        // Helper to compute average RGB of a currently bound texture via FBO pixel read
        const getAverageRGB = (tex, width, height) => {
            const fbo = gl.createFramebuffer();
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

            // Note: If width/height > max limits, we'd need to resize.
            // But reading a large tensor completely is slow.
            // Faster: bind it and draw it tiny, then read.
            // For now, we'll read a 256x256 subsample from the center.
            const size = Math.min(256, width, height);
            const cx = Math.floor(width / 2 - size / 2);
            const cy = Math.floor(height / 2 - size / 2);

            const pixels = new Uint8Array(size * size * 4);
            gl.readPixels(cx, cy, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteFramebuffer(fbo);

            let rSum = 0, gSum = 0, bSum = 0;
            const len = size * size;
            for (let i = 0; i < pixels.length; i += 4) {
                // Approximate linear back from 8-bit read
                rSum += Math.pow(pixels[i] / 255.0, 2.2);
                gSum += Math.pow(pixels[i + 1] / 255.0, 2.2);
                bSum += Math.pow(pixels[i + 2] / 255.0, 2.2);
            }
            return [rSum / len, gSum / len, bSum / len];
        };

        try {
            this._pushUndo(); // Save state before auto-match

            // 1. Get Reference mean
            const refTex = this.renderer.referenceShelf[shelfSlot];
            // The shelf texture size is same as WebGL canvas
            const refMean = getAverageRGB(refTex, this.renderer.canvas.width, this.renderer.canvas.height);

            // 2. Get Current image mean (without current grade)
            // It's tricky to read the RAW texture because it's Float16 and readPixels usually demands Uint8.
            // Read from the processed output instead, but reset offset first.
            const prevOffset = [...this.offset];
            this.offset = [0, 0, 0];
            this.renderer.setOffset(0, 0, 0);
            this.render(); // force render without offset

            const curMean = getAverageRGB(this.renderer.textures.image, this.renderer.imageWidth || 512, this.renderer.imageHeight || 512);

            // 3. Compute difference in Scene-Linear 
            // We want to add an offset such that curMean + offset = refMean
            const dr = refMean[0] - curMean[0];
            const dg = refMean[1] - curMean[1];
            const db = refMean[2] - curMean[2];

            // Add difference to previous offset, blend it 80% to avoid extreme blowing out
            this.offset = [
                prevOffset[0] + dr * 0.8,
                prevOffset[1] + dg * 0.8,
                prevOffset[2] + db * 0.8
            ];

            // Broadcast to renderer and UI
            this.renderer.setOffset(this.offset[0], this.offset[1], this.offset[2]);
            if (this._lastRenderContent) this._lastRenderContent(); // update HUD wheels
            this.render();

            console.log(`[Radiance] Shot Match applied. Offset shifted by [${dr.toFixed(3)}, ${dg.toFixed(3)}, ${db.toFixed(3)}]`);

        } catch (e) {
            console.error("[Radiance] Auto match failed:", e);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          SCOPES
    // ═══════════════════════════════════════════════════════════════════════════

    toggleScope(scope) {
        if (scope === 'histogram') {
            this.showHistogram = !this.showHistogram;
            this.histogramCanvas.style.display = this.showHistogram ? 'block' : 'none';
            this.histogramLabel.style.display = this.showHistogram ? 'block' : 'none';
        } else if (scope === 'waveform') {
            this.showWaveform = !this.showWaveform;
            this.waveformCanvas.style.display = this.showWaveform ? 'block' : 'none';
            this.waveformLabel.style.display = this.showWaveform ? 'block' : 'none';
            this.paradeToggle.style.display = this.showWaveform ? 'block' : 'none';
        } else if (scope === 'vectorscope') {
            this.showVectorscope = !this.showVectorscope;
            this.vectorscopeCanvas.style.display = this.showVectorscope ? 'block' : 'none';
            this.vectorscopeLabel.style.display = this.showVectorscope ? 'block' : 'none';
        }

        const any = this.showHistogram || this.showWaveform || this.showVectorscope;
        this.scopePanel.style.display = any ? 'flex' : 'none';
        if (this.image) this.updateScopes();
    }

    // v3.0 #6: Route histogram scope to renderer.renderHistogram() for GPU-based rendering
    _renderGPUHistogram() {
        if (this.renderer && this.histogramCanvas && this.showHistogram) {
            // Use log scale for HDR images (data_range max > 1.0)
            const isHDR = this.hdrData && this.hdrData.data_range && this.hdrData.data_range[1] > 1.05;
            this.renderer.renderHistogram(this.histogramCanvas, isHDR);
        }
    }


    // v2.2: HDR-aware histogram — reads float32 data, maps to 256 bins (log scale for >1.0)
    _updateHistogramHDR() {
        const fd = this.hdrData.data;
        const ch = this.hdrData.channels || 3;
        const total = fd.length / ch;
        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);

        // Map: [0,1] → bins 0-200, [1,∞) → bins 200-255 (log scale)
        const toBin = (v) => {
            if (v <= 0) return 0;
            if (v <= 1.0) return Math.floor(v * 200);
            // log2 scale for >1.0: each stop = ~10 bins
            return Math.min(255, 200 + Math.floor(Math.log2(v) * 10));
        };

        for (let i = 0; i < total; i++) {
            const r = fd[i * ch];
            const g = ch > 1 ? fd[i * ch + 1] : r;
            const b = ch > 2 ? fd[i * ch + 2] : r;
            hR[toBin(r)]++;
            hG[toBin(g)]++;
            hB[toBin(b)]++;
        }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);
        this.histogramData = { hR, hG, hB, max };

        const hCtx = this.histogramCtx;
        const w = this.histogramCanvas.width, h = this.histogramCanvas.height;
        hCtx.fillStyle = '#0a0a0f';
        hCtx.fillRect(0, 0, w, h);

        // Draw HDR reference line at bin 200 (= 1.0)
        hCtx.strokeStyle = 'rgba(255,255,255,0.15)';
        hCtx.lineWidth = 1;
        hCtx.beginPath();
        const refX = (200 / 255) * w;
        hCtx.moveTo(refX, 0); hCtx.lineTo(refX, h);
        hCtx.stroke();
        hCtx.fillStyle = 'rgba(255,255,255,0.3)';
        hCtx.font = '7px monospace';
        hCtx.fillText('1.0', refX + 2, 8);

        const draw = (hist, color) => {
            hCtx.strokeStyle = color; hCtx.lineWidth = 1; hCtx.beginPath();
            for (let i = 0; i < 256; i++) {
                const y = h - (hist[i] / max) * h;
                if (i === 0) hCtx.moveTo(i, y); else hCtx.lineTo(i, y);
            }
            hCtx.stroke();
        };
        hCtx.globalAlpha = 0.7;
        draw(hR, '#ff4444'); draw(hG, '#44ff44'); draw(hB, '#4444ff');
        hCtx.globalAlpha = 1;
    }

    toggleZdepth() {
        // If Z-Depth image is available, toggle it
        if (this.zdepthImage || (this.frameZdepthImages && this.frameZdepthImages.length > 0)) {
            this.showZdepth = !this.showZdepth;
            // Disable other overlays
            if (this.showZdepth) {
                this.zebra = false;
                this.falseColor = false;
                this.focusPeaking = false;
            }
        }
        // If no Z-Depth, toggle Zebra as fallback
        else {
            this.zebra = !this.zebra;
            if (this.zebra) {
                this.falseColor = false;
                this.focusPeaking = false;
                this.showZdepth = false;
            }
        }
        this.render();
    }

    cycleCompareMode() {
        if (!this.image && !this.renderer) { this.compareMode = 'none'; return; }

        const wasOff = this.compareMode === 'none';
        this.compareMode = wasOff ? 'wipe' : 'none';

        if (this.compareMode === 'wipe') {
            // Auto-grab the current rendered frame as the B (reference) side.
            // Then we switch to wipe mode — any subsequent grading changes
            // show on the RIGHT (A=live) vs LEFT (B=reference).
            if (this.renderer && this.renderer.grabReferenceStill) {
                this.renderer.grabReferenceStill(0);
                // Activate ref slot 0 — sets wipeRefEnabled = true internally
                this.renderer.swapReferenceShelf(0);
            }
            this.wipePosition = 0.5;
            if (this.renderer) this.renderer.setWipe(this.wipePosition, true);
        } else {
            // Disable wipe on both paths
            if (this.renderer) this.renderer.setWipe(0.5, false);
            if (this.renderer && this.renderer.clearReferenceShelf) this.renderer.clearReferenceShelf();
        }

        this._updateCompareBtn();
        this.render();
    }

    /** Sync the A|B toolbar button appearance to the current compare state. */
    _updateCompareBtn() {
        if (!this._compareBtnEl) return;
        const on = this.compareMode === 'wipe';
        this._compareBtnEl.style.borderColor  = on ? '#00a8ff' : 'rgba(255,255,255,0.15)';
        this._compareBtnEl.style.background   = on ? 'rgba(0,168,255,0.15)' : 'rgba(255,255,255,0.05)';
        this._compareBtnEl.style.color        = on ? '#00a8ff' : '#ccc';
    }

    /** Draw the wipe split-line, A/B labels and drag handle on top of the rendered canvas. */
    _drawWipeSplitOverlay(ctx, w, h) {
        if (this.compareMode !== 'wipe') return;

        const wipeX = Math.round(w * this.wipePosition);

        // Crisp anti-alias-off line
        ctx.save();
        ctx.globalAlpha = 1.0;

        // Drop-shadow glow for depth
        ctx.shadowColor  = 'rgba(0,0,0,0.8)';
        ctx.shadowBlur   = 4;

        // Outer white line
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth   = 2;
        ctx.beginPath();
        ctx.moveTo(wipeX, 0);
        ctx.lineTo(wipeX, h);
        ctx.stroke();

        ctx.shadowBlur = 0;

        // ── Labels ──────────────────────────────────────────────────────────
        const PAD = 8, LBL_H = 22, LBL_W = 32;
        const labelY = Math.max(10, Math.min(h - LBL_H - 10, h * 0.08));
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // B label (left / reference side)
        const bX = Math.max(LBL_W / 2 + PAD, wipeX - PAD - LBL_W / 2);
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.beginPath();
        ctx.roundRect(bX - LBL_W / 2, labelY, LBL_W, LBL_H, 4);
        ctx.fill();
        ctx.fillStyle = '#aad4ff';
        ctx.fillText('B', bX, labelY + LBL_H / 2);

        // A label (right / live/graded side)
        const aX = Math.min(w - LBL_W / 2 - PAD, wipeX + PAD + LBL_W / 2);
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.beginPath();
        ctx.roundRect(aX - LBL_W / 2, labelY, LBL_W, LBL_H, 4);
        ctx.fill();
        ctx.fillStyle = '#a0ffb0';
        ctx.fillText('A', aX, labelY + LBL_H / 2);

        // ── Drag Handle ──────────────────────────────────────────────────────
        const cy = h / 2;
        const R_OUTER = 12, R_INNER = 5;

        // Outer disc
        ctx.fillStyle = 'rgba(20,24,36,0.85)';
        ctx.beginPath();
        ctx.arc(wipeX, cy, R_OUTER, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(wipeX, cy, R_OUTER, 0, Math.PI * 2);
        ctx.stroke();

        // Left/Right arrows
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('⇔', wipeX, cy);

        ctx.restore();
    }

    updateScopes() {
        // Debounce scope updates for better performance
        if (this.scopeUpdateTimer) {
            clearTimeout(this.scopeUpdateTimer);
        }
        this.scopeUpdateTimer = setTimeout(() => {
            if (this.showHistogram) this.updateHistogram();
            if (this.showWaveform) this.updateWaveform();
            if (this.showVectorscope) this.updateVectorscope();
            if (this.scopeOverlay) this.renderOverlay();
        }, this.scopeDebounceMs);
    }

    updateHistogram() {
        if (!this.image || !this.renderer) return;

        // Use renderHistogram() which adds log-scale grid, HDR dotted line, and labels
        this.renderer.renderHistogram(this.histogramCanvas, this.scopeLogView || false);
    }

    toggleParadeMode() {
        this.waveformParadeMode = !this.waveformParadeMode;
        if (this.showWaveform) {
            this.updateWaveform();
        }
    }

    updateWaveform() {
        if (!this.image || !this.renderer) return;

        // v2.5: GPU-Accelerated Waveform (32-bit HDR)
        const tex = this.renderer.textures.image;
        if (tex) {
            this.renderer.renderScope('waveform', this.waveformCanvas, tex, this.renderer.isLinearTexture, this.waveformParadeMode);
        }
    }


    updateVectorscope() {
        if (!this.image || !this.renderer) return;

        // v2.5: GPU-Accelerated Vectorscope
        const tex = this.renderer.textures.image;
        if (tex) {
            this.renderer.renderScope('vectorscope', this.vectorscopeCanvas, tex, this.renderer.isLinearTexture);

            // Draw Pro Overlays (Skin Tone Line, Targets) on top of GPU result
            const vCtx = this.vectorscopeCtx;
            const size = this.vectorscopeCanvas.width;
            const cx = size / 2, cy = size / 2, rad = size / 2 - 10;

            // Skin Tone Line (I-axis in YIQ, approx 123 deg)
            vCtx.strokeStyle = 'rgba(255, 120, 80, 0.4)';
            vCtx.lineWidth = 1.5;
            vCtx.setLineDash([4, 4]);
            const skinAng = (123 - 90) * Math.PI / 180;
            vCtx.beginPath();
            vCtx.moveTo(cx, cy);
            vCtx.lineTo(cx + Math.cos(skinAng) * rad, cy + Math.sin(skinAng) * rad);
            vCtx.stroke();
            vCtx.setLineDash([]);

            // Rec.709 Targets
            const targets = [
                { a: 103, c: '#f44', n: 'R' }, { a: 167, c: '#ff4', n: 'Y' },
                { a: 241, c: '#4f4', n: 'G' }, { a: 283, c: '#4ff', n: 'C' },
                { a: 347, c: '#44f', n: 'B' }, { a: 61, c: '#f4f', n: 'M' }
            ];
            targets.forEach(t => {
                const ang = (t.a - 90) * Math.PI / 180;
                vCtx.strokeStyle = t.c; vCtx.lineWidth = 1;
                vCtx.strokeRect(cx + Math.cos(ang) * rad * 0.75 - 3, cy + Math.sin(ang) * rad * 0.75 - 3, 6, 6);
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          OVERLAY RENDERING
    // ═══════════════════════════════════════════════════════════════════════════

    renderOverlay() {
        const ctx = this.overlayCtx;
        const w = this.overlayCanvas.width, h = this.overlayCanvas.height;
        ctx.clearRect(0, 0, w, h);

        // Grid
        if (this.showGrid) this.drawGrid(ctx, w, h);

        // Scope overlay
        if (this.scopeOverlay && this.histogramData) this.drawHistogramOverlay(ctx, w, h);

        // Interactive Mask / Power Window Overlay
        if (this.maskState && this.maskState.type > 0 && this.maskState.showOverlay && !this.wipeEnabled) {
            this.drawMaskInteractiveOverlay(ctx);
        }

        // Sprint 4: Render persistent probe dots
        if (this._probeMemory && this._probeMemory.length > 0) {
            this._probeMemory.forEach((p, idx) => {
                // Map image coords to canvas coords
                const cx = p.imgX * this.zoom + this.panX;
                const cy = p.imgY * this.zoom + this.panY;
                ctx.save();
                ctx.fillStyle = p.color || '#fff';
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(cx, cy, 5, 0, Math.PI * 2);
                ctx.fill(); ctx.stroke();
                // Label
                ctx.fillStyle = p.color || '#fff';
                ctx.font = 'bold 9px monospace';
                ctx.fillText(`P${idx + 1} ${p.hex}`, cx + 7, cy + 4);
                ctx.restore();
            });
        }

        // Sprint 4: Render annotations (persistent lines)
        this._renderAnnotationLines(ctx, this._annotationLines || []);
        if (this._isAnnotating && this._annotationCurrentLine) {
            this._renderAnnotationLines(ctx, [this._annotationCurrentLine]);
        }
    }

    // Render a set of annotation line objects onto the given 2D canvas context
    _renderAnnotationLines(ctx, lines) {
        lines.forEach(line => {
            if (!line.pts || line.pts.length < 2) return;
            ctx.save();
            ctx.strokeStyle = line.color || '#ff4';
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.shadowColor = 'rgba(0,0,0,0.6)';
            ctx.shadowBlur = 3;
            ctx.beginPath();
            ctx.moveTo(line.pts[0].x, line.pts[0].y);
            for (let i = 1; i < line.pts.length; i++) ctx.lineTo(line.pts[i].x, line.pts[i].y);
            ctx.stroke();
            ctx.restore();
        });
    }

    // Called during live Alt+drag drawing to show the current stroke in real-time
    _drawAnnotations() {
        this.renderOverlay(); // full redraw (debounce handled by RAF via renderOverlay)
    }

    getMaskHandleAt(x, y) {
        if (!this.maskState || this.maskState.type === 0) return null;

        const m = this.maskState;
        const cx = m.center[0] * this.imageWidth;
        const cy = m.center[1] * this.imageHeight;
        const sw = m.scale[0] * this.imageWidth;
        const sh = m.scale[1] * this.imageHeight;
        const rot = m.rotation;

        // Hit distance in image space, adjusted for zoom
        const hitRadius = 12 / this.zoom;

        // 1. Center
        if (Math.hypot(x - cx, y - cy) < hitRadius) return 'center';

        // 2. Rotation Handle (above top edge)
        const rotHandleDist = (sh / 2) + (25 / this.zoom);
        const rx = cx + rotHandleDist * Math.sin(rot);
        const ry = cy - rotHandleDist * Math.cos(rot);
        if (Math.hypot(x - rx, y - ry) < hitRadius) return 'rotation';

        // 3. Scale Handles
        const cosR = Math.cos(rot), sinR = Math.sin(rot);

        // Scale X handles (right/left)
        const ptsX = [
            { x: cx + (sw / 2) * cosR, y: cy + (sw / 2) * sinR },
            { x: cx - (sw / 2) * cosR, y: cy - (sw / 2) * sinR }
        ];
        for (const p of ptsX) if (Math.hypot(x - p.x, y - p.y) < hitRadius) return 'scale_x';

        // Scale Y handles (top/bottom)
        const ptsY = [
            { x: cx - (sh / 2) * sinR, y: cy + (sh / 2) * cosR },
            { x: cx + (sh / 2) * sinR, y: cy - (sh / 2) * cosR }
        ];
        for (const p of ptsY) if (Math.hypot(x - p.x, y - p.y) < hitRadius) return 'scale_y';

        return null;
    }

    drawMaskInteractiveOverlay(ctx) {
        if (!this.maskState) return;
        const m = this.maskState;
        const cx = m.center[0] * this.imageWidth;
        const cy = m.center[1] * this.imageHeight;
        const sw = m.scale[0] * this.imageWidth;
        const sh = m.scale[1] * this.imageHeight;
        const rot = m.rotation;

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);

        const accent = this.theme.accent;
        const handleSize = 5 / this.zoom;

        // 1. Draw Shape Outline
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.5 / this.zoom;
        ctx.setLineDash([4 / this.zoom, 4 / this.zoom]);

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rot);

        ctx.beginPath();
        if (m.type === 1) { // Circle
            ctx.ellipse(0, 0, sw / 2, sh / 2, 0, 0, Math.PI * 2);
        } else if (m.type === 2) { // Box
            ctx.rect(-sw / 2, -sh / 2, sw, sh);
        }
        ctx.stroke();
        ctx.restore();
        ctx.setLineDash([]);

        // 2. Draw Handles
        const drawHandle = (hx, hy, isDot = false) => {
            ctx.fillStyle = accent;
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1 / this.zoom;
            ctx.beginPath();
            if (isDot) ctx.arc(hx, hy, handleSize * 1.2, 0, Math.PI * 2);
            else ctx.rect(hx - handleSize, hy - handleSize, handleSize * 2, handleSize * 2);
            ctx.fill();
            ctx.stroke();
        };

        // Center
        drawHandle(cx, cy, true);

        // Rotation line and handle
        const rotHandleDist = (sh / 2) + (25 / this.zoom);
        const rx = cx + rotHandleDist * Math.sin(rot);
        const ry = cy - rotHandleDist * Math.cos(rot);

        ctx.beginPath();
        ctx.moveTo(cx + (sh / 2) * Math.sin(rot), cy - (sh / 2) * Math.cos(rot));
        ctx.lineTo(rx, ry);
        ctx.stroke();
        drawHandle(rx, ry, true);

        // Scale Handles
        const cosR = Math.cos(rot), sinR = Math.sin(rot);
        drawHandle(cx + (sw / 2) * cosR, cy + (sw / 2) * sinR);
        drawHandle(cx - (sw / 2) * cosR, cy - (sw / 2) * sinR);
        drawHandle(cx - (sh / 2) * sinR, cy + (sh / 2) * cosR);
        drawHandle(cx + (sh / 2) * sinR, cy - (sh / 2) * cosR);

        ctx.restore();
    }

    drawGrid(ctx, w, h) {
        // Grid mode: 1=thirds, 2=safe areas, 3=center, 4=all

        // Rule of thirds (mode 1 or 4)
        if (this.gridMode === 1 || this.gridMode === 4) {
            ctx.strokeStyle = 'rgba(255,255,255,0.2)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 2; i++) {
                ctx.beginPath();
                ctx.moveTo(w * i / 3, 0); ctx.lineTo(w * i / 3, h);
                ctx.moveTo(0, h * i / 3); ctx.lineTo(w, h * i / 3);
                ctx.stroke();
            }
        }

        // Safe areas (mode 2 or 4, or via safeAreaMode)
        const showSafeFromGrid = this.gridMode === 2 || this.gridMode === 4;
        const showActionSafe = showSafeFromGrid || this.safeAreaMode === 'action' || this.safeAreaMode === 'both';
        const showTitleSafe = showSafeFromGrid || this.safeAreaMode === 'title' || this.safeAreaMode === 'both';

        // Action Safe (93% - broadcast safe)
        if (showActionSafe) {
            ctx.strokeStyle = 'rgba(0, 200, 255, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([8, 4]);
            const actionMargin = 0.035; // 3.5% margin = 93% visible
            ctx.beginPath();
            ctx.rect(
                w * actionMargin, h * actionMargin,
                w * (1 - 2 * actionMargin), h * (1 - 2 * actionMargin)
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = 'rgba(0, 200, 255, 0.6)';
            ctx.font = '9px sans-serif';
            ctx.fillText('Action Safe 93%', w * actionMargin + 4, h * actionMargin + 12);
        }

        // Title Safe (80% - text safe)
        if (showTitleSafe) {
            ctx.strokeStyle = 'rgba(255, 200, 0, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            const titleMargin = 0.10; // 10% margin = 80% visible
            ctx.beginPath();
            ctx.rect(
                w * titleMargin, h * titleMargin,
                w * (1 - 2 * titleMargin), h * (1 - 2 * titleMargin)
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = 'rgba(255, 200, 0, 0.6)';
            ctx.font = '9px sans-serif';
            ctx.fillText('Title Safe 80%', w * titleMargin + 4, h * titleMargin + 12);
        }

        // Center cross (mode 3 or 4)
        if (this.gridMode === 3 || this.gridMode === 4) {
            ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(w / 2 - 30, h / 2); ctx.lineTo(w / 2 + 30, h / 2);
            ctx.moveTo(w / 2, h / 2 - 30); ctx.lineTo(w / 2, h / 2 + 30);
            ctx.stroke();

            // Center circle
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, 5, 0, Math.PI * 2);
            ctx.stroke();
        }
    }


    drawHistogramOverlay(ctx, w, h) {
        const { hR, hG, hB, max } = this.histogramData;
        const hx = w - 130, hy = 8, hw = 120, hh = 40;

        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(hx - 4, hy - 4, hw + 8, hh + 8);

        const draw = (hist, color) => {
            ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = hx + (i / 255) * hw, y = hy + hh - (hist[i] / max) * hh;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };
        ctx.globalAlpha = 0.6;
        draw(hR, '#f44'); draw(hG, '#4f4'); draw(hB, '#44f');
        ctx.globalAlpha = 1;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    setupEventListeners() {
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            this._lastCanvasRect = rect;
            this._canvasScaleX = this.canvas.width / rect.width;
            this._canvasScaleY = this.canvas.height / rect.height;
            const mx = (e.clientX - rect.left) * this._canvasScaleX;
            const my = (e.clientY - rect.top) * this._canvasScaleY;

            // v2.5: Exponential zoom for smoother response at all scales (Nuke/Resolve style)
            const sensitivity = 0.001;
            const factor = Math.exp(-e.deltaY * sensitivity);
            const newZoom = Math.max(0.01, Math.min(200, this.zoom * factor));

            this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
            this.panY = my - (my - this.panY) * (newZoom / this.zoom);
            this.zoom = newZoom;
            this.updateBottomBar();
            this.render();
        });

        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this._lastCanvasRect = rect; // Cache for performance during mousemove
            this._canvasScaleX = this.canvas.width / rect.width;
            this._canvasScaleY = this.canvas.height / rect.height;

            const mx = (e.clientX - rect.left) * this._canvasScaleX;
            const my = (e.clientY - rect.top) * this._canvasScaleY;
            const x = (mx - this.panX) / this.zoom;
            const y = (my - this.panY) / this.zoom;

            if (this.compareMode === 'wipe' && e.button === 0 && Math.abs(mx - this.canvas.width * this.wipePosition) < 20) {
                this.isDraggingWipe = true;
                this.canvas.style.cursor = 'col-resize';
                return;
            }

            // Check Mask UI Handles
            const handle = this.getMaskHandleAt(x, y);
            if (handle && e.button === 0) {
                this.maskDragMode = handle;
                this.maskDragStart = { x, y };
                this.maskStateStart = JSON.parse(JSON.stringify(this.maskState));
                this.canvas.style.cursor = handle === 'rotation' ? 'alias' : 'crosshair';
                return;
            }



            if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                e.preventDefault(); // Prevent middle-click auto-scroll which swallows mouseup
                this.isPanning = true;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.canvas.style.cursor = 'grabbing';
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this._lastCanvasRect) this._lastCanvasRect = this.canvas.getBoundingClientRect();
            const rect = this._lastCanvasRect;
            const mx = (e.clientX - rect.left) * (this._canvasScaleX || 1);
            const my = (e.clientY - rect.top) * (this._canvasScaleY || 1);
            // Track mouse X for wipe drag hit-test in the WebGL path
            this._lastMouseX = mx;

            if (this.isDraggingWipe) {
                // Safety: check if left button is still pressed. e.buttons & 1 is the left button.
                // This handles cases where mouseup might have been swallowed by other UI elements.
                if (e.buttons !== undefined && !(e.buttons & 1)) {
                    this.isDraggingWipe = false;
                    this.updateCursor(e);
                    return;
                }
                this.wipePosition = Math.max(0.02, Math.min(0.98, mx / this.canvas.width));
                if (this.renderer) this.renderer.setWipe(this.wipePosition, true);
                this.render();
                return;
            }

            // Cursor feedback: show col-resize only when hovering near the split line
            if (this.compareMode === 'wipe') {
                const wipeX = this.canvas.width * this.wipePosition;
                this.canvas.style.cursor = Math.abs(mx - wipeX) < 20 ? 'col-resize' : 'crosshair';
            }

            if (this.maskDragMode) {
                const x = (mx - this.panX) / this.zoom;
                const y = (my - this.panY) / this.zoom;
                const m = this.maskState;
                const s = this.maskStateStart;

                if (this.maskDragMode === 'center') {
                    m.center[0] = s.center[0] + (x - this.maskDragStart.x) / this.imageWidth;
                    m.center[1] = s.center[1] + (y - this.maskDragStart.y) / this.imageHeight;
                } else if (this.maskDragMode === 'scale_x' || this.maskDragMode === 'scale_y') {
                    const dx = x - (m.center[0] * this.imageWidth);
                    const dy = y - (m.center[1] * this.imageHeight);
                    const cosR = Math.cos(-m.rotation);
                    const sinR = Math.sin(-m.rotation);
                    const tx = dx * cosR - dy * sinR;
                    const ty = dx * sinR + dy * cosR;

                    if (this.maskDragMode === 'scale_x') {
                        m.scale[0] = Math.max(0.01, Math.abs(tx) * 2 / this.imageWidth);
                    } else {
                        m.scale[1] = Math.max(0.01, Math.abs(ty) * 2 / this.imageHeight);
                    }
                } else if (this.maskDragMode === 'rotation') {
                    const dx = x - (m.center[0] * this.imageWidth);
                    const dy = y - (m.center[1] * this.imageHeight);
                    m.rotation = Math.atan2(dy, dx) + Math.PI / 2;
                }

                if (this.activeTab === 'masks' && this.tabContentContainer) {
                    // Update the visible GUI sliders without fully rebuilding the DOM to avoid losing focus
                    const centerInputs = this.tabContentContainer.querySelectorAll('.knob-value');
                }

                if (this.renderer) this.renderer.setMask(m);
                this.render();
                return;
            }

            if (this.isPanning) {
                if (e.buttons !== undefined && !(e.buttons & 1) && !(e.buttons & 4)) {
                    this.isPanning = false;
                    this.canvas.style.cursor = 'crosshair';
                    return;
                }
                // Panning strictly relies on clientX delta, scaling isn't necessary for delta-drag
                this.panX += e.clientX - this.lastMouseX;
                this.panY += e.clientY - this.lastMouseY;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.render();
            }


            this.updateCursor(e);
            this.updateProbe(e);
        });

        // Click-to-Focus for DoF

        window.addEventListener('mouseup', (e) => {
            this.isPanning = false;
            if (this.isDraggingWipe) {
                this.isDraggingWipe = false;
                // Restore cursor based on current hover position
                if (this.compareMode === 'wipe') {
                    const wipeX = this.canvas.width * this.wipePosition;
                    this.canvas.style.cursor = Math.abs((this._lastMouseX || -999) - wipeX) < 20 ? 'col-resize' : 'crosshair';
                } else {
                    this.canvas.style.cursor = 'crosshair';
                }
            }

            if (this.maskDragMode) {
                this.maskDragMode = null;
                // Re-render the mask tab to update sliders if it's the active tab
                if (this.activeTab === 'masks' && this._lastRenderContent) {
                    this._lastRenderContent();
                }
            }

            if (!this.isAnnotating && !this.maskDragMode && this.compareMode !== 'wipe') {
                this.canvas.style.cursor = 'crosshair';
            }
        });

        this.canvas.addEventListener('click', (e) => {
            if (this.dofEnabled && !this.isAnnotating && !this.isPanning && !this.isDraggingWipe) {
                const rect = this.canvas.getBoundingClientRect();
                const scaleX = this.canvas.width / rect.width;
                const scaleY = this.canvas.height / rect.height;
                const mx = (e.clientX - rect.left) * scaleX;
                const my = (e.clientY - rect.top) * scaleY;
                const imgX = Math.floor((mx - this.panX) / this.zoom);
                const imgY = Math.floor((my - this.panY) / this.zoom);

                if (this.imageData && imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                    if (this.zdepthImage) {
                        // Sample Z-depth
                        const off = document.createElement('canvas'); off.width = 1; off.height = 1;
                        const ctx = off.getContext('2d');
                        ctx.drawImage(this.zdepthImage, imgX, imgY, 1, 1, 0, 0, 1, 1);
                        const d = ctx.getImageData(0, 0, 1, 1).data[0] / 255;

                        this.focusDistance = d;
                        // v2.2: Update HUD focus control (replaces undefined focusSlider)
                        if (this.focusControl) {
                            const finput = this.focusControl.querySelector('input[type="range"]');
                            if (finput) { finput.value = d; finput.dispatchEvent(new Event('input')); }
                        }
                        if (this.renderer) {
                            this.renderer.setFocusDistance(d);
                            this.render();
                        }
                    }
                }
            }
        });

        // Multi-probe memory (Sprint 4): right-click saves up to 4 persistent probe points
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (!this._lastProbe) return;
            if (!this._probeMemory) this._probeMemory = [];
            if (this._probeMemory.length >= 4) this._probeMemory.shift(); // FIFO cap at 4
            this._probeMemory.push({ ...this._lastProbe, color: ['#f55', '#5f5', '#55f', '#ff5'][this._probeMemory.length] });
            this.renderOverlay(); // Re-draw to show new probe dot
            this._termLog?.('info', `[Probe] Stored ${this._probeMemory.length}/4: Disp ${this._lastProbe.dispStr}  |  Linear ${this._lastProbe.linStr}`);
        });

        // Alt+drag annotation drawing (Sprint 4)
        this._annotationLines = [];
        this._isAnnotating = false;
        this._annotationCurrentLine = null;

        this.canvas.addEventListener('mousedown', (eAnn) => {
            if (eAnn.altKey && eAnn.button === 0) {
                eAnn.preventDefault();
                this._isAnnotating = true;
                const rect = this.canvas.getBoundingClientRect();
                const mx = (eAnn.clientX - rect.left) * (this._canvasScaleX || 1);
                const my = (eAnn.clientY - rect.top) * (this._canvasScaleY || 1);
                const colors = ['#ff4', '#f55', '#5f5', '#55f', '#f5f', '#5ff'];
                this._annotationCurrentLine = {
                    pts: [{ x: mx, y: my }],
                    color: colors[this._annotationLines.length % colors.length]
                };
            }
        }, { capture: true });

        document.addEventListener('mousemove', (eAnn) => {
            if (!this._isAnnotating || !this._annotationCurrentLine) return;
            const rect = this.canvas.getBoundingClientRect();
            const mx = (eAnn.clientX - rect.left) * (this._canvasScaleX || 1);
            const my = (eAnn.clientY - rect.top) * (this._canvasScaleY || 1);
            this._annotationCurrentLine.pts.push({ x: mx, y: my });
            this._drawAnnotations();
        });

        document.addEventListener('mouseup', () => {
            if (this._isAnnotating && this._annotationCurrentLine && this._annotationCurrentLine.pts.length > 1) {
                this._annotationLines.push(this._annotationCurrentLine);
            }
            this._isAnnotating = false;
            this._annotationCurrentLine = null;
        });

        // Shift+Alt+Click clears all annotations
        this.canvas.addEventListener('click', (eAnn) => {
            if (eAnn.altKey && eAnn.shiftKey) {
                this._annotationLines = [];
                this._probeMemory = [];
                this.renderOverlay();
                this._termLog?.('info', '[Annotation] All annotations and probes cleared.');
            }
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.infoLeft.innerHTML = '';
            // Don't clear infoRight as it contains static info
        });

        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.canvasWrapper);
        }
    }

    resize() {
        this._lastCanvasRect = null; // Invalidate cache
        const rect = this.canvasWrapper.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        this.canvas.width = Math.floor(rect.width);
        this.canvas.height = Math.floor(rect.height);
        this.overlayCanvas.width = this.canvas.width;
        this.overlayCanvas.height = this.canvas.height;
        this.render();
    }

    updateCursor(e) {
        if (!this.image) return;
        const rect = this._lastCanvasRect || this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this._canvasScaleX || 1);
        const my = (e.clientY - rect.top) * (this._canvasScaleY || 1);
        const imgX = Math.floor((mx - this.panX) / this.zoom);
        const imgY = Math.floor((my - this.panY) / this.zoom);
        if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
            this.cursorInfo.textContent = `X: ${imgX} Y: ${imgY}`;
        } else {
            this.cursorInfo.textContent = 'X: — Y: —';
        }
    }

    updateProbe(e) {
        if (!this.image || !this.imageData) { this.infoLeft.innerHTML = ''; return; }
        const rect = this._lastCanvasRect || this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this._canvasScaleX || 1);
        const my = (e.clientY - rect.top) * (this._canvasScaleY || 1);
        const imgX = Math.floor((mx - this.panX) / this.zoom);
        const imgY = Math.floor((my - this.panY) / this.zoom);

        if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
            const idx = (imgY * this.imageWidth + imgX) * 4;
            let r = this.imageData[idx], g = this.imageData[idx + 1], b = this.imageData[idx + 2], a = this.imageData[idx + 3];

            // Check for HDR float data
            let floatR, floatG, floatB;

            if (this.hdrData && this.hdrData.data) {
                // NPY is usually HWC or CHW?
                // Our backend saves as (H, W, C).
                // shape[0]=H, shape[1]=W, shape[2]=C.
                const channels = this.hdrData.shape[2] || 3;
                const hIdx = (imgY * this.imageWidth + imgX) * channels;

                floatR = this.hdrData.data[hIdx];
                if (channels === 1) {
                    floatG = floatR; floatB = floatR;
                } else {
                    floatG = this.hdrData.data[hIdx + 1];
                    floatB = channels >= 3 ? this.hdrData.data[hIdx + 2] : 0;
                }
            } else {
                // Fallback to 8-bit normalized
                floatR = r / 255;
                floatG = g / 255;
                floatB = b / 255;
            }

            let floatVals = '';
            let evVal = '';
            const luma_f = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 255;

            if (this.hdrData && this.hdrData.data) {
                const ch = this.hdrData.channels || 3;
                const hIdx = (imgY * this.imageWidth + imgX) * ch;
                const fr = this.hdrData.data[hIdx];
                const fg = ch > 1 ? this.hdrData.data[hIdx + 1] : fr;
                const fb = ch > 2 ? this.hdrData.data[hIdx + 2] : fr;
                floatVals = `<span style="color:${this.theme.accent}">F: ${(fr).toFixed(4)} ${(fg).toFixed(4)} ${(fb).toFixed(4)}</span> | `;

                const curLuma = fr * 0.2126 + fg * 0.7152 + fb * 0.0722;
                const stops = Math.log2(Math.max(1e-6, curLuma) / 0.18);
                evVal = `<span style="color:#d49dff">EV: ${stops > 0 ? '+' : ''}${stops.toFixed(1)}</span> | `;
            } else {
                const stops = Math.log2(Math.max(1e-6, luma_f) / 0.18);
                evVal = `<span style="color:#d49dff">EV: ${stops > 0 ? '+' : ''}${stops.toFixed(1)}</span> | `;
            }

            this.lastPixelColor = { r, g, b, a };
            const luma = (r * 0.2126 + g * 0.7152 + b * 0.0722).toFixed(0);
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

            // Compute display (gamma-encoded) values and scene-linear values
            // Display: 8-bit normalized 0-255 or [0-1] 
            const dispR = (r / 255).toFixed(4);
            const dispG = (g / 255).toFixed(4);
            const dispB = (b / 255).toFixed(4);

            // Scene-linear: use HDR sidecar if available (true float), else sRGB gamma decode
            let linR, linG, linB;
            if (this.hdrData && this.hdrData.data) {
                const ch = this.hdrData.channels || 3;
                const hIdx2 = (imgY * this.imageWidth + imgX) * ch;
                linR = this.hdrData.data[hIdx2];
                linG = ch > 1 ? this.hdrData.data[hIdx2 + 1] : linR;
                linB = ch > 2 ? this.hdrData.data[hIdx2 + 2] : linR;
            } else {
                // sRGB → linear approximation (IEC 61966-2-1)
                const sRGBtoLin = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
                linR = sRGBtoLin(r / 255);
                linG = sRGBtoLin(g / 255);
                linB = sRGBtoLin(b / 255);
            }

            const linStr = `${linR.toFixed(4)} ${linG.toFixed(4)} ${linB.toFixed(4)}`;
            const dispStr = `${dispR} ${dispG} ${dispB}`;
            const isHDRPick = !!(this.hdrData && this.hdrData.data);
            const pickLabel = isHDRPick ? '🟢 HDR Linear' : '⚪ Linear~';

            this.colorInfo.textContent = `RGB: ${r} ${g} ${b}`;

            // v3.5 Pro Probing UI — dual display+linear readout (Nuke-style)
            this.infoLeft.innerHTML = `
                <span style="color:#888; margin-right:8px;">X:${imgX.toString().padStart(4, '0')} Y:${imgY.toString().padStart(4, '0')}</span>
                ${evVal}
                <span style="color:#aaa; margin-right:6px;">Disp: ${dispStr}</span>
                <span style="color:${isHDRPick ? this.theme.accent : '#888'}; margin-right:6px;" title="${pickLabel}: scene-linear float values">${pickLabel}: ${linStr}</span>
                <span style="color:#777">Hex: ${hex} | L: ${luma}</span>
            `;

            // Store probe for multi-probe display (Sprint 4)
            this._lastProbe = { imgX, imgY, dispStr, linStr, hex, isHDRPick };

            // Draw pixel loupe on overlay
            if (this.showLoupe) {
                this.renderOverlay(); // Clear and redraw first
                this.drawLoupe(mx, my, imgX, imgY);
            }
        } else {
            this.infoLeft.innerHTML = '';
            this.colorInfo.textContent = 'RGB: — — —';
            this.lastPixelColor = null;
        }

        // Update fixed right info stats continuously
        const zoomPct = Math.round(this.zoom * 100);
        const depth = (this.hdrData && this.hdrData.data) ? '32b FP' : '8b INT';

        let indicators = '';
        if (this.gamutWarning) indicators += '<span style="color:#f0f; font-weight:bold">GW</span> ';
        if (this.clippingMonitor) indicators += '<span style="color:#f00; font-weight:bold">CLP</span> ';
        if (this.falseColor) indicators += '<span style="color:#fc0; font-weight:bold">FC</span> ';

        this.infoRight.innerHTML = `
            ${indicators}
            <span>${depth}</span>
            <span>CH: ${this.channel.toUpperCase()}</span>
            <span>RES: ${this.imageWidth}x${this.imageHeight}</span>
            <span>ZOOM: ${zoomPct}%</span>
            <span>FRM: ${this.currentFrame + 1}/${this.totalFrames}</span>
        `;

        // Toggle Legend visibility
        if (this.fcLegend) {
            this.fcLegend.style.display = this.falseColor ? 'flex' : 'none';
        }

    }

    updateBottomBar() {
        if (!this.infoRight) return;
        const zoomPct = Math.round((this.zoom || 1) * 100);
        const w = this.imageWidth || 0;
        const h = this.imageHeight || 0;
        this.infoRight.innerHTML = `
            <span>CH: ${(this.channel || 'rgb').toUpperCase()}</span>
            <span>RES: ${w}x${h}</span>
            <span>ZOOM: ${zoomPct}%</span>
            <span>FRM: ${(this.currentFrame || 0) + 1}/${this.totalFrames || 1}</span>
        `;
    }

    drawLoupe(mx, my, imgX, imgY) {
        if (!this.overlayCtx) return;
        const ctx = this.overlayCtx;
        const size = this.loupeSize;
        const mag = this.loupeMagnification;
        const numPixels = Math.floor(size / mag);
        const halfPixels = Math.floor(numPixels / 2);

        // Position loupe in corner opposite to cursor
        let lx = mx > this.canvas.width / 2 ? 10 : this.canvas.width - size - 10;
        let ly = my > this.canvas.height / 2 ? 10 : this.canvas.height - size - 10;

        // Draw loupe background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(lx - 1, ly - 1, size + 2, size + 2, 4);
        ctx.fill();
        ctx.stroke();

        // Draw magnified pixels
        for (let py = -halfPixels; py <= halfPixels; py++) {
            for (let px = -halfPixels; px <= halfPixels; px++) {
                const sx = imgX + px, sy = imgY + py;
                const dx = lx + (px + halfPixels) * mag;
                const dy = ly + (py + halfPixels) * mag;

                // Stop if we are outside the loupe box bounds
                if (px + halfPixels >= numPixels || py + halfPixels >= numPixels) continue;

                if (sx >= 0 && sx < this.imageWidth && sy >= 0 && sy < this.imageHeight) {
                    let r, g, b;
                    // v2.2: Read HDR float data for loupe when available
                    if (this.hdrData && this.hdrData.data) {
                        const channels = this.hdrData.shape ? (this.hdrData.shape[2] || 3) : (this.hdrData.channels || 3);
                        const hIdx = (sy * this.imageWidth + sx) * channels;

                        // Tonemap for display: simple Reinhard per-channel
                        const fr = this.hdrData.data[hIdx];
                        const fg = channels > 1 ? this.hdrData.data[hIdx + 1] : fr;
                        const fb = channels > 2 ? this.hdrData.data[hIdx + 2] : fr;

                        // Apply exposure
                        const em = Math.pow(2, this.exposure || 0);
                        r = Math.min(255, Math.max(0, Math.round((fr * em / (fr * em + 1)) * 255)));
                        g = Math.min(255, Math.max(0, Math.round((fg * em / (fg * em + 1)) * 255)));
                        b = Math.min(255, Math.max(0, Math.round((fb * em / (fb * em + 1)) * 255)));
                    } else {
                        const idx = (sy * this.imageWidth + sx) * 4;
                        r = this.imageData[idx]; g = this.imageData[idx + 1]; b = this.imageData[idx + 2];
                    }
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                } else {
                    ctx.fillStyle = '#111';
                }

                ctx.fillRect(dx, dy, mag, mag);

                // Highlight center pixel
                if (px === 0 && py === 0) {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(dx + 0.5, dy + 0.5, mag - 1, mag - 1);
                }
            }
        }

        // Draw border
        ctx.strokeStyle = this.theme.accent;
        ctx.lineWidth = 1;
        ctx.strokeRect(lx, ly, size, size);

        // Draw crosshair
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx + size / 2, ly); ctx.lineTo(lx + size / 2, ly + size);
        ctx.moveTo(lx, ly + size / 2); ctx.lineTo(lx + size, ly + size / 2);
        ctx.stroke();
    }

    toggleMetadata() {
        const isVisible = this.metadataPanel.style.display !== 'none';
        this.metadataPanel.style.display = isVisible ? 'none' : 'block';
        if (!isVisible) this.updateInfo();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                     OCIO COLOR MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Fetch OCIO config info from the backend.
     * Call once on viewer init to populate the OCIO dropdown.
     */
    async ocioInit() {
        try {
            const resp = await fetch('/radiance/ocio/config');
            if (!resp.ok) return;
            this._ocioConfig = await resp.json();

            if (this._ocioConfig.loaded) {
                console.log(`[Radiance OCIO] Config: ${this._ocioConfig.name} (${this._ocioConfig.display_view_pairs?.length || 0} transforms)`);
                this._ocioPopulateDropdown();
            } else {
                console.log('[Radiance OCIO] No config loaded');
            }
        } catch (e) {
            // OCIO endpoint not available — silently degrade
            this._ocioConfig = null;
        }
    }

    /**
     * Load an OCIO config from a specific file path.
     */
    async ocioLoadConfig(configPath) {
        try {
            const resp = await fetch('/radiance/ocio/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: configPath }),
            });
            const data = await resp.json();
            if (data.status === 'success') {
                this._ocioConfig = data.config;
                this._ocioPopulateDropdown();
                this._termLog?.('info', `[OCIO] Loaded: ${this._ocioConfig.name}`);
                return true;
            } else {
                this._termLog?.('error', `[OCIO] ${data.error}`);
                return false;
            }
        } catch (e) {
            this._termLog?.('error', `[OCIO] Load failed: ${e.message}`);
            return false;
        }
    }

    /**
     * Apply an OCIO display/view transform by baking it to a 3D LUT
     * and uploading to the WebGL renderer.
     *
     * @param {string} display  - OCIO display name (e.g., "sRGB")
     * @param {string} view     - OCIO view name (e.g., "ACES 1.0 SDR-video")
     * @param {number} size     - LUT cube size (33 = fast, 65 = quality)
     */
    async ocioApplyDisplayView(display, view, size = 33) {
        if (!this.renderer) return;

        try {
            this._termLog?.('info', `[OCIO] Baking ${size}³ LUT: ${display} / ${view}...`);

            const resp = await fetch('/radiance/ocio/bake', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ display, view, size }),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ error: resp.statusText }));
                this._termLog?.('error', `[OCIO] Bake failed: ${err.error}`);
                return;
            }

            // Response is raw float32 binary — convert to Float32Array
            const buffer = await resp.arrayBuffer();
            const lutData = new Float32Array(buffer);

            // Upload to WebGL via existing 3D LUT pipeline
            this.renderer.loadLUT(lutData, size);
            // OCIO-FIX: This LUT already contains the full display transform
            // including sRGB OETF. Clear the analytical display LUT mode so
            // it doesn't stack on top, and flag the shader to skip the final
            // linearToSRGB to prevent double-gamma (orange cast / blown highlights).
            this.renderer.setDisplayLutMode(0);
            this.renderer.setLutIsDisplayTransform(true);
            this.render();

            this._ocioActiveTransform = `${display} / ${view}`;
            this._termLog?.('info', `[OCIO] Active: ${this._ocioActiveTransform} (${size}³)`);

        } catch (e) {
            this._termLog?.('error', `[OCIO] Apply failed: ${e.message}`);
        }
    }

    /**
     * Apply a direct color space → color space OCIO transform.
     */
    async ocioApplyColorSpace(src, dst, size = 33) {
        if (!this.renderer) return;

        try {
            const resp = await fetch('/radiance/ocio/bake', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ src, dst, size }),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ error: resp.statusText }));
                this._termLog?.('error', `[OCIO] Bake failed: ${err.error}`);
                return;
            }

            const buffer = await resp.arrayBuffer();
            const lutData = new Float32Array(buffer);
            this.renderer.loadLUT(lutData, size);
            // Colorspace transforms are NOT display transforms — the shader
            // must still apply the final linearToSRGB OETF after this LUT.
            this.renderer.setDisplayLutMode(0);
            this.renderer.setLutIsDisplayTransform(false);
            this.render();

            this._ocioActiveTransform = `${src} → ${dst}`;
            this._termLog?.('info', `[OCIO] Active: ${this._ocioActiveTransform} (${size}³)`);

        } catch (e) {
            this._termLog?.('error', `[OCIO] Apply failed: ${e.message}`);
        }
    }

    /**
     * Remove the active OCIO LUT — reverts to built-in analytical LUTs.
     */
    ocioClear() {
        if (this.renderer && this.renderer.textures.lut) {
            this.renderer.gl.deleteTexture(this.renderer.textures.lut);
            this.renderer.textures.lut = null;
        }
        // OCIO-FIX: Reset display-transform flag so normal linearToSRGB resumes.
        if (this.renderer) this.renderer.setLutIsDisplayTransform(false);
        this._ocioActiveTransform = null;
        this.render();
        this._termLog?.('info', '[OCIO] Cleared — using built-in LUTs');
    }

    /**
     * Populate the HUD OCIO dropdown (if the HUD has one).
     * @private
     */
    _ocioPopulateDropdown() {
        const dropdown = this.container?.querySelector?.('.radiance-ocio-select');
        if (!dropdown || !this._ocioConfig?.display_view_pairs) return;

        // Clear existing options
        dropdown.innerHTML = '<option value="">— OCIO: None —</option>';

        for (const pair of this._ocioConfig.display_view_pairs) {
            const opt = document.createElement('option');
            opt.value = JSON.stringify({ display: pair.display, view: pair.view });
            opt.textContent = pair.label;
            dropdown.appendChild(opt);
        }

        // Wire up change handler
        dropdown.onchange = async () => {
            const val = dropdown.value;
            if (!val) {
                this.ocioClear();
                return;
            }
            try {
                const { display, view } = JSON.parse(val);
                await this.ocioApplyDisplayView(display, view);
            } catch (e) {
                console.error('[Radiance OCIO] Dropdown error:', e);
            }
        };
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          IMAGE
    // ═══════════════════════════════════════════════════════════════════════════

    async setImage(src, data = {}) {
        // Support both URL string and Image element
        if (src instanceof HTMLImageElement || src instanceof HTMLCanvasElement) {
            // Direct image element passed
            this.image = src;
            this.imageWidth = src.width;
            this.imageHeight = src.height;

            // Create ImageData for CPU processing
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = src.width;
            tempCanvas.height = src.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(src, 0, 0);
            this.imageData = tempCtx.getImageData(0, 0, src.width, src.height).data;

            // Load to WebGL renderer
            if (this.renderer) {
                this.renderer.loadImageTexture(src);
            }

            this.imageSrc = src.src || '';
            this.viewerData = data;
        } else {
            // URL string path
            this.imageSrc = src;
            this.viewerData = data;
            // Standard 8/16-bit image loading
            await this.loadStandardImage(src);
        }

        // Load depth map if provided
        if (data.depth_path) {
            const depthImg = new Image();
            depthImg.crossOrigin = 'anonymous';
            depthImg.onload = () => {
                this.zdepthImage = depthImg;
                if (this.renderer) this.renderer.loadDepthTexture(depthImg);
            };
            depthImg.src = data.depth_path;
        }

        this.fitToView();
        this.render();
        this.updateInfo();
        this.updateScopes();

        // Update Curve Editor Histogram
        if (this.curveEditor && this.image) {
            this.curveEditor.updateHistogram(this.image);
        }

        // v4.1: Refresh pipeline precision badge whenever a new image is loaded
        this._updateBitDepthBadge();
    }

    async loadStandardImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';

            img.onload = () => {
                this.image = img;
                this.imageWidth = img.width;
                this.imageHeight = img.height;

                // Create ImageData for CPU processing
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = img.width;
                tempCanvas.height = img.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(img, 0, 0);
                this.imageData = tempCtx.getImageData(0, 0, img.width, img.height).data;

                // Load to WebGL renderer
                if (this.renderer) {
                    this.renderer.loadImageTexture(img);
                }

                resolve();
            };

            img.onerror = reject;
            img.src = src;
        });
    }

    // v3.5: Load HDR data — supports .rhdr (fp16), .npy (f32), .exr, .hdr (RGBE),
    // .tif/.tiff (16/32-bit), and .f32 (RF32 binary). Format is auto-detected
    // from magic bytes by _parseHDRBuffer — no extension matching required.
    async loadHDRData(hdr_path) {
        const response = await fetch(hdr_path);
        const arrayBuffer = await response.arrayBuffer();

        const parsed = await this._parseHDRBuffer(arrayBuffer);

        if (!parsed) throw new Error('Failed to parse HDR sidecar');

        const height = parsed.shape[0];
        const width = parsed.shape[1];
        const channels = parsed.shape[2] || 3;

        const hdrData = {
            data: parsed.data,        // Float32Array (CPU reads, scopes, probe)
            fp16data: parsed.fp16data,    // Uint16Array (GPU HALF_FLOAT, RHDR only)
            width, height, channels,
            shape: parsed.shape,
            format: parsed.format || 'npy',
            isLinear: parsed.isLinear !== false  // true for all float/HDR formats
        };

        if (this.renderer) {
            if (hdrData.fp16data) {
                // .rhdr: prefer HALF_FLOAT — half the VRAM, 2× faster upload
                this.renderer.loadFloat16Texture(hdrData.fp16data, width, height, channels);
            } else {
                this.renderer.loadFloat32Texture(hdrData.data, width, height, channels);
            }
            // Override isLinearTexture for formats that are normalized SDR
            // (tiff_u8 is display-encoded; tiff_u16/f32, exr, rgbe, rf32 are scene-linear)
            if (hdrData.format === 'tiff_u8') {
                this.renderer.isLinearTexture = false;
            }
        }

        this.imageWidth = width;
        this.imageHeight = height;
        this.hdrData = hdrData;

        console.log(`[Radiance] Loaded ${hdrData.format.toUpperCase()} ${width}×${height}×${channels}ch (isLinear=${hdrData.isLinear})`);
        this.createPlaceholderImage(width, height);
        // v4.1: Update pipeline badge to reflect new HDR input precision
        this._updateBitDepthBadge();
    }

    // v3.1: _loadRHDR removed — consolidated into _parseRHDR (single implementation)
    // The loadHDRData method now uses _parseHDRBuffer → _parseRHDR.



    // v3.5: Load an image/HDR File object dropped onto the viewer.
    // Routes: float/HDR formats → _parseHDRBuffer → float32 GPU texture
    //         standard images (PNG/JPG) → ImageBitmap → loadImageTexture
    async _loadDroppedImageFile(file) {
        const name = file.name.toLowerCase();
        const FLOAT_EXTS = /\.(exr|hdr|tif|tiff|f32|rf32|rhdr|npy)$/;

        try {
            if (FLOAT_EXTS.test(name)) {
                // Float / HDR path
                const buffer = await file.arrayBuffer();
                const parsed = await this._parseHDRBuffer(buffer);
                if (!parsed) { console.warn('[Radiance] Failed to parse dropped file:', name); return; }

                const H = parsed.shape[0], W = parsed.shape[1], C = parsed.shape[2] || 3;
                this.hdrData = {
                    data: parsed.data, fp16data: parsed.fp16data,
                    width: W, height: H, channels: C,
                    shape: parsed.shape, format: parsed.format,
                    isLinear: parsed.isLinear !== false
                };
                this.imageWidth = W;
                this.imageHeight = H;

                if (this.renderer) {
                    if (parsed.fp16data) {
                        this.renderer.loadFloat16Texture(parsed.fp16data, W, H, C);
                    } else {
                        this.renderer.loadFloat32Texture(parsed.data, W, H, C);
                    }
                    if (parsed.format === 'tiff_u8') this.renderer.isLinearTexture = false;
                }

                this.createPlaceholderImage(W, H);
                this.fitToView(); this.render();
                this.updateScopes && this.updateScopes();
                this.updateInfo && this.updateInfo();
                console.log(`[Radiance] Loaded dropped ${parsed.format.toUpperCase()} ${W}×${H}×${C}ch`);

            } else {
                // Standard image (PNG, JPG, WebP) — use browser decode
                const bitmap = await createImageBitmap(file);
                this.hdrData = null;
                this.image = bitmap;
                this.imageWidth = bitmap.width;
                this.imageHeight = bitmap.height;
                if (this.renderer) this.renderer.loadImageTexture(bitmap);
                this.fitToView(); this.render();
                this.updateInfo && this.updateInfo();
                console.log(`[Radiance] Loaded dropped image ${bitmap.width}×${bitmap.height}`);
            }
        } catch (err) {
            console.error('[Radiance] Error loading dropped file:', err);
        }
    }

    createPlaceholderImage(width, height) {
        // Create a small placeholder for 2D canvas operations
        const placeholder = document.createElement('canvas');
        placeholder.width = width;
        placeholder.height = height;
        const ctx = placeholder.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#fff';
        ctx.font = '24px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('WebGL2 HDR Mode', width / 2, height / 2);

        this.image = placeholder;
        // v3.0 FIX: Store .data (Uint8ClampedArray), not ImageData object.
        // The rest of the code indexes this.imageData[i] for pixel values.
        this.imageData = ctx.getImageData(0, 0, width, height).data;
    }

    setCompareImage(img) {
        this.compareImage = img;
        this.diffCanvas = null; // Clear difference cache
        if (this.compareMode === 'none') this.compareMode = 'wipe';
        this.render();
    }

    togglePlayback() {
        if (this.videoMode && this.videoEl) {
            // Video element mode — delegate to native play/pause
            if (this.videoEl.paused) {
                this.videoEl.play();
            } else {
                this.videoEl.pause();
            }
            return; // isPlaying/icon updated via event listeners in loadVideo()
        }

        // Frame-sequence mode
        this.isPlaying = !this.isPlaying;
        this._updatePlayBtn();

        if (this.isPlaying) {
            this.lastFrameTime = performance.now();
            this._seqPlaybackLoop();
        }
    }

    _updatePlayBtn() {
        if (this.playBtn) this.playBtn.textContent = this.isPlaying ? '⏸' : '▶';
        if (this.videoMode && this.videoEl) {
            if (this.playBtn) this.playBtn.textContent = this.videoEl.paused ? '▶' : '⏸';
        }
    }

    // ── Frame-sequence playback loop (RAF, timing-compensated) ───────────────
    _seqPlaybackLoop() {
        if (!this.isPlaying || this.videoMode) return;

        const now = performance.now();
        const interval = 1000 / (this.playbackFps || 24);

        if (now - this.lastFrameTime >= interval) {
            this.nextFrame();
            this.lastFrameTime = now - ((now - this.lastFrameTime) % interval);
        }

        this._seqRAF = requestAnimationFrame(() => this._seqPlaybackLoop());
    }

    // ── Video file playback ───────────────────────────────────────────────────
    /**
     * Load a File or Blob (mp4/webm/mov/avi/mkv) into the viewer for real-time playback.
     * Frames are captured via an offscreen canvas and pushed into the WebGL renderer.
     */
    loadVideo(fileOrUrl) {
        this.unloadVideo(); // clean previous state

        const url = (fileOrUrl instanceof File || fileOrUrl instanceof Blob)
            ? URL.createObjectURL(fileOrUrl)
            : fileOrUrl;

        const vid = document.createElement('video');
        vid.src = url;
        vid.loop = this.loop;
        vid.muted = true; // required for autoplay policies
        vid.playsInline = true;
        vid.crossOrigin = 'anonymous';
        vid.preload = 'metadata';
        vid.playbackRate = this.playbackSpeed || 1.0;

        // Offscreen canvas for frame capture
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: false });

        this._videoCanvas = canvas;
        this._videoCtx = ctx;
        this.videoEl = vid;
        this.videoMode = true;
        this._videoUrl = url;

        vid.addEventListener('loadedmetadata', () => {
            canvas.width = vid.videoWidth;
            canvas.height = vid.videoHeight;
            this.totalFrames = Math.max(1, Math.round(vid.duration * (this._videoNativeFps || 25)));
            this.currentFrame = 0;
            this._updateVideoTimeline();
            this._updateScrubberRange();
        });

        vid.addEventListener('play', () => {
            this.isPlaying = true;
            this._updatePlayBtn();
            this._videoRenderLoop();
        });

        vid.addEventListener('pause', () => {
            this.isPlaying = false;
            this._updatePlayBtn();
            if (this._videoRAF) { cancelAnimationFrame(this._videoRAF); this._videoRAF = null; }
            // Still capture the paused frame
            this._captureVideoFrame();
        });

        vid.addEventListener('ended', () => {
            this.isPlaying = false;
            this._updatePlayBtn();
            if (this._videoRAF) { cancelAnimationFrame(this._videoRAF); this._videoRAF = null; }
        });

        vid.addEventListener('timeupdate', () => {
            // Update scrubber & timecode during playback
            this._updateVideoTimeline();
        });

        vid.addEventListener('seeked', () => {
            this._captureVideoFrame();
        });

        // Show video info in bottom bar
        vid.addEventListener('loadeddata', () => {
            this._updateVideoTimeline();
            this._captureVideoFrame();
        });

        this.videoEl = vid;
        this._updatePlayBtn();

        // Show transport if hidden
        if (this.transportPanel) this.transportPanel.style.display = 'flex';
    }

    unloadVideo() {
        if (this._videoRAF) { cancelAnimationFrame(this._videoRAF); this._videoRAF = null; }
        if (this.videoEl) {
            this.videoEl.pause();
            this.videoEl.src = '';
            this.videoEl.load();
            this.videoEl = null;
        }
        if (this._videoUrl && this._videoUrl.startsWith('blob:')) {
            URL.revokeObjectURL(this._videoUrl);
        }
        this._videoUrl = null;
        this._videoCanvas = null;
        this._videoCtx = null;
        this.videoMode = false;
        this.isPlaying = false;
        this._updatePlayBtn();
    }

    // ── RAF loop for video frame capture → WebGL ─────────────────────────────
    _videoRenderLoop() {
        if (!this.videoMode || !this.videoEl || this.videoEl.paused || this.videoEl.ended) {
            this._videoRAF = null;
            return;
        }
        this._captureVideoFrame();
        this._videoRAF = requestAnimationFrame(() => this._videoRenderLoop());
    }

    _captureVideoFrame() {
        const vid = this.videoEl;
        const canvas = this._videoCanvas;
        const ctx = this._videoCtx;
        if (!vid || !canvas || !ctx || vid.readyState < 2) return;

        if (canvas.width !== vid.videoWidth || canvas.height !== vid.videoHeight) {
            canvas.width = vid.videoWidth;
            canvas.height = vid.videoHeight;
        }

        ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);

        // Push into WebGL renderer via ImageBitmap for zero-copy GPU path
        if (this.renderer) {
            if (typeof createImageBitmap !== 'undefined') {
                createImageBitmap(canvas).then(bmp => {
                    if (this.renderer && this.videoMode) {
                        this.renderer.loadImageTexture(bmp);
                        bmp.close();
                        this.render();
                    }
                }).catch(() => {
                    // Fallback: direct canvas
                    if (this.renderer && this.videoMode) {
                        this.renderer.loadImageTexture(canvas);
                        this.render();
                    }
                });
            } else {
                this.renderer.loadImageTexture(canvas);
                this.render();
            }
        }

        this.imageWidth = canvas.width;
        this.imageHeight = canvas.height;
        this._updateVideoTimeline();
    }

    _updateVideoTimeline() {
        const vid = this.videoEl;
        if (!vid) return;
        const t = vid.currentTime;
        const dur = vid.duration || 1;
        const pct = t / dur;

        // Update scrubber
        if (this._videoScrubber) this._videoScrubber.value = String(Math.round(pct * 10000));

        // Update timecode display
        const fmt = (s) => {
            const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
            const fr = Math.floor((s % 1) * (this._videoNativeFps || 25));
            return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}:${String(fr).padStart(2, '0')}`;
        };
        if (this._videoTimecode) this._videoTimecode.textContent = fmt(t);
        if (this._videoDuration) this._videoDuration.textContent = fmt(dur);

        // Frame counter approximation
        const fps = this._videoNativeFps || 25;
        const approxFrame = Math.round(t * fps);
        if (this.frameCounter) this.frameCounter.textContent = `${approxFrame} / ${Math.round(dur * fps)}`;
    }

    _updateScrubberRange() {
        if (this._videoScrubber) {
            this._videoScrubber.min = '0';
            this._videoScrubber.max = '10000';
            this._videoScrubber.step = '1';
        }
    }

    // ── Video seek to position ────────────────────────────────────────────────
    seekVideoTo(pct) {
        if (!this.videoEl || isNaN(this.videoEl.duration)) return;
        this.videoEl.currentTime = Math.max(0, Math.min(1, pct)) * this.videoEl.duration;
    }

    // playbackLoop stays as alias for the sequence version
    playbackLoop() { this._seqPlaybackLoop(); }

    nextFrame() {
        if (!this.totalFrames || this.totalFrames <= 1) return;

        let next = this.currentFrame + 1;
        if (next >= this.totalFrames) {
            if (this.loop) {
                next = 0;
            } else {
                next = this.totalFrames - 1;
                this.isPlaying = false;
                if (this.playBtn) this.playBtn.textContent = '▶';
            }
        }

        this.setFrame(next);
    }

    prevFrame() {
        if (!this.totalFrames || this.totalFrames <= 1) return;

        let prev = this.currentFrame - 1;
        if (prev < 0) {
            if (this.loop) {
                prev = this.totalFrames - 1;
            } else {
                prev = 0;
            }
        }

        this.setFrame(prev);
    }

    setFrame(idx) {
        if (idx === this.currentFrame) return;
        this.currentFrame = idx;

        // Update Display
        if (this.frameHDRData[idx]) {
            // We have HDR data for this frame
            const npy = this.frameHDRData[idx];
            this.hdrData = npy;
            this.imageWidth = npy.width;
            this.imageHeight = npy.height;

            if (this.renderer) {
                if (npy.fp16data) {
                    this.renderer.loadFloat16Texture(npy.fp16data, npy.width, npy.height, npy.channels);
                } else {
                    this.renderer.loadFloat32Texture(npy.data, npy.width, npy.height, npy.channels);
                }
            }
        } else if (this.frameImages[idx]) {
            // Fallback to PNG
            this.hdrData = null;
            this.image = this.frameImages[idx];
            if (this.renderer) this.renderer.loadImageTexture(this.image);
        }

        // Update Z-Depth for the new frame
        if (this.frameZdepthImages && this.frameZdepthImages[idx]) {
            this.zdepthImage = this.frameZdepthImages[idx];
            if (this.renderer) this.renderer.loadDepthTexture(this.zdepthImage);
        } else {
            this.zdepthImage = null;
        }

        this.render();
        this.updateInfo();
        this.updateFrameDisplay();

        // Update Scopes
        // Note: Real-time scopes update from displayed texture, so just calling updateScopes() is enough
        // assuming updateScopes pulls from renderer's texture.
        if (this.activeTab === 'scopes') {
            this.renderScopesTab(this.tabContentContainer);
        }
    }

    updateFrameDisplay() {
        if (this.videoMode) return; // video mode manages its own timeline

        // Update frame counter text
        if (this.frameCounter && this.frameCounter.tagName !== 'INPUT') {
            this.frameCounter.textContent = `${this.currentFrame + 1} / ${this.totalFrames}`;
        }

        // Sync scrubber position
        if (this._videoScrubber && this.totalFrames > 1) {
            const pct = this.currentFrame / Math.max(1, this.totalFrames - 1);
            this._videoScrubber.value = String(Math.round(pct * 10000));
            this._videoScrubber.style.background =
                `linear-gradient(to right, ${this.theme.accent} ${pct * 100}%, rgba(255,255,255,0.15) ${pct * 100}%)`;
        }

        // Timecode in HH:MM:SS:FF format based on playbackFps
        if (this._videoTimecode && this.playbackFps) {
            const fps = this.playbackFps;
            const totalSec = this.currentFrame / fps;
            const h = Math.floor(totalSec / 3600);
            const m = Math.floor((totalSec % 3600) / 60);
            const s = Math.floor(totalSec % 60);
            const f = this.currentFrame % Math.round(fps);
            this._videoTimecode.textContent =
                `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}:${String(f).padStart(2, '0')}`;
        }

        if (this._videoDuration && this.playbackFps && this.totalFrames > 1) {
            const fps = this.playbackFps;
            const totalSec = (this.totalFrames - 1) / fps;
            const h = Math.floor(totalSec / 3600);
            const m = Math.floor((totalSec % 3600) / 60);
            const s = Math.floor(totalSec % 60);
            const f = (this.totalFrames - 1) % Math.round(fps);
            this._videoDuration.textContent =
                `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}:${String(f).padStart(2, '0')}`;
        }

        // Show transport if we have frames
        if (this.totalFrames > 1 && this.transportPanel) {
            this.transportPanel.style.display = 'flex';
        }
    }

    // Existing fitToView...
    fitToView() {
        if (!this.image) return;
        const w = this.canvas.width, h = this.canvas.height;
        if (w < 10 || h < 10) {
            // Retry if canvas is somehow not ready
            requestAnimationFrame(() => this.fitToView());
            return;
        }

        let z = Math.min(w / this.imageWidth, h / this.imageHeight);
        // Add a small margin (5%) if it's tight
        z = z * 0.95;

        this.zoom = z;
        this.panX = (w - this.imageWidth * this.zoom) / 2;
        this.panY = (h - this.imageHeight * this.zoom) / 2;
        this.updateBottomBar();
        this.render();
    }

    setZoom(z) {
        const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
        // Zoom towards center of view, not 0,0
        const oldZ = this.zoom;

        // Calculate world point at center
        const wx = (cx - this.panX) / oldZ;
        const wy = (cy - this.panY) / oldZ;

        this.zoom = z;
        // Recalculate pan to keep world point at center
        this.panX = cx - wx * this.zoom;
        this.panY = cy - wy * this.zoom;

        this.updateInfo();
        this.render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RENDERING
    // ═══════════════════════════════════════════════════════════════════════════

    // v3.5: Throttled rendering entry point
    requestRender() {
        if (this._renderRequested) return;
        this._renderRequested = true;
        requestAnimationFrame(() => {
            this._renderRequested = false;
            this.render();
        });
    }

    // v3.5: Debounced scope update
    requestScopeUpdate() {
        if (this._scopeUpdateTimer) clearTimeout(this._scopeUpdateTimer);
        this._scopeUpdateTimer = setTimeout(() => {
            this.updateScopes();
            this._scopeUpdateTimer = null;
        }, this.scopeDebounceMs || 150);
    }

    render() {
        if (this.canvas.width === 0 || this.canvas.height === 0) return;

        // ═══════════════════════════════════════════════════════════════
        // GPU-ACCELERATED WEBGL RENDERING PATH (Primary - Always On)
        // ═══════════════════════════════════════════════════════════════
        const useWebGL = this.useWebGL && this.renderer && this.renderer.textures.image;

        if (useWebGL) {
            // Ensure WebGL canvas is backend-only (hidden)
            this.glCanvas.style.visibility = 'hidden';

            // 1. Resize/Init WebGL Canvas to Image Size (Texture size)
            if (this.glCanvas.width !== this.imageWidth || this.glCanvas.height !== this.imageHeight) {
                this.glCanvas.width = this.imageWidth;
                this.glCanvas.height = this.imageHeight;
            }

            // 2. Update renderer state from UI controls (GPU parameters)
            this.renderer.setExposure(this.exposure || 0.0);
            const gArr = Array.isArray(this.gamma) ? this.gamma : [this.gamma || 1, this.gamma || 1, this.gamma || 1];
            this.renderer.setGamma(gArr[0], gArr[1], gArr[2]);
            this.renderer.setSaturation(this.saturation !== undefined ? this.saturation : 1.0);

            // Analytics State
            this.renderer.setFalseColor(this.falseColor || false);
            this.renderer.setZebra(this.zebra || false);
            this.renderer.setZebraThreshold(this.zebraThreshold || 0.95);
            this.renderer.setGamutWarning(this.gamutWarning || false);
            this.renderer.setClippingMonitor(this.clippingMonitor || false);

            // v3.1: Masking (Power Windows)
            this.renderer.setMask(this.maskState);

            // v3.4 FIX: Qualifier must be pushed every frame, not just when the masks tab is open.
            // Pass showMask=true only while the masks tab is active — switching away hides the overlay.
            if (this.renderer.setQualifier && this.qualifierState) {
                const qState = this.activeTab === 'masks'
                    ? this.qualifierState
                    : { ...this.qualifierState, showMask: false };
                this.renderer.setQualifier(qState);
            }

            // v2.2: Channel isolation + focus peaking + display LUT on GPU
            const chMap = { 'rgb': 0, 'r': 1, 'g': 2, 'b': 3, 'luma': 4, 'a': 5 };
            this.renderer.setChannelMode(chMap[this.channel] || 0);
            this.renderer.setFocusPeaking(this.focusPeaking || false, this.focusPeakingThreshold || 120);
            const lutMap = {
                'None': 0,
                'sRGB (Display)': 1,
                'Rec.709 (Broadcast)': 2,
                'Filmic (Cinematic)': 3,
                'Log C3 (ARRI)': 4,
                'Log C4 (ARRI)': 11,
                'Canon Log': 12,
                'Fuji F-Log': 14,
                'Blackmagic Gen5': 15,
                'Linear to Log': 6,
                'Reinhard Tonemap': 8,
                'ACES Filmic': 9,
                'LogC3 (ARRI EI800)': 4,
                'LogC4 (ARRI Alexa 35)': 11,
                'F-Log2 (Fujifilm)': 14,
                'C-Log3 (Canon)': 12,
                'Log3G10 (RED IPP2)': 19,
                'DaVinci Intermediate': 20,
                'BMD Film Gen5': 15,
                'V-Log (Panasonic)': 13,
                'RED Log3G10': 19,
                'N-Log (Nikon)': 21,
                'Linear to Log (Generic)': 6,
                'IDT: LogC3 → Linear': 29, 
                'IDT: LogC4 → Linear': 22,
                'IDT: V-Log → Linear': 30, 
                'IDT: Log3G10 → Linear': 25,
                'IDT: DaVinci → Linear': 26,
                'IDT: BMD Gen5 → Linear': 27,
                'IDT: N-Log → Linear': 28,
                'IDT: F-Log2 → Linear': 24,
                'IDT: C-Log3 → Linear': 23
            };
            this.renderer.setDisplayLutMode(lutMap[this.displayLut] || 0);
            this.renderer.setInputLutMode(lutMap[this.inputSpace] || 0);

            // v2.3: Denoise & Depth Eval
            this.renderer.setDenoise(this.denoise || 0.0);
            this.renderer.setShowDepth(this.showZdepth || false);

            // v3.4: Printer Lights + Soft Clip
            this.renderer.setPrinterLights(this.printerR || 0, this.printerG || 0, this.printerB || 0);
            this.renderer.setSoftClip(this.softClip || 0.0);

            // ── Effects panel: push ALL state every frame ──────────────────
            // Grain
            this.renderer.setGrain(this.grain || 0.0);
            this.renderer.setGrainSize(this.grainSize || 1.0);
            this.renderer.setGrainColor(this.grainColor || 0.0);
            this.renderer.setGrainAnimate(this.grainAnimate || false);
            // Bloom / Halation / Diffusion
            this.renderer.setBloom(this.bloom || 0.0);
            if (this.renderer && this.renderer.setBloomThreshold) this.renderer.setBloomThreshold(this.bloomThreshold ?? 1.0);
            this.renderer.setHalation(this.halation || 0.0);
            if (this.renderer && this.renderer.setHalationRadius) this.renderer.setHalationRadius(this.halationRadius ?? 1.0);
            if (this.renderer && this.renderer.setHalationThreshold) this.renderer.setHalationThreshold(this.halationThreshold ?? 0.35);
            this.renderer.setDiffusion(this.diffusion || 0.0);
            // Lens Distortion + Chromatic Aberration
            this.renderer.setLensDistortion(this.lensDistortion || 0.0, this.lensFringe || 0.0);
            // Vignette
            this.renderer.setVignette(this.vignetteIntensity || 0.0, this.vignetteFalloff !== undefined ? this.vignetteFalloff : 0.5);
            // Bokeh physics (always, not only when DoF enabled — affects CA+highlight)
            this.renderer.setBokehPhysics(this.bokehHighlightBias || 0.0, this.bokehSoapBubble || 0.0, this.bokehOpticalVig || 0.0);
            // Aperture shape (always pushed for CA / anamorphic ratio in fringe mode)
            this.renderer.setApertureShape(this.apertureBlades || 0, this.apertureRotation || 0.0, this.apertureAnamorphic || 1.0);
            // Anamorphic streaks
            if (this.renderer && this.renderer.setAnamorphicStreaks) this.renderer.setAnamorphicStreaks(this.anamorphicStreaks || 0.0);
            if (this.renderer && this.renderer.setStreakThreshold) this.renderer.setStreakThreshold(this.streakThreshold ?? 0.85);
            if (this.renderer && this.renderer.setStreakLength) this.renderer.setStreakLength(this.streakLength ?? 0.08);

            // Time + frame must always be updated so grain, zebra-blink, and other
            // time-driven effects animate even when DoF is disabled.
            this.renderer.setTime(performance.now() / 1000.0);
            this.renderer.setFrame(this.currentFrame || 0);

            // DoF controls
            if (this.dofEnabled && this.renderer.textures.depth) {
                this.renderer.setDoFEnabled(true);
                this.renderer.setFocusDistance(this.focusDistance || 0.5);
                this.renderer.setAperture(this.aperture || 0.0);
            } else {
                this.renderer.setDoFEnabled(false);
            }

            // ── A/B Wipe: pass position to GPU shader every frame ──────────
            if (this.compareMode === 'wipe') {
                this.renderer.setWipe(this.wipePosition, true);
            } else {
                this.renderer.setWipe(0.5, false);
            }

            // Render to WebGL canvas (GPU)
            const lutStrength = this.lutIntensity !== undefined ? this.lutIntensity : 1.0;
            this.renderer.render(lutStrength);

            // 3. Draw WebGL output to Main Canvas via 2D context with transform
            const ctx = this.ctx;
            const w = this.canvas.width, h = this.canvas.height;

            ctx.fillStyle = this.theme.bg;
            ctx.fillRect(0, 0, w, h);

            ctx.save();
            ctx.translate(this.panX, this.panY);
            ctx.scale(this.zoom, this.zoom);

            // Draw the GPU-rendered image
            ctx.drawImage(this.glCanvas, 0, 0);

            ctx.restore();

            // Overlay the crisp split-line + labels on top of the GL output
            this._drawWipeSplitOverlay(ctx, w, h);

            this.updateBottomBar();
            this.renderOverlay();
            return; // WebGL path complete
        } else {
            if (this.glCanvas) this.glCanvas.style.visibility = 'hidden';
        }

        // ═══════════════════════════════════════════════════════════════
        // 2D CANVAS FALLBACK (CPU rendering - legacy/fallback only)
        // ═══════════════════════════════════════════════════════════════
        const ctx = this.ctx;
        const w = this.canvas.width, h = this.canvas.height;
        if (w === 0 || h === 0) return;

        ctx.fillStyle = this.theme.bg;
        ctx.fillRect(0, 0, w, h);

        if (!this.image) {
            ctx.fillStyle = this.theme.textDim;
            ctx.font = '12px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No image loaded', w / 2, h / 2);
            return;
        }

        // Ensure high quality scaling
        ctx.imageSmoothingEnabled = this.zoom < 1.0; // Smooth when downscaling, pixelated when upscaling?
        if (this.zoom > 2.0) ctx.imageSmoothingEnabled = false; // Pixel art look for high zoom
        else ctx.imageSmoothingQuality = 'high';

        const cmpImg = this.compareImage || (this.frameImages && this.frameImages[this.currentFrame]) || this.videoEl || this.image;

        if (this.compareMode === 'sidebyside' && cmpImg) {
            this.renderSideBySide(ctx, w, h, cmpImg);
        } else if (this.compareMode === 'difference' && cmpImg) {
            this.renderDifference(ctx, w, h, cmpImg);
        } else {
            ctx.save();
            ctx.translate(this.panX, this.panY);
            ctx.scale(this.zoom, this.zoom);
            this.renderImage(ctx, this.image);
            ctx.restore();

            // Canvas fallback wipe: for non-WebGL mode, draw the split overlay only.
            // The actual pixel split is handled by the GPU path above; in SDR canvas
            // mode there is no graded B frame so we just show the overlay.
            if (this.compareMode === 'wipe') this._drawWipeSplitOverlay(ctx, w, h);
        }

        this.updateBottomBar();
        this.renderOverlay();
        this._updateAnalysisIndicator();
    }

    // P1: Visual viewport border for active analysis modes
    _updateAnalysisIndicator() {
        let borderColor = 'transparent';
        let label = '';
        if (this.falseColor) { borderColor = '#FFD600'; label = 'FALSE COLOR'; }
        else if (this.zebra) { borderColor = '#FF4444'; label = 'ZEBRA'; }
        else if (this.focusPeaking) { borderColor = '#FF6600'; label = 'FOCUS PEAK'; }
        else if (this.showZdepth) { borderColor = '#00BBFF'; label = 'Z-DEPTH'; }
        else if (this.gamutWarning) { borderColor = '#FF00FF'; label = 'GAMUT WARN'; }
        else if (this.clippingMonitor) { borderColor = '#FF2222'; label = 'CLIPPING'; }

        this.canvasWrapper.style.boxShadow = borderColor !== 'transparent'
            ? `inset 0 0 0 2px ${borderColor}`
            : 'none';

        // Floating label in top-right of viewport
        if (!this._analysisLabel) {
            this._analysisLabel = document.createElement('div');
            this._analysisLabel.style.cssText = `
                position: absolute; top: 6px; right: 6px; z-index: 95;
                padding: 2px 8px; border-radius: 3px; font-size: 10px;
                font-family: ${this.theme?.mono || 'monospace'}; font-weight: 600;
                letter-spacing: 0.5px; pointer-events: none; transition: opacity 0.15s;
            `;
            this.canvasWrapper.appendChild(this._analysisLabel);
        }
        if (label) {
            this._analysisLabel.textContent = label;
            this._analysisLabel.style.background = borderColor;
            this._analysisLabel.style.color = '#000';
            this._analysisLabel.style.opacity = '1';
        } else {
            this._analysisLabel.style.opacity = '0';
        }

        // Auto-tonemap indicator (shows when HDR + no explicit display LUT)
        if (!this._autoTonemapBadge) {
            this._autoTonemapBadge = document.createElement('div');
            this._autoTonemapBadge.style.cssText = `
                position: absolute; top: 6px; left: 6px; z-index: 95;
                padding: 2px 8px; border-radius: 3px; font-size: 10px;
                font-family: ${this.theme?.mono || 'monospace'}; font-weight: 600;
                letter-spacing: 0.5px; pointer-events: none; transition: opacity 0.15s;
                background: rgba(255, 160, 0, 0.85); color: #000;
            `;
            this.canvasWrapper.appendChild(this._autoTonemapBadge);
        }
        const isAutoACES = this.renderer && this.renderer.isLinearTexture &&
            (this.displayLut === 'None' || !this.displayLut);
        if (isAutoACES) {
            this._autoTonemapBadge.textContent = 'ACES (Auto)';
            this._autoTonemapBadge.style.opacity = '1';
        } else {
            this._autoTonemapBadge.style.opacity = '0';
        }
    }

    renderImage(ctx, img) {
        // Z-Depth override
        if (this.showZdepth && this.zdepthImage) {
            ctx.drawImage(this.zdepthImage, 0, 0);
            return;
        }

        const gammaScalar = Array.isArray(this.gamma) ? (this.gamma[0] + this.gamma[1] + this.gamma[2]) / 3 : (this.gamma || 1.0);
        if (this.exposure !== 0 || gammaScalar !== 1.0 || this.channel !== 'rgb' || this.falseColor || this.zebra || this.focusPeaking || this.displayLut !== 'None') {
            this.renderProcessed(ctx, img);
        } else {
            ctx.drawImage(img, 0, 0);
        }
    }

    // renderWipe: deprecated — wipe is now handled entirely by the WebGL shader
    // and _drawWipeSplitOverlay(). Kept as a no-op to avoid call-site errors.
    renderWipe(ctx, w, h, cmpImg) {}

    renderSideBySide(ctx, w, h, cmpImg) {
        const hw = w / 2;
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, hw, h); ctx.clip();
        ctx.translate(this.panX * 0.5, this.panY); ctx.scale(this.zoom * 0.5, this.zoom);
        ctx.drawImage(this.renderer ? this.renderer.canvas : this.image, 0, 0); ctx.restore();

        ctx.save(); ctx.beginPath(); ctx.rect(hw, 0, hw, h); ctx.clip();
        ctx.translate(hw + this.panX * 0.5, this.panY); ctx.scale(this.zoom * 0.5, this.zoom);
        ctx.drawImage(cmpImg, 0, 0); ctx.restore();

        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(hw, 0); ctx.lineTo(hw, h); ctx.stroke();
    }

    renderDifference(ctx, w, h, cmpImg) {
        if (!this.diffCanvas) {
            this.diffCanvas = document.createElement('canvas');
            this.diffCanvas.width = this.imageWidth;
            this.diffCanvas.height = this.imageHeight;
            const dCtx = this.diffCanvas.getContext('2d');

            // Draw A
            dCtx.drawImage(this.renderer ? this.renderer.canvas : this.image, 0, 0);
            const dA = dCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);

            // Draw B to temporary canvas to get data
            const tmp = document.createElement('canvas');
            tmp.width = this.imageWidth; tmp.height = this.imageHeight;
            const tCtx = tmp.getContext('2d');
            tCtx.drawImage(cmpImg, 0, 0);
            const dB = tCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);

            // Compute Difference
            for (let i = 0; i < dA.data.length; i += 4) {
                const r = Math.abs(dA.data[i] - dB.data[i]);
                const g = Math.abs(dA.data[i + 1] - dB.data[i + 1]);
                const b = Math.abs(dA.data[i + 2] - dB.data[i + 2]);

                // Visualization: Boost difference
                dA.data[i] = Math.min(255, r * 4);
                dA.data[i + 1] = Math.min(255, g * 4);
                dA.data[i + 2] = Math.min(255, b * 4);
                dA.data[i + 3] = 255;
            }
            dCtx.putImageData(dA, 0, 0);
        }

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        ctx.drawImage(this.diffCanvas, 0, 0);
        ctx.restore();
    }

    renderProcessed(ctx, img) {
        // Reuse offscreen canvas to avoid severe GC pressure (33MB+ per 4K frame)
        if (!this._processCanvas) {
            this._processCanvas = document.createElement('canvas');
        }
        const off = this._processCanvas;
        if (off.width !== this.imageWidth || off.height !== this.imageHeight) {
            off.width = this.imageWidth; off.height = this.imageHeight;
        }
        const offCtx = off.getContext('2d');
        offCtx.drawImage(img, 0, 0);

        const imageData = offCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);
        const data = imageData.data;
        const expMult = Math.pow(2, this.exposure);
        const gammaAvg = Array.isArray(this.gamma) ? (this.gamma[0] + this.gamma[1] + this.gamma[2]) / 3 : (this.gamma || 1.0);
        const invGamma = 1.0 / gammaAvg;

        for (let i = 0; i < data.length; i += 4) {
            let r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;

            // v2.2: Linearize sRGB PNG data before grading (fixes double-gamma bug)
            const srgb2lin = (c) => c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
            r = srgb2lin(r); g = srgb2lin(g); b = srgb2lin(b);

            // Apply exposure in linear space
            r *= expMult; g *= expMult; b *= expMult;

            // Apply LUT / Color Space
            if (this.displayLut !== 'None') {
                [r, g, b] = this.applyColorTransform(r, g, b, this.displayLut);
            }

            // Apply Gamma
            if (gammaAvg !== 1.0) {
                r = Math.pow(Math.max(0, r), invGamma);
                g = Math.pow(Math.max(0, g), invGamma);
                b = Math.pow(Math.max(0, b), invGamma);
            }

            if (this.channel === 'r') { g = r; b = r; }
            else if (this.channel === 'g') { r = g; b = g; }
            else if (this.channel === 'b') { r = b; g = b; }
            else if (this.channel === 'a') {
                // Alpha channel - show as grayscale with checkerboard for transparency
                const a = data[i + 3] / 255;
                const x = (i / 4) % this.imageWidth, y = Math.floor((i / 4) / this.imageWidth);
                const checker = ((Math.floor(x / 8) + Math.floor(y / 8)) % 2) === 0 ? 0.3 : 0.5;
                r = g = b = a * 1.0 + (1 - a) * checker;
            }
            else if (this.channel === 'luma') { const l = r * 0.2126 + g * 0.7152 + b * 0.0722; r = g = b = l; }

            if (this.falseColor) {
                const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
                const fc = this.getFalseColor(l);
                r = fc.r; g = fc.g; b = fc.b;
            }

            if (this.zebra) {
                const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
                const x = (i / 4) % this.imageWidth, y = Math.floor((i / 4) / this.imageWidth);
                if (l > 0.95 && (x + y) % 8 < 4) { r = 1; g = 0; b = 0; }
                if (l < 0.02 && (x + y) % 8 < 4) { r = 0; g = 0; b = 1; }
            }

            // Clip final values
            data[i] = Math.min(255, Math.max(0, r * 255));
            data[i + 1] = Math.min(255, Math.max(0, g * 255));
            data[i + 2] = Math.min(255, Math.max(0, b * 255));
        }

        // Focus peaking - apply as post-process for edge detection
        if (this.focusPeaking) {
            this.applyFocusPeaking(imageData, this.imageWidth, this.imageHeight);
        }

        offCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(off, 0, 0);
    }

    applyFocusPeaking(imageData, width, height) {
        const data = imageData.data;
        const threshold = this.focusPeakingThreshold;

        // Create edge buffer
        const edges = new Uint8Array(width * height);

        // Sobel edge detection on luminance
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;

                // Get luminance of surrounding pixels
                const getLuma = (ox, oy) => {
                    const i = ((y + oy) * width + (x + ox)) * 4;
                    return data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
                };

                // Sobel operators
                const gx = -getLuma(-1, -1) + getLuma(1, -1) +
                    -2 * getLuma(-1, 0) + 2 * getLuma(1, 0) +
                    -getLuma(-1, 1) + getLuma(1, 1);

                const gy = -getLuma(-1, -1) - 2 * getLuma(0, -1) - getLuma(1, -1) +
                    getLuma(-1, 1) + 2 * getLuma(0, 1) + getLuma(1, 1);

                const magnitude = Math.sqrt(gx * gx + gy * gy);
                edges[y * width + x] = magnitude > threshold ? 1 : 0;
            }
        }

        // Apply peaks as colored overlay
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (edges[y * width + x]) {
                    const idx = (y * width + x) * 4;
                    // Red highlight for sharp edges
                    data[idx] = 255;
                    data[idx + 1] = Math.floor(data[idx + 1] * 0.3);
                    data[idx + 2] = Math.floor(data[idx + 2] * 0.3);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          METADATA OVERLAY
    // ═══════════════════════════════════════════════════════════════════════════

    createMetadataOverlay() {
        if (this.metadataOverlay) return; // Already created — prevent duplicates

        this.metadataOverlay = document.createElement('div');
        this.metadataOverlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            padding: 10px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            z-index: 90;
        `;

        // Top Left: Resolution + Bit Depth
        this.metaTL = document.createElement('div');
        this.metaTL.style.cssText = 'color: rgba(255,255,255,0.7); font-family: monospace; font-size: 11px; text-shadow: 1px 1px 2px #000;';
        this.metadataOverlay.appendChild(this.metaTL);

        // This info will be populated by updateInfo()

        // Top Right: Zoom / Pan (optional, maybe just Zoom)
        this.metaTR = document.createElement('div');
        this.metaTR.style.cssText = 'color: rgba(255,255,255,0.7); font-family: monospace; font-size: 11px; text-shadow: 1px 1px 2px #000; align-self: flex-end; position: absolute; top: 10px; right: 10px;';
        this.metadataOverlay.appendChild(this.metaTR);

        // Add to container (behind controls, above image)
        this.container.appendChild(this.metadataOverlay);

        this.container.addEventListener('wheel', () => {
            // Update Zoom display on scroll
            requestAnimationFrame(() => this.updateInfo());
        }, { passive: true });
    }

    getFalseColor(l) {
        if (l < 0.01) return { r: 0.1, g: 0, b: 0.3 };
        if (l < 0.08) return { r: 0, g: 0, b: 0.8 };
        if (l < 0.20) return { r: 0.3, g: 0.3, b: 0.3 };
        if (l < 0.40) return { r: 0, g: 0.7, b: 0 };
        if (l < 0.60) return { r: 0.8, g: 0.8, b: 0 };
        if (l < 0.80) return { r: 1, g: 0.5, b: 0 };
        if (l < 0.95) return { r: 1, g: 0, b: 0 };
        return { r: 1, g: 0, b: 1 };
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RUN & PROMPT
    // ═══════════════════════════════════════════════════════════════════════════

    runWorkflow() {
        if (this.isQueueing) return;
        this.isQueueing = true;

        const oldText = this.runButton ? this.runButton.innerHTML : '▶ RUN';
        if (this.runButton) this.runButton.innerHTML = '⏳ QUEUING...';

        app.queuePrompt(0).then(() => {
            this.isQueueing = false;
            if (this.runButton) this.runButton.innerHTML = oldText;
        }).catch(() => {
            this.isQueueing = false;
            if (this.runButton) {
                this.runButton.innerHTML = '❌ ERROR';
                setTimeout(() => this.runButton.innerHTML = oldText, 1500);
            }
        });
    }

    updateInfo() {
        if (!this.image) {
            this.dimensionInfo.textContent = '—×—';
            this.zoomInfo.textContent = '—';
            return;
        }

        const z = (this.zoom * 100).toFixed(0);
        this.dimensionInfo.textContent = `${this.imageWidth}×${this.imageHeight}`;
        this.zoomInfo.textContent = `${z}%`;

        // v4.1: Pipeline precision badge (replaces old simple bit-depth text)
        this._updateBitDepthBadge();

        // Update metadata panel if visible
        if (this.metadataContent && this.metadataPanel.style.display !== 'none') {
            let hdrStatus = '—';
            let formatStr = 'PNG 8-bit';
            if (this.hdrData) {
                const hasHighValues = this.hdrData.data &&
                    Array.from(this.hdrData.data.slice(0, 1000)).some(v => v > 1.0 || v < 0.0);
                hdrStatus = hasHighValues ? '✓ HDR Content' : '✗ Standard Range';
                if (this.hdrData.format === 'rhdr') {
                    formatStr = 'RHDR fp16 (primary)';
                } else if (this.hdrData.format === 'rhdr_f32') {
                    formatStr = 'RHDR fp32 (primary)';
                } else {
                    formatStr = 'Float32 HDR';
                }
            } else if (this.imageData) {
                hdrStatus = '✗ Standard Range';
            }

            // v4.1: Get full pipeline info from renderer for metadata panel
            const pipeInfo = this.renderer ? this.renderer.getPipelineInfo() : null;
            const pipeLabel = pipeInfo ? pipeInfo.label : '—';
            const pipeColor = pipeInfo
                ? (pipeInfo.mode === 'f32' ? '#4ade80' : pipeInfo.mode === 'f16' ? '#60a5fa' : this.theme.textDim)
                : this.theme.textDim;

            // Also refresh the status bar badge
            this._updateBitDepthBadge();

            this.metadataContent.innerHTML = `
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">Resolution:</span><br/>
                    <span style="color:${this.theme.accent}">${this.imageWidth} × ${this.imageHeight}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.hdrData ? '#4ade80' : this.theme.text}">${formatStr}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">Pipeline:</span><br/>
                    <span style="color:${pipeColor}">${pipeLabel}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">Channels:</span><br/>
                    <span>${this.hdrData ? `${this.hdrData.channels || 3}ch` : 'RGBA (4)'}</span>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="color:${this.theme.textDim}">HDR Status:</span><br/>
                    <span style="color:${hdrStatus.includes('✓') ? '#4ade80' : '#94a3b8'}">${hdrStatus}</span>
                </div>
                <div>
                    <span style="color:${this.theme.textDim}">Zoom:</span><br/>
                    <span>${z}%</span>
                </div>
            `;
        }
    }

    togglePromptPanel() {
        this.showPromptPanel = !this.showPromptPanel;

        if (this.showPromptPanel) {
            if (this.promptButton) {
                this.promptButton.classList.add('active');
                this.promptButton.style.background = this.theme.accent;
                this.promptButton.style.color = '#fff';
            }
            // v2.4: Integrate with HUD — panel-embedded mode
            this.showControls = true;
            if (this.rightControlPanel) {
                this.rightControlPanel.style.display = 'flex';
            } else if (this.controlsPanel) {
                this.controlsPanel.style.display = 'flex';
                this.controlsPanel.style.opacity = '1';
                this.controlsPanel.style.pointerEvents = 'auto';
            }
            if (this.controlsToggle) {
                this.controlsToggle.style.color = this.theme.accent;
            }

            // Switch to prompt tab
            const promptTab = this._hudTabs?.find(t => t.dataset?.tabId === 'prompt');
            if (promptTab) {
                promptTab.click();
            }
        } else {
            if (this.promptButton) {
                this.promptButton.classList.remove('active');
                this.promptButton.style.background = '';
                this.promptButton.style.color = this.theme.textDim;
            }
            // If we were toggling prompt specifically, maybe we hide HUD?
            // Actually, stay consistent: if promptPanel is false, but showControls is true, keep HUD open but prompt tab is still active.
            // But usually 'P' toggles the prompt UI. If prompt is now in HUD, 'P' should toggle HUD.
            this.toggleControls(); // Toggle entire HUD for 'P' key consistency
            if (this.promptPanel) this.promptPanel.remove();
            this.promptPanel = null;
        }
    }

    renderPromptTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; flex: 1; gap: 16px; padding: 12px; min-height: 0; overflow-y: auto; color: #fff;';
        const t = this.theme;

        // v2.4: Move Run Button to top of prompt tab
        const runBtnWrapper = document.createElement('div');
        runBtnWrapper.style.cssText = 'padding: 5px 0 15px 0; display: flex; justify-content: center; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 5px;';

        const runBtn = document.createElement('button');
        runBtn.innerHTML = '▶ RUN WORKFLOW';
        runBtn.style.cssText = `
        background: linear-gradient(135deg, #134e13 0%, #0a2e0a 100%);
        color: #6eff6e;
        border: 1px solid rgba(110, 255, 110, 0.3);
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 13px;
        font-weight: 900;
        cursor: pointer;
        letter-spacing: 1.5px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        font-family: ${t.font};
        box-shadow: 0 4px 6px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.05);
        text-shadow: 0 0 10px rgba(110, 255, 110, 0.3);
    `;
        runBtn.onmouseover = () => {
            runBtn.style.background = 'linear-gradient(135deg, #1a631a 0%, #0d3b0d 100%)';
            runBtn.style.border = '1px solid rgba(110, 255, 110, 0.5)';
            runBtn.style.boxShadow = '0 0 20px rgba(79, 255, 79, 0.15), 0 6px 8px rgba(0,0,0,0.4)';
            runBtn.style.transform = 'translateY(-1px)';
        };
        runBtn.onmouseout = () => {
            runBtn.style.background = 'linear-gradient(135deg, #134e13 0%, #0a2e0a 100%)';
            runBtn.style.border = '1px solid rgba(110, 255, 110, 0.3)';
            runBtn.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.05)';
            runBtn.style.transform = 'none';
        };
        this.runButton = runBtn;
        runBtn.onclick = () => this.runWorkflow();

        runBtnWrapper.appendChild(runBtn);
        container.appendChild(runBtnWrapper);

        // Get Generation Settings Nodes
        const nodes = app.graph._nodes.filter(n => {
            const nodeType = n.type || "";
            const comfyClass = n.comfyClass || nodeType;

            const isEncoder = comfyClass.includes("CinematicPromptEncoder");
            const isUnet = comfyClass === "CheckpointLoaderSimple" || comfyClass === "CheckpointLoader" || comfyClass === "UNETLoader" || comfyClass.includes("DualCLIPLoader") || comfyClass === "RadianceUnifiedLoader";
            const isLatent = comfyClass === "EmptyLatentImage" || comfyClass === "EmptySD3LatentImage" || comfyClass === "RadianceResolution";

            return isEncoder || isUnet || isLatent;
        });

        // Sort nodes to ensure CinematicPromptEncoder is rendered first, and RadianceUnifiedLoader is rendered last
        nodes.sort((a, b) => {
            const aClass = a.comfyClass || a.type || "";
            const bClass = b.comfyClass || b.type || "";

            const aIsEncoder = aClass.includes("CinematicPromptEncoder");
            const bIsEncoder = bClass.includes("CinematicPromptEncoder");
            const aIsUnified = aClass === "RadianceUnifiedLoader";
            const bIsUnified = bClass === "RadianceUnifiedLoader";

            if (aIsEncoder && !bIsEncoder) return -1;
            if (!aIsEncoder && bIsEncoder) return 1;

            if (aIsUnified && !bIsUnified) return 1;
            if (!aIsUnified && bIsUnified) return -1;

            return 0;
        });

        if (nodes.length === 0) {
            const msg = document.createElement('div');
            msg.textContent = "No Generation Nodes found.";
            msg.style.color = t.textDim;
            msg.style.fontSize = "11px";
            container.appendChild(msg);
            return;
        }

        nodes.forEach(node => {
            const wrapper = document.createElement('div');
            wrapper.style.cssText = `display: flex; flex-direction: column; gap: 12px; padding: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);`;

            const label = document.createElement('div');
            label.textContent = (node.title || node.type).toUpperCase();
            label.style.cssText = `color: ${t.accent}; font-size: 11px; font-weight: 900; cursor: pointer; letter-spacing: 1px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px; margin-bottom: 6px;`;
            label.onclick = () => { app.canvas.centerOnNode(node); app.canvas.selectNode(node); };
            wrapper.appendChild(label);

            if (node.widgets) {
                // Main text area (base_prompt)
                const mainPrompt = node.widgets.find(w => w.name === 'base_prompt');
                if (mainPrompt) {
                    const pc = document.createElement('div');
                    pc.style.cssText = 'display: flex; flex-direction: column; gap: 2px;';
                    const textarea = document.createElement('textarea');
                    textarea.value = mainPrompt.value || "";
                    textarea.placeholder = "Enter your base cinematic prompt...";
                    textarea.style.cssText = `width: 100%; height: 80px; background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 12px; padding: 8px; resize: vertical; font-family: ${t.mono}; outline: none; transition: border-color 0.2s;`;
                    textarea.onfocus = () => textarea.style.borderColor = t.accent;
                    textarea.onblur = () => textarea.style.borderColor = 'rgba(255,255,255,0.1)';
                    textarea.oninput = (e) => {
                        mainPrompt.value = e.target.value;
                        if (mainPrompt.callback) mainPrompt.callback(mainPrompt.value);
                        node.setDirtyCanvas(true);
                    };
                    pc.appendChild(textarea);
                    wrapper.appendChild(pc);
                }

                // Flex container for horizontal layout
                const grid = document.createElement('div');
                grid.style.cssText = 'display: flex; flex-direction: row; gap: 14px; align-items: flex-end; flex-wrap: wrap;';

                node.widgets.forEach(w => {
                    if (w.name === 'base_prompt' || w.name === 'prompt_preview' || w.type === 'converted-widget' || w.name === '_temp') return;

                    const wWrap = document.createElement('div');
                    wWrap.style.cssText = 'display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 100px;';

                    const wl = document.createElement('div');
                    wl.textContent = w.name.replace(/_/g, ' ').toUpperCase();
                    wl.style.cssText = `color: ${t.textDim}; font-size: 9px; font-weight: 700; opacity: 1.0; letter-spacing: 0.2px;`;
                    wWrap.appendChild(wl);

                    let input;
                    if (w.type === 'combo') {
                        input = document.createElement('select');
                        input.style.cssText = `width: 100%; background: #0a0a0f; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 11px; padding: 4px; cursor: pointer; outline: none;`;
                        if (w.options && w.options.values) {
                            w.options.values.forEach(v => {
                                const opt = document.createElement('option');
                                opt.value = v; opt.textContent = v;
                                input.appendChild(opt);
                            });
                        }
                        input.value = w.value;
                        input.onchange = (e) => {
                            w.value = e.target.value;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                    } else if (w.type === 'number' || typeof w.value === 'number') {
                        input = document.createElement('input');
                        input.type = 'number';
                        input.value = w.value;
                        if (w.options) {
                            if (w.options.min !== undefined) input.min = w.options.min;
                            if (w.options.max !== undefined) input.max = w.options.max;
                            if (w.options.step !== undefined) input.step = w.options.step;
                        }
                        input.style.cssText = `width: 100%; background: #0a0a0f; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 11px; padding: 4px; outline: none;`;
                        input.oninput = (e) => {
                            let val = parseFloat(e.target.value);
                            if (w.options) {
                                if (w.options.min !== undefined) val = Math.max(w.options.min, val);
                                if (w.options.max !== undefined) val = Math.min(w.options.max, val);
                            }
                            w.value = val;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                    } else if (w.type === 'toggle' || typeof w.value === 'boolean') {
                        const tr = document.createElement('div');
                        tr.style.cssText = 'display: flex; align-items: center; gap: 8px;';
                        const ck = document.createElement('input');
                        ck.type = 'checkbox'; ck.checked = w.value;
                        ck.onchange = (e) => {
                            w.value = e.target.checked;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                        tr.appendChild(ck);
                        const cl = document.createElement('span'); cl.textContent = w.value ? 'ON' : 'OFF'; cl.style.fontSize = '10px';
                        ck.addEventListener('change', () => cl.textContent = ck.checked ? 'ON' : 'OFF');
                        tr.appendChild(cl);
                        input = tr;
                    } else {
                        input = document.createElement('textarea');
                        input.value = w.value || "";
                        input.style.cssText = `width: 100%; height: 30px; background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; color: ${t.text}; font-size: 11px; padding: 4px; resize: vertical; font-family: ${t.mono}; outline: none;`;
                        input.oninput = (e) => {
                            w.value = e.target.value;
                            if (w.callback) w.callback(w.value);
                            node.setDirtyCanvas(true);
                        };
                    }

                    if (input) wWrap.appendChild(input);
                    grid.appendChild(wWrap);
                });
                wrapper.appendChild(grid);
            }
            container.appendChild(wrapper);
        });

    }


    // ═══════════════════════════════════════════════════════════════════════════
    //                          HUD / CONTROLS
    // ═══════════════════════════════════════════════════════════════════════════

    getHUDDockMode() {
        return localStorage.getItem('radiance_hud_dock_mode') === 'floating' ? 'floating' : 'panel';
    }

    applyHUDDockMode() {
        if (!this.controlsPanel) return;
        const mode = this.getHUDDockMode();
        const panel = this.controlsPanel;
        if (mode === 'floating') {
            panel.classList.remove('radiance-panel-embedded');
            if (panel.parentNode !== document.body) {
                if (panel.parentNode) panel.parentNode.removeChild(panel);
                document.body.appendChild(panel);
            }

            const width = parseInt(localStorage.getItem('radiance_hud_float_width') || '520');
            const height = parseInt(localStorage.getItem('radiance_hud_float_height') || '720');
            const left = parseInt(localStorage.getItem('radiance_hud_float_left') || String(Math.max(16, window.innerWidth - width - 24)));
            const top = parseInt(localStorage.getItem('radiance_hud_float_top') || '72');
            const clampedLeft = Math.min(Math.max(8, left), Math.max(8, window.innerWidth - 120));
            const clampedTop = Math.min(Math.max(8, top), Math.max(8, window.innerHeight - 80));

            panel.style.position = 'fixed';
            panel.style.left = `${clampedLeft}px`;
            panel.style.top = `${clampedTop}px`;
            panel.style.width = `${Math.max(320, width)}px`;
            panel.style.height = `${Math.max(240, height)}px`;
            panel.style.zIndex = '10000';
            panel.style.display = 'flex';
            panel.style.overflow = 'hidden';
            panel.style.pointerEvents = 'auto';
            panel.style.opacity = '1';
            if (this.rightControlPanel) this.rightControlPanel.style.display = 'none';
        } else {
            if (this.rightControlPanel) this.rightControlPanel.style.display = 'flex';
            panel.classList.add('radiance-panel-embedded');
            if (this.rightControlPanel && panel.parentNode !== this.rightControlPanel) {
                if (panel.parentNode) panel.parentNode.removeChild(panel);
                this.rightControlPanel.appendChild(panel);
            }
            panel.style.left = '';
            panel.style.top = '';
            panel.style.width = '';
            panel.style.height = '';
            panel.style.position = '';
            panel.style.zIndex = '';
        }
    }

    enableFloatingHUDDrag(gripRow) {
        if (!gripRow || gripRow._radianceDragBound) return;
        gripRow._radianceDragBound = true;
        gripRow.addEventListener('mousedown', (e) => {
            if (e.button !== 0 || this.getHUDDockMode() !== 'floating') return;
            if (e.target?.closest?.('[data-radiance-hud-action]')) return;
            e.preventDefault();
            const panel = this.controlsPanel;
            const rect = panel.getBoundingClientRect();
            const startX = e.clientX;
            const startY = e.clientY;
            const startLeft = rect.left;
            const startTop = rect.top;

            const onMove = (me) => {
                const left = Math.min(Math.max(8, startLeft + me.clientX - startX), Math.max(8, window.innerWidth - 120));
                const top = Math.min(Math.max(8, startTop + me.clientY - startY), Math.max(8, window.innerHeight - 80));
                panel.style.left = `${left}px`;
                panel.style.top = `${top}px`;
                localStorage.setItem('radiance_hud_float_left', Math.round(left));
                localStorage.setItem('radiance_hud_float_top', Math.round(top));
            };
            const onUp = () => {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    }

    createHUD() {
        // Singleton pattern: Check if HUD already exists
        if (RadianceViewer.singletonHUD) {
            this.controlsPanel = RadianceViewer.singletonHUD;
            this.applyHUDDockMode();

            // v2.4: Sync active instance and re-render content immediately on creation if HUD already exists
            RadianceViewer.activeInstance = this;
            if (this._lastRenderContent) this._lastRenderContent();
            return;
        }

        if (this._hudResizeListener) {
            window.removeEventListener('resize', this._hudResizeListener);
            this._hudResizeListener = null;
        }

        const t = this.theme;
        this.controlsPanel = document.createElement('div');
        this.controlsPanel.className = 'radiance-glass-dock radiance-panel-embedded';
        this.controlsPanel.id = 'radiance-singleton-hud';
        RadianceViewer.singletonHUD = this.controlsPanel;

        // v3.0 #15: High Contrast Mode Initialization
        // Restores accessibility preference from localStorage and applies the CSS hook.
        this.highContrast = localStorage.getItem('radiance_high_contrast') === '1';
        if (this.highContrast) {
            this.controlsPanel.classList.add('high-contrast');
        }

        // ─── Core helpers ────────────────────────────────────────────────────────
        const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
        const MIN_W = 320, MIN_H = 120;

        const persistSize = () => {
            localStorage.setItem('radiance_hud_width', Math.round(this.hudPanelWidth));
            if (this.hudPanelHeight) localStorage.setItem('radiance_hud_height2', Math.round(this.hudPanelHeight));
            else localStorage.removeItem('radiance_hud_height2');
            if (this.getHUDDockMode() === 'floating' && this.controlsPanel) {
                const rect = this.controlsPanel.getBoundingClientRect();
                localStorage.setItem('radiance_hud_float_width', Math.round(rect.width));
                localStorage.setItem('radiance_hud_float_height', Math.round(rect.height));
            }
        };

        // Panel-embedded mode: position/size controlled by rightControlPanel — these are no-ops
        const applyHUDPosition = (x, y) => { /* no-op in panel mode */ };
        const applyHUDSize = (w, h) => { /* no-op in panel mode */ };

        // No window resize re-clamping needed in panel mode


        // ─── Panel-embedded mode: no floating resize handles needed ─────────────
        // (Resize is handled by the rightControlPanel's left-edge drag handle)



        // ─── Inner scroll wrapper (all HUD content lives here) ───────────────────
        const innerWrap = document.createElement('div');
        innerWrap.style.cssText = `
            display: flex; flex-direction: column; gap: 10px;
            padding: 10px 14px 12px;
            overflow-x: hidden; overflow-y: auto;
            flex: 1; min-height: 0;
            box-sizing: border-box;
        `;
        this._hudInnerWrap = innerWrap;
        this.controlsPanel.appendChild(innerWrap);

        // ─── Title bar: minimize | grip | reset ──────────────────────────────────
        const tabsHeader = document.createElement('div');
        tabsHeader.style.cssText = `
            display: flex; flex-direction: column; gap: 0;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            padding-bottom: 6px; margin-bottom: 0;
            user-select: none; flex-shrink: 0;
        `;

        // Panel-embedded: gripRow is now a simple header bar (no drag)
        const gripRow = document.createElement('div');
        gripRow.style.cssText = `
            display: flex; align-items: center; justify-content: space-between;
            height: 28px; padding: 0 8px; gap: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        `;

        // Minimize / expand button
        const minBtn = document.createElement('div');
        minBtn.title = this.hudMinimized ? 'Expand panel' : 'Collapse panel';
        const updateMinBtn = () => {
            minBtn.innerHTML = this.hudMinimized
                ? `<svg width="13" height="13" viewBox="0 0 13 13"><line x1="2" y1="6.5" x2="11" y2="6.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="6.5" y1="2" x2="6.5" y2="11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`
                : `<svg width="13" height="13" viewBox="0 0 13 13"><line x1="2" y1="6.5" x2="11" y2="6.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;
            minBtn.title = this.hudMinimized ? 'Expand panel' : 'Collapse panel';
        };
        updateMinBtn();
        minBtn.style.cssText = `
            width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
            color: rgba(255,255,255,0.3); cursor: pointer; border-radius: 6px;
            transition: color 0.15s, background 0.15s; flex-shrink: 0;
        `;
        minBtn.dataset.radianceHudAction = '1';
        minBtn.onmouseenter = () => { minBtn.style.color = '#fff'; minBtn.style.background = 'rgba(255,255,255,0.08)'; };
        minBtn.onmouseleave = () => { minBtn.style.color = 'rgba(255,255,255,0.3)'; minBtn.style.background = ''; };
        minBtn.addEventListener('mousedown', e => e.stopPropagation());
        minBtn.onclick = () => {
            this.hudMinimized = !this.hudMinimized;
            localStorage.setItem('radiance_hud_minimized', this.hudMinimized);
            updateMinBtn();
            if (innerWrap._contentNode) {
                innerWrap._contentNode.style.display = this.hudMinimized ? 'none' : '';
            }
            // Also hide the footer inside innerWrap
            if (innerWrap._footerNode) {
                innerWrap._footerNode.style.display = this.hudMinimized ? 'none' : '';
            }
            this.controlsPanel.style.height = 'auto';
            this.hudPanelHeight = null;
            persistSize();
        };

        // Title label for the panel
        const panelTitle = document.createElement('div');
        panelTitle.textContent = 'GRADING CONTROLS';
        panelTitle.style.cssText = `flex: 1; font-size: 10px; font-weight: 700; letter-spacing: 1.2px; color: rgba(255,255,255,0.28); user-select: none;`;

        const dockBtn = document.createElement('div');
        dockBtn.dataset.radianceHudAction = '1';
        dockBtn.style.cssText = `
            width: 26px; height: 24px; display: flex; align-items: center; justify-content: center;
            color: rgba(255,255,255,0.36); cursor: pointer; border-radius: 6px;
            transition: color 0.15s, background 0.15s; flex-shrink: 0; font-size: 12px;
        `;
        const updateDockBtn = () => {
            const floating = this.getHUDDockMode() === 'floating';
            dockBtn.textContent = floating ? '⇲' : '⇱';
            dockBtn.title = floating ? 'Dock controls to right panel' : 'Float controls panel';
        };
        updateDockBtn();
        dockBtn.onmouseenter = () => { dockBtn.style.color = '#fff'; dockBtn.style.background = 'rgba(255,255,255,0.08)'; };
        dockBtn.onmouseleave = () => { dockBtn.style.color = 'rgba(255,255,255,0.36)'; dockBtn.style.background = ''; };
        dockBtn.onclick = () => {
            const next = this.getHUDDockMode() === 'floating' ? 'panel' : 'floating';
            localStorage.setItem('radiance_hud_dock_mode', next);
            this.applyHUDDockMode();
            updateDockBtn();
        };

        gripRow.appendChild(minBtn);
        gripRow.appendChild(panelTitle);
        gripRow.appendChild(dockBtn);
        this.enableFloatingHUDDrag(gripRow);

        tabsHeader.appendChild(gripRow);

        // Tabs row
        const tabsRow = document.createElement('div');
        tabsRow.style.cssText = `display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 6px 10px 6px;`;

        const activeTabStyle = `background: rgba(255,255,255,0.1); color: ${t.text}; border-bottom: 2px solid ${t.accent}`;
        const inactiveTabStyle = `background: transparent; color: ${t.textDim}; border-bottom: 2px solid transparent`;
        const baseTabStyle = `flex: 1 1 auto; min-width: 72px; text-align: center; padding: 6px 4px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; cursor: pointer; border-radius: 4px; transition: all 0.2s;`;

        let activeTab = 'primaries';
        this.activeTab = 'primaries';
        const tabContentContainer = document.createElement('div');
        this.tabContentContainer = tabContentContainer;

        const tabs = [
            { id: 'prompt', label: 'PROMPT' },
            { id: 'primaries', label: 'PRIMARIES' },
            { id: 'curves', label: 'CURVES' },
            { id: 'effects', label: 'EFFECTS' },
            { id: 'masks', label: 'MASKS' },
            { id: 'view', label: 'VIEW' },
            { id: 'terminal', label: '> _TERM' }
        ];
        this._hudTabs = [];

        const renderTabs = () => {
            tabsRow.innerHTML = '';
            tabs.forEach(tab => {
                const btn = document.createElement('div');
                btn.textContent = tab.label;
                btn.dataset.tabId = tab.id;
                btn.style.cssText = baseTabStyle + (activeTab === tab.id ? activeTabStyle : inactiveTabStyle);
                btn.onmouseover = () => { if (activeTab !== tab.id) btn.style.background = 'rgba(255,255,255,0.05)'; };
                btn.onmouseout = () => { if (activeTab !== tab.id) btn.style.background = 'transparent'; };
                btn.onclick = () => {
                    const active = RadianceViewer.activeInstance || this;
                    activeTab = tab.id;
                    active.activeTab = tab.id;
                    // Switching any tab clears depth/overlay modes that override rendering
                    active.showZdepth = false;
                    if (active.renderer) active.renderer.setShowDepth(false);
                    renderTabs();
                    renderContent();
                    active.render();
                };
                this._hudTabs.push(btn);
                tabsRow.appendChild(btn);
            });
        };

        tabsHeader.appendChild(tabsRow);
        innerWrap.appendChild(tabsHeader);


        const renderContent = () => {
            const active = RadianceViewer.activeInstance || this;
            tabContentContainer.innerHTML = '';
            tabContentContainer.style.cssText = 'display: flex; flex-direction: column; flex: 1; min-height: 0; overflow-y: auto;';

            if (activeTab === 'prompt') {
                active.renderPromptTab(tabContentContainer);
            } else if (activeTab === 'primaries') {
                active.renderPrimariesTab(tabContentContainer);
            } else if (activeTab === 'curves') {
                active.renderCurvesTab(tabContentContainer);
            } else if (activeTab === 'effects') {
                active.renderEffectsTab(tabContentContainer);
                active.renderLensTab(tabContentContainer);
            } else if (activeTab === 'masks') {
                active.renderQualifiersTab(tabContentContainer);
                active.renderMasksTab(tabContentContainer);
            } else if (activeTab === 'view') {
                active.renderViewTab(tabContentContainer);
            } else if (activeTab === 'terminal') {
                active.renderTerminalTab(tabContentContainer);
            }
        };

        innerWrap.appendChild(tabContentContainer);

        // Initialize
        renderTabs();
        renderContent();
        // Save reference for undo/redo panel refresh
        this._lastRenderContent = renderContent;

        // Track nodes for minimize toggle
        innerWrap._contentNode = tabContentContainer;
        if (this.hudMinimized) tabContentContainer.style.display = 'none';

        // Footer: A/B Bypass + Reset All
        const footer = document.createElement('div');
        footer.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 4px;';
        if (this.hudMinimized) footer.style.display = 'none';
        innerWrap._footerNode = footer;

        // A/B Bypass Toggle
        const bypassBtn = document.createElement('div');
        bypassBtn.onclick = () => {
            const active = RadianceViewer.activeInstance || this;
            active._gradingBypassed = !active._gradingBypassed;
            if (active._gradingBypassed) {
                // Save current state and set identity
                active._savedGrading = {
                    exposure: active.exposure, lift: [...(active.lift || [0, 0, 0])], gamma: [...(active.gamma || [1, 1, 1])], gain: [...(active.gain || [1, 1, 1])],
                    temperature: active.temperature, tint: active.tint, contrast: active.contrast, pivot: active.pivot, saturation: active.saturation,
                    grain: active.grain, grainSize: active.grainSize ?? 1.0, grainColor: active.grainColor ?? 0.0, grainAnimate: active.grainAnimate ?? false,
                    denoise: active.denoise,
                    bloom: active.bloom ?? 0.0, bloomThreshold: active.bloomThreshold ?? 1.0, halation: active.halation ?? 0.0,
                    halationRadius: active.halationRadius ?? 1.0, halationThreshold: active.halationThreshold ?? 0.35, diffusion: active.diffusion ?? 0.0,
                    lensDistortion: active.lensDistortion, lensFringe: active.lensFringe,
                    vignetteIntensity: active.vignetteIntensity, vignetteFalloff: active.vignetteFalloff,
                    bokehHighlightBias: active.bokehHighlightBias ?? 0.0, bokehSoapBubble: active.bokehSoapBubble ?? 0.0, bokehOpticalVig: active.bokehOpticalVig ?? 0.0,
                    apertureBlades: active.apertureBlades ?? 0, apertureRotation: active.apertureRotation ?? 0.0, apertureAnamorphic: active.apertureAnamorphic ?? 1.0,
                    anamorphicStreaks: active.anamorphicStreaks ?? 0.0, streakThreshold: active.streakThreshold ?? 0.85, streakLength: active.streakLength ?? 0.08,
                };
                if (active.renderer) {
                    active.renderer.setExposure(0); active.renderer.setLift(0, 0, 0); active.renderer.setGamma(1, 1, 1); active.renderer.setGain(1, 1, 1);
                    active.renderer.setTemperature(0); active.renderer.setTint(0); active.renderer.setContrast(1); active.renderer.setPivot(0.5); active.renderer.setSaturation(1);
                    active.renderer.setGrain(0); active.renderer.setGrainSize(1.0); active.renderer.setGrainColor(0.0); active.renderer.setGrainAnimate(false);
                    active.renderer.setDenoise(0); active.renderer.setBloom(0); active.renderer.setHalation(0); active.renderer.setDiffusion(0);
                    active.renderer.setLensDistortion(0, 0); active.renderer.setVignette(0, 0.5);
                    active.renderer.setBokehPhysics(0, 0, 0); active.renderer.setApertureShape(0, 0, 1.0);
                    if (active.renderer.setAnamorphicStreaks) active.renderer.setAnamorphicStreaks(0);
                }
                bypassBtn.textContent = '● BYPASSED'; bypassBtn.style.color = '#ff6b6b'; bypassBtn.style.borderColor = 'rgba(255,100,100,0.3)';
            } else {
                // Restore saved state
                const s = active._savedGrading;
                if (s && active.renderer) {
                    active.renderer.setExposure(s.exposure); active.renderer.setLift(s.lift[0], s.lift[1], s.lift[2]);
                    active.renderer.setGamma(s.gamma[0], s.gamma[1], s.gamma[2]); active.renderer.setGain(s.gain[0], s.gain[1], s.gain[2]);
                    active.renderer.setTemperature(s.temperature); active.renderer.setTint(s.tint); active.renderer.setContrast(s.contrast); active.renderer.setPivot(s.pivot); active.renderer.setSaturation(s.saturation);
                    active.renderer.setGrain(s.grain); active.renderer.setGrainSize(s.grainSize ?? 1.0); active.renderer.setGrainColor(s.grainColor ?? 0.0); active.renderer.setGrainAnimate(s.grainAnimate ?? false);
                    active.renderer.setDenoise(s.denoise);
                    active.renderer.setBloom(s.bloom ?? 0.0); if (active.renderer.setBloomThreshold) active.renderer.setBloomThreshold(s.bloomThreshold ?? 1.0);
                    active.renderer.setHalation(s.halation ?? 0.0); if (active.renderer.setHalationRadius) active.renderer.setHalationRadius(s.halationRadius ?? 1.0); if (active.renderer.setHalationThreshold) active.renderer.setHalationThreshold(s.halationThreshold ?? 0.35); active.renderer.setDiffusion(s.diffusion ?? 0.0);
                    active.renderer.setLensDistortion(s.lensDistortion, s.lensFringe); active.renderer.setVignette(s.vignetteIntensity, s.vignetteFalloff);
                    active.renderer.setBokehPhysics(s.bokehHighlightBias ?? 0.0, s.bokehSoapBubble ?? 0.0, s.bokehOpticalVig ?? 0.0);
                    active.renderer.setApertureShape(s.apertureBlades ?? 0, s.apertureRotation ?? 0.0, s.apertureAnamorphic ?? 1.0);
                    if (active.renderer.setAnamorphicStreaks) active.renderer.setAnamorphicStreaks(s.anamorphicStreaks ?? 0.0);
                    if (active.renderer.setStreakThreshold) active.renderer.setStreakThreshold(s.streakThreshold ?? 0.85);
                    if (active.renderer.setStreakLength) active.renderer.setStreakLength(s.streakLength ?? 0.08);
                }
                bypassBtn.textContent = 'A/B'; bypassBtn.style.color = '#666'; bypassBtn.style.borderColor = 'rgba(255,255,255,0.08)';
            }
            active.render();
        };
        footer.appendChild(bypassBtn);

        // Undo / Redo Buttons
        const undoRedoGroup = document.createElement('div');
        undoRedoGroup.style.cssText = 'display: flex; gap: 4px;';

        const undoBtn = document.createElement('div');
        undoBtn.textContent = '↶';
        undoBtn.title = 'Undo (Ctrl+Z)';
        undoBtn.style.cssText = 'font-size: 14px; color: #555; cursor: pointer; padding: 0 4px; user-select: none; transition: color 0.15s;';
        undoBtn.onmouseenter = () => { undoBtn.style.color = this._undoStack.length > 0 ? this.theme.accent : '#555'; };
        undoBtn.onmouseleave = () => { undoBtn.style.color = '#555'; };
        undoBtn.onclick = () => { (RadianceViewer.activeInstance || this).undo(); };

        const redoBtn = document.createElement('div');
        redoBtn.textContent = '↷';
        redoBtn.title = 'Redo (Ctrl+Shift+Z)';
        redoBtn.style.cssText = 'font-size: 14px; color: #555; cursor: pointer; padding: 0 4px; user-select: none; transition: color 0.15s;';
        redoBtn.onmouseenter = () => {
            const active = RadianceViewer.activeInstance || this;
            redoBtn.style.color = active._redoStack.length > 0 ? active.theme.accent : '#555';
        };
        redoBtn.onmouseleave = () => { redoBtn.style.color = '#555'; };
        redoBtn.onclick = () => { (RadianceViewer.activeInstance || this).redo(); };

        undoRedoGroup.appendChild(undoBtn);
        undoRedoGroup.appendChild(redoBtn);
        footer.appendChild(undoRedoGroup);

        // Reset All Button
        const resetBtn = document.createElement('div');
        resetBtn.textContent = 'RESET ALL';
        resetBtn.style.cssText = 'font-size: 9px; color: #666; cursor: pointer; letter-spacing: 1px;';
        resetBtn.onclick = () => {
            const active = RadianceViewer.activeInstance || this;
            // Push undo before resetting
            active._pushUndo();
            // Primaries
            active.exposure = 0.0;
            active.lift = [0, 0, 0]; active.gamma = [1, 1, 1]; active.gain = [1, 1, 1];
            active.temperature = 0.0; active.tint = 0.0;
            active.contrast = 1.0; active.pivot = 0.5; active.saturation = 1.0;
            // Film / Effects
            active.grain = 0.0; active.grainSize = 1.0; active.grainColor = 0.0; active.grainAnimate = false;
            active.denoise = 0.0; active.bloom = 0.0; active.bloomThreshold = 1.0; active.halation = 0.0; active.halationRadius = 1.0; active.halationThreshold = 0.35; active.diffusion = 0.0;
            active.anamorphicStreaks = 0.0; active.streakThreshold = 0.85; active.streakLength = 0.08;
            // Lens
            active.focusDistance = 0.5; active.aperture = 0.0; active.dofEnabled = false;
            active.apertureBlades = 0; active.apertureRotation = 0.0; active.apertureAnamorphic = 1.0;
            active.lensDistortion = 0.0; active.lensFringe = 0.0;
            active.vignetteIntensity = 0.0; active.vignetteFalloff = 0.5;
            active.bokehHighlightBias = 0.0; active.bokehSoapBubble = 0.0; active.bokehOpticalVig = 0.0;
            active.activeLensSignature = null; active._activeFilmPreset = 'No Grain';
            // Curves
            if (active.curveEditor) active.curveEditor.resetAllChannels?.();
            // Qualifier
            if (active.qualifierState) {
                active.qualifierState.enabled = false; active.qualifierState.showMask = false;
            }

            if (active.renderer) {
                active.renderer.setExposure(0);
                active.renderer.setLift(0, 0, 0); active.renderer.setGamma(1, 1, 1); active.renderer.setGain(1, 1, 1);
                active.renderer.setTemperature(0); active.renderer.setTint(0);
                active.renderer.setContrast(1); active.renderer.setPivot(0.5); active.renderer.setSaturation(1);
                active.renderer.setGrain(0); active.renderer.setGrainSize(1.0); active.renderer.setGrainColor(0.0); active.renderer.setGrainAnimate(false);
                active.renderer.setDenoise(0); active.renderer.setBloom(0); active.renderer.setHalation(0); active.renderer.setDiffusion(0);
                active.renderer.setDoFEnabled(false); active.renderer.setFocusDistance(0.5); active.renderer.setAperture(0);
                active.renderer.setApertureShape(0, 0, 1.0);
                active.renderer.setLensDistortion(0, 0); active.renderer.setVignette(0, 0.5);
                active.renderer.setBokehPhysics(0, 0, 0);
                if (active.renderer.setAnamorphicStreaks) active.renderer.setAnamorphicStreaks(0);
                active.renderer.setCurveMix(0);
                if (active.qualifierState && active.renderer.setQualifier) {
                    active.renderer.setQualifier(active.qualifierState);
                }
            }
            active._gradingBypassed = false;
            active.render();
            renderContent();
        };
        footer.appendChild(resetBtn);
        innerWrap.appendChild(footer);

        // Keyboard shortcuts for undo/redo (scoped to document when HUD is visible)
        if (!this._undoKeyListener) {
            this._undoKeyListener = (e) => {
                // Only respond when HUD is visible and not in a text input
                if (!RadianceViewer.singletonHUD || RadianceViewer.singletonHUD.style.opacity === '0') return;
                const active = RadianceViewer.activeInstance;
                if (!active) return;

                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
                    e.preventDefault();
                    active.undo();
                } else if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
                    e.preventDefault();
                    active.redo();
                } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
                    e.preventDefault();
                    active.redo();
                }

                // --- Printer Lights (1-4) ---
                if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                    const step = e.shiftKey ? -0.01 : 0.01;
                    let changed = false;

                    if (e.key === '1') { // Red
                        active.offset[0] += step; changed = true;
                    } else if (e.key === '2') { // Green
                        active.offset[1] += step; changed = true;
                    } else if (e.key === '3') { // Blue
                        active.offset[2] += step; changed = true;
                    } else if (e.key === '4') { // Master
                        active.offset[0] += step; active.offset[1] += step; active.offset[2] += step;
                        changed = true;
                    }

                    if (changed) {
                        e.preventDefault();
                        active._pushUndoDebounced();
                        if (active.renderer) active.renderer.setOffset(active.offset[0], active.offset[1], active.offset[2]);
                        active.render();
                        if (active._lastRenderContent) active._lastRenderContent(); // Refresh knobs
                    }
                }
            };
            document.addEventListener('keydown', this._undoKeyListener);
        }

        // ═══════════════════════════════════════════════════════════════════════════
        //               TRANSPORT CONTROLS — Real-time Video + Frame Sequence
        // ═══════════════════════════════════════════════════════════════════════════

        if (this.transportPanel) this.transportPanel.remove();

        this.transportPanel = document.createElement('div');
        this.transportPanel.style.cssText = `
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            display: ${this.totalFrames > 1 || this.videoMode ? 'flex' : 'none'};
            flex-direction: column;
            align-items: stretch;
            gap: 0;
            background: rgba(8, 8, 14, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 0;
            backdrop-filter: blur(12px);
            z-index: 101;
            width: min(640px, 90vw);
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
            overflow: hidden;
            pointer-events: auto;
        `;

        // ── Top: Video drop zone / filename strip ──────────────────────────────
        const fileStrip = document.createElement('div');
        fileStrip.style.cssText = `
            display: flex; align-items: center; gap: 8px;
            padding: 5px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 10px; font-family: monospace;
            color: rgba(255,255,255,0.35);
            cursor: pointer;
        `;
        const fileIcon = document.createElement('span');
        fileIcon.innerHTML = `<svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <rect x="2" y="1" width="8" height="11" rx="1" stroke="currentColor" stroke-width="1.1"/>
            <path d="M5 6l3 1.5L5 9V6z" fill="currentColor"/>
        </svg>`;
        fileIcon.style.cssText = 'opacity: 0.6; flex-shrink: 0; display: flex; align-items: center;';
        this._fileNameLabel = document.createElement('span');
        this._fileNameLabel.textContent = 'Drop video  •  MP4 / WebM / MOV  •  or click to browse';
        this._fileNameLabel.style.cssText = 'flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';

        const unloadBtn = document.createElement('div');
        unloadBtn.innerHTML = '✕';
        unloadBtn.title = 'Unload video';
        unloadBtn.style.cssText = `
            font-size: 10px; color: rgba(255,255,255,0.25); cursor: pointer; padding: 2px 4px;
            border-radius: 4px; transition: color 0.15s; display: none;
        `;
        unloadBtn.onmouseenter = () => { unloadBtn.style.color = '#ff6b6b'; };
        unloadBtn.onmouseleave = () => { unloadBtn.style.color = 'rgba(255,255,255,0.25)'; };
        unloadBtn.onclick = (e) => {
            e.stopPropagation();
            this.unloadVideo();
            this._fileNameLabel.textContent = 'Drop video  •  MP4 / WebM / MOV  •  or click to browse';
            unloadBtn.style.display = 'none';
            if (this.totalFrames <= 1) this.transportPanel.style.display = 'none';
        };

        fileStrip.appendChild(fileIcon);
        fileStrip.appendChild(this._fileNameLabel);
        fileStrip.appendChild(unloadBtn);

        // Hidden file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'video/mp4,video/webm,video/quicktime,video/x-matroska,video/*';
        fileInput.style.display = 'none';
        fileInput.onchange = (e) => {
            const f = e.target.files[0];
            if (!f) return;
            this._fileNameLabel.textContent = f.name;
            unloadBtn.style.display = '';
            this.loadVideo(f);
        };
        fileStrip.appendChild(fileInput);

        fileStrip.onclick = (e) => {
            if (e.target === unloadBtn) return;
            fileInput.click();
        };

        // Drag-and-drop on canvasWrapper — accepts video AND image/HDR files
        this.canvasWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            if ([...e.dataTransfer.types].includes('Files')) {
                e.dataTransfer.dropEffect = 'copy';
                fileStrip.style.background = 'rgba(0,168,255,0.08)';
            }
        });
        this.canvasWrapper.addEventListener('dragleave', () => {
            fileStrip.style.background = '';
        });
        this.canvasWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            fileStrip.style.background = '';
            const files = [...e.dataTransfer.files];

            // Priority: video first, then image/HDR
            const videoFile = files.find(f => f.type.startsWith('video/'));
            if (videoFile) {
                this._fileNameLabel.textContent = videoFile.name;
                unloadBtn.style.display = '';
                this.transportPanel.style.display = 'flex';
                this.loadVideo(videoFile);
                return;
            }

            // Image / HDR / float file
            const IMAGE_EXTS = /\.(exr|hdr|tif|tiff|f32|rf32|rhdr|npy|png|jpg|jpeg|webp)$/i;
            const imgFile = files.find(f =>
                IMAGE_EXTS.test(f.name) ||
                f.type.startsWith('image/') ||
                f.type === 'application/octet-stream'
            );
            if (imgFile) {
                this._fileNameLabel.textContent = imgFile.name;
                this._loadDroppedImageFile(imgFile);
            }
        });

        this.transportPanel.appendChild(fileStrip);

        // ── Scrubber / Progress bar ────────────────────────────────────────────
        const scrubberWrap = document.createElement('div');
        scrubberWrap.style.cssText = 'position: relative; padding: 4px 12px 2px; display: flex; align-items: center; gap: 8px;';

        // Timecode start
        this._videoTimecode = document.createElement('span');
        this._videoTimecode.textContent = '00:00:00:00';
        this._videoTimecode.style.cssText = 'font-size: 9.5px; font-family: monospace; color: rgba(255,255,255,0.45); min-width: 62px; user-select: none;';

        const scrubber = document.createElement('input');
        scrubber.type = 'range';
        scrubber.min = '0'; scrubber.max = '10000'; scrubber.step = '1'; scrubber.value = '0';
        scrubber.style.cssText = `
            flex: 1; height: 3px; cursor: pointer;
            accent-color: ${this.theme.accent};
            -webkit-appearance: none; appearance: none;
            background: linear-gradient(to right, ${this.theme.accent} 0%, rgba(255,255,255,0.15) 0%);
        `;
        this._videoScrubber = scrubber;
        this.frameCounter = scrubber; // alias so updateFrameDisplay() doesn't crash

        let _scrubbing = false;
        scrubber.addEventListener('mousedown', () => { _scrubbing = true; });
        scrubber.addEventListener('mouseup', () => { _scrubbing = false; });
        scrubber.addEventListener('input', () => {
            const pct = parseInt(scrubber.value) / 10000;
            // Update scrubber fill
            scrubber.style.background = `linear-gradient(to right, ${this.theme.accent} ${pct * 100}%, rgba(255,255,255,0.15) ${pct * 100}%)`;
            if (this.videoMode && this.videoEl) {
                this.seekVideoTo(pct);
            } else if (this.totalFrames > 1) {
                const frameIdx = Math.round(pct * (this.totalFrames - 1));
                this.setFrame(frameIdx);
            }
        });

        // Duration label
        this._videoDuration = document.createElement('span');
        this._videoDuration.textContent = '00:00:00:00';
        this._videoDuration.style.cssText = 'font-size: 9.5px; font-family: monospace; color: rgba(255,255,255,0.28); min-width: 62px; text-align: right; user-select: none;';

        scrubberWrap.appendChild(this._videoTimecode);
        scrubberWrap.appendChild(scrubber);
        scrubberWrap.appendChild(this._videoDuration);
        this.transportPanel.appendChild(scrubberWrap);

        // ── Controls row ──────────────────────────────────────────────────────
        const ctrlRow = document.createElement('div');
        ctrlRow.style.cssText = 'display: flex; align-items: center; gap: 8px; padding: 6px 12px 8px; justify-content: space-between;';

        // Left group: loop | step-back | prev | play | next | step-fwd
        const leftGroup = document.createElement('div');
        leftGroup.style.cssText = 'display: flex; align-items: center; gap: 6px;';

        const mkBtn = (html, title, onClick, small = false) => {
            const b = document.createElement('div');
            b.innerHTML = html;
            b.title = title;
            b.style.cssText = `
                color: ${small ? 'rgba(255,255,255,0.55)' : '#ddd'}; cursor: pointer;
                font-size: ${small ? '12px' : '15px'}; width: ${small ? '18px' : '22px'};
                display: flex; align-items: center; justify-content: center;
                transition: color 0.12s; user-select: none;
            `;
            b.onmouseenter = () => { b.style.color = '#fff'; };
            b.onmouseleave = () => { b.style.color = small ? 'rgba(255,255,255,0.55)' : '#ddd'; };
            b.onclick = onClick;
            return b;
        };

        const loopBtn = mkBtn('∞', 'Toggle Loop', () => {
            this.loop = !this.loop;
            if (this.videoEl) this.videoEl.loop = this.loop;
            loopBtn.style.color = this.loop ? this.theme.accent : 'rgba(255,255,255,0.55)';
        }, true);
        loopBtn.style.fontSize = '17px';
        loopBtn.style.color = this.loop ? this.theme.accent : 'rgba(255,255,255,0.55)';

        const stepBackBtn = mkBtn('⇤', 'Step back 1 frame (Shift+←)', () => {
            if (this.videoMode && this.videoEl) {
                const fps = this._videoNativeFps || 25;
                this.videoEl.currentTime = Math.max(0, this.videoEl.currentTime - 1 / fps);
            } else this.prevFrame();
        }, true);

        const prevBtn = mkBtn('⏮', 'Previous frame (←)', () => {
            if (this.videoMode && this.videoEl) this.videoEl.currentTime = 0;
            else this.prevFrame();
        }, false);

        const playBtn = document.createElement('div');
        playBtn.textContent = '▶';
        playBtn.title = 'Play / Pause (Space)';
        playBtn.style.cssText = `
            color: #fff; cursor: pointer; font-size: 18px; width: 30px; height: 30px;
            display: flex; align-items: center; justify-content: center;
            background: ${this.theme.accent}22; border: 1px solid ${this.theme.accent}55;
            border-radius: 50%; transition: background 0.15s; user-select: none; flex-shrink: 0;
        `;
        playBtn.onmouseenter = () => { playBtn.style.background = this.theme.accent + '44'; };
        playBtn.onmouseleave = () => { playBtn.style.background = this.theme.accent + '22'; };
        playBtn.onclick = () => this.togglePlayback();
        this.playBtn = playBtn;

        const nextBtn = mkBtn('⏭', 'Next frame (→)', () => {
            if (this.videoMode && this.videoEl) this.videoEl.currentTime = this.videoEl.duration;
            else this.nextFrame();
        }, false);

        const stepFwdBtn = mkBtn('⇥', 'Step forward 1 frame (Shift+→)', () => {
            if (this.videoMode && this.videoEl) {
                const fps = this._videoNativeFps || 25;
                this.videoEl.currentTime = Math.min(this.videoEl.duration, this.videoEl.currentTime + 1 / fps);
            } else this.nextFrame();
        }, true);

        leftGroup.appendChild(loopBtn);
        leftGroup.appendChild(stepBackBtn);
        leftGroup.appendChild(prevBtn);
        leftGroup.appendChild(playBtn);
        leftGroup.appendChild(nextBtn);
        leftGroup.appendChild(stepFwdBtn);

        // Center: frame counter
        this.frameCounter = document.createElement('div');
        this.frameCounter.textContent = '1 / 1';
        this.frameCounter.style.cssText = 'color: rgba(255,255,255,0.35); font-size: 10px; font-family: monospace; min-width: 70px; text-align: center; user-select: none;';

        // Right group: FPS selector | Speed selector
        const rightGroup = document.createElement('div');
        rightGroup.style.cssText = 'display: flex; align-items: center; gap: 8px;';

        const mkSel = (label, options, current, onChange) => {
            const wrap = document.createElement('div');
            wrap.style.cssText = 'display: flex; align-items: center; gap: 4px;';
            const lbl = document.createElement('span');
            lbl.textContent = label;
            lbl.style.cssText = 'font-size: 9px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 0.5px;';
            const sel = document.createElement('select');
            sel.style.cssText = `
                background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.7);
                border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;
                padding: 2px 4px; font-size: 10px; font-family: monospace; cursor: pointer; outline: none;
            `;
            options.forEach(([val, txt]) => {
                const opt = document.createElement('option');
                opt.value = String(val); opt.textContent = txt;
                if (String(val) === String(current)) opt.selected = true;
                sel.appendChild(opt);
            });
            sel.onchange = (e) => onChange(e.target.value);
            wrap.appendChild(lbl);
            wrap.appendChild(sel);
            return { wrap, sel };
        };

        const { wrap: fpsWrap, sel: fpsSel } = mkSel('FPS',
            [[12, '12'], [15, '15'], [23.976, '23.97'], [24, '24'], [25, '25'], [29.97, '29.97'], [30, '30'], [48, '48'], [50, '50'], [60, '60']],
            this.playbackFps || 24,
            (v) => {
                this.playbackFps = parseFloat(v);
                if (this.videoEl) this._videoNativeFps = this.playbackFps;
            }
        );
        this._fpsSel = fpsSel;

        const { wrap: spdWrap, sel: spdSel } = mkSel('×',
            [[0.25, '¼'], [0.5, '½'], [1, '1'], [1.5, '1.5'], [2, '2'], [4, '4']],
            this.playbackSpeed || 1,
            (v) => {
                this.playbackSpeed = parseFloat(v);
                if (this.videoEl) this.videoEl.playbackRate = this.playbackSpeed;
            }
        );

        rightGroup.appendChild(fpsWrap);
        rightGroup.appendChild(spdWrap);

        // v3.0 #8.2: GPU Cache Purge Button
        const purgeBtn = mkBtn('◎', 'Clear GPU Frame Cache', () => {
            if (this.renderer) {
                this.renderer.clearFrameCache();
                // If sequence mode, clear the frameImages array to force reload from disk/network
                if (this.totalFrames > 1) {
                    this.frameImages = [];
                    this.updateFrameDisplay();
                }
                this._termLog('info', '[GPU] Cache purged. Memory released.');
            }
        }, true);
        purgeBtn.style.marginLeft = '4px';
        rightGroup.appendChild(purgeBtn);

        ctrlRow.appendChild(leftGroup);
        ctrlRow.appendChild(this.frameCounter);
        ctrlRow.appendChild(rightGroup);
        this.transportPanel.appendChild(ctrlRow);

        // ── Space bar shortcut ─────────────────────────────────────────────────
        // Wire into existing docKeyHandler if possible, otherwise add here
        const spaceHandler = (e) => {
            if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                this.togglePlayback();
            }
            if (e.code === 'ArrowLeft' && e.shiftKey) { stepBackBtn.onclick(); }
            if (e.code === 'ArrowRight' && e.shiftKey) { stepFwdBtn.onclick(); }
        };
        if (!this._transportSpaceHandler) {
            this._transportSpaceHandler = spaceHandler;
            document.addEventListener('keydown', this._transportSpaceHandler);
        }

        // Float transport over the canvas
        this.canvasWrapper.appendChild(this.transportPanel);


        this.applyHUDDockMode();

        // Relative ordering for info bar
        if (this.bottomInfoBar) {
            this.container.appendChild(this.bottomInfoBar);
        }
    }

    renderPrimariesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; gap: 12px; padding: 10px 4px;';

        const createMini = (lbl, min, max, val, step, cb) => {
            const k = this.createKnob(lbl, min, max, val, step, cb);
            k.style.maxWidth = '80px';
            return k;
        };

        // ═════════════════════════════════════════════════════════════════════
        // 1. TOP BAR: Exp | Temp | Tint | Contrast | Pivot | Mid/Detail
        // ═════════════════════════════════════════════════════════════════════
        const topBar = document.createElement('div');
        topBar.style.cssText = 'display: flex; flex-wrap: wrap; gap: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; justify-content: space-evenly;';

        // Exposure: Slider 0..10, but allow keyboard/manual push to 12.0 (shader limit)
        topBar.appendChild(createMini('EXP', -10.0, 10.0, this.exposure || 0.0, 0.1, v => {
            this.exposure = Math.max(-12.0, Math.min(12.0, v));
            if (this.renderer) this.renderer.setExposure(this.exposure);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        topBar.appendChild(createMini('TEMP', -2.0, 2.0, this.temperature || 0.0, 0.05, v => {
            this.temperature = v;
            if (this.renderer) this.renderer.setTemperature(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        topBar.appendChild(createMini('TINT', -2.0, 2.0, this.tint || 0.0, 0.05, v => {
            this.tint = v;
            if (this.renderer) this.renderer.setTint(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        topBar.appendChild(createMini('CONTRAST', 0.2, 3.0, this.contrast || 1.0, 0.02, v => {
            this.contrast = Math.max(0.0, Math.min(5.0, v));
            if (this.renderer) this.renderer.setContrast(this.contrast);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        topBar.appendChild(createMini('PIVOT', 0.0, 1.0, this.pivot || 0.5, 0.05, v => {
            this.pivot = v;
            if (this.renderer) this.renderer.setPivot(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        // Midtone Detail
        topBar.appendChild(createMini('M.DETAIL', -1.0, 1.0, this.midDetail || 0.0, 0.05, v => {
            this.midDetail = v;
            if (this.renderer) this.renderer.setMidDetail(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        container.appendChild(topBar);

        // ═════════════════════════════════════════════════════════════════════
        // 1b. INTEGRATED SCOPES
        // ═════════════════════════════════════════════════════════════════════
        const scopesWrapper = document.createElement('div');
        scopesWrapper.style.cssText = 'border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 4px; margin-bottom: 8px;';

        // Pass a sub-container to renderScopesTab so it doesn't affect the main layout
        const scopesInner = document.createElement('div');
        this.renderScopesTab(scopesInner);
        // Overwrite some container styles for integration
        scopesInner.style.padding = '0';
        scopesInner.style.gap = '4px';

        scopesWrapper.appendChild(scopesInner);
        container.appendChild(scopesWrapper);

        // ═════════════════════════════════════════════════════════════════════
        // 1c. RGB PRINTER LIGHTS
        // ═════════════════════════════════════════════════════════════════════
        const printerRow = document.createElement('div');
        printerRow.style.cssText = 'display: flex; flex-direction: column; gap: 3px; padding: 4px 0 6px; border-bottom: 1px solid rgba(255,255,255,0.06);';

        const printerLabel = document.createElement('div');
        printerLabel.textContent = 'PRINTER LIGHTS';
        printerLabel.style.cssText = 'font-size: 9px; font-weight: bold; color: #666; letter-spacing: 0.08em; padding-left: 2px;';
        printerRow.appendChild(printerLabel);

        const printerStrips = document.createElement('div');
        printerStrips.style.cssText = 'display: flex; flex-direction: column; gap: 3px;';

        const makePrinterStrip = (label, color, hexColor, getVal, setVal) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; align-items: center; gap: 6px;';

            const lbl = document.createElement('span');
            lbl.textContent = label;
            lbl.style.cssText = `font-size: 9px; font-weight: bold; color: ${hexColor}; width: 10px; flex-shrink: 0;`;
            row.appendChild(lbl);

            const track = document.createElement('div');
            track.style.cssText = `
                flex: 1; height: 10px; background: linear-gradient(to right, #111 0%, ${hexColor}33 50%, ${hexColor}88 100%);
                border-radius: 5px; border: 1px solid rgba(255,255,255,0.1); position: relative; cursor: ew-resize;
            `;

            const thumb = document.createElement('div');
            const pct = (getVal() + 50) / 100;
            thumb.style.cssText = `
                position: absolute; top: 50%; transform: translate(-50%, -50%);
                left: ${pct * 100}%; width: 10px; height: 10px;
                border-radius: 50%; background: ${hexColor}; border: 1px solid #fff;
                box-shadow: 0 0 4px ${hexColor}; pointer-events: none;
            `;
            track.appendChild(thumb);

            const valLbl = document.createElement('span');
            valLbl.textContent = getVal() > 0 ? `+${getVal()}` : `${getVal()}`;
            valLbl.style.cssText = 'font-size: 9px; color: #888; width: 28px; text-align: right; font-family: monospace;';

            const dblClick = () => {
                setVal(0);
                this.requestRender();
                this.requestScopeUpdate();
                const newPct = 0.5;
                thumb.style.left = `${newPct * 100}%`;
                valLbl.textContent = '0';
            };
            track.ondblclick = dblClick;

            let dragging = false;
            track.onmousedown = (e) => {
                dragging = true;
                e.preventDefault();
            };
            document.addEventListener('mousemove', (e) => {
                if (!dragging) return;
                const rect = track.getBoundingClientRect();
                const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const val = Math.round(pct * 100 - 50);
                setVal(val);
                thumb.style.left = `${pct * 100}%`;
                valLbl.textContent = val > 0 ? `+${val}` : `${val}`;
                this.requestRender();
                this.requestScopeUpdate();
            });
            document.addEventListener('mouseup', () => { dragging = false; });

            row.appendChild(track);
            row.appendChild(valLbl);
            return row;
        };

        printerStrips.appendChild(makePrinterStrip('R', 'red', '#ff4444',
            () => this.printerR || 0, v => { this.printerR = v; if (this.renderer) this.renderer.setPrinterLights(this.printerR, this.printerG || 0, this.printerB || 0); }));
        printerStrips.appendChild(makePrinterStrip('G', 'green', '#44ff44',
            () => this.printerG || 0, v => { this.printerG = v; if (this.renderer) this.renderer.setPrinterLights(this.printerR || 0, this.printerG, this.printerB || 0); }));
        printerStrips.appendChild(makePrinterStrip('B', 'blue', '#4488ff',
            () => this.printerB || 0, v => { this.printerB = v; if (this.renderer) this.renderer.setPrinterLights(this.printerR || 0, this.printerG || 0, this.printerB); }));

        printerRow.appendChild(printerStrips);
        container.appendChild(printerRow);

        // ═════════════════════════════════════════════════════════════════════
        // 2. COLOR WHEELS TABS: Primary | Log
        // ═════════════════════════════════════════════════════════════════════
        const wheelsWrapper = document.createElement('div');
        wheelsWrapper.style.cssText = 'display: flex; flex-direction: column; gap: 4px; padding: 4px 0;';

        const wheelTabs = document.createElement('div');
        wheelTabs.style.cssText = 'display: flex; gap: 8px; justify-content: center; margin-bottom: 4px;';

        const renderWheels = () => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; justify-content: space-between; gap: 4px;';

            if (this.activeWheelTab === 'LOG') {
                row.appendChild(this.createColorWheel('SHADOW', -0.2, 0.2, this.logShadow || [0, 0, 0], 0.005, (r, g, b) => {
                    this.logShadow = [r, g, b];
                    if (this.renderer) this.renderer.setLogShadow(r, g, b);
                    this.requestRender();
                    this.requestScopeUpdate();
                }));
                row.appendChild(this.createColorWheel('MIDTONE', -0.5, 0.5, this.logMidtone || [0, 0, 0], 0.01, (r, g, b) => {
                    this.logMidtone = [r, g, b];
                    if (this.renderer) this.renderer.setLogMidtone(r, g, b);
                    this.requestRender();
                    this.requestScopeUpdate();
                }));
                row.appendChild(this.createColorWheel('HILIGHT', -0.5, 1.5, this.logHighlight || [0, 0, 0], 0.01, (r, g, b) => {
                    this.logHighlight = [r, g, b];
                    if (this.renderer) this.renderer.setLogHighlight(r, g, b);
                    this.requestRender();
                    this.requestScopeUpdate();
                }));
            } else {
                // Lift
                row.appendChild(this.createColorWheel('LIFT', -0.2, 0.2, this.lift || [0, 0, 0], 0.005, (r, g, b) => {
                    this.lift = [r, g, b];
                    if (this.renderer) this.renderer.setLift(r, g, b);
                    this.requestRender();
                    this.requestScopeUpdate();
                }));
                // Gamma
                row.appendChild(this.createColorWheel('GAMMA', -0.5, 0.5, this.gamma ? this.gamma.map(x => x - 1.0) : [0, 0, 0], 0.01, (r, g, b) => {
                    this.gamma = [Math.max(0.1, 1.0 + r), Math.max(0.1, 1.0 + g), Math.max(0.1, 1.0 + b)];
                    if (this.renderer) this.renderer.setGamma(this.gamma[0], this.gamma[1], this.gamma[2]);
                    this.requestRender();
                    this.requestScopeUpdate();
                }));
                // Gain
                row.appendChild(this.createColorWheel('GAIN', -0.5, 1.5, this.gain ? this.gain.map(x => x - 1.0) : [0, 0, 0], 0.01, (r, g, b) => {
                    this.gain = [Math.max(0, 1.0 + r), Math.max(0, 1.0 + g), Math.max(0, 1.0 + b)];
                    if (this.renderer) this.renderer.setGain(this.gain[0], this.gain[1], this.gain[2]);
                    this.requestRender();
                    this.requestScopeUpdate();
                }));
            }

            // Offset (Global) is shared
            row.appendChild(this.createColorWheel('OFFSET', -0.5, 0.5, this.offset || [0, 0, 0], 0.005, (r, g, b) => {
                this.offset = [r, g, b];
                if (this.renderer) this.renderer.setOffset(r, g, b);
                this.requestRender();
                this.requestScopeUpdate();
            }));

            return row;
        };

        if (this.colorScience === undefined) this.colorScience = 0;
        if (!this.activeWheelTab) this.activeWheelTab = 'PRIMARY';
        let wheelContainer = renderWheels();

        ['PRIMARY', 'LOG'].forEach(tab => {
            const btn = document.createElement('div');
            btn.textContent = tab;
            btn.style.cssText = `font-size: 10px; font-weight: bold; cursor: pointer; padding: 2px 8px; border-radius: 4px; color: ${this.activeWheelTab === tab ? '#fff' : '#666'}; background: ${this.activeWheelTab === tab ? 'rgba(255,255,255,0.1)' : 'transparent'};`;
            btn.onclick = () => {
                this.activeWheelTab = tab;
                const newWheels = renderWheels();
                wheelsWrapper.replaceChild(newWheels, wheelContainer);
                wheelContainer = newWheels;
                Array.from(wheelTabs.children).forEach(c => {
                    const isA = c.textContent === tab;
                    c.style.color = isA ? '#fff' : '#666';
                    c.style.background = isA ? 'rgba(255,255,255,0.1)' : 'transparent';
                });
            };
            wheelTabs.appendChild(btn);
        });

        wheelsWrapper.appendChild(wheelTabs);

        const csSelect = document.createElement('select');
        csSelect.style.cssText = 'background: #222; color: #fff; border: 1px solid #444; border-radius: 4px; padding: 2px 4px; font-size: 10px; outline: none; margin-left: auto; cursor: pointer;';
        const optLin = document.createElement('option'); optLin.value = 0; optLin.textContent = 'Linear (sRGB)';
        const optAces = document.createElement('option'); optAces.value = 1; optAces.textContent = 'ACEScct';
        csSelect.appendChild(optLin);
        csSelect.appendChild(optAces);
        csSelect.value = this.colorScience || 0;
        csSelect.onchange = (e) => {
            this.colorScience = parseInt(e.target.value);
            if (this.renderer) this.renderer.setColorScience(this.colorScience);
            this.render();
        };

        wheelTabs.appendChild(csSelect);
        wheelsWrapper.appendChild(wheelContainer);
        container.appendChild(wheelsWrapper);


        // ═════════════════════════════════════════════════════════════════════
        // 3. BOTTOM BAR: Boost | Shadows | Highlights | Sat | Hue | Luma Mix
        // ═════════════════════════════════════════════════════════════════════
        const botBar = document.createElement('div');
        botBar.style.cssText = 'display: flex; flex-wrap: wrap; gap: 2px 0; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; justify-content: space-evenly;';

        // Color Boost
        botBar.appendChild(createMini('C.BOOST', 0.0, 2.0, this.colorBoost || 0.0, 0.05, v => {
            this.colorBoost = v;
            if (this.renderer) this.renderer.setColorBoost(v);
            this.render();
        }));

        // Shadows
        botBar.appendChild(createMini('SHADOWS', -1.0, 1.0, this.shadows || 0.0, 0.05, v => {
            this.shadows = v;
            if (this.renderer) this.renderer.setShadows(v);
            this.render();
        }));

        // Highlights
        botBar.appendChild(createMini('HILIGHT', -1.0, 1.0, this.highlights || 0.0, 0.05, v => {
            this.highlights = v;
            if (this.renderer) this.renderer.setHighlights(v);
            this.render();
        }));

        // Saturation
        botBar.appendChild(createMini('SAT', 0.0, 3.0, this.saturation || 1.0, 0.05, v => {
            this.saturation = v;
            if (this.renderer) this.renderer.setSaturation(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        // Hue
        botBar.appendChild(createMini('HUE', 0.0, 360.0, this.hueShift || 0.0, 1.0, v => {
            this.hueShift = v;
            if (this.renderer) this.renderer.setHueShift(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));

        // Luma Mix
        botBar.appendChild(createMini('LUMA MIX', 0.0, 1.0, this.lumaMix !== undefined ? this.lumaMix : 1.0, 0.05, v => {
            this.lumaMix = v;
            if (this.renderer) this.renderer.setLumaMix(v);
            this.requestRender();
            this.requestScopeUpdate();
        }));


        container.appendChild(botBar);
    }


    renderEffectsTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; flex: 1; gap: 10px; padding: 10px; min-height: 0; overflow-y: auto;';

        // ─── Grain Knobs Row ──────────────────────────────
        const grid = document.createElement('div');
        grid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px 0; justify-content: space-evenly;';

        // Grain Amount
        grid.appendChild(this.createKnob('GRAIN', 0.0, 1.0, this.grain || 0.0, 0.05, v => {
            this.grain = v;
            if (this.renderer) this.renderer.setGrain(v);
            this.requestRender();
        }));

        // Grain Size
        grid.appendChild(this.createKnob('SIZE', 1.0, 4.0, this.grainSize || 1.0, 0.1, v => {
            this.grainSize = v;
            if (this.renderer) this.renderer.setGrainSize(v);
            this.requestRender();
        }));

        // Color Grain
        grid.appendChild(this.createKnob('COLOR', 0.0, 1.0, this.grainColor || 0.0, 0.05, v => {
            this.grainColor = v;
            if (this.renderer) this.renderer.setGrainColor(v);
            this.requestRender();
        }));

        // Denoise
        grid.appendChild(this.createKnob('DENOISE', 0.0, 1.0, this.denoise || 0.0, 0.05, v => {
            this.denoise = v;
            if (this.renderer) this.renderer.setDenoise(v);
            this.requestRender();
        }));

        container.appendChild(grid);

        // Animate toggle — below grain knobs
        const animRow = document.createElement('div');
        animRow.style.cssText = 'display: flex; align-items: center; gap: 10px; margin-top: 2px; margin-bottom: 4px;';

        const animBtn = document.createElement('div');
        const isAnimOn = !!this.grainAnimate;
        animBtn.textContent = isAnimOn ? '⏵ GRAIN ANIMATE  ON' : '⏸ GRAIN ANIMATE  OFF';
        animBtn.style.cssText = `
            font-size: 9px; font-weight: 600; letter-spacing: 0.08em; cursor: pointer;
            padding: 3px 10px; border-radius: 3px; user-select: none;
            border: 1px solid ${isAnimOn ? 'rgba(255,180,60,0.5)' : 'rgba(255,255,255,0.1)'};
            background: ${isAnimOn ? 'rgba(255,180,60,0.12)' : 'rgba(255,255,255,0.03)'};
            color: ${isAnimOn ? '#ffa830' : '#555'};
            transition: all 0.15s;
        `;
        const animNote = document.createElement('div');
        animNote.style.cssText = 'font-size: 8px; color: #444;';
        animNote.textContent = 'Static for stills · Animated for video';

        animBtn.onclick = () => {
            this.grainAnimate = !this.grainAnimate;
            if (this.renderer) this.renderer.setGrainAnimate(this.grainAnimate);
            animBtn.textContent = this.grainAnimate ? '⏵ GRAIN ANIMATE  ON' : '⏸ GRAIN ANIMATE  OFF';
            animBtn.style.borderColor = this.grainAnimate ? 'rgba(255,180,60,0.5)' : 'rgba(255,255,255,0.1)';
            animBtn.style.background = this.grainAnimate ? 'rgba(255,180,60,0.12)' : 'rgba(255,255,255,0.03)';
            animBtn.style.color = this.grainAnimate ? '#ffa830' : '#555';
            this.render();
        };
        animRow.appendChild(animBtn);
        animRow.appendChild(animNote);
        container.appendChild(animRow);

        // ─── Film Stock Presets ───────────────────────────
        const presetLabel = document.createElement('div');
        presetLabel.style.cssText = 'color: #888; font-size: 10px; text-transform: uppercase; margin-top: 6px;';
        presetLabel.textContent = 'Film Stock Presets';
        container.appendChild(presetLabel);

        const presets = document.createElement('div');
        presets.style.cssText = 'display: flex; flex-wrap: wrap; gap: 6px;';

        const INACTIVE_PRESET = `background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);color:#888;padding:4px 8px;border-radius:4px;font-size:10px;cursor:pointer;transition:all 0.15s;`;
        const ACTIVE_PRESET = `background:rgba(0,168,255,0.12);border:1px solid rgba(0,168,255,0.45);color:#00a8ff;padding:4px 8px;border-radius:4px;font-size:10px;cursor:pointer;transition:all 0.15s;`;

        // Track which grain preset is active via a module-level label key
        if (!this._activeFilmPreset) this._activeFilmPreset = 'No Grain';

        const allPresetBtns = [];
        const addPreset = (lbl, gVal, sVal, cVal) => {
            const b = document.createElement('div');
            b.textContent = lbl;
            const isActive = this._activeFilmPreset === lbl;
            b.style.cssText = isActive ? ACTIVE_PRESET : INACTIVE_PRESET;
            allPresetBtns.push({ btn: b, lbl });
            b.onmouseenter = () => { if (this._activeFilmPreset !== lbl) b.style.background = 'rgba(255,255,255,0.1)'; };
            b.onmouseleave = () => { if (this._activeFilmPreset !== lbl) b.style.background = 'rgba(255,255,255,0.04)'; };
            b.onclick = () => {
                this._activeFilmPreset = lbl;
                // Update all pills
                allPresetBtns.forEach(({ btn, lbl: l }) => { btn.style.cssText = l === lbl ? ACTIVE_PRESET : INACTIVE_PRESET; });
                this.grain = gVal;
                this.grainSize = sVal;
                this.grainColor = cVal;
                if (this.renderer) {
                    this.renderer.setGrain(gVal);
                    this.renderer.setGrainSize(sVal);
                    this.renderer.setGrainColor(cVal);
                }
                this.render();
            };
            presets.appendChild(b);
        };

        //                    Label              Grain  Size   Color
        addPreset('No Grain', 0.00, 1.0, 0.0);
        addPreset('Super 8', 0.50, 3.5, 0.4);
        addPreset('16mm', 0.35, 2.5, 0.3);
        addPreset('Super 16', 0.28, 2.0, 0.25);
        addPreset('35mm', 0.15, 1.5, 0.15);
        addPreset('65mm / IMAX', 0.06, 1.0, 0.05);
        addPreset('Kodak 500T', 0.12, 1.3, 0.1);
        addPreset('Kodak 50D', 0.05, 1.0, 0.05);
        addPreset('Kodachrome 64', 0.08, 1.1, 0.1);
        addPreset('Ektachrome 100', 0.10, 1.2, 0.15);
        addPreset('Technicolor', 0.22, 2.2, 0.35);
        addPreset('Fuji Eterna', 0.10, 1.2, 0.08);
        addPreset('CineStill 800T', 0.18, 1.8, 0.2);
        addPreset('Tri-X 400', 0.25, 2.0, 0.0);
        addPreset('Classic Noir', 0.32, 2.8, 0.0);
        addPreset('Push (+2 Stop)', 0.45, 3.2, 0.25);
        addPreset('Expired 35mm', 0.38, 2.6, 0.45);
        addPreset('Vintage 1920s', 0.65, 4.0, 0.1);
        addPreset('Polaroid 600', 0.28, 3.0, 0.2);
        addPreset('Digital Noise', 0.08, 1.0, 0.0);

        container.appendChild(presets);
    }

    renderLensTab(container) {
        // ── Idempotent guard: remove any previous lens content to prevent duplication
        const existingLens = container.querySelector('[data-radiance-lens-tab]');
        if (existingLens) existingLens.remove();

        const lensWrap = document.createElement('div');
        lensWrap.setAttribute('data-radiance-lens-tab', '1');
        lensWrap.style.cssText = 'display: flex; flex-direction: column; gap: 10px; padding: 10px;';

        // 1. Focus & DoF Top Row
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; margin-bottom: 5px; justify-content: space-between;';

        const dofLeft = document.createElement('div');
        dofLeft.style.cssText = 'display: flex; align-items: center; gap: 8px;';

        const dofCheck = document.createElement('div');
        dofCheck.innerHTML = `
        <input type="checkbox" id="dof-enable" ${this.dofEnabled ? 'checked' : ''}>
        <label for="dof-enable" style="color: #ccc; font-size: 11px; margin-left: 4px;">Enable DoF</label>
    `;
        dofCheck.querySelector('input').onchange = (e) => {
            this.dofEnabled = e.target.checked;
            if (this.renderer) this.renderer.setDoFEnabled(this.dofEnabled);
            this.requestRender();
        };
        dofLeft.appendChild(dofCheck);

        // Pick Focus Tool
        const pickBtn = document.createElement('button');
        pickBtn.textContent = '◎ Pick Focus';
        pickBtn.style.cssText = 'background: #333; color: #ccc; border: 1px solid #555; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px;';
        pickBtn.onclick = () => this.activateFocusPicker(() => {
            // Safe re-render: just replace the lens section
            this.renderLensTab(this.tabContentContainer);
        });
        dofLeft.appendChild(pickBtn);

        topRow.appendChild(dofLeft);
        lensWrap.appendChild(topRow);

        // 2. Main Knobs Row (Focus, Aperture, Distortion)
        const knobsRow = document.createElement('div');
        knobsRow.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px 0; justify-content: space-evenly;';

        knobsRow.appendChild(this.createKnob('FOCUS', 0.0, 1.0, this.focusDistance || 0.5, 0.01, v => {
            this.focusDistance = v;
            if (this.renderer) this.renderer.setFocusDistance(v);
            this.requestRender();
        }));

        knobsRow.appendChild(this.createKnob('SIZE', 0.0, 1.0, this.aperture || 0.0, 0.01, v => {
            this.aperture = v;
            if (this.renderer) this.renderer.setAperture(v);
            this.requestRender();
        }));

        knobsRow.appendChild(this.createKnob('DISTORT', -0.5, 0.5, this.lensDistortion || 0.0, 0.01, v => {
            this.lensDistortion = v;
            if (this.renderer) this.renderer.setLensDistortion(v, this.lensFringe || 0.0);
            this.requestRender();
        }));

        lensWrap.appendChild(knobsRow);

        // 3. Bokeh Shape Presets
        const shapeGroup = document.createElement('div');
        shapeGroup.style.marginTop = '10px';
        shapeGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 6px; text-transform: uppercase;">Bokeh Shape</div>';

        // ─── Preset Buttons Row ──────────────────────────
        const presetRow = document.createElement('div');
        presetRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;';

        const presetBtnStyle = `background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: #aaa; padding: 4px 10px; border-radius: 4px; font-size: 10px; cursor: pointer; transition: background 0.15s; `;
        const presetBtnActiveStyle = `background: rgba(100, 140, 255, 0.2); border: 1px solid rgba(100, 140, 255, 0.4); color: #9cf; `;

        const bokehPresets = [
            { label: '● Circle', blades: 0, angle: 0 },
            { label: '⬠ Pent', blades: 5, angle: 0 },
            { label: '⬡ Hex', blades: 6, angle: 0 },
            { label: '⬡ Oct', blades: 8, angle: 0 },
        ];

        const signaturePresets = [
            { label: 'Zeiss MP', blades: 0, angle: 0, anamorphic: 1.0, distort: 0.0, fringe: 0.0, halation: 0.05, diffusion: 0.0 },
            { label: 'Cooke S4', blades: 8, angle: 22, anamorphic: 1.0, distort: 0.04, fringe: 0.35, halation: 0.15, diffusion: 0.2 },
            { label: 'Anamorphic', blades: 0, angle: 0, anamorphic: 2.0, distort: 0.18, fringe: 0.75, halation: 0.1, halRadius: 1.0, diffusion: 0.15, streaks: 0.5, streakThreshold: 0.78, streakLength: 0.12 },
            { label: 'Petzval', blades: 0, angle: 0, anamorphic: 1.0, distort: -0.1, fringe: 0.0, opticalVig: 0.5, highlight: 0.4 },
            { label: 'Dreamy', blades: 0, angle: 0, anamorphic: 1.0, distort: 0.0, fringe: 0.2, halation: 0.4, diffusion: 0.6 }
        ];

        const currentBlades = this.apertureBlades || 0;
        bokehPresets.forEach(p => {
            const btn = document.createElement('div');
            btn.textContent = p.label;
            const isActive = currentBlades === p.blades;
            btn.style.cssText = presetBtnStyle + (isActive ? presetBtnActiveStyle : '');
            btn.onmouseenter = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.12)'; };
            btn.onmouseleave = () => { if (!isActive) btn.style.background = 'rgba(255,255,255,0.05)'; };
            btn.onclick = () => {
                this.apertureBlades = p.blades;
                this.apertureRotation = p.angle;
                if (this.renderer) this.renderer.setApertureShape(p.blades, p.angle, this.apertureAnamorphic || 1.0);
                this.render();
                // Safe: renderLensTab is now idempotent (removes old content first)
                this.renderLensTab(this.tabContentContainer);
            };
            presetRow.appendChild(btn);
        });

        shapeGroup.appendChild(presetRow);

        // ─── Lens Signature Row ──────────────────────────
        const sigLabel = document.createElement('div');
        sigLabel.style.cssText = 'color: #888; font-size: 10px; margin-bottom: 6px; text-transform: uppercase; margin-top: 8px;';
        sigLabel.textContent = 'Lens Signatures';
        shapeGroup.appendChild(sigLabel);

        const sigRow = document.createElement('div');
        sigRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;';

        signaturePresets.forEach(p => {
            const btn = document.createElement('div');
            btn.textContent = p.label;
            btn.dataset.sigLabel = p.label;
            const isActiveSig = this.activeLensSignature === p.label;
            btn.style.cssText = presetBtnStyle + (isActiveSig ? presetBtnActiveStyle : '');
            btn.onmouseenter = () => { if (this.activeLensSignature !== p.label) btn.style.background = 'rgba(255,255,255,0.12)'; };
            btn.onmouseleave = () => { if (this.activeLensSignature !== p.label) btn.style.background = 'rgba(255,255,255,0.05)'; };
            btn.onclick = () => {
                const isAlreadyActive = this.activeLensSignature === p.label;

                if (isAlreadyActive) {
                    // Toggle OFF — reset to neutral
                    this.activeLensSignature = null;
                    this.apertureBlades = 0; this.apertureRotation = 0;
                    this.apertureAnamorphic = 1.0;
                    this.lensDistortion = 0.0; this.lensFringe = 0.0;
                    this.halation = 0.0; this.diffusion = 0.0;
                    this.halationRadius = 1.0; this.halationThreshold = 0.35; this.bloomThreshold = 1.0;
                    this.bokehOpticalVig = 0.0; this.bokehHighlightBias = 0.0;
                    this.anamorphicStreaks = 0.0;
                    this.streakThreshold = 0.85; this.streakLength = 0.08;
                } else {
                    // Apply preset
                    this.activeLensSignature = p.label;
                    this.apertureBlades = p.blades;
                    this.apertureRotation = p.angle;
                    this.apertureAnamorphic = p.anamorphic || 1.0;
                    this.lensDistortion = p.distort || 0.0;
                    this.lensFringe = p.fringe || 0.0;
                    this.halation = p.halation || 0.0;
                    this.halationRadius = p.halRadius || 1.0;
                    this.halationThreshold = p.halThreshold || 0.35;
                    this.bloomThreshold = p.bloomThreshold || 1.0;
                    this.diffusion = p.diffusion || 0.0;
                    this.bokehOpticalVig = p.opticalVig || 0.0;
                    this.bokehHighlightBias = p.highlight || 0.0;
                    this.anamorphicStreaks = p.streaks || 0.0;
                    this.streakThreshold = p.streakThreshold || 0.85;
                    this.streakLength = p.streakLength || 0.08;
                }

                if (this.renderer) {
                    this.renderer.setApertureShape(this.apertureBlades, this.apertureRotation, this.apertureAnamorphic);
                    this.renderer.setLensDistortion(this.lensDistortion, this.lensFringe);
                    this.renderer.setHalation(this.halation);
                    if (this.renderer.setBloomThreshold) this.renderer.setBloomThreshold(this.bloomThreshold);
                    if (this.renderer.setHalationRadius) this.renderer.setHalationRadius(this.halationRadius);
                    if (this.renderer.setHalationThreshold) this.renderer.setHalationThreshold(this.halationThreshold);
                    this.renderer.setDiffusion(this.diffusion);
                    this.renderer.setBokehPhysics(this.bokehHighlightBias, this.bokehSoapBubble || 0.0, this.bokehOpticalVig);
                    if (this.renderer.setAnamorphicStreaks) this.renderer.setAnamorphicStreaks(this.anamorphicStreaks);
                    if (this.renderer.setStreakThreshold) this.renderer.setStreakThreshold(this.streakThreshold);
                    if (this.renderer.setStreakLength) this.renderer.setStreakLength(this.streakLength);
                }
                this.render();
                // Safe: renderLensTab is now idempotent (removes old content first)
                this.renderLensTab(this.tabContentContainer);
            };
            sigRow.appendChild(btn);
        });
        shapeGroup.appendChild(sigRow);

        // ─── Manual Shape Knobs ──────────────────────────
        const shapeGrid = document.createElement('div');
        shapeGrid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px 0; justify-content: space-evenly;';

        shapeGrid.appendChild(this.createKnob('BLADES', 0, 9, this.apertureBlades || 0, 1, v => {
            this.apertureBlades = Math.round(v);
            if (this.renderer) this.renderer.setApertureShape(this.apertureBlades, this.apertureRotation || 0, this.apertureAnamorphic || 1.0);
            this.render();
        }));

        shapeGrid.appendChild(this.createKnob('ANGLE', 0, 360, this.apertureRotation || 0, 1, v => {
            this.apertureRotation = v;
            if (this.renderer) this.renderer.setApertureShape(this.apertureBlades || 0, this.apertureRotation, this.apertureAnamorphic || 1.0);
            this.render();
        }));

        shapeGrid.appendChild(this.createKnob('RATIO', 1.0, 2.0, this.apertureAnamorphic || 1.0, 0.05, v => {
            this.apertureAnamorphic = v;
            if (this.renderer) this.renderer.setApertureShape(this.apertureBlades || 0, this.apertureRotation || 0, this.apertureAnamorphic);
            this.render();
        }));

        shapeGroup.appendChild(shapeGrid);
        lensWrap.appendChild(shapeGroup);

        // 4. Optical Filters
        const filterGroup = document.createElement('div');
        filterGroup.style.marginTop = '10px';
        filterGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase;">Optical Filters</div>';

        const filterGrid = document.createElement('div');
        filterGrid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px 0; justify-content: space-evenly;';

        filterGrid.appendChild(this.createKnob('FRINGE', 0.0, 2.0, this.lensFringe || 0.0, 0.05, v => {
            this.lensFringe = v;
            if (this.renderer) this.renderer.setLensDistortion(this.lensDistortion || 0.0, v);
            this.render();
        }));

        filterGrid.appendChild(this.createKnob('VIG INT', 0.0, 1.0, this.vignetteIntensity || 0.0, 0.01, v => {
            this.vignetteIntensity = v;
            if (this.renderer) this.renderer.setVignette(v, this.vignetteFalloff || 0.5);
            this.render();
        }));

        filterGrid.appendChild(this.createKnob('VIG FALL', 0.1, 1.0, this.vignetteFalloff || 0.5, 0.05, v => {
            this.vignetteFalloff = v;
            if (this.renderer) this.renderer.setVignette(this.vignetteIntensity || 0.0, v);
            this.render();
        }));

        filterGroup.appendChild(filterGrid);
        lensWrap.appendChild(filterGroup);

        // 5. Lens Effects (Bloom, Halation, Diffusion)
        const fxGroup = document.createElement('div');
        fxGroup.style.marginTop = '10px';
        fxGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase;">Lens Effects</div>';

        const fxGrid = document.createElement('div');
        fxGrid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px 0; justify-content: space-evenly;';

        const diffPresets = [
            { label: 'Pro-Mist 1/4', bloom: 0.15, threshold: 1.0, halation: 0.1, halRadius: 0.7, halThreshold: 0.35, diffusion: 0.2 },
            { label: 'Pro-Mist 1/2', bloom: 0.25, threshold: 0.95, halation: 0.15, halRadius: 0.85, halThreshold: 0.32, diffusion: 0.35 },
            { label: 'Glimmerglass', bloom: 0.4, threshold: 0.85, halation: 0.05, halRadius: 0.6, halThreshold: 0.38, diffusion: 0.15 },
            { label: 'H.Black Magic', bloom: 0.2, threshold: 0.9, halation: 0.3, halRadius: 1.0, halThreshold: 0.28, diffusion: 0.4 },
            { label: 'Strong Preview', bloom: 0.65, threshold: 0.65, halation: 0.55, halRadius: 1.0, halThreshold: 0.22, diffusion: 0.25, streaks: 0.35, streakThreshold: 0.7, streakLength: 0.14 },
        ];

        const dfRow = document.createElement('div');
        dfRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;';
        diffPresets.forEach(p => {
            const btn = document.createElement('button');
            btn.textContent = p.label;
            btn.style.cssText = presetBtnStyle;
            btn.onmouseenter = () => btn.style.background = 'rgba(255,255,255,0.12)';
            btn.onmouseleave = () => btn.style.background = 'rgba(255,255,255,0.05)';
            btn.onclick = () => {
                this.bloom = p.bloom;
                this.bloomThreshold = p.threshold;
                this.halation = p.halation;
                this.halationRadius = p.halRadius;
                this.halationThreshold = p.halThreshold;
                this.diffusion = p.diffusion;
                if (p.streaks !== undefined) this.anamorphicStreaks = p.streaks;
                if (p.streakThreshold !== undefined) this.streakThreshold = p.streakThreshold;
                if (p.streakLength !== undefined) this.streakLength = p.streakLength;
                if (this.renderer) {
                    this.renderer.setBloom(p.bloom);
                    if (this.renderer.setBloomThreshold) this.renderer.setBloomThreshold(p.threshold);
                    this.renderer.setHalation(p.halation);
                    if (this.renderer.setHalationRadius) this.renderer.setHalationRadius(p.halRadius);
                    if (this.renderer.setHalationThreshold) this.renderer.setHalationThreshold(p.halThreshold);
                    this.renderer.setDiffusion(p.diffusion);
                    if (p.streaks !== undefined && this.renderer.setAnamorphicStreaks) this.renderer.setAnamorphicStreaks(p.streaks);
                    if (p.streakThreshold !== undefined && this.renderer.setStreakThreshold) this.renderer.setStreakThreshold(p.streakThreshold);
                    if (p.streakLength !== undefined && this.renderer.setStreakLength) this.renderer.setStreakLength(p.streakLength);
                }
                this.render();
                // Safe: renderLensTab is now idempotent (removes old content first)
                this.renderLensTab(this.tabContentContainer);
            };
            dfRow.appendChild(btn);
        });
        fxGroup.appendChild(dfRow);

        fxGrid.appendChild(this.createKnob('BLOOM', 0.0, 2.0, this.bloom || 0.0, 0.01, v => {
            this.bloom = v;
            if (this.renderer) this.renderer.setBloom(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('BLOOM THR', 0.5, 2.0, this.bloomThreshold ?? 1.0, 0.05, v => {
            this.bloomThreshold = v;
            if (this.renderer && this.renderer.setBloomThreshold) this.renderer.setBloomThreshold(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('HALATION', 0.0, 2.0, this.halation || 0.0, 0.01, v => {
            this.halation = v;
            if (this.renderer) this.renderer.setHalation(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('HAL RADIUS', 0.0, 1.0, this.halationRadius ?? 1.0, 0.05, v => {
            this.halationRadius = v;
            if (this.renderer && this.renderer.setHalationRadius) this.renderer.setHalationRadius(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('HAL THR', 0.05, 1.0, this.halationThreshold ?? 0.35, 0.01, v => {
            this.halationThreshold = v;
            if (this.renderer && this.renderer.setHalationThreshold) this.renderer.setHalationThreshold(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('DIFFUSION', 0.0, 1.0, this.diffusion || 0.0, 0.01, v => {
            this.diffusion = v;
            if (this.renderer) this.renderer.setDiffusion(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('STREAK', 0.0, 2.0, this.anamorphicStreaks || 0.0, 0.01, v => {
            this.anamorphicStreaks = v;
            if (this.renderer && this.renderer.setAnamorphicStreaks) this.renderer.setAnamorphicStreaks(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('STR THR', 0.1, 1.0, this.streakThreshold ?? 0.85, 0.01, v => {
            this.streakThreshold = v;
            if (this.renderer && this.renderer.setStreakThreshold) this.renderer.setStreakThreshold(v);
            this.render();
        }));

        fxGrid.appendChild(this.createKnob('STR LEN', 0.02, 0.25, this.streakLength ?? 0.08, 0.01, v => {
            this.streakLength = v;
            if (this.renderer && this.renderer.setStreakLength) this.renderer.setStreakLength(v);
            this.render();
        }));

        fxGroup.appendChild(fxGrid);
        lensWrap.appendChild(fxGroup);

        // 6. Bokeh Physics
        const bokehGroup = document.createElement('div');
        bokehGroup.style.marginTop = '10px';
        bokehGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 4px; text-transform: uppercase;">Bokeh Physics</div>';

        const bokehGrid = document.createElement('div');
        bokehGrid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px 0; justify-content: space-evenly;';

        bokehGrid.appendChild(this.createKnob('HIGHLIGHT', 0.0, 5.0, this.bokehHighlightBias || 0.0, 0.1, v => {
            this.bokehHighlightBias = v;
            if (this.renderer) this.renderer.setBokehPhysics(this.bokehHighlightBias, this.bokehSoapBubble || 0.0, this.bokehOpticalVig || 0.0);
            this.render();
        }));

        bokehGrid.appendChild(this.createKnob('SOAP BBL', 0.0, 2.0, this.bokehSoapBubble || 0.0, 0.05, v => {
            this.bokehSoapBubble = v;
            if (this.renderer) this.renderer.setBokehPhysics(this.bokehHighlightBias || 0.0, this.bokehSoapBubble, this.bokehOpticalVig || 0.0);
            this.render();
        }));

        bokehGrid.appendChild(this.createKnob('OPTIC VIG', 0.0, 1.0, this.bokehOpticalVig || 0.0, 0.05, v => {
            this.bokehOpticalVig = v;
            if (this.renderer) this.renderer.setBokehPhysics(this.bokehHighlightBias || 0.0, this.bokehSoapBubble || 0.0, this.bokehOpticalVig);
            this.render();
        }));

        bokehGroup.appendChild(bokehGrid);
        lensWrap.appendChild(bokehGroup);

        // Append the complete lens wrapper to the container
        container.appendChild(lensWrap);
    }

    renderCurvesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; align-items: stretch; gap: 0; padding: 0;';

        // ════════════════════════════════════════════════════════════════
        //  DaVinci Resolve–style two-panel layout:
        //    LEFT:  Curve editor canvas (fills available space)
        //    RIGHT: Edit controls + Soft Clip panel (~190px sidebar)
        // ════════════════════════════════════════════════════════════════

        const mainRow = document.createElement('div');
        mainRow.style.cssText = 'display: flex; flex: 1; min-height: 280px; border: 1px solid rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden; background: #1a1a22;';
        container.appendChild(mainRow);

        // ── LEFT: Curve Editor Canvas ───────────────────────────────────
        const editorContainer = document.createElement('div');
        editorContainer.style.cssText = 'flex: 1; position: relative; min-width: 0;';
        mainRow.appendChild(editorContainer);

        // ── RIGHT: Controls Sidebar ─────────────────────────────────────
        const sidebar = document.createElement('div');
        sidebar.style.cssText = `
            width: 190px; min-width: 190px; background: #1e1e28;
            border-left: 1px solid rgba(255,255,255,0.06);
            display: flex; flex-direction: column; overflow-y: auto;
            font-family: ${this.theme.mono};
        `;
        mainRow.appendChild(sidebar);

        // ─────────────────────────────────────────────────────────────────
        //  Initialize Curve Editor
        // ─────────────────────────────────────────────────────────────────
        if (!this.curveEditor) {
            this.curveEditor = new RadianceCurveEditor(280, 280, this.theme, (data, secData) => {
                if (this.renderer) {
                    this.renderer.updateCurveLut(data);
                    // Note: setCurveSlope() removed (v4.2) — u_curveSlope was a dead
                    // uniform after FIX 6 replaced slope-based with ratio-based HDR
                    // extrapolation. The topVal sample at uv=(1,0.5) handles it fully.

                    if (secData) {
                        this.renderer.updateSecondaryCurveLut(secData);
                        const isIdentity = secData.every((v, i) => i % 4 === 3 ? true : Math.abs(v - 0.5) < 0.001);
                        if (!isIdentity) {
                            this.renderer.setSecondaryCurveMix(1.0);
                        } else {
                            this.renderer.setSecondaryCurveMix(0.0);
                        }
                    }
                    this.renderer.setCurveMix(this.curveMix !== undefined ? this.curveMix : 1.0);
                    this.render();
                }
            });
            if (this.image) this.curveEditor.updateHistogram(this.image);
            this.curveEditor.notifyChange();
        }

        editorContainer.appendChild(this.curveEditor.canvas);
        this.curveEditor.canvas.style.cssText = 'width: 100%; height: 100%; display: block;';

        // Observe resize — clean up previous observer if tab was re-rendered
        if (this._curveResizeObs) {
            this._curveResizeObs.disconnect();
        }
        this._curveResizeObs = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    this.curveEditor.resize(Math.round(width), Math.round(height));
                }
            }
        });
        this._curveResizeObs.observe(editorContainer);

        // ─────────────────────────────────────────────────────────────────
        //  SIDEBAR: Edit Section
        // ─────────────────────────────────────────────────────────────────
        const editSection = document.createElement('div');
        editSection.style.cssText = 'padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.06);';
        sidebar.appendChild(editSection);

        // "Edit" header with pencil + channel buttons
        const editHeader = document.createElement('div');
        editHeader.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 10px;';
        editSection.appendChild(editHeader);

        const editLabel = document.createElement('div');
        editLabel.textContent = 'Edit';
        editLabel.style.cssText = 'color: #999; font-size: 11px; font-weight: 600; margin-right: 4px;';
        editHeader.appendChild(editLabel);

        // Pencil icon
        const editPen = document.createElement('div');
        editPen.innerHTML = '✎';
        editPen.style.cssText = 'color: #666; font-size: 12px; margin-right: 2px;';
        editHeader.appendChild(editPen);

        // Y / R / G / B channel buttons — DaVinci colored squares
        const channelMap = [
            { ch: 'RGB', label: 'Y', color: '#e0e0e0', bg: 'rgba(255,255,255,0.12)' },
            { ch: 'R', label: 'R', color: '#ff4444', bg: 'rgba(255,68,68,0.2)' },
            { ch: 'G', label: 'G', color: '#44cc44', bg: 'rgba(68,204,68,0.2)' },
            { ch: 'B', label: 'B', color: '#4488ff', bg: 'rgba(68,136,255,0.2)' }
        ];

        const channelBtns = {};

        const refreshChannelBtns = () => {
            channelMap.forEach(({ ch, color, bg }) => {
                const btn = channelBtns[ch];
                if (!btn) return;
                const isActive = this.curveEditor.activeChannel === ch;
                btn.style.background = isActive ? bg : 'rgba(255,255,255,0.04)';
                btn.style.borderColor = isActive ? color : 'rgba(255,255,255,0.1)';
                btn.style.color = isActive ? '#fff' : color;
                btn.style.boxShadow = isActive ? `0 0 6px ${color}44` : 'none';
            });
        };

        channelMap.forEach(({ ch, label, color, bg }) => {
            const btn = document.createElement('div');
            btn.textContent = label;
            btn.style.cssText = `
                width: 22px; height: 20px;
                background: rgba(255,255,255,0.04);
                color: ${color};
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 3px;
                display: flex; align-items: center; justify-content: center;
                font-size: 11px; font-weight: bold; cursor: pointer;
                transition: all 0.12s;
            `;
            btn.onmouseenter = () => { if (this.curveEditor.activeChannel !== ch) btn.style.background = 'rgba(255,255,255,0.08)'; };
            btn.onmouseleave = () => refreshChannelBtns();
            btn.onclick = () => {
                this.curveEditor.setActiveChannel(ch);
                refreshChannelBtns();
                refreshGainInputs();
            };
            channelBtns[ch] = btn;
            editHeader.appendChild(btn);
        });

        // Copy to All Button
        const copyBtn = document.createElement('div');
        copyBtn.textContent = '📋';
        copyBtn.title = 'Copy RGB to All Channels';
        copyBtn.style.cssText = 'width: 22px; height: 20px; background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); border-radius: 3px; display: flex; align-items: center; justify-content: center; font-size: 10px; cursor: pointer; transition: all 0.15s; margin-left: 2px;';
        copyBtn.onmouseenter = () => copyBtn.style.background = 'rgba(255,255,255,0.1)';
        copyBtn.onmouseleave = () => copyBtn.style.background = 'rgba(0,0,0,0.4)';
        copyBtn.onclick = () => {
            this.curveEditor.copyRGBToAll();
            this._lastRenderContent();
        };
        editHeader.appendChild(copyBtn);

        refreshChannelBtns();

        // Per-channel gain rows (DaVinci "intensity" inputs)
        const gainContainer = document.createElement('div');
        gainContainer.style.cssText = 'display: flex; flex-direction: column; gap: 4px;';
        editSection.appendChild(gainContainer);

        const gainInputs = {};

        const refreshGainInputs = () => {
            Object.keys(gainInputs).forEach(key => {
                const gainKey = key === 'RGB' ? 'Y' : key;
                gainInputs[key].value = this.curveEditor.channelGain[gainKey];
            });
        };

        const createGainRow = (ch, label, dotColor) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; align-items: center; gap: 6px; height: 24px;';

            // Color dot
            const dot = document.createElement('div');
            dot.style.cssText = `width: 6px; height: 6px; border-radius: 50%; background: ${dotColor}; flex-shrink: 0;`;
            row.appendChild(dot);

            // Numeric input
            const input = document.createElement('input');
            input.type = 'number';
            input.min = '0'; input.max = '200'; input.step = '1';
            const gainKey = ch === 'RGB' ? 'Y' : ch;
            input.value = this.curveEditor.channelGain[gainKey];
            input.style.cssText = `
                width: 50px; background: #2a2a35; color: #ddd;
                border: 1px solid rgba(255,255,255,0.1); border-radius: 3px;
                font-size: 11px; font-family: ${this.theme.mono};
                padding: 2px 6px; text-align: center; outline: none;
            `;
            input.onfocus = () => { input.style.borderColor = 'rgba(255,255,255,0.3)'; };
            input.onblur = () => { input.style.borderColor = 'rgba(255,255,255,0.1)'; };
            input.oninput = () => {
                const v = Math.max(0, Math.min(200, parseInt(input.value) || 0));
                this.curveEditor.channelGain[gainKey] = v;
                this.curveEditor.notifyChange();
            };
            gainInputs[ch] = input;
            row.appendChild(input);

            // Reset ↺ per channel
            const resetBtn = document.createElement('div');
            resetBtn.innerHTML = '↺';
            resetBtn.title = `Reset ${label}`;
            resetBtn.style.cssText = 'color: #555; cursor: pointer; font-size: 14px; margin-left: auto; transition: color 0.15s;';
            resetBtn.onmouseenter = () => { resetBtn.style.color = '#aaa'; };
            resetBtn.onmouseleave = () => { resetBtn.style.color = '#555'; };
            resetBtn.onclick = () => {
                this.curveEditor.channelGain[gainKey] = 100;
                input.value = 100;
                if (this.curveEditor.activeChannel === ch) {
                    this.curveEditor.resetActiveChannel();
                }
                this.curveEditor.notifyChange();
                this._lastRenderContent();
            };
            row.appendChild(resetBtn);

            return row;
        };

        gainContainer.appendChild(createGainRow('RGB', 'Y (Master)', '#e0e0e0'));
        gainContainer.appendChild(createGainRow('R', 'Red', '#ff4444'));
        gainContainer.appendChild(createGainRow('G', 'Green', '#44cc44'));
        gainContainer.appendChild(createGainRow('B', 'Blue', '#4488ff'));

        // ─────────────────────────────────────────────────────────────────
        //  SIDEBAR: Soft Clip Section
        // ─────────────────────────────────────────────────────────────────
        const softClipSection = document.createElement('div');
        softClipSection.style.cssText = 'padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.06);';
        sidebar.appendChild(softClipSection);

        // Header with toggle + R/G/B buttons
        const scHeader = document.createElement('div');
        scHeader.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 10px;';
        softClipSection.appendChild(scHeader);

        const scLabel = document.createElement('div');
        scLabel.textContent = 'Soft Clip';
        scLabel.style.cssText = 'color: #999; font-size: 11px; font-weight: 600; margin-right: 4px;';
        scHeader.appendChild(scLabel);

        // Enable toggle (pen icon)
        const scToggle = document.createElement('div');
        scToggle.innerHTML = '✎';
        scToggle.style.cssText = `
            color: ${this.curveEditor.softClipEnabled ? '#fff' : '#555'};
            font-size: 12px; cursor: pointer; transition: color 0.15s; margin-right: 2px;
        `;
        scToggle.onclick = () => {
            this.curveEditor.softClipEnabled = !this.curveEditor.softClipEnabled;
            scToggle.style.color = this.curveEditor.softClipEnabled ? '#fff' : '#555';
            this.curveEditor.notifyChange();
            this.render();
        };
        scHeader.appendChild(scToggle);

        // R / G / B channel toggles for Soft Clip
        ['R', 'G', 'B'].forEach(ch => {
            const colors = { R: '#ff4444', G: '#44cc44', B: '#4488ff' };
            const btn = document.createElement('div');
            btn.textContent = ch;
            const isOn = this.curveEditor.softClipChannels[ch];
            btn.style.cssText = `
                width: 22px; height: 20px;
                background: ${isOn ? colors[ch] + '33' : 'rgba(255,255,255,0.04)'};
                color: ${isOn ? '#fff' : colors[ch]};
                border: 1px solid ${isOn ? colors[ch] : 'rgba(255,255,255,0.1)'};
                border-radius: 3px;
                display: flex; align-items: center; justify-content: center;
                font-size: 10px; font-weight: bold; cursor: pointer;
                transition: all 0.12s;
            `;
            btn.onclick = () => {
                this.curveEditor.softClipChannels[ch] = !this.curveEditor.softClipChannels[ch];
                const on = this.curveEditor.softClipChannels[ch];
                btn.style.background = on ? colors[ch] + '33' : 'rgba(255,255,255,0.04)';
                btn.style.color = on ? '#fff' : colors[ch];
                btn.style.borderColor = on ? colors[ch] : 'rgba(255,255,255,0.1)';
                this.curveEditor.notifyChange();
            };
            scHeader.appendChild(btn);
        });

        // Soft Clip sliders
        const scSliders = document.createElement('div');
        scSliders.style.cssText = 'display: flex; flex-direction: column; gap: 6px;';
        softClipSection.appendChild(scSliders);

        const createSCSlider = (label, paramKey, min, max, step) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; align-items: center; gap: 6px;';

            const lbl = document.createElement('div');
            lbl.textContent = label;
            lbl.style.cssText = 'color: #777; font-size: 10px; min-width: 55px;';
            row.appendChild(lbl);

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = min; slider.max = max; slider.step = step;
            slider.value = this.curveEditor.softClipParams[paramKey];
            slider.style.cssText = 'flex: 1; accent-color: #888; height: 3px; cursor: pointer;';
            slider.oninput = () => {
                this.curveEditor.softClipParams[paramKey] = parseFloat(slider.value);
                this.curveEditor.notifyChange();
                this.render();
            };
            row.appendChild(slider);

            return row;
        };

        scSliders.appendChild(createSCSlider('Low', 'low', '0', '0.5', '0.005'));
        scSliders.appendChild(createSCSlider('Low Soft', 'lowSoft', '0', '0.5', '0.005'));
        scSliders.appendChild(createSCSlider('High', 'high', '0.5', '1.0', '0.005'));
        scSliders.appendChild(createSCSlider('High Soft', 'highSoft', '0', '0.5', '0.005'));

        // ─────────────────────────────────────────────────────────────────
        //  SIDEBAR: Levels Section
        // ─────────────────────────────────────────────────────────────────
        const levelsSection = document.createElement('div');
        levelsSection.style.cssText = 'padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.06);';
        sidebar.appendChild(levelsSection);

        const levelsTitle = document.createElement('div');
        levelsTitle.textContent = 'Levels';
        levelsTitle.style.cssText = 'color: #999; font-size: 11px; font-weight: 600; margin-bottom: 8px;';
        levelsSection.appendChild(levelsTitle);

        const createLevelSlider = (label, min, max, val, setter) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 4px;';
            const lbl = document.createElement('div');
            lbl.style.cssText = 'color: #777; font-size: 10px; min-width: 55px;';
            lbl.textContent = label;
            row.appendChild(lbl);

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = min; slider.max = max; slider.step = '1';
            slider.value = val;
            slider.style.cssText = 'flex: 1; accent-color: #888; height: 3px; cursor: pointer;';

            const valLabel = document.createElement('div');
            valLabel.style.cssText = `color: #888; font-size: 9px; min-width: 24px; text-align: right; font-family: ${this.theme.mono};`;
            valLabel.textContent = val;

            slider.oninput = (e) => {
                const v = parseInt(e.target.value) || 0;
                valLabel.textContent = v;
                setter(v);
            };
            row.appendChild(slider);
            row.appendChild(valLabel);
            return row;
        };

        let _inBlackSlider, _inWhiteSlider;
        const _updateSliderConstraints = () => {
            if (!_inBlackSlider || !_inWhiteSlider) return;
            const bv = parseInt(_inBlackSlider.value) || 0;
            const wv = parseInt(_inWhiteSlider.value) || 255;
            _inBlackSlider.max = String(wv - 5);
            _inWhiteSlider.min = String(bv + 5);
        };

        const inBlackRow = createLevelSlider('In Black', 0, 250, this.curveEditor.levels.inBlack, (v) => {
            this.curveEditor.setLevels(v, this.curveEditor.levels.inWhite);
            _updateSliderConstraints();
        });
        const inWhiteRow = createLevelSlider('In White', 5, 255, this.curveEditor.levels.inWhite, (v) => {
            this.curveEditor.setLevels(this.curveEditor.levels.inBlack, v);
            _updateSliderConstraints();
        });

        _inBlackSlider = inBlackRow.querySelector('input[type="range"]');
        _inWhiteSlider = inWhiteRow.querySelector('input[type="range"]');
        _updateSliderConstraints();

        levelsSection.appendChild(inBlackRow);
        levelsSection.appendChild(inWhiteRow);

        // ─────────────────────────────────────────────────────────────────
        //  SIDEBAR: Presets & Mix
        // ─────────────────────────────────────────────────────────────────
        const bottomSection = document.createElement('div');
        bottomSection.style.cssText = 'padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.06);';
        sidebar.appendChild(bottomSection);

        // HDR Range + Presets row
        const rangePresetsRow = document.createElement('div');
        rangePresetsRow.style.cssText = 'display: flex; gap: 6px; margin-bottom: 8px;';
        bottomSection.appendChild(rangePresetsRow);

        // HDR Range Dropdown
        const rangeSelect = document.createElement('select');
        rangeSelect.title = 'HDR Range Visualization';
        rangeSelect.style.cssText = `
            flex: 0 0 auto; background: #2a2a35; color: #e2a;
            border: 1px solid rgba(221,34,170,0.3); border-radius: 3px;
            font-size: 10px; padding: 2px 6px; outline: none; cursor: pointer;
            font-family: ${this.theme.mono}; font-weight: bold;
        `;
        ['1x', '2x', '4x'].forEach(r => {
            const opt = document.createElement('option');
            opt.value = parseFloat(r);
            opt.textContent = r;
            rangeSelect.appendChild(opt);
        });
        rangeSelect.onchange = (e) => {
            this.curveEditor.rangeY = parseFloat(e.target.value);
            this.curveEditor.draw();
        };
        rangePresetsRow.appendChild(rangeSelect);

        // Presets dropdown
        const presetsSelect = document.createElement('select');
        presetsSelect.style.cssText = `
            flex: 1; background: #2a2a35; color: #ccc;
            border: 1px solid rgba(255,255,255,0.1); border-radius: 3px;
            font-size: 10px; padding: 2px 6px; outline: none; cursor: pointer;
            font-family: ${this.theme.mono};
        `;
        ['Presets...', 'Punchy', 'High Contrast', 'Flat / Log', 'Shadow Lift', 'S-Curve', 'Film Print', 'Bleach Bypass', 'Cross Process'].forEach(p => {
            const opt = document.createElement('option');
            opt.value = p === 'Presets...' ? '' : p;
            opt.textContent = p;
            presetsSelect.appendChild(opt);
        });
        presetsSelect.onchange = (e) => {
            if (e.target.value) {
                this.curveEditor.applyIndustryPreset(e.target.value);
                e.target.value = '';
                this._lastRenderContent();
            }
        };
        rangePresetsRow.appendChild(presetsSelect);

        // Mix slider
        const mixRow = document.createElement('div');
        mixRow.style.cssText = 'display: flex; align-items: center; gap: 6px;';

        const mixLabel = document.createElement('div');
        mixLabel.textContent = 'MIX';
        mixLabel.style.cssText = 'color: #666; font-size: 9px; font-weight: 700; min-width: 26px;';
        mixRow.appendChild(mixLabel);

        const mixSlider = document.createElement('input');
        mixSlider.type = 'range';
        mixSlider.min = '0'; mixSlider.max = '100'; mixSlider.step = '1';
        mixSlider.value = String(Math.round((this.curveMix !== undefined ? this.curveMix : 1.0) * 100));
        mixSlider.style.cssText = 'flex: 1; accent-color: #888; height: 3px; cursor: pointer;';

        const mixValue = document.createElement('div');
        mixValue.style.cssText = `color: #888; font-size: 9px; min-width: 28px; text-align: right; font-family: ${this.theme.mono};`;
        mixValue.textContent = Math.round((this.curveMix !== undefined ? this.curveMix : 1.0) * 100) + '%';

        mixSlider.oninput = (e) => {
            this.curveMix = (parseInt(e.target.value) || 0) / 100;
            if (this.renderer) this.renderer.setCurveMix(this.curveMix);
            mixValue.textContent = e.target.value + '%';
            this.render();
        };
        mixRow.appendChild(mixSlider);
        mixRow.appendChild(mixValue);
        bottomSection.appendChild(mixRow);

        // Reset buttons
        const resetRow = document.createElement('div');
        resetRow.style.cssText = 'display: flex; gap: 6px; margin-top: 10px;';
        bottomSection.appendChild(resetRow);

        const resetChBtn = document.createElement('button');
        resetChBtn.textContent = '↺ Channel';
        resetChBtn.style.cssText = `
            flex: 1; padding: 4px; background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08); border-radius: 3px;
            color: #888; font-size: 9px; cursor: pointer; transition: all 0.15s;
            font-family: ${this.theme.mono};
        `;
        resetChBtn.onmouseenter = () => { resetChBtn.style.color = '#fff'; resetChBtn.style.borderColor = 'rgba(255,255,255,0.2)'; };
        resetChBtn.onmouseleave = () => { resetChBtn.style.color = '#888'; resetChBtn.style.borderColor = 'rgba(255,255,255,0.08)'; };
        resetChBtn.onclick = () => { this.curveEditor.resetActiveChannel(); this._lastRenderContent(); };
        resetRow.appendChild(resetChBtn);

        const resetAllBtn = document.createElement('button');
        resetAllBtn.textContent = '⟲ All';
        resetAllBtn.style.cssText = `
            flex: 1; padding: 4px; background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08); border-radius: 3px;
            color: #888; font-size: 9px; cursor: pointer; transition: all 0.15s;
            font-family: ${this.theme.mono};
        `;
        resetAllBtn.onmouseenter = () => { resetAllBtn.style.color = '#ff4d4d'; resetAllBtn.style.borderColor = 'rgba(255,0,0,0.2)'; };
        resetAllBtn.onmouseleave = () => { resetAllBtn.style.color = '#888'; resetAllBtn.style.borderColor = 'rgba(255,255,255,0.08)'; };
        resetAllBtn.onclick = () => {
            this.curveEditor.resetAll();
            this.curveMix = 1.0;
            if (this.renderer) this.renderer.setCurveMix(1.0);
            this._lastRenderContent();
        };
        resetRow.appendChild(resetAllBtn);

        // Snapshot row
        const snapRow = document.createElement('div');
        snapRow.style.cssText = 'display: flex; gap: 6px; margin-top: 4px;';
        bottomSection.appendChild(snapRow);

        const mkSnapBtn = (text, title, onClick) => {
            const btn = document.createElement('button');
            btn.textContent = text;
            btn.title = title;
            btn.style.cssText = `
                flex: 1; padding: 4px; background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08); border-radius: 3px;
                color: #888; font-size: 9px; cursor: pointer; transition: all 0.15s;
                font-family: ${this.theme.mono};
            `;
            btn.onmouseenter = () => { btn.style.color = '#fff'; btn.style.borderColor = 'rgba(255,255,255,0.2)'; };
            btn.onmouseleave = () => { btn.style.color = '#888'; btn.style.borderColor = 'rgba(255,255,255,0.08)'; };
            btn.onclick = onClick;
            return btn;
        };

        snapRow.appendChild(mkSnapBtn('📌 Snap', 'Save reference snapshot (O)', () => {
            this.curveEditor.saveSnapshot();
        }));
        snapRow.appendChild(mkSnapBtn('✕ Clear', 'Clear snapshot', () => {
            this.curveEditor.clearSnapshot();
        }));

        // ─────────────────────────────────────────────────────────────────
        //  SIDEBAR: Hue Curves (collapsible)
        // ─────────────────────────────────────────────────────────────────
        const hueSection = document.createElement('div');
        hueSection.style.cssText = 'padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.06);';
        sidebar.appendChild(hueSection);

        const hueHeader = document.createElement('div');
        hueHeader.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 6px;';

        const hueLabel = document.createElement('div');
        hueLabel.textContent = 'Hue Curves';
        hueLabel.style.cssText = 'color: #999; font-size: 11px; font-weight: 600;';
        hueHeader.appendChild(hueLabel);
        hueSection.appendChild(hueHeader);

        const hueBtns = document.createElement('div');
        hueBtns.style.cssText = 'display: flex; gap: 4px;';
        hueSection.appendChild(hueBtns);

        [
            { ch: 'HueVsHue', label: 'HvH', color: '#ffaaaa' },
            { ch: 'HueVsSat', label: 'HvS', color: '#fff144' },
            { ch: 'HueVsLuma', label: 'HvL', color: '#44ffaa' }
        ].forEach(({ ch, label, color }) => {
            const btn = document.createElement('div');
            btn.textContent = label;
            btn.style.cssText = `
                padding: 3px 8px; background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08); border-radius: 3px;
                color: ${color}; font-size: 10px; font-weight: bold; cursor: pointer;
                transition: all 0.12s;
            `;
            btn.onmouseenter = () => { btn.style.background = 'rgba(255,255,255,0.08)'; };
            btn.onmouseleave = () => { btn.style.background = this.curveEditor.activeChannel === ch ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)'; };
            btn.onclick = () => {
                this.curveEditor.setActiveChannel(ch);
                refreshChannelBtns();
                this._lastRenderContent();
            };
            hueBtns.appendChild(btn);
        });

        // ── v4.1: Pipeline Bit-Depth Selector ────────────────────────────────
        // Industry-standard 3-button group: INT 8 / FLOAT 16 / FLOAT 32
        // Matches the precision selector found in Nuke, Flame, and Baselight.
        const precSection = document.createElement('div');
        precSection.style.cssText = 'padding: 12px 10px; display: flex; flex-direction: column; gap: 6px;';
        sidebar.appendChild(precSection);

        const precHeader = document.createElement('div');
        precHeader.style.cssText = 'display: flex; align-items: center; justify-content: space-between;';
        const precTitle = document.createElement('div');
        precTitle.textContent = 'BIT DEPTH';
        precTitle.style.cssText = `color: ${this.theme.textDim}; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px;`;
        precHeader.appendChild(precTitle);
        precSection.appendChild(precHeader);

        const precBtnRow = document.createElement('div');
        precBtnRow.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 3px;';

        const precModes = [
            { mode: 'u8', label: 'INT 8', color: '#707088' },
            { mode: 'f16', label: 'FP 16', color: '#60a5fa' },
            { mode: 'f32', label: 'FP 32', color: '#4ade80' }
        ];

        const getCurrentMode = () => this.renderer ? this.renderer.pipelinePrecision : (localStorage.getItem('radiance_pipeline_precision') || 'f32');

        const precBtns = {};
        const refreshPrecBtns = () => {
            const cur = getCurrentMode();
            precModes.forEach(({ mode, color }) => {
                const btn = precBtns[mode];
                if (!btn) return;
                const active = cur === mode;
                btn.style.background = active ? `${color}22` : 'rgba(255,255,255,0.03)';
                btn.style.borderColor = active ? color : 'rgba(255,255,255,0.08)';
                btn.style.color = active ? color : '#555';
            });
        };

        precModes.forEach(({ mode, label, color }) => {
            const btn = document.createElement('button');
            btn.textContent = label;
            btn.style.cssText = `
                padding: 4px 2px; border-radius: 3px; cursor: pointer;
                border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03);
                transition: all 0.15s ease; font-family: ${this.theme.mono};
                font-size: 9px; font-weight: 700; letter-spacing: 0.3px;
            `;
            precBtns[mode] = btn;

            btn.addEventListener('mouseenter', () => {
                if (getCurrentMode() !== mode) btn.style.background = 'rgba(255,255,255,0.07)';
            });
            btn.addEventListener('mouseleave', () => refreshPrecBtns());
            btn.addEventListener('click', () => {
                this._setPipelinePrecision(mode);
                refreshPrecBtns();
            });

            precBtnRow.appendChild(btn);
        });

        precSection.appendChild(precBtnRow);

        // Restore saved precision
        const savedPrec = localStorage.getItem('radiance_pipeline_precision') || 'f32';
        if (this.renderer && this.renderer.pipelinePrecision !== savedPrec) {
            this._setPipelinePrecision(savedPrec);
        }
        refreshPrecBtns();
        this._updateBitDepthBadge();
    }


    renderQualifiersTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; flex: 1; gap: 10px; padding: 10px; min-height: 0; overflow-y: auto;';

        // Initialize state if missing
        if (!this.qualifierState) {
            this.qualifierState = {
                enabled: false,
                showMask: false,
                h: 0.0, hW: 0.1, hS: 0.05,
                s: 0.5, sW: 0.5, sS: 0.1,
                l: 0.5, lW: 0.5, lS: 0.1
            };
        }

        const update = () => {
            if (this.renderer) {
                this.renderer.setQualifier(this.qualifierState);
                this.requestRender();
            }
        };

        // 1. Top Controls (Enable, Show Mask, Eyedropper)
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 5px;';

        // Enable Toggle
        // Helper for consistent checkbox style
        const createToggle = (id, labelText, isChecked, onChange) => {
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'display: flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.03); padding: 4px 8px; border-radius: 4px;';
            const check = document.createElement('input');
            check.type = 'checkbox';
            check.id = id;
            check.checked = isChecked;
            check.style.cssText = 'accent-color: #00ffcc; cursor: pointer;';
            check.onchange = onChange;
            const lbl = document.createElement('label');
            lbl.htmlFor = id;
            lbl.textContent = labelText;
            lbl.style.cssText = 'color: #ccc; font-size: 11px; cursor: pointer;';
            wrapper.appendChild(check);
            wrapper.appendChild(lbl);
            return wrapper;
        };

        // Active Toggle
        topRow.appendChild(createToggle('qual-enable', 'Active', this.qualifierState.enabled, (e) => {
            this.qualifierState.enabled = e.target.checked;
            update();
        }));

        // Show Mask Toggle
        topRow.appendChild(createToggle('qual-mask', 'Show Mask', this.qualifierState.showMask, (e) => {
            this.qualifierState.showMask = e.target.checked;
            update();
        }));

        // Eyedropper
        const pickerBtn = document.createElement('button');
        pickerBtn.textContent = '⚲ PICK COLOR';
        pickerBtn.style.cssText = 'background: #2a2a30; color: #fff; border: 1px solid #444; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 10px; font-weight: bold; transition: all 0.2s;';
        pickerBtn.onmouseover = () => { pickerBtn.style.borderColor = '#00ffcc'; pickerBtn.style.boxShadow = '0 0 8px rgba(0, 255, 204, 0.3)'; };
        pickerBtn.onmouseout = () => { pickerBtn.style.borderColor = '#444'; pickerBtn.style.boxShadow = 'none'; };
        if (this.isPicking) {
            pickerBtn.style.borderColor = '#00ffcc';
            pickerBtn.style.boxShadow = '0 0 8px rgba(0, 255, 204, 0.3)';
            pickerBtn.style.color = '#00ffcc';
        }
        pickerBtn.onclick = () => {
            this.activateEyedropper(update);
            // Re-render to update pick button state
            container.innerHTML = '';
            this.renderMasksTab(container);
        };
        topRow.appendChild(pickerBtn);

        container.appendChild(topRow);

        // Compact HSL Grid Layout
        const gridContainer = document.createElement('div');
        gridContainer.style.cssText = 'display: flex; flex-direction: column; gap: 4px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; margin-bottom: 12px;';

        // Headers
        const headerRow = document.createElement('div');
        headerRow.style.cssText = 'display: flex; gap: 8px; margin-bottom: 2px; padding-left: 40px;';
        ['CENTER', 'WIDTH', 'SOFT'].forEach(text => {
            const h = document.createElement('div');
            h.textContent = text;
            h.style.cssText = 'color: #888; font-size: 9px; font-weight: bold; width: 44px; text-align: center; letter-spacing: 0.5px;';
            headerRow.appendChild(h);
        });
        gridContainer.appendChild(headerRow);

        // Helper to create knob row WITHOUT individual labels
        const createRow = (label, param, labelColor) => {
            const row = document.createElement('div');
            row.style.cssText = 'display: flex; gap: 8px; align-items: center;';

            const title = document.createElement('div');
            title.textContent = label;
            title.style.cssText = `color: ${labelColor}; font-size: 10px; width: 32px; font-weight: bold; text-align: right; margin-right: 4px;`;
            row.appendChild(title);

            // Center
            row.appendChild(this.createKnob('', 0, 1, this.qualifierState[param], 0.01, (v) => {
                this.qualifierState[param] = v;
                update();
            }));

            // Width
            row.appendChild(this.createKnob('', 0, 1, this.qualifierState[param + 'W'], 0.01, (v) => {
                this.qualifierState[param + 'W'] = v;
                update();
            }));

            // Soft
            row.appendChild(this.createKnob('', 0, 0.5, this.qualifierState[param + 'S'], 0.01, (v) => {
                this.qualifierState[param + 'S'] = v;
                update();
            }));

            return row;
        };

        gridContainer.appendChild(createRow('HUE', 'h', '#ff6b6b'));
        gridContainer.appendChild(createRow('SAT', 's', '#4cd137'));
        gridContainer.appendChild(createRow('LUM', 'l', '#dcdde1'));

        container.appendChild(gridContainer);
    }

    renderMasksTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; flex: 1; gap: 10px; padding: 10px; min-height: 0; overflow-y: auto;';

        const update = () => {
            if (this.renderer) {
                this.renderer.setMask(this.maskState);
                this.requestRender();
            }
        };

        // 1. Top Controls (Type, Invert, Show Overlay)
        const topRow = document.createElement('div');
        topRow.style.cssText = 'display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 8px;';

        // Mask Type
        const typeSelect = document.createElement('select');
        typeSelect.style.cssText = 'background: #333; color: #ccc; border: 1px solid #555; font-size: 11px; padding: 4px; border-radius: 4px; cursor: pointer;';
        ['None', 'Circle', 'Box'].forEach((label, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = label;
            if (this.maskState.type === i) opt.selected = true;
            typeSelect.appendChild(opt);
        });
        typeSelect.onchange = (e) => {
            this.maskState.type = parseInt(e.target.value);
            update();
        };
        topRow.appendChild(typeSelect);

        const checkGroup = document.createElement('div');
        checkGroup.style.cssText = 'display: flex; gap: 8px;';

        // Helper for consistent checkbox style (reused from qualifier)
        const createToggle = (id, labelText, isChecked, onChange) => {
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'display: flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.03); padding: 4px 8px; border-radius: 4px;';
            const check = document.createElement('input');
            check.type = 'checkbox';
            check.id = id;
            check.checked = isChecked;
            check.style.cssText = 'accent-color: #00ffcc; cursor: pointer;';
            check.onchange = onChange;
            const lbl = document.createElement('label');
            lbl.htmlFor = id;
            lbl.textContent = labelText;
            lbl.style.cssText = 'color: #ccc; font-size: 11px; cursor: pointer;';
            wrapper.appendChild(check);
            wrapper.appendChild(lbl);
            return wrapper;
        };

        // Invert
        checkGroup.appendChild(createToggle('mask-invert', 'Invert', this.maskState.invert, (e) => {
            this.maskState.invert = e.target.checked;
            update();
        }));

        // Show Overlay
        checkGroup.appendChild(createToggle('mask-overlay', 'Overlay', this.maskState.showOverlay, (e) => {
            this.maskState.showOverlay = e.target.checked;
            update();
        }));

        topRow.appendChild(checkGroup);
        container.appendChild(topRow);

        // 2. Transform Controls (2x2 Grid)
        const gridGroup = document.createElement('div');
        gridGroup.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr; gap: 8px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;';

        const createCell = (label, controls) => {
            const cell = document.createElement('div');
            cell.style.cssText = 'display: flex; flex-direction: column; gap: 4px;';
            const title = document.createElement('div');
            title.textContent = label;
            title.style.cssText = 'color: #888; font-size: 9px; font-weight: bold; letter-spacing: 0.5px;';
            cell.appendChild(title);
            const ctrlGroup = document.createElement('div');
            ctrlGroup.style.cssText = 'display: flex; gap: 10px; align-items: center;';
            controls.forEach(c => ctrlGroup.appendChild(c));
            cell.appendChild(ctrlGroup);
            return cell;
        };

        // Row 1: Center & Scale
        gridGroup.appendChild(createCell('CENTER', [
            this.createKnob('X', 0, 1, this.maskState.center[0], 0.01, (v) => { this.maskState.center[0] = v; update(); }),
            this.createKnob('Y', 0, 1, this.maskState.center[1], 0.01, (v) => { this.maskState.center[1] = v; update(); })
        ]));

        gridGroup.appendChild(createCell('SCALE', [
            this.createKnob('X', 0.01, 2, this.maskState.scale[0], 0.01, (v) => { this.maskState.scale[0] = v; update(); }),
            this.createKnob('Y', 0.01, 2, this.maskState.scale[1], 0.01, (v) => { this.maskState.scale[1] = v; update(); })
        ]));

        // Row 2: Rotation & Feather
        gridGroup.appendChild(createCell('ROTATION', [
            this.createKnob('Rad', -Math.PI, Math.PI, this.maskState.rotation, 0.01, (v) => { this.maskState.rotation = v; update(); })
        ]));

        gridGroup.appendChild(createCell('FEATHER', [
            this.createKnob('Soft', 0, 1, this.maskState.feather, 0.01, (v) => { this.maskState.feather = v; update(); })
        ]));

        container.appendChild(gridGroup);
    }

    activateEyedropper(callback) {
        if (this.isPicking) return;
        this.isPicking = true;

        // Show overlay helper
        const overlay = document.createElement('div');
        overlay.textContent = 'Click to Pick Color';
        overlay.style.cssText = 'position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); color: #fff; padding: 5px 10px; border-radius: 4px; pointer-events: none; z-index: 200;';
        this.container.appendChild(overlay);

        this.container.style.cursor = 'crosshair';

        const clickHandler = (e) => {
            // Get click coordinate relative to image
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Map to image coordinates
            // Need inverse fitToView transform... 
            // Simplified: if we click on canvas, we read displayed pixel? 
            // Or read from source data? Source data is better.

            // Calculate UV
            // Pan/Zoom aware mapping
            // x_img_scaled = (uv.x * width * zoom) + pan.x
            const imgX = (x - this.panX) / this.zoom;
            const imgY = (y - this.panY) / this.zoom;

            if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                // Determine color
                let r = 0, g = 0, b = 0;

                if (this.hdrData) {
                    // Float sample
                    const ix = Math.floor(imgX);
                    const iy = Math.floor(imgY);
                    const idx = (iy * this.imageWidth + ix) * this.hdrData.channels;
                    const d = this.hdrData.fp16data ? this.hdrData.data : this.hdrData.data; // both are float32 in js wrapper usually
                    // Wait, .data is Float32Array
                    r = d[idx];
                    g = d[idx + 1];
                    b = d[idx + 2];
                } else if (this.imageData) {
                    // 8-bit sample from canvas/image? 
                    // We can't easily read image.data without context usually.
                    // But we created a temp context in Scopes...
                    // Let's assume we can read from renderer's buffer? 
                    // Or just use the temp canvas approach if needed. 
                    // For now, if no HDR, maybe skip or use approximation.
                    // Actually, if we have Image object, we can draw to canvas.
                }

                // Convert to HSL
                // r,g,b are linear or sRGB? HDR data is likely linear? 
                // Let's assume linear.

                // RGB to HSL logic (JS version)
                const max = Math.max(r, g, b);
                const min = Math.min(r, g, b);
                let h, s, l = (max + min) / 2;

                if (max === min) {
                    h = s = 0; // achromatic
                } else {
                    const d = max - min;
                    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
                    switch (max) {
                        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                        case g: h = (b - r) / d + 2; break;
                        case b: h = (r - g) / d + 4; break;
                    }
                    h /= 6;
                }

                // Set Qualifiers
                this.qualifierState.h = h;
                this.qualifierState.s = s;
                this.qualifierState.l = l;
                this.qualifierState.enabled = true;

                // Refresh UI sliders? They need to re-render or update value
                // Since createKnob sets initial value, re-rendering tab is easiest
                // But we passed 'update' callback which just renders WebGL.
                // We need to re-render controls.
                this.renderQualifiersTab(this.tabContentContainer);

                if (this.renderer) {
                    this.renderer.setQualifier(this.qualifierState);
                    this.render();
                }
            }

            // Cleanup
            this.container.style.cursor = 'default';
            overlay.remove();
            this.container.removeEventListener('click', clickHandler);
            this.isPicking = false;
        };

        this.container.addEventListener('click', clickHandler);
    }

    activateFocusPicker(callback) {
        if (this.isPickingFocus) return;
        this.isPickingFocus = true;

        const overlay = document.createElement('div');
        overlay.textContent = 'Click to Set Focus';
        overlay.style.cssText = 'position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: rgba(106,138,255,0.85); color: #fff; padding: 6px 12px; border-radius: 4px; pointer-events: none; z-index: 200; font-size: 11px; font-weight: bold; box-shadow: 0 4px 12px rgba(0,0,0,0.5);';
        this.container.appendChild(overlay);

        this.container.style.cursor = 'crosshair';

        const clickHandler = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const imgX = (x - this.panX) / this.zoom;
            const imgY = (y - this.panY) / this.zoom;

            if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
                if (this.zdepthImage) {
                    // Sample from depth image
                    const pickCanvas = document.createElement('canvas');
                    pickCanvas.width = this.zdepthImage.width;
                    pickCanvas.height = this.zdepthImage.height;
                    const pctx = pickCanvas.getContext('2d');
                    pctx.drawImage(this.zdepthImage, 0, 0);

                    const ix = Math.floor(imgX * (this.zdepthImage.width / this.imageWidth));
                    const iy = Math.floor(imgY * (this.zdepthImage.height / this.imageHeight));

                    const pix = pctx.getImageData(ix, iy, 1, 1).data;
                    // Depth is usually stored in R channel (grayscale)
                    // In ComfyUI/Radiance, normalize it to 0..1
                    this.focusDistance = pix[0] / 255.0;

                    if (this.renderer) {
                        this.renderer.setFocusDistance(this.focusDistance);
                        this.render();
                    }
                    if (callback) callback(this.focusDistance);
                } else {
                    console.warn("[Radiance] No depth map available to pick focus from.");
                }
            }

            this.container.style.cursor = 'default';
            overlay.remove();
            this.container.removeEventListener('click', clickHandler);
            this.isPickingFocus = false;
        };

        this.container.addEventListener('click', clickHandler);
    }

    renderViewTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; flex: 1; gap: 12px; padding: 12px; min-height: 0; overflow-y: auto;';

        // 0. Neural Network Monitor (Real-time 3D)
        const neuralGroup = document.createElement('div');
        neuralGroup.style.marginBottom = '12px';
        neuralGroup.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; padding: 0 4px;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="width: 6px; height: 6px; background: #00ffcc; border-radius: 50%; box-shadow: 0 0 8px #00ffcc; animation: pulse 2s infinite;"></div>
                    <div style="color: #00ffcc; font-size: 10px; text-transform: uppercase; font-weight: 800; letter-spacing: 1.5px; font-family: 'Inter', sans-serif;">Neural Topology</div>
                </div>
                <div style="font-size: 8px; color: rgba(0, 255, 204, 0.4); font-family: monospace; letter-spacing: 1px;">CORE_DEEP_LINK_v2.0</div>
            </div>
        `;

        const monitorContainer = document.createElement('div');
        monitorContainer.style.cssText = `
            background: #010409;
            border: 1px solid rgba(0, 255, 204, 0.15);
            border-radius: 8px;
            position: relative;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5), inset 0 0 20px rgba(0,255,180,0.02);
            overflow: hidden;
            height: 250px;
        `;

        // Add a scanline overlay in CSS if not present
        if (!document.getElementById('radiance-monitor-styles')) {
            const style = document.createElement('style');
            style.id = 'radiance-monitor-styles';
            style.innerHTML = `
                @keyframes pulse {
                    0% { opacity: 0.4; transform: scale(0.9); }
                    50% { opacity: 1; transform: scale(1.1); }
                    100% { opacity: 0.4; transform: scale(0.9); }
                }
            `;
            document.head.appendChild(style);
        }

        neuralGroup.appendChild(monitorContainer);
        container.appendChild(neuralGroup);

        // Initialize or re-attach the monitor
        if (this.neuralMonitor) {
            this.neuralMonitor.dispose();
        }
        this.neuralMonitor = new RadianceNeuralMonitor(monitorContainer);
        // Ensure current progress is reflected if already generating
        if (this._lastProgress !== undefined) {
            this.neuralMonitor.setProgress(this._lastProgress);
        }

        // 1. Comparison Wipe
        const wipeGroup = document.createElement('div');
        wipeGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">A/B Comparison (Split Screen)</div>';

        const wipeControls = document.createElement('div');
        wipeControls.style.cssText = 'display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.03); padding: 8px; border-radius: 6px;';

        const wipeCheck = document.createElement('input');
        wipeCheck.type = 'checkbox';
        wipeCheck.id = 'wipe-enable';
        wipeCheck.checked = this.wipeEnabled || false;
        wipeCheck.onchange = (e) => {
            this.wipeEnabled = e.target.checked;
            if (this.renderer) this.renderer.setWipe(this.wipe || 0.5, this.wipeEnabled);
            this.render();
        };
        wipeControls.appendChild(wipeCheck);

        const wipeLbl = document.createElement('label');
        wipeLbl.htmlFor = 'wipe-enable';
        wipeLbl.textContent = 'Enable Wipe';
        wipeLbl.style.cssText = 'color: #ccc; font-size: 11px; cursor: pointer; flex: 1;';
        wipeControls.appendChild(wipeLbl);

        const wipeSlider = document.createElement('input');
        wipeSlider.type = 'range'; wipeSlider.min = '0'; wipeSlider.max = '100'; wipeSlider.step = '1';
        wipeSlider.value = String(Math.round((this.wipe || 0.5) * 100));
        wipeSlider.style.cssText = 'width: 100px; accent-color: #00ffcc; height: 4px; cursor: pointer;';
        wipeSlider.oninput = (e) => {
            this.wipe = parseInt(e.target.value) / 100;
            if (this.renderer) this.renderer.setWipe(this.wipe, this.wipeEnabled);
            this.render();
        };
        wipeControls.appendChild(wipeSlider);

        wipeGroup.appendChild(wipeControls);
        container.appendChild(wipeGroup);

        // 2. Reference Image
        const refGroup = document.createElement('div');
        refGroup.style.marginTop = '4px';
        refGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Reference Gallery</div>';

        const refRow = document.createElement('div');
        refRow.style.cssText = 'display: flex; gap: 8px; align-items: center;';

        const grabBtn = document.createElement('button');
        grabBtn.textContent = '⛶ GRAB STILL';
        grabBtn.style.cssText = 'background: #2a2a30; color: #fff; border: 1px solid #444; padding: 6px 12px; border-radius: 4px; font-size: 10px; cursor: pointer; font-weight: bold; flex: 1;';
        grabBtn.onclick = () => this.grabStill();
        refRow.appendChild(grabBtn);

        // Clear Gallery Button
        const clearBtn = document.createElement('button');
        clearBtn.textContent = '🗑 CLEAR';
        clearBtn.style.cssText = 'background: #3a2020; color: #ff8888; border: 1px solid #552222; padding: 6px 12px; border-radius: 4px; font-size: 10px; cursor: pointer; font-weight: bold;';
        clearBtn.onclick = () => {
            if (this.renderer) this.renderer.clearReferenceShelf();
            if (this.galleryGrid) {
                this.galleryGrid.innerHTML = '';
                this.savedStills = 0;
            }
        };
        refRow.appendChild(clearBtn);

        const refCheckGroup = document.createElement('div');
        refCheckGroup.style.cssText = 'display: flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.03); padding: 5px 10px; border-radius: 4px;';

        const refCheck = document.createElement('input');
        refCheck.type = 'checkbox';
        refCheck.id = 'wipe-ref';
        refCheck.checked = this.wipeRefEnabled || false;
        refCheck.onchange = (e) => {
            this.wipeRefEnabled = e.target.checked;
            if (this.renderer) this.renderer.setWipeRef(this.wipeRefEnabled);
            this.render();
        };
        refCheckGroup.appendChild(refCheck);

        const refLbl = document.createElement('label');
        refLbl.htmlFor = 'wipe-ref';
        refLbl.textContent = 'Use Ref';
        refLbl.style.cssText = 'color: #888; font-size: 10px; cursor: pointer;';
        refRow.appendChild(refCheckGroup);

        refGroup.appendChild(refRow);

        // Add Gallery Grid container below the buttons
        this.galleryGrid = document.createElement('div');
        this.galleryGrid.style.cssText = 'display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; margin-bottom: 8px;';
        refGroup.appendChild(this.galleryGrid);

        container.appendChild(refGroup);

        // 3. Overlays / Grids
        const gridGroup = document.createElement('div');
        gridGroup.style.marginTop = '4px';
        gridGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Composition Overlays</div>';

        const gridSelect = document.createElement('select');
        gridSelect.style.cssText = 'width: 100%; background: #1a1a20; color: #ccc; border: 1px solid #333; padding: 6px; border-radius: 4px; font-size: 11px; cursor: pointer;';

        const gridOptions = [
            { id: 0, label: 'None' },
            { id: 1, label: 'Rule of Thirds' },
            { id: 2, label: '2.39:1 Cinematic Mask' },
            { id: 3, label: 'Center Cross' }
        ];

        gridOptions.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt.id;
            o.textContent = opt.label;
            if (opt.id === (this.gridMode || 0)) o.selected = true;
            gridSelect.appendChild(o);
        });

        gridSelect.onchange = (e) => {
            this.gridMode = parseInt(e.target.value);
            if (this.renderer) this.renderer.setGridMode(this.gridMode);
            this.render();
        };
        gridGroup.appendChild(gridSelect);
        container.appendChild(gridGroup);

        // 4. Grade Presets & Export
        const exportGroup = document.createElement('div');
        exportGroup.style.marginTop = '4px';
        exportGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Presets & Export</div>';

        const exportRow = document.createElement('div');
        exportRow.style.cssText = 'display: flex; gap: 8px;';

        const saveBtn = document.createElement('button');
        saveBtn.textContent = '⤓ SAVE PRESET';
        saveBtn.style.cssText = 'background: #1a1a20; color: #fff; border: 1px solid #333; padding: 6px; border-radius: 4px; font-size: 10px; cursor: pointer; flex: 1;';
        saveBtn.onclick = () => this.saveGrade();
        exportRow.appendChild(saveBtn);

        const cubeBtn = document.createElement('button');
        cubeBtn.textContent = '⤒ EXPORT .CUBE';
        cubeBtn.style.cssText = 'background: #1a1a20; color: #ffca28; border: 1px solid #ffca2855; padding: 6px; border-radius: 4px; font-size: 10px; cursor: pointer; flex: 1; font-weight: bold;';
        cubeBtn.onclick = () => this.exportToCube();
        exportRow.appendChild(cubeBtn);

        const cdlExportBtn = document.createElement('button');
        cdlExportBtn.textContent = '⤒ EXPORT .CDL';
        cdlExportBtn.style.cssText = 'background: #1a1a20; color: #88ff88; border: 1px solid #88ff8855; padding: 6px; border-radius: 4px; font-size: 10px; cursor: pointer; flex: 1; font-weight: bold;';
        cdlExportBtn.onclick = () => this.exportToCDL();
        exportRow.appendChild(cdlExportBtn);

        const cdlImportBtn = document.createElement('button');
        cdlImportBtn.textContent = '⤓ IMPORT .CDL';
        cdlImportBtn.style.cssText = 'background: #1a1a20; color: #88ff88; border: 1px solid #88ff8855; padding: 6px; border-radius: 4px; font-size: 10px; cursor: pointer; flex: 1; font-weight: bold;';

        // Setup hidden file input for CDL import
        const cdlInput = document.createElement('input');
        cdlInput.type = 'file';
        cdlInput.accept = '.cdl';
        cdlInput.style.display = 'none';
        cdlInput.onchange = (e) => {
            if (e.target.files && e.target.files.length > 0) {
                this.importFromCDL(e.target.files[0]);
            }
        };
        exportRow.appendChild(cdlInput);

        cdlImportBtn.onclick = () => cdlInput.click();
        exportRow.appendChild(cdlImportBtn);

        exportGroup.appendChild(exportRow);

        // Preset List
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        const presetNames = Object.keys(presets);
        if (presetNames.length > 0) {
            const list = document.createElement('div');
            list.style.cssText = 'display: flex; flex-direction: column; gap: 4px; margin-top: 8px; max-height: 100px; overflow-y: auto; background: rgba(0,0,0,0.2); border-radius: 4px; padding: 4px;';
            presetNames.forEach(name => {
                const item = document.createElement('div');
                item.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.03); border-radius: 3px; font-size: 10px; color: #ccc;';

                const nameBtn = document.createElement('div');
                nameBtn.textContent = name;
                nameBtn.style.cssText = 'cursor: pointer; flex: 1;';
                nameBtn.onclick = () => this.loadGrade(name);
                item.appendChild(nameBtn);

                const delBtn = document.createElement('div');
                delBtn.textContent = '✕';
                delBtn.style.cssText = 'cursor: pointer; color: #666; padding: 0 4px;';
                delBtn.onmouseover = () => delBtn.style.color = '#ff6b6b';
                delBtn.onmouseout = () => delBtn.style.color = '#666';
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (confirm(`Delete preset "${name}" ? `)) this.deleteGrade(name);
                };
                item.appendChild(delBtn);

                list.appendChild(item);
            });
            exportGroup.appendChild(list);
        }

        container.appendChild(exportGroup);

        // 5. Accessibility settings
        const accessGroup = document.createElement('div');
        accessGroup.style.marginTop = '4px';
        accessGroup.innerHTML = '<div style="color: #888; font-size: 10px; margin-bottom: 8px; text-transform: uppercase; font-weight: bold;">Accessibility</div>';

        const hcRow = document.createElement('div');
        hcRow.style.cssText = 'display: flex; align-items: center; gap: 10px; background: rgba(255,255,255,0.03); padding: 8px; border-radius: 6px;';

        const hcCheck = document.createElement('input');
        hcCheck.type = 'checkbox';
        hcCheck.id = 'hud-high-contrast';
        hcCheck.checked = this.highContrast || false;
        hcCheck.onchange = (e) => {
            this.highContrast = e.target.checked;
            localStorage.setItem('radiance_high_contrast', this.highContrast ? '1' : '0');
            if (this.highContrast) {
                RadianceViewer.singletonHUD?.classList.add('high-contrast');
            } else {
                RadianceViewer.singletonHUD?.classList.remove('high-contrast');
            }
        };
        hcRow.appendChild(hcCheck);

        const hcLbl = document.createElement('label');
        hcLbl.htmlFor = 'hud-high-contrast';
        hcLbl.textContent = 'High Contrast Mode';
        hcLbl.style.cssText = 'color: #ccc; font-size: 11px; cursor: pointer; flex: 1;';
        hcRow.appendChild(hcLbl);

        accessGroup.appendChild(hcRow);
        container.appendChild(accessGroup);
    }

    async grabStill() {
        if (!this.renderer || !this.canvas) return;

        // 1. Tell WebGL to grab the full-res texture directly into its GPU shelf
        const slot = this.renderer.grabReferenceStill();

        // 2. Downscale canvas for UI thumbnail to save memory
        const thumbCanvas = document.createElement('canvas');
        const aspect = this.canvas.width / this.canvas.height;
        thumbCanvas.width = 160;
        thumbCanvas.height = 160 / aspect;
        const ctx = thumbCanvas.getContext('2d');
        ctx.drawImage(this.canvas, 0, 0, thumbCanvas.width, thumbCanvas.height);

        const thumb = new Image();
        thumb.src = thumbCanvas.toDataURL('image/jpeg', 0.8);

        thumb.onload = () => {
            // Flash screen
            this.canvas.style.filter = 'brightness(2)';
            setTimeout(() => { this.canvas.style.filter = ''; }, 100);

            // Create UI Thumbnail
            if (this.galleryGrid) {
                const thumbWrapper = document.createElement('div');
                thumbWrapper.style.cssText = `
                width: 72px; height: 48px; position: relative; cursor: pointer;
                border: 2px solid transparent; border-radius: 4px; overflow: hidden;
                transition: border 0.2s;
            `;

                thumb.style.cssText = 'width: 100%; height: 100%; object-fit: cover; display: block;';
                thumbWrapper.appendChild(thumb);

                // Number label
                const lbl = document.createElement('div');
                lbl.textContent = (slot + 1).toString();
                lbl.style.cssText = 'position: absolute; bottom: 2px; right: 4px; font-size: 10px; font-weight: bold; color: white; text-shadow: 0 1px 2px black;';
                thumbWrapper.appendChild(lbl);

                thumbWrapper.onclick = () => {
                    this.renderer.swapReferenceShelf(slot);
                    // Update border
                    Array.from(this.galleryGrid.children).forEach(c => c.style.borderColor = 'transparent');
                    thumbWrapper.style.borderColor = '#6a8aff';

                    // Auto-enable wipe
                    const wipeCheck = document.getElementById('wipe-ref');
                    if (wipeCheck) {
                        wipeCheck.checked = true;
                        this.wipeRefEnabled = true;
                    }
                    this.render();
                };

                // Add MATCH button inside thumbnail
                const matchBtn = document.createElement('div');
                matchBtn.textContent = 'MATCH';
                matchBtn.style.cssText = `
                position: absolute; top: 2px; right: 2px;
                background: rgba(40,40,255,0.8); color: white;
                font-size: 8px; font-weight: bold; padding: 2px 4px;
                border-radius: 2px; cursor: pointer; display: none;
            `;

                thumbWrapper.onmouseenter = () => matchBtn.style.display = 'block';
                thumbWrapper.onmouseleave = () => matchBtn.style.display = 'none';

                matchBtn.onclick = (e) => {
                    e.stopPropagation(); // prevent wipe trigger
                    this.matchGrade(slot);
                };
                thumbWrapper.appendChild(matchBtn);

                this.galleryGrid.appendChild(thumbWrapper);
                this.savedStills = (this.savedStills || 0) + 1;
            }
        };
    }

    renderScopesTab(container) {
        container.style.cssText = 'display: flex; flex-direction: column; align-items: stretch; gap: 8px; padding: 8px;';

        // ─── Mode Selector Bar ─────────────────────────────
        const modeBar = document.createElement('div');
        modeBar.style.cssText = 'display: flex; gap: 4px; width: 100%; margin-bottom: 4px;';

        const modes = [
            { id: 'parade', label: 'Parade' },
            { id: 'waveform', label: 'Waveform' },
            { id: 'histogram', label: 'Histogram' },
            { id: 'vectorscope', label: 'Vector' },
            { id: 'falsecolor', label: 'False Color' },
        ];

        modes.forEach(m => {
            const btn = document.createElement('div');
            btn.textContent = m.label;
            const isActive = this.scopeMode === m.id;
            btn.style.cssText = `
                flex: 1; text-align: center; padding: 8px 0;
                background: ${isActive ? 'rgba(106,138,255,0.25)' : 'rgba(255,255,255,0.04)'};
                color: ${isActive ? '#8aafff' : '#888'};
                border: 1px solid ${isActive ? 'rgba(106,138,255,0.5)' : 'rgba(255,255,255,0.12)'};
                border-radius: 4px; font-size: 11px; font-weight: 700; cursor: pointer;
                transition: all 0.15s;
            `;
            btn.onmouseenter = () => { if (!isActive) { btn.style.background = 'rgba(255,255,255,0.12)'; btn.style.color = '#fff'; } };
            btn.onmouseleave = () => { if (!isActive) { btn.style.background = 'rgba(255,255,255,0.04)'; btn.style.color = '#888'; } };
            btn.onclick = () => {
                this.scopeMode = m.id;
                localStorage.setItem('radiance_scope_mode', m.id);
                this._lastRenderContent();
            };
            modeBar.appendChild(btn);
        });
        container.appendChild(modeBar);

        // ─── Options Row: Log/Lin toggle + Source toggle ─────
        if (!this.scopeLogView) this.scopeLogView = localStorage.getItem('radiance_scope_log') === '1';

        const optRow = document.createElement('div');
        optRow.style.cssText = 'display: flex; gap: 10px; width: 100%; align-items: center; margin-bottom: 4px;';

        const makeOptBtn = (label, active, onClick) => {
            const b = document.createElement('div');
            b.textContent = label;
            b.style.cssText = `
                padding: 5px 12px; border-radius: 4px; font-size: 11px; font-weight: bold; cursor: pointer;
                background: ${active ? 'rgba(255,200,80,0.22)' : 'rgba(255,255,255,0.06)'};
                color: ${active ? '#ffc844' : '#777'};
                border: 1px solid ${active ? 'rgba(255,200,80,0.6)' : 'rgba(255,255,255,0.12)'};
                transition: all 0.15s;
            `;
            b.onclick = onClick;
            return b;
        };

        optRow.appendChild(makeOptBtn('LOG VIEW', this.scopeLogView, () => {
            this.scopeLogView = !this.scopeLogView;
            localStorage.setItem('radiance_scope_log', this.scopeLogView ? '1' : '0');
            this._lastRenderContent();
        }));

        const logNote = document.createElement('div');
        logNote.textContent = this.scopeLogView ? 'LogC · shadows expanded' : 'Linear · 0–255';
        logNote.style.cssText = 'font-size: 10px; color: #666; margin-left: auto; font-weight: 600;';
        optRow.appendChild(logNote);
        container.appendChild(optRow);

        // ─── Scope Canvas ──────────────────────────────────
        const isSquare = this.scopeMode === 'vectorscope';
        // v3.3: Max resolution scopes (1000px width) with ultra-tall canvases to fill the panel
        const cW = 1000, cH = isSquare ? 1000 : 750;

        const canvas = document.createElement('canvas');
        canvas.width = cW;
        canvas.height = cH;
        canvas.style.cssText = `background: #010102; border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 8px; width: 100%; height: auto; aspect-ratio: ${cW}/${cH}; box-shadow: inset 0 0 30px rgba(0,0,0,0.9);`;
        container.appendChild(canvas);

        // ─── Extract Pixel Data ─────────────────────────────
        if (!this.image) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#444'; ctx.font = '12px monospace';
            ctx.fillText('No Image', cW / 2 - 30, cH / 2);
            return;
        }

        const sampleW = 1000;
        const sampleH = Math.round(this.image.height * (sampleW / this.image.width));
        const tmp = document.createElement('canvas');
        tmp.width = sampleW; tmp.height = sampleH;
        const tctx = tmp.getContext('2d');

        // Sample from graded GL canvas if possible, else fall back to raw image
        const srcCanvas = (this.glCanvas && this.glCanvas.width > 0) ? this.glCanvas : this.image;
        tctx.drawImage(srcCanvas, 0, 0, sampleW, sampleH);
        const imgData = tctx.getImageData(0, 0, sampleW, sampleH);

        // ─── Optional Log Transform ─────────────────────────
        // Applies a simplified LogC-style curve (log base ~300) to pixels so that
        // the scope shadows are expanded and highlights compressed — exactly as
        // broadcast monitors with "log assist" display work.
        let pixels;
        if (this.scopeLogView) {
            pixels = this._scopeApplyLogCurve(imgData.data);
        } else {
            pixels = imgData.data;
        }

        const ctx = canvas.getContext('2d');

        // ─── Render Based on Mode ───────────────────────────
        const logFlag = this.scopeLogView;
        switch (this.scopeMode) {
            case 'parade': this._drawScopeParade(ctx, pixels, sampleW, sampleH, cW, cH, logFlag); break;
            case 'waveform': this._drawScopeWaveform(ctx, pixels, sampleW, sampleH, cW, cH, logFlag); break;
            case 'histogram': this._drawScopeHistogram(ctx, pixels, cW, cH, logFlag); break;
            case 'vectorscope': this._drawScopeVectorscope(ctx, pixels, cW, cH); break;
            case 'falsecolor': this._drawScopeFalseColor(ctx, pixels, sampleW, sampleH, cW, cH); break;
        }
    }

    // ─── Log Curve Transform for Scopes ──────────────────────────────────────
    // Converts linear 8-bit scope pixels to a log-like representation.
    // Based on a LogC-inspired curve: maps 0–255 through log(1 + v*c)/log(1+c)
    // where c = 299 gives a roughly Arri LogC3-shaped response.
    // Output is a NEW Uint8ClampedArray (source data is not mutated).
    _scopeApplyLogCurve(data) {
        const out = new Uint8ClampedArray(data.length);
        const C = 299.0;
        const logC1 = Math.log(1.0 + C);

        // Pre-build LUT for speed
        const lut = new Uint8Array(256);
        for (let i = 0; i < 256; i++) {
            const lin = i / 255.0;
            const logVal = Math.log(1.0 + lin * C) / logC1;
            lut[i] = Math.round(logVal * 255);
        }

        for (let i = 0; i < data.length; i += 4) {
            out[i] = lut[data[i]];
            out[i + 1] = lut[data[i + 1]];
            out[i + 2] = lut[data[i + 2]];
            out[i + 3] = data[i + 3];
        }
        return out;
    }

    // ─── RGB Parade ──────────────────────────────────────────
    _drawScopeParade(ctx, data, imgW, imgH, w, h, logView) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        const secW = Math.floor(w / 3);
        const channels = [
            { idx: 0, color: 'rgba(255,80,80,', label: 'R', x: 0 },
            { idx: 1, color: 'rgba(80,255,80,', label: 'G', x: secW },
            { idx: 2, color: 'rgba(80,120,255,', label: 'B', x: secW * 2 }
        ];

        const step = Math.max(1, Math.floor(imgW / secW));

        channels.forEach(ch => {
            // Plot dots
            ctx.globalAlpha = 1.0;
            for (let col = 0; col < imgW; col += step) {
                const x = ch.x + Math.floor((col / imgW) * secW);
                const hist = new Uint32Array(256);
                for (let row = 0; row < imgH; row++) {
                    hist[data[(row * imgW + col) * 4 + ch.idx]]++;
                }
                for (let v = 0; v < 256; v++) {
                    if (hist[v] > 0) {
                        const intensity = Math.min(hist[v] / (imgH * 0.08), 1.0);
                        const alpha = intensity * 0.7 + 0.15;
                        ctx.fillStyle = ch.color + alpha + ')';
                        ctx.fillRect(x, h - (v / 255) * h, 1, 2);
                    }
                }
            }

            // Label
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = ch.color + '0.8)';
            ctx.font = '18px monospace';
            ctx.fillText(ch.label, ch.x + 8, 22);
        });

        // Separators
        ctx.strokeStyle = '#333'; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(secW, 0); ctx.lineTo(secW, h);
        ctx.moveTo(secW * 2, 0); ctx.lineTo(secW * 2, h);
        ctx.stroke();

        // Guide lines: 0%, 50%, 100% with log labels if needed
        const guides = logView
            ? [{ v: 0, lbl: '0' }, { v: 0.5, lbl: '~18%' }, { v: 0.74, lbl: '~90%' }, { v: 1, lbl: '100' }]
            : [{ v: 0, lbl: '0' }, { v: 0.5, lbl: '50%' }, { v: 1, lbl: '100' }];
        ctx.strokeStyle = '#333'; ctx.setLineDash([6, 6]);
        ctx.font = '14px monospace'; ctx.fillStyle = '#444';
        guides.forEach(g => {
            const y = h - g.v * h;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            if (g.lbl) ctx.fillText(g.lbl, 4, y - 4);
        });
        ctx.setLineDash([]);
        if (logView) {
            ctx.fillStyle = '#554400'; ctx.font = '16px monospace';
            ctx.fillText('LOG', w - 40, h - 6);
        }
    }

    // ─── Luma Waveform ───────────────────────────────────────
    _drawScopeWaveform(ctx, data, imgW, imgH, w, h, logView) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        // IRE guide lines — in log view these map to approximate log stops
        const guides = logView
            ? [
                { v: 0, lbl: '0' },
                { v: 0.18, lbl: '~black' },
                { v: 0.50, lbl: '~18%' },
                { v: 0.74, lbl: '~90%' },
                { v: 1.0, lbl: '100' }
            ]
            : [
                { v: 0, lbl: '0' },
                { v: 0.25, lbl: '25' },
                { v: 0.50, lbl: '50' },
                { v: 0.75, lbl: '75' },
                { v: 1.0, lbl: '100' }
            ];

        ctx.strokeStyle = '#2a2a2a'; ctx.lineWidth = 2;
        ctx.font = '16px monospace'; ctx.fillStyle = '#444';
        guides.forEach(g => {
            const y = h - g.v * h;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            ctx.fillText(g.lbl, 4, y - 4);
        });

        // Plot luma dots
        const step = Math.max(1, Math.floor(imgW / w));
        ctx.globalAlpha = 0.08;
        for (let col = 0; col < imgW; col += step) {
            const x = Math.floor((col / imgW) * w);
            for (let row = 0; row < imgH; row += 2) {
                const idx = (row * imgW + col) * 4;
                const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                const y = h - (luma / 255) * h;
                const bright = Math.floor(40 + luma * 0.6);
                ctx.fillStyle = `rgb(${bright}, ${Math.floor(bright * 1.4)}, ${bright})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        ctx.globalAlpha = 1.0;

        // Label
        ctx.fillStyle = '#5a5'; ctx.font = '18px monospace';
        ctx.fillText(logView ? 'LUMA·LOG' : 'LUMA', 8, 22);
    }

    // ─── Histogram ───────────────────────────────────────────
    _drawScopeHistogram(ctx, data, w, h, logView) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
        for (let i = 0; i < data.length; i += 4) {
            hR[data[i]]++;
            hG[data[i + 1]]++;
            hB[data[i + 2]]++;
        }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);

        // Grid
        ctx.strokeStyle = '#222'; ctx.lineWidth = 2;
        for (let i = 1; i < 4; i++) {
            const x = (i / 4) * w;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }

        const drawCurve = (hist, color) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * w;
                const y = h - (hist[i] / max) * h * 0.95;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };

        // Fill under curves
        const fillCurve = (hist, color) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(0, h);
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * w;
                const y = h - (hist[i] / max) * h * 0.95;
                ctx.lineTo(x, y);
            }
            ctx.lineTo(w, h);
            ctx.fill();
        };

        ctx.globalAlpha = 0.15;
        fillCurve(hR, '#ff4444');
        fillCurve(hG, '#44ff44');
        fillCurve(hB, '#4488ff');
        ctx.globalAlpha = 0.8;
        drawCurve(hR, '#ff4444');
        drawCurve(hG, '#44ff44');
        drawCurve(hB, '#4488ff');
        ctx.globalAlpha = 1.0;

        // Labels
        ctx.fillStyle = '#666'; ctx.font = '16px monospace';
        ctx.fillText(logView ? 'LOG·0' : '0', 4, h - 6);
        ctx.fillText(logView ? 'LOG·255' : '255', w - 75, h - 6);
        if (logView) {
            ctx.fillStyle = '#554400';
            ctx.fillText('LOG', w - 40, 20);
        }
    }

    // ─── Vectorscope ─────────────────────────────────────────
    _drawScopeVectorscope(ctx, data, w, h) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, w, h);

        const cx = w / 2, cy = h / 2;
        const rad = Math.min(cx, cy) - 10;

        // Graticule rings
        ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 2;
        [0.25, 0.5, 0.75, 1.0].forEach(r => {
            ctx.beginPath(); ctx.arc(cx, cy, rad * r, 0, Math.PI * 2); ctx.stroke();
        });

        // Crosshair
        ctx.strokeStyle = '#1a1a1a';
        ctx.beginPath(); ctx.moveTo(cx, cy - rad); ctx.lineTo(cx, cy + rad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx - rad, cy); ctx.lineTo(cx + rad, cy); ctx.stroke();

        // Rec.709 color targets
        const targets = [
            { a: 103, c: '#f33', l: 'R' },
            { a: 167, c: '#ff0', l: 'Yl' },
            { a: 241, c: '#0f0', l: 'G' },
            { a: 283, c: '#0ff', l: 'Cy' },
            { a: 347, c: '#33f', l: 'B' },
            { a: 61, c: '#f0f', l: 'Mg' },
        ];
        targets.forEach(t => {
            const ang = (t.a - 90) * Math.PI / 180;
            const tx = cx + Math.cos(ang) * rad * 0.75;
            const ty = cy + Math.sin(ang) * rad * 0.75;
            ctx.fillStyle = t.c;
            ctx.beginPath(); ctx.arc(tx, ty, 6, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#555'; ctx.font = '16px monospace';
            ctx.fillText(t.l, tx + 10, ty + 6);
        });

        // Skin Tone Indicator (I-Line)
        ctx.strokeStyle = 'rgba(255, 140, 100, 0.4)';
        ctx.lineWidth = 3;
        ctx.setLineDash([6, 6]);
        const iLineAng = (123 - 90) * Math.PI / 180;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(iLineAng) * rad * 0.9, cy + Math.sin(iLineAng) * rad * 0.9);
        ctx.stroke();
        ctx.setLineDash([]);

        // Plot pixels
        ctx.globalAlpha = 0.04;
        const step = Math.max(1, Math.floor(data.length / 4 / 25000));
        for (let i = 0; i < data.length; i += 4 * step) {
            const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
            const y = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const u = (b - y) * 0.492;
            const v = (r - y) * 0.877;
            ctx.fillStyle = `rgb(${data[i]}, ${data[i + 1]}, ${data[i + 2]})`;
            ctx.fillRect(cx + u * rad * 2.2, cy - v * rad * 2.2, 1, 1);
        }
        ctx.globalAlpha = 1.0;
    }

    // ─── False Color ─────────────────────────────────────────
    _drawScopeFalseColor(ctx, data, imgW, imgH, w, h) {
        // Create false color image from luminance
        const outImg = ctx.createImageData(w, h);
        const out = outImg.data;

        // Scale factors
        const sx = imgW / w, sy = imgH / h;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const srcX = Math.floor(x * sx), srcY = Math.floor(y * sy);
                const idx = (srcY * imgW + srcX) * 4;
                const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                const ire = luma / 255; // 0..1

                let r, g, b;
                if (ire < 0.02) {
                    // Under black — purple
                    r = 80; g = 0; b = 120;
                } else if (ire < 0.10) {
                    // Deep shadows — blue
                    r = 20; g = 40; b = 180;
                } else if (ire < 0.25) {
                    // Shadows — cyan
                    r = 0; g = 140; b = 180;
                } else if (ire < 0.40) {
                    // Low mid — teal
                    r = 0; g = 160; b = 100;
                } else if (ire < 0.55) {
                    // Mid — green (proper exposure)
                    r = 40; g = 180; b = 40;
                } else if (ire < 0.68) {
                    // Upper mid — yellow-green
                    r = 160; g = 180; b = 0;
                } else if (ire < 0.80) {
                    // Highlights — yellow
                    r = 220; g = 200; b = 0;
                } else if (ire < 0.90) {
                    // Hot highlights — orange
                    r = 240; g = 120; b = 0;
                } else if (ire < 0.97) {
                    // Near clipping — red
                    r = 230; g = 30; b = 30;
                } else {
                    // Clipped — hot pink/white
                    r = 255; g = 50; b = 150;
                }

                const oi = (y * w + x) * 4;
                out[oi] = r; out[oi + 1] = g; out[oi + 2] = b; out[oi + 3] = 255;
            }
        }

        ctx.putImageData(outImg, 0, 0);

        // Legend
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(w - 144, 4, 140, 24);
        ctx.fillStyle = '#aaa'; ctx.font = '18px monospace';
        ctx.fillText('FALSE COLOR', w - 138, 22);

        // IRE scale bar
        const barX = w - 14, barH = h - 20, barY = 18;
        const ireColors = [
            [80, 0, 120], [20, 40, 180], [0, 140, 180], [0, 160, 100],
            [40, 180, 40], [160, 180, 0], [220, 200, 0], [240, 120, 0],
            [230, 30, 30], [255, 50, 150]
        ];
        const segH = barH / ireColors.length;
        ireColors.forEach((c, i) => {
            ctx.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
            ctx.fillRect(barX, barY + (ireColors.length - 1 - i) * segH, 10, segH);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         UNDO / REDO STACK
    // ═══════════════════════════════════════════════════════════════════════════

    _captureGradingState() {
        return {
            exposure: this.exposure || 0.0,
            lift: this.lift ? [...this.lift] : [0, 0, 0],
            gamma: this.gamma ? (Array.isArray(this.gamma) ? [...this.gamma] : [this.gamma, this.gamma, this.gamma]) : [1, 1, 1],
            gain: this.gain ? [...this.gain] : [1, 1, 1],
            temperature: this.temperature || 0.0,
            tint: this.tint || 0.0,
            contrast: this.contrast || 1.0,
            pivot: this.pivot ?? 0.5,
            saturation: this.saturation || 1.0,
            grain: this.grain || 0.0,
            grainSize: this.grainSize || 1.0,
            grainColor: this.grainColor || 0.0,
            grainAnimate: this.grainAnimate || false,
            denoise: this.denoise || 0.0,
            bloom: this.bloom || 0.0,
            halation: this.halation || 0.0,
            diffusion: this.diffusion || 0.0,
            lensDistortion: this.lensDistortion || 0.0,
            lensFringe: this.lensFringe || 0.0,
            vignetteIntensity: this.vignetteIntensity || 0.0,
            vignetteFalloff: this.vignetteFalloff ?? 0.5,
            bokehHighlightBias: this.bokehHighlightBias || 0.0,
            bokehSoapBubble: this.bokehSoapBubble || 0.0,
            bokehOpticalVig: this.bokehOpticalVig || 0.0,
            apertureBlades: this.apertureBlades || 0,
            apertureRotation: this.apertureRotation || 0.0,
            apertureAnamorphic: this.apertureAnamorphic || 1.0,
            anamorphicStreaks: this.anamorphicStreaks || 0.0,
            // v3.4 additions
            printerR: this.printerR || 0,
            printerG: this.printerG || 0,
            printerB: this.printerB || 0,
            softClip: this.softClip || 0.0,
        };
    }

    _restoreGradingState(snapshot) {
        this.exposure = snapshot.exposure;
        this.lift = [...snapshot.lift];
        this.gamma = [...snapshot.gamma];
        this.gain = [...snapshot.gain];
        this.temperature = snapshot.temperature;
        this.tint = snapshot.tint;
        this.contrast = snapshot.contrast;
        this.pivot = snapshot.pivot;
        this.saturation = snapshot.saturation;
        this.grain = snapshot.grain;
        this.grainSize = snapshot.grainSize ?? 1.0;
        this.grainColor = snapshot.grainColor ?? 0.0;
        this.grainAnimate = snapshot.grainAnimate ?? false;
        this.denoise = snapshot.denoise;
        this.bloom = snapshot.bloom ?? 0.0;
        this.halation = snapshot.halation ?? 0.0;
        this.diffusion = snapshot.diffusion ?? 0.0;
        this.lensDistortion = snapshot.lensDistortion;
        this.lensFringe = snapshot.lensFringe;
        this.vignetteIntensity = snapshot.vignetteIntensity;
        this.vignetteFalloff = snapshot.vignetteFalloff;
        this.bokehHighlightBias = snapshot.bokehHighlightBias ?? 0.0;
        this.bokehSoapBubble = snapshot.bokehSoapBubble ?? 0.0;
        this.bokehOpticalVig = snapshot.bokehOpticalVig ?? 0.0;
        this.apertureBlades = snapshot.apertureBlades ?? 0;
        this.apertureRotation = snapshot.apertureRotation ?? 0.0;
        this.apertureAnamorphic = snapshot.apertureAnamorphic ?? 1.0;
        this.anamorphicStreaks = snapshot.anamorphicStreaks ?? 0.0;
        // v3.4: Printer Lights + Soft Clip
        this.printerR = snapshot.printerR ?? 0;
        this.printerG = snapshot.printerG ?? 0;
        this.printerB = snapshot.printerB ?? 0;
        this.softClip = snapshot.softClip ?? 0.0;

        if (this.renderer) {
            this.renderer.setExposure(this.exposure);
            this.renderer.setLift(this.lift[0], this.lift[1], this.lift[2]);
            this.renderer.setGamma(this.gamma[0], this.gamma[1], this.gamma[2]);
            this.renderer.setGain(this.gain[0], this.gain[1], this.gain[2]);
            this.renderer.setTemperature(this.temperature);
            this.renderer.setTint(this.tint);
            this.renderer.setContrast(this.contrast);
            this.renderer.setPivot(this.pivot);
            this.renderer.setSaturation(this.saturation);
            this.renderer.setGrain(this.grain);
            this.renderer.setGrainSize(this.grainSize);
            this.renderer.setGrainColor(this.grainColor);
            this.renderer.setGrainAnimate(this.grainAnimate);
            this.renderer.setDenoise(this.denoise);
            this.renderer.setBloom(this.bloom);
            this.renderer.setHalation(this.halation);
            this.renderer.setDiffusion(this.diffusion);
            this.renderer.setLensDistortion(this.lensDistortion, this.lensFringe);
            this.renderer.setVignette(this.vignetteIntensity, this.vignetteFalloff);
            this.renderer.setBokehPhysics(this.bokehHighlightBias, this.bokehSoapBubble, this.bokehOpticalVig);
            this.renderer.setApertureShape(this.apertureBlades, this.apertureRotation, this.apertureAnamorphic);
            if (this.renderer.setAnamorphicStreaks) this.renderer.setAnamorphicStreaks(this.anamorphicStreaks);
            this.renderer.setPrinterLights(this.printerR, this.printerG, this.printerB);
            this.renderer.setSoftClip(this.softClip);
        }
        this.render();
    }

    _pushUndo() {
        const state = this._captureGradingState();
        this._undoStack.push(state);
        if (this._undoStack.length > this._undoMaxSize) {
            this._undoStack.shift(); // Drop oldest
        }
        this._redoStack = []; // Clear redo on new action
        this._lastUndoPush = performance.now();
    }

    // Debounced undo push for rapid-fire inputs (scroll wheel, printer lights)
    // Only captures if last push was >500ms ago — prevents flooding the stack
    _pushUndoDebounced() {
        const now = performance.now();
        if (!this._lastUndoPush || (now - this._lastUndoPush) > 500) {
            this._pushUndo();
        }
    }

    undo() {
        if (this._undoStack.length === 0) return;
        // Save current state to redo before restoring
        this._redoStack.push(this._captureGradingState());
        const prev = this._undoStack.pop();
        this._restoreGradingState(prev);
        // Re-render the controls panel to update knob visuals
        if (this.controlsPanel && this._lastRenderContent) this._lastRenderContent();
    }

    redo() {
        if (this._redoStack.length === 0) return;
        // Save current state to undo before restoring
        this._undoStack.push(this._captureGradingState());
        const next = this._redoStack.pop();
        this._restoreGradingState(next);
        if (this.controlsPanel && this._lastRenderContent) this._lastRenderContent();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         GRADE EXPORT & PRESETS
    // ═══════════════════════════════════════════════════════════════════════════

    saveGrade(name) {
        if (!name) name = prompt("Enter preset name:", "New Grade");
        if (!name) return;

        const state = this._captureGradingState();
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        presets[name] = state;
        localStorage.setItem('radiance_presets', JSON.stringify(presets));
        console.log(`[Radiance] Preset "${name}" saved.`);
        if (this._lastRenderContent) this._lastRenderContent(); // Refresh UI
    }

    loadGrade(name) {
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        const state = presets[name];
        if (state) {
            this._pushUndo();
            this._restoreGradingState(state);
            console.log(`[Radiance] Preset "${name}" applied.`);
        }
    }

    deleteGrade(name) {
        const presets = JSON.parse(localStorage.getItem('radiance_presets') || '{}');
        delete presets[name];
        localStorage.setItem('radiance_presets', JSON.stringify(presets));
        if (this._lastRenderContent) this._lastRenderContent();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         TERMINAL / SCRIPT EDITOR
    // ═══════════════════════════════════════════════════════════════════════════

    renderTerminalTab(container) {
        const t = this.theme;
        const STORAGE_SCRIPT = 'radiance_term_script';
        const STORAGE_HIST = 'radiance_term_history';
        const MAX_HIST = 50;

        // ── History ────────────────────────────────────────────────────────
        let history = JSON.parse(localStorage.getItem(STORAGE_HIST) || '[]');
        let histIdx = history.length; // Points past the last entry (fresh line)

        // ── Root layout ────────────────────────────────────────────────────
        const root = document.createElement('div');
        root.style.cssText = `display:flex;flex-direction:column;gap:0;height:100%;font-family:'JetBrains Mono',monospace;`;

        // ── Toolbar ────────────────────────────────────────────────────────
        const toolbar = document.createElement('div');
        toolbar.style.cssText = `display:flex;gap:6px;align-items:center;padding:8px 6px;border-bottom:1px solid rgba(255,255,255,0.07);flex-shrink:0;background:rgba(255,255,255,0.02);`;

        const mkBtn = (label, title, color) => {
            const b = document.createElement('div');
            b.textContent = label;
            b.title = title;
            b.style.cssText = `font-size:10px;font-weight:700;letter-spacing:0.5px;padding:4px 12px;border-radius:4px;cursor:pointer;border:1px solid rgba(255,255,255,0.1);color:${color || '#ccc'};background:rgba(255,255,255,0.04);transition:all 0.2s;user-select:none;`;
            b.onmouseenter = () => {
                b.style.background = 'rgba(255,255,255,0.1)';
                b.style.borderColor = color ? color + '88' : 'rgba(255,255,255,0.3)';
            };
            b.onmouseleave = () => {
                b.style.background = 'rgba(255,255,255,0.04)';
                b.style.borderColor = 'rgba(255,255,255,0.1)';
            };
            return b;
        };

        const runBtn = mkBtn('▶ RUN', 'Run script  (Ctrl+Enter)', t.accent);
        const clearBtn = mkBtn('⌫ CLEAR', 'Clear output  (Ctrl+L)', '#888');
        const resetBtn = mkBtn('↺ RESET', 'Reset Python namespace', '#e87');

        runBtn.style.borderColor = t.accent + '33';
        runBtn.style.color = t.accent;

        // Snippet presets
        const snippets = [
            ['[AUDIT] Find Model', 'if "prompt" in globals():\n    [print(f"Node {k}: {v[\'inputs\'][\"ckpt_name\"]}") for k,v in prompt.items() if "inputs" in v and "ckpt_name" in v["inputs"]]'],
            ['[AUDIT] Dependencies', 'import importlib.util\npackages = ["cv2", "numpy", "torch", "imageio", "PIL"]\nfor p in packages:\n    spec = importlib.util.find_spec(p)\n    print(f"{p:10}: {\"INSTALLED\" if spec else \"MISSING\"}")'],
            ['[IMG] Gamut Check', 'if "image" in globals():\n    # Simple check for values that would be out of sRGB gamut\n    out_of_gamut = torch.any(image > 1.0) or torch.any(image < 0.0)\n    print(f"Wide Gamut / HDR Detected: {out_of_gamut}")\n    if out_of_gamut:\n        print(f"Max Component Value: {image.max():.3f}")'],
            ['[IMG] Center RGB', 'if "image" in globals():\n    h, w = image.shape[1:3]\n    pixel = image[0, h//2, w//2, :3].tolist()\n    print(f"Center Pixel [R,G,B]: {[round(c, 4) for c in pixel]}")'],
            ['[DATA] Hist Plot', 'if "image" in globals():\n    h = torch.histc(image, bins=10, min=0, max=1)\n    [print(f"[{i*10}%-{(i+1)*10}%]: {\'#\' * int(v/image.numel()*100)}") for i,v in enumerate(h)]'],
            ['[UTIL] CDL Export', 'print(\"ASC CDL XML (Identity):\")\nprint(\"\"\"<ColorCorrection id=\\\"radiance\\\">\n  <SOPNode>\n    <Slope>1.0 1.0 1.0</Slope>\n    <Offset>0.0 0.0 0.0</Offset>\n    <Power>1.0 1.0 1.0</Power>\n  </SOPNode>\n  <SatNode>\n    <Saturation>1.000000</Saturation>\n  </SatNode>\n</ColorCorrection>\"\"\")'],
            ['[SYS] VRAM Stats', 'if torch.cuda.is_available():\n    m = torch.cuda.mem_get_info()\n    print(f"VRAM Free: {m[0]/1e9:.2f} GB / {m[1]/1e9:.2f} GB")'],
            ['[SYS] Memory Nuke', 'import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None; print("Pipeline Memory Flushed.")'],
        ];

        const snipSelect = document.createElement('select');
        snipSelect.title = 'Insert Radiance Snippet';
        snipSelect.innerHTML = '<option value="">◎ Snippets</option>' +
            snippets.map((s, i) => `<option value="${i}">${s[0]}</option>`).join('');
        snipSelect.style.cssText = `font-size:11px; font-weight:bold; background:#0d0d14; color:${t.accent}; border:1px solid ${t.accent}33; border-radius:4px; padding:3px 12px; cursor:pointer; outline:none; appearance:none; text-align:center; min-width:120px; transition: border-color 0.2s, background 0.2s;`;
        snipSelect.onmouseenter = () => snipSelect.style.borderColor = 'rgba(255,255,255,0.3)';
        snipSelect.onmouseleave = () => snipSelect.style.borderColor = 'rgba(255,255,255,0.1)';

        toolbar.appendChild(runBtn);
        toolbar.appendChild(clearBtn);
        toolbar.appendChild(resetBtn);

        // Spacer
        const spacer = document.createElement('div');
        spacer.style.flex = '1';
        toolbar.appendChild(spacer);
        toolbar.appendChild(snipSelect);

        root.appendChild(toolbar);

        // ── Editor ─────────────────────────────────────────────────────────
        const editorWrap = document.createElement('div');
        editorWrap.style.cssText = `flex:0 0 200px;position:relative;border-bottom:1px solid rgba(255,255,255,0.1);overflow:hidden;`;

        // Line numbers
        const lineNums = document.createElement('div');
        lineNums.style.cssText = `position:absolute;left:0;top:0;bottom:0;width:32px;background:rgba(0,0,0,0.35);border-right:1px solid rgba(255,255,255,0.06);padding-top:8px;font-size:11px;line-height:1.6;text-align:right;padding-right:6px;color:rgba(255,255,255,0.2);overflow:hidden;user-select:none;pointer-events:none;`;

        const editor = document.createElement('textarea');
        editor.spellcheck = false;
        editor.placeholder = '# Python script — Ctrl+Enter to run\nprint("Hello from Radiance Terminal")';
        editor.value = localStorage.getItem(STORAGE_SCRIPT) || '';
        editor.style.cssText = `width:100%;height:100%;box-sizing:border-box;padding:8px 8px 8px 40px;background:rgba(0,0,0,0.45);color:#e2e2f0;border:none;outline:none;resize:none;font-size:12px;font-family:'JetBrains Mono','Fira Code',monospace;line-height:1.6;tab-size:4;`;

        const updateLineNums = () => {
            const lines = editor.value.split('\n').length;
            lineNums.innerHTML = Array.from({ length: lines }, (_, i) => i + 1).join('<br>');
        };
        updateLineNums();

        editorWrap.appendChild(lineNums);
        editorWrap.appendChild(editor);
        root.appendChild(editorWrap);

        // ── Resize handle between editor and output ─────────────────────────
        const resizeHandle = document.createElement('div');
        resizeHandle.style.cssText = `height:4px;background:transparent;cursor:ns-resize;flex-shrink:0;z-index:10;`;
        resizeHandle.title = 'Drag to resize';
        resizeHandle.onmouseenter = () => resizeHandle.style.background = t.accent + '66';
        resizeHandle.onmouseleave = () => resizeHandle.style.background = 'transparent';
        let _rszDragging = false;
        resizeHandle.addEventListener('mousedown', e => {
            _rszDragging = true; e.preventDefault();
            const startY = e.clientY;
            const startH = editorWrap.offsetHeight;
            const onMove = me => {
                const newH = Math.max(60, Math.min(600, startH + me.clientY - startY));
                editorWrap.style.flex = `0 0 ${newH}px`;
            };
            const onUp = () => {
                _rszDragging = false;
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
        root.appendChild(resizeHandle);

        // ── Output panel ────────────────────────────────────────────────────
        const outputPanel = document.createElement('div');
        outputPanel.style.cssText = `flex:1;overflow-y:auto;padding:10px 12px;background:rgba(0,0,0,0.6);font-size:11px;font-family:'JetBrains Mono',monospace;line-height:1.6;min-height:0;scrollbar-width:thin;`;

        const appendOutput = (text, isError, cmdEcho) => {
            if (cmdEcho) {
                const cmdLine = document.createElement('div');
                cmdLine.style.cssText = `color:rgba(0,168,255,0.7);margin-top:6px;font-weight:600;`;
                cmdLine.textContent = '>>> ' + cmdEcho.split('\n')[0] + (cmdEcho.includes('\n') ? '...' : '');
                outputPanel.appendChild(cmdLine);
            }
            if (!text && !isError) {
                const ok = document.createElement('div');
                ok.style.cssText = `color:rgba(100,200,100,0.5);font-size:10px;padding-left:12px;`;
                ok.textContent = '✓ OK';
                outputPanel.appendChild(ok);
                return;
            }
            const lines = text.split('\n');
            lines.forEach(line => {
                if (!line && lines.length === 1) return;
                const el = document.createElement('div');
                el.style.color = isError ? '#ff7070' : '#b0f0b0';
                el.textContent = line;
                outputPanel.appendChild(el);
            });
            // Auto-scroll to bottom
            outputPanel.scrollTop = outputPanel.scrollHeight;
        };

        // Separator line
        const sep = document.createElement('div');
        sep.style.cssText = `height:1px;background:rgba(255,255,255,0.05);margin:6px 0;`;
        outputPanel.appendChild(sep);

        root.appendChild(outputPanel);

        // ── Status bar ─────────────────────────────────────────────────────
        const statusBar = document.createElement('div');
        statusBar.style.cssText = `display:flex;justify-content:space-between;padding:4px 10px;background:rgba(10,10,15,0.8);font-size:9px;color:rgba(255,255,255,0.3);letter-spacing:0.5px;flex-shrink:0;border-top:1px solid rgba(255,255,255,0.05);`;
        statusBar.innerHTML = `
            <div style="display:flex;gap:15px;align-items:center;">
                <span style="color:#666;font-weight:600;">PYTHON 3.11 REPL</span>
                <span id="rad-term-copy" style="cursor:pointer;color:${t.accent};opacity:0.6;transition:opacity 0.2s;">[COPY OUTPUT]</span>
                <span id="rad-term-clean" style="cursor:pointer;color:#e87;opacity:0.6;transition:opacity 0.2s;">[CLEAN MEM]</span>
            </div>
            <div style="display:flex;gap:12px;align-items:center;">
                <a href="https://radiance.fxtd.org/" target="_blank" style="color:rgba(0,168,255,0.5);text-decoration:none;" onmouseover="this.style.color='#00a8ff'" onmouseout="this.style.color='rgba(0,168,255,0.5)'">📖 DOCS</a>
                <span id="rad-term-status">READY</span>
            </div>
        `;
        root.appendChild(statusBar);

        // Status bar interactions
        const copyBtn = statusBar.querySelector('#rad-term-copy');
        copyBtn.onmouseenter = () => copyBtn.style.opacity = '1';
        copyBtn.onmouseleave = () => copyBtn.style.opacity = '0.6';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(outputPanel.innerText);
            copyBtn.textContent = '[COPIED!]';
            setTimeout(() => copyBtn.textContent = '[COPY OUTPUT]', 1500);
        };

        const cleanBtn = statusBar.querySelector('#rad-term-clean');
        cleanBtn.onmouseenter = () => cleanBtn.style.opacity = '1';
        cleanBtn.onmouseleave = () => cleanBtn.style.opacity = '0.6';
        cleanBtn.onclick = () => {
            outputPanel.innerHTML = '';
            resetNamespace();
        };

        const setStatus = (msg, color) => {
            const el = statusBar.querySelector('#rad-term-status');
            if (el) {
                el.textContent = msg.toUpperCase();
                el.style.color = color || 'rgba(255,255,255,0.3)';
            }
        };

        // ── Run script ─────────────────────────────────────────────────────
        const runScript = async () => {
            const code = editor.value.trim();
            if (!code) return;

            // Persist script
            localStorage.setItem(STORAGE_SCRIPT, editor.value);

            // Add to history
            if (history[history.length - 1] !== code) {
                history.push(code);
                if (history.length > MAX_HIST) history.shift();
                localStorage.setItem(STORAGE_HIST, JSON.stringify(history));
            }
            histIdx = history.length;

            setStatus('EXECUTING...', t.accent);
            runBtn.style.opacity = '0.5';

            try {
                const resp = await fetch('/radiance/terminal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code })
                });
                const data = await resp.json();
                appendOutput(data.output || '', data.status === 'error', code);
                setStatus(data.status === 'error' ? '✗ ERROR' : '✓ OK', data.status === 'error' ? '#ff7070' : '#80e080');
            } catch (e) {
                appendOutput(`Network error: ${e.message}`, true, code);
                setStatus('✗ NET ERROR', '#ff7070');
            } finally {
                runBtn.style.opacity = '1';
            }
        };

        // ── Reset namespace ────────────────────────────────────────────────
        const resetNamespace = async () => {
            try {
                const resp = await fetch('/radiance/terminal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: '__radiance_reset__ = True', _reset: true })
                });
                const data = await resp.json();
                const msg = document.createElement('div');
                msg.style.cssText = `color:#e87;font-size:10px;margin-top:8px;font-style:italic;opacity:0.8;`;
                msg.textContent = '— SYSTEM: PYTHON NAMESPACE RESET COMPLETE —';
                outputPanel.appendChild(msg);
                outputPanel.scrollTop = outputPanel.scrollHeight;
                setStatus('RESET OK', '#e87');
            } catch (e) {
                setStatus('RESET FAILED', '#ff7070');
            }
        };

        // ── Event wiring ────────────────────────────────────────────────────
        editor.addEventListener('input', () => {
            updateLineNums();
            localStorage.setItem(STORAGE_SCRIPT, editor.value);
        });

        editor.addEventListener('scroll', () => {
            lineNums.scrollTop = editor.scrollTop;
        });

        editor.addEventListener('keydown', e => {
            // Ctrl+Enter → Run
            if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); runScript(); return; }
            // Ctrl+L → Clear output
            if (e.ctrlKey && e.key === 'l') { e.preventDefault(); outputPanel.innerHTML = ''; setStatus('READY'); return; }

            // Tab → 4 spaces
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = editor.selectionStart;
                const end = editor.selectionEnd;
                editor.value = editor.value.substring(0, start) + '    ' + editor.value.substring(end);
                editor.selectionStart = editor.selectionEnd = start + 4;
                updateLineNums();
                return;
            }

            // Expert Feature: Auto-indent on Enter
            if (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey) {
                const start = editor.selectionStart;
                const lines = editor.value.substring(0, start).split('\n');
                const lastLine = lines[lines.length - 1];
                const indentMatch = lastLine.match(/^\s*/);
                const indent = indentMatch ? indentMatch[0] : '';

                // Allow a tiny delay so the actual newline is inserted by browser, then we add indent
                setTimeout(() => {
                    const newPos = editor.selectionStart;
                    editor.value = editor.value.substring(0, newPos) + indent + editor.value.substring(newPos);
                    editor.selectionStart = editor.selectionEnd = newPos + indent.length;
                    updateLineNums();
                }, 0);
            }

            // ↑↓ History (Alt + Up/Down)
            if (e.altKey && e.key === 'ArrowUp') {
                e.preventDefault();
                if (histIdx > 0) { histIdx--; editor.value = history[histIdx]; updateLineNums(); }
            }
            if (e.altKey && e.key === 'ArrowDown') {
                e.preventDefault();
                if (histIdx < history.length - 1) { histIdx++; editor.value = history[histIdx]; }
                else { histIdx = history.length; editor.value = ''; }
                updateLineNums();
            }
        });

        runBtn.onclick = () => runScript();
        clearBtn.onclick = () => { outputPanel.innerHTML = ''; setStatus('READY'); };
        resetBtn.onclick = () => {
            if (confirm('Reset Python namespace and clear all variables?')) {
                resetNamespace();
            }
        };

        snipSelect.onchange = () => {
            const idx = parseInt(snipSelect.value);
            if (!isNaN(idx) && snippets[idx]) {
                editor.value = snippets[idx][1];
                updateLineNums();
                localStorage.setItem(STORAGE_SCRIPT, editor.value);
            }
            snipSelect.value = '';
        };

        container.appendChild(root);
    }


    exportToCDL() {
        console.log("[Radiance] Generating ASC CDL (.cdl)...");

        // ASC CDL maps directly:
        // Slope = (gain * exposure_stops)
        // Offset = lift + global_offset
        // Power = gamma
        // Saturation = saturation

        const expMult = Math.pow(2.0, this.exposure || 0);

        const slopeX = ((this.gain[0] || 1) * expMult).toFixed(6);
        const slopeY = ((this.gain[1] || 1) * expMult).toFixed(6);
        const slopeZ = ((this.gain[2] || 1) * expMult).toFixed(6);

        const offsetX = ((this.lift[0] || 0) + (this.offset[0] || 0)).toFixed(6);
        const offsetY = ((this.lift[1] || 0) + (this.offset[1] || 0)).toFixed(6);
        const offsetZ = ((this.lift[2] || 0) + (this.offset[2] || 0)).toFixed(6);

        const powerX = (this.gamma[0] || 1).toFixed(6);
        const powerY = (this.gamma[1] || 1).toFixed(6);
        const powerZ = (this.gamma[2] || 1).toFixed(6);

        const sat = (this.saturation || 1.0).toFixed(6);

        const cdl = `<?xml version="1.0" encoding="UTF-8"?>
<ColorDecisionList xmlns="urn:ASC:CDL:v1.2">
  <ColorDecision>
<!-- Radiance Viewer Grade -->
<ColorCorrection id="radiance_grade">
  <SOPNode>
    <Slope>${slopeX} ${slopeY} ${slopeZ}</Slope>
    <Offset>${offsetX} ${offsetY} ${offsetZ}</Offset>
    <Power>${powerX} ${powerY} ${powerZ}</Power>
  </SOPNode>
  <SatNode>
    <Saturation>${sat}</Saturation>
  </SatNode>
</ColorCorrection>
  </ColorDecision>
</ColorDecisionList>
`;
        const blob = new Blob([cdl], { type: 'text/xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "radiance_grade.cdl";
        a.click();
        URL.revokeObjectURL(url);
    }

    importFromCDL(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(text, "text/xml");

            try {
                this._pushUndo();

                const getVal = (tag) => {
                    const el = xmlDoc.getElementsByTagName(tag)[0];
                    if (!el) return null;
                    return el.textContent.trim().split(/\\s+/).map(Number);
                };

                const slope = getVal("Slope");
                const offset = getVal("Offset");
                const power = getVal("Power");
                const satData = getVal("Saturation");

                if (slope) {
                    this.gain = slope;
                    this.exposure = 0; // Slope fully encodes exposure
                }

                if (offset) {
                    this.lift = offset;
                    this.offset = [0, 0, 0]; // Offset fully encoded in lift (or vice versa)
                }

                if (power) {
                    this.gamma = power;
                }

                if (satData) {
                    this.saturation = satData[0];
                }

                // Update renderer
                if (this.renderer) {
                    this.renderer.setExposure(this.exposure);
                    this.renderer.setLift(this.lift[0], this.lift[1], this.lift[2]);
                    this.renderer.setGain(this.gain[0], this.gain[1], this.gain[2]);
                    this.renderer.setGamma(this.gamma[0], this.gamma[1], this.gamma[2]);
                    this.renderer.setOffset(this.offset[0], this.offset[1], this.offset[2]);
                    this.renderer.setSaturation(this.saturation);
                }

                this.render();
                if (this._lastRenderContent) this._lastRenderContent(); // refresh UI panels

                console.log("[Radiance] CDL Imported successfully.");

            } catch (err) {
                console.error("[Radiance] Failed to parse CDL:", err);
                alert("Failed to parse CDL file. Ensure it is ASC CDL v1.2 XML format.");
            }
        };
        reader.readAsText(file);
    }

    exportToCube() {
        console.log("[Radiance] Generating 3D LUT (.cube)...");
        const size = 33;
        let cube = `TITLE "Radiance Export"\nLUT_3D_SIZE ${size} \nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n\n`;

        // Helper to apply math (matching radiance_webgl.js and Python apply_grading)
        const applyMath = (c) => {
            let r = c[0], g = c[1], b = c[2];

            // 1. Offset
            r += this.offset[0] || 0; g += this.offset[1] || 0; b += this.offset[2] || 0;

            // 2. Exposure (Stops)
            const expMult = Math.pow(2.0, this.exposure || 0);
            r *= expMult; g *= expMult; b *= expMult;

            // 3. White Balance (Temp / Tint usually skipped in LUT for neutral grey, but adding for completeness)
            // Skipping WB here as Temp is usually done globally before grading, but could be added.

            // 4. Lift (Pivoted at White)
            const luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const liftPivot = Math.max(0.0, Math.min(1.0, 1.0 - luma));
            r += (this.lift[0] || 0) * liftPivot;
            g += (this.lift[1] || 0) * liftPivot;
            b += (this.lift[2] || 0) * liftPivot;

            // 5. Gain
            r *= (this.gain[0] || 1); g *= (this.gain[1] || 1); b *= (this.gain[2] || 1);

            // 6. Gamma
            r = Math.sign(r) * Math.pow(Math.abs(r), 1.0 / (this.gamma[0] || 1));
            g = Math.sign(g) * Math.pow(Math.abs(g), 1.0 / (this.gamma[1] || 1));
            b = Math.sign(b) * Math.pow(Math.abs(b), 1.0 / (this.gamma[2] || 1));

            // 7. Contrast & Pivot
            const con = this.contrast || 1.0;
            const piv = this.pivot || 0.18;
            r = (r - piv) * con + piv;
            g = (g - piv) * con + piv;
            b = (b - piv) * con + piv;

            // 8. Log Wheels (Shadow/Midtone/Highlight)
            // Precise reimplementation of `applyLogWheels` from glsl
            const logLuma = r * 0.2126 + g * 0.7152 + b * 0.0722;

            // Shadow curve log_s(x)
            let logS = 0;
            if (logLuma <= 0.45) {
                if (logLuma <= 0.33) logS = 1.0;
                else logS = 1.0 - (logLuma - 0.33) / (0.45 - 0.33);
            }

            // Highlight curve log_h(x)
            let logH = 0;
            if (logLuma >= 0.55) {
                if (logLuma >= 0.66) logH = 1.0;
                else logH = (logLuma - 0.55) / (0.66 - 0.55);
            }

            // Midtone curve log_m(x)
            const logM = 1.0 - logS - logH;

            const ls = this.logShadow || [1, 1, 1];
            const lm = this.logMidtone || [1, 1, 1];
            const lh = this.logHighlight || [1, 1, 1];

            r = r * (logS * ls[0] + logM * lm[0] + logH * lh[0]);
            g = g * (logS * ls[1] + logM * lm[1] + logH * lh[1]);
            b = b * (logS * ls[2] + logM * lm[2] + logH * lh[2]);

            // 9. Saturation
            const luma2 = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const sat = this.saturation || 1.0;
            r = luma2 + (r - luma2) * sat;
            g = luma2 + (g - luma2) * sat;
            b = luma2 + (b - luma2) * sat;

            return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
        };

        for (let b = 0; b < size; b++) {
            for (let g = 0; g < size; g++) {
                for (let r = 0; r < size; r++) {
                    const result = applyMath([r / (size - 1), g / (size - 1), b / (size - 1)]);
                    cube += `${result[0].toFixed(6)} ${result[1].toFixed(6)} ${result[2].toFixed(6)}\n`;
                }
            }
        }

        const blob = new Blob([cube], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "radiance_grade.cube";
        a.click();
        URL.revokeObjectURL(url);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         KNOB CONTROL WIDGET
    // ═══════════════════════════════════════════════════════════════════════════

    createKnob(label, min, max, initial, step, callback) {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 60px; width: 70px; flex: 1; max-width: 90px; position: relative;';

        // 1. Label
        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 9.5px; font-weight: 500; font-family: ${this.theme.font}; letter-spacing: 0.4px; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; text-align: center;`;

        // 2. Knob Wrapper (for centering value)
        const knobWrapper = document.createElement('div');
        knobWrapper.style.cssText = 'position: relative; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;';

        const size = 48;
        const strokeWidth = 3;
        const radius = (size - strokeWidth) / 2;
        const circumference = 2 * Math.PI * radius;

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", size);
        svg.setAttribute("height", size);
        svg.style.cssText = "cursor: ns-resize; transform: rotate(-90deg); touch-action: none;";

        // Track (Background Ring)
        const track = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        track.setAttribute("cx", size / 2);
        track.setAttribute("cy", size / 2);
        track.setAttribute("r", radius);
        track.setAttribute("stroke", "rgba(255, 255, 255, 0.1)");
        track.setAttribute("stroke-width", strokeWidth);
        track.setAttribute("fill", "transparent");

        // Progress Arc
        const progress = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        progress.setAttribute("cx", size / 2);
        progress.setAttribute("cy", size / 2);
        progress.setAttribute("r", radius);
        progress.setAttribute("stroke", this.theme.accent);
        progress.setAttribute("stroke-width", strokeWidth);
        progress.setAttribute("fill", "transparent");
        progress.setAttribute("stroke-dasharray", circumference);
        progress.setAttribute("stroke-dashoffset", circumference);
        progress.style.transition = "stroke-dashoffset 0.05s linear";

        svg.appendChild(track);
        svg.appendChild(progress);

        // 3. Value Display
        const valDisplay = document.createElement('div');
        valDisplay.textContent = initial.toFixed(2);
        valDisplay.style.cssText = `
        font-family: ${this.theme.mono};
        font-size: 10px;
        color: ${this.theme.text};
        position: absolute;
        pointer-events: none;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
        z-index: 10;
    `;

        knobWrapper.appendChild(svg);
        knobWrapper.appendChild(valDisplay);

        container.appendChild(lbl);
        container.appendChild(knobWrapper);

        // Logic
        let currentValue = initial;
        let startY = 0;
        let startVal = 0;

        const updateVisuals = (val) => {
            // Normalized 0..1
            const t = (val - min) / (max - min);
            const offset = circumference - (t * circumference);
            progress.setAttribute("stroke-dashoffset", offset);

            // v3.1: Clamp/Danger Indicator (P2 UX Fix)
            // If we are at the extreme edges (within 1% of range), add a glow
            const isAtLimit = (val <= min + (max - min) * 0.001) || (val >= max - (max - min) * 0.001);

            if (isAtLimit && val !== initial) {
                progress.setAttribute("stroke", "#ff4a4a"); // Red alert
                progress.style.filter = "drop-shadow(0 0 3px #ff4a4a)";
                valDisplay.style.color = "#ff4a4a";
            } else {
                progress.setAttribute("stroke", val === initial ? "rgba(255,255,255,0.3)" : this.theme.accent);
                progress.style.filter = "none";
                valDisplay.style.color = this.theme.text;
            }

            lbl.style.color = val === initial ? this.theme.textDim : (isAtLimit ? "#ff4a4a" : this.theme.accent);
            valDisplay.textContent = val.toFixed(2);
        };

        updateVisuals(initial);

        // Expose method to update externally (for presets)
        container.updateValue = (val) => {
            currentValue = val;
            updateVisuals(val);
        };

        // Interaction
        svg.onpointerdown = (e) => {
            e.preventDefault();
            // Capture undo state before this drag
            this._pushUndo();
            startY = e.clientY;
            startVal = currentValue;

            svg.setPointerCapture(e.pointerId);

            const onMove = (em) => {
                const delta = startY - em.clientY; // Standard drag delta
                // Sensitivity: Ctrl = fine (0.1x), Shift = coarse (3x)
                const range = max - min;
                let sensitivity = range * 0.005;
                if (em.ctrlKey || em.metaKey) sensitivity *= 0.1;  // Fine mode
                if (em.shiftKey) sensitivity *= 3.0;  // Coarse mode

                let newVal = startVal + (delta * sensitivity);
                newVal = Math.max(min, Math.min(newVal, max));

                // Snap to step
                if (step) newVal = Math.round(newVal / step) * step;

                currentValue = newVal;
                updateVisuals(currentValue);
                callback(currentValue);
            };

            const onUp = () => {
                svg.removeEventListener('pointermove', onMove);
                svg.removeEventListener('pointerup', onUp);
                svg.releasePointerCapture(e.pointerId);
            };

            svg.addEventListener('pointermove', onMove);
            svg.addEventListener('pointerup', onUp);
        };

        svg.ondblclick = () => {
            this._pushUndo();
            currentValue = initial;
            updateVisuals(initial);
            callback(initial);
        };

        // Right-click numeric entry
        svg.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Remove any existing input
            const existingInput = container.querySelector('.knob-numeric-input');
            if (existingInput) { existingInput.remove(); return; }

            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'knob-numeric-input';
            input.value = currentValue.toFixed(Math.max(0, -Math.log10(step || 0.01)));
            input.style.cssText = `
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 48px; text-align: center; font-size: 10px;
            font-family: ${this.theme.mono}; color: ${this.theme.text};
            background: rgba(0, 0, 0, 0.9); border: 1px solid ${this.theme.accent};
            border-radius: 4px; padding: 2px 4px; outline: none; z-index: 100;
        `;
            container.appendChild(input);
            input.focus();
            input.select();

            const apply = () => {
                let val = parseFloat(input.value);
                if (!isNaN(val)) {
                    this._pushUndo();
                    val = Math.max(min, Math.min(val, max));
                    if (step) val = Math.round(val / step) * step;
                    currentValue = val;
                    updateVisuals(currentValue);
                    callback(currentValue);
                }
                input.remove();
            };

            input.onkeydown = (ke) => {
                if (ke.key === 'Enter') { ke.preventDefault(); apply(); }
                else if (ke.key === 'Escape') { ke.preventDefault(); input.remove(); }
                ke.stopPropagation();
            };
            input.onblur = () => { setTimeout(() => { if (input.parentNode) apply(); }, 50); };
        });

        return container;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                    PER-CHANNEL COLOR WHEEL WIDGET
    // ═══════════════════════════════════════════════════════════════════════════

    createColorWheel(label, min, max, defaults, step, callback) {
        // defaults = [r, g, b] (Relative offsets usually 0,0,0)
        // Autodesk Flame Style: Wheel controls Chroma (Hue/Sat), Ring controls Master (Luma).

        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 120px; flex: 1; position: relative;';

        // Label
        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);`;
        container.appendChild(lbl);

        // SVG Constants
        const size = 110;
        const center = size / 2;
        const wheelRadius = 38;
        const ringOuter = 52;
        const ringInner = 46;
        const puckRadius = 5;

        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", size);
        svg.setAttribute("height", size);
        svg.style.cssText = "cursor: default; touch-action: none; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.4));";

        // 1. Master Ring Background
        const ringBg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        ringBg.setAttribute("cx", center); ringBg.setAttribute("cy", center);
        ringBg.setAttribute("r", (ringOuter + ringInner) / 2);
        ringBg.setAttribute("fill", "none");
        ringBg.setAttribute("stroke", "rgba(0,0,0,0.4)");
        ringBg.setAttribute("stroke-width", ringOuter - ringInner);
        svg.appendChild(ringBg);

        // 2. Master Ring Progress
        const ringProgress = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        const ringCirc = 2 * Math.PI * ((ringOuter + ringInner) / 2);
        ringProgress.setAttribute("cx", center); ringProgress.setAttribute("cy", center);
        ringProgress.setAttribute("r", (ringOuter + ringInner) / 2);
        ringProgress.setAttribute("fill", "none");
        ringProgress.setAttribute("stroke", this.theme.accent);
        ringProgress.setAttribute("stroke-width", ringOuter - ringInner - 2);
        ringProgress.setAttribute("stroke-dasharray", ringCirc);
        ringProgress.setAttribute("stroke-dashoffset", ringCirc); // Start empty
        ringProgress.setAttribute("stroke-linecap", "round");
        ringProgress.style.transformOrigin = "center";
        ringProgress.style.transform = "rotate(-90deg)"; // Start from top
        ringProgress.style.transition = "stroke-dashoffset 0.05s linear, stroke 0.2s";
        svg.appendChild(ringProgress);

        // 3. Inner Wheel Background
        const wheelBg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        wheelBg.setAttribute("cx", center); wheelBg.setAttribute("cy", center);
        wheelBg.setAttribute("r", wheelRadius);
        wheelBg.setAttribute("fill", "#111");
        wheelBg.setAttribute("stroke", "rgba(255,255,255,0.05)");
        svg.appendChild(wheelBg);

        // Crosshair
        const crossStyle = "stroke: rgba(255,255,255,0.05); stroke-width: 1;";
        const hLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        hLine.setAttribute("x1", center - wheelRadius); hLine.setAttribute("x2", center + wheelRadius);
        hLine.setAttribute("y1", center); hLine.setAttribute("y2", center); hLine.setAttribute("style", crossStyle);
        svg.appendChild(hLine);
        const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        vLine.setAttribute("x1", center); vLine.setAttribute("x2", center);
        vLine.setAttribute("y1", center - wheelRadius); vLine.setAttribute("y2", center + wheelRadius);
        vLine.setAttribute("style", crossStyle);
        svg.appendChild(vLine);

        // 4. Puck (Hue/Sat handle)
        const puck = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        puck.setAttribute("r", puckRadius);
        puck.setAttribute("fill", "#fff");
        puck.setAttribute("stroke", "#000");
        puck.setAttribute("stroke-width", "2");
        puck.style.cursor = "crosshair";
        puck.style.filter = "drop-shadow(0 2px 4px rgba(0,0,0,0.5))";
        svg.appendChild(puck);

        container.appendChild(svg);

        // Data State
        let wheelR = 0, wheelG = 0, wheelB = 0; // Chroma offsets
        let masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0; // Master offset

        const updateVisuals = () => {
            // Puck Position
            const angleR = 0, angleG = 2 * Math.PI / 3, angleB = 4 * Math.PI / 3;
            let dx = wheelR * Math.cos(angleR) + wheelG * Math.cos(angleG) + wheelB * Math.cos(angleB);
            let dy = wheelR * Math.sin(angleR) + wheelG * Math.sin(angleG) + wheelB * Math.sin(angleB);

            const pxSens = wheelRadius / 0.5;
            let px = dx * pxSens;
            let py = dy * pxSens;

            const dist = Math.sqrt(px * px + py * py);
            if (dist > wheelRadius) {
                px = (px / dist) * wheelRadius;
                py = (py / dist) * wheelRadius;
            }

            puck.setAttribute("cx", center + px);
            puck.setAttribute("cy", center + py);

            // Ring Position
            const range = max - min;
            const t = Math.max(0, Math.min(1, (masterVal - min) / range));
            const offset = ringCirc * (1 - t);
            ringProgress.setAttribute("stroke-dashoffset", offset);
            ringProgress.setAttribute("stroke", Math.abs(masterVal - (defaults[0] + defaults[1] + defaults[2]) / 3) < 0.001 ? "rgba(255,255,255,0.2)" : this.theme.accent);
        };

        // Initialize from defaults
        const initialAvg = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
        wheelR = defaults[0] - initialAvg;
        wheelG = defaults[1] - initialAvg;
        wheelB = defaults[2] - initialAvg;
        masterVal = initialAvg;
        updateVisuals();

        // Interaction
        svg.onpointerdown = (e) => {
            e.preventDefault();
            this._pushUndo();
            svg.setPointerCapture(e.pointerId);

            const rect = svg.getBoundingClientRect();
            const startX = e.clientX, startY = e.clientY;
            const distToCenter = Math.sqrt(Math.pow(startX - (rect.left + center), 2) + Math.pow(startY - (rect.top + center), 2));

            const isRingDrag = distToCenter > wheelRadius;
            const initialMaster = masterVal;

            const onMove = (em) => {
                if (isRingDrag) {
                    const deltaY = startY - em.clientY;
                    let sens = (max - min) * 0.005;
                    if (em.ctrlKey) sens *= 0.1;
                    if (em.shiftKey) sens *= 3.0;
                    masterVal = Math.max(min, Math.min(max, initialMaster + deltaY * sens));
                } else {
                    let dx = em.clientX - (rect.left + center);
                    let dy = em.clientY - (rect.top + center);
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > wheelRadius) {
                        dx = (dx / dist) * wheelRadius;
                        dy = (dy / dist) * wheelRadius;
                    }
                    const mag = dist / wheelRadius;
                    const ang = Math.atan2(dy, dx);
                    const strength = mag * 0.5;
                    wheelR = Math.cos(ang) * strength;
                    wheelG = Math.cos(ang - 2 * Math.PI / 3) * strength;
                    wheelB = Math.cos(ang - 4 * Math.PI / 3) * strength;
                }
                updateVisuals();
                callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
            };

            const onUp = () => {
                svg.removeEventListener('pointermove', onMove);
                svg.removeEventListener('pointerup', onUp);
                svg.releasePointerCapture(e.pointerId);
            };

            svg.addEventListener('pointermove', onMove);
            svg.addEventListener('pointerup', onUp);
        };

        svg.ondblclick = (e) => {
            const rect = svg.getBoundingClientRect();
            const distToCenter = Math.sqrt(Math.pow(e.clientX - (rect.left + center), 2) + Math.pow(e.clientY - (rect.top + center), 2));
            this._pushUndo();
            if (distToCenter > wheelRadius) {
                masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
            } else {
                wheelR = 0; wheelG = 0; wheelB = 0;
            }
            updateVisuals();
            callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
        };

        svg.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this._pushUndo();
            wheelR = 0; wheelG = 0; wheelB = 0;
            masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
            updateVisuals();
            callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
        });

        container.updateValue = (r, g, b) => {
            const m = (r + g + b) / 3.0;
            wheelR = r - m; wheelG = g - m; wheelB = b - m;
            masterVal = m;
            updateVisuals();
        };

        return container;
    }



    createControlRow(label, min, max, initial, step, callback) {
        // v3.0: Compact horizontal layout (Label + Value on top, Slider below)
        const row = document.createElement('div');
        row.style.cssText = 'display: flex; flex-direction: column; gap: 4px; width: 100%;';
        // ... (Legacy code maintained for Lens/other controls if needed)
        // Actually, we are replacing all grading controls with knobs.
        // Keeping this for backward compatibility or non-grading controls.

        const metaRow = document.createElement('div');
        metaRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center; width: 100%;';

        const lbl = document.createElement('div');
        lbl.textContent = label;
        lbl.style.cssText = `
    color: ${this.theme.textDim};
    font-size: 11px;
    font-family: ${this.theme.font};
    min-width: 50px;
    font-weight: 500;
    `;

        const value = document.createElement('div');
        value.textContent = initial.toFixed(2);
        value.style.cssText = `
    font-family: ${this.theme.mono};
    font-size: 11px;
    color: ${this.theme.accent};
    min-width: 40px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    `;

        metaRow.appendChild(lbl);
        metaRow.appendChild(value);

        const sliderContainer = document.createElement('div');
        sliderContainer.style.cssText = 'position: relative; height: 4px; background: #333; border-radius: 2px; margin-top: 2px;';

        const sliderFill = document.createElement('div');
        sliderFill.style.cssText = `
    position: absolute; left: 0; top: 0; height: 100%; background: #445;
    width: 50%; pointer-events: none; border-radius: 2px;
    `;

        const sliderInput = document.createElement('input');
        sliderInput.type = 'range';
        sliderInput.min = min; sliderInput.max = max; sliderInput.step = step; sliderInput.value = initial;
        sliderInput.style.cssText = `
    position: absolute; left: 0; top: -6px; width: 100%; height: 16px; opacity: 0; cursor: ew-resize; margin: 0;
    `;

        const updateVisuals = (val) => {
            const pct = ((val - min) / (max - min)) * 100;
            sliderFill.style.width = pct + '%';
            sliderFill.style.background = this.theme.accent;
            value.textContent = parseFloat(val).toFixed(2);
        };
        updateVisuals(initial);

        sliderInput.oninput = (e) => {
            const v = parseFloat(e.target.value);
            updateVisuals(v);
            callback(v);
        };

        // Reset on double click
        sliderInput.ondblclick = () => {
            sliderInput.value = initial;
            updateVisuals(initial);
            callback(initial);
        };

        sliderContainer.appendChild(sliderFill);
        sliderContainer.appendChild(sliderInput);

        row.appendChild(metaRow);
        row.appendChild(sliderContainer);

        return row;
    }

    toggleControls() {
        this.showControls = !this.showControls;
        // In panel-embedded mode, toggle the entire rightControlPanel (not just the HUD)
        if (this.rightControlPanel) {
            this.rightControlPanel.style.display = this.showControls ? 'flex' : 'none';
        } else if (this.controlsPanel) {
            this.controlsPanel.style.display = this.showControls ? 'flex' : 'none';
        }
        this.controlsToggle.style.color = this.showControls ? this.theme.accent : this.theme.textDim;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          PROGRESS & STATUS
    // ═══════════════════════════════════════════════════════════════════════════

    setupProgressUI() {
        const t = this.theme;
        this.progressContainer = document.createElement('div');
        this.progressContainer.style.cssText = `
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: rgba(0, 0, 0, 0.5);
    z-index: 50;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s ease;
    `;

        this.progressBar = document.createElement('div');
        this.progressBar.style.cssText = `
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, ${t.accent}, #4f4);
    transition: width 0.1s linear;
    box-shadow: 0 0 10px ${t.accent};
    `;

        this.progressText = document.createElement('div');
        this.progressText.style.cssText = `
    position: absolute;
    bottom: 6px;
    right: 10px;
    font-size: 10px;
    font-family: monospace;
    color: rgba(255, 255, 255, 0.8);
    text-shadow: 0 1px 2px black;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s ease;
    background: rgba(0, 0, 0, 0.6);
    padding: 2px 6px;
    border-radius: 4px;
    `;

        this.progressContainer.appendChild(this.progressBar);
        this.container.appendChild(this.progressContainer);
        this.container.appendChild(this.progressText);

        // API Events
        api.addEventListener("execution_start", () => {
            this.progressStart = Date.now();
            this.progressHistory = [];
            this.showProgress(true);
        });

        api.addEventListener("progress", ({ detail }) => {
            const { value, max } = detail;
            const pct = (value / max) * 100;
            this.progressBar.style.width = `${pct}% `;

            // ETA Calculation
            const now = Date.now();
            if (value > 0) {
                const elapsed = (now - this.progressStart) / 1000;
                const timePerStep = elapsed / value;
                const remaining = (max - value) * timePerStep;

                // Simple formatting
                const eta = remaining < 60 ? `${remaining.toFixed(1)} s` : `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)} s`;
                this.progressText.textContent = `Step ${value}/${max} | ETA: ${eta}`;
            }
        });

        api.addEventListener("executed", ({ detail }) => {
            // Hide progress eventually if queue empty, but ComfyUI usually handles global progress
        });

        api.addEventListener("status", ({ detail }) => {
            if (!detail || detail.exec_info.queue_remaining === 0) {
                this.showProgress(false);
            }
        });
    }

    showProgress(show) {
        this.progressContainer.style.opacity = show ? 1 : 0;
        this.progressText.style.opacity = show ? 1 : 0;
        if (!show) {
            setTimeout(() => {
                this.progressBar.style.width = '0%';
                this.progressText.textContent = '';
            }, 300); // Wait for fade out
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          COLOR TRANSFORMS
    // ═══════════════════════════════════════════════════════════════════════════

    applyColorTransform(r, g, b, mode) {
        // Helper: Linear to sRGB
        const lin2srgb = (c) => c > 0.0031308 ? 1.055 * Math.pow(c, 1 / 2.4) - 0.055 : 12.92 * c;
        // Helper: Linear to Rec.709
        const lin2rec709 = (c) => c < 0.018 ? 4.5 * c : 1.099 * Math.pow(c, 0.45) - 0.099;

        switch (mode) {
            case 'sRGB':
                // Assume Input is Linear, Output sRGB
                return [lin2srgb(r), lin2srgb(g), lin2srgb(b)];
            case 'Rec.709':
                // Assume Input is Linear, Output Rec.709
                return [lin2rec709(r), lin2rec709(g), lin2rec709(b)];
            case 'LogC3':
                // ARRI LogC3 to Rec709 LUT approximation
                r = (r > 0.1496582 ? (Math.pow(10.0, (r - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (r / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                g = (g > 0.1496582 ? (Math.pow(10.0, (g - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (g / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                b = (b > 0.1496582 ? (Math.pow(10.0, (b - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (b / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                return [lin2rec709(Math.max(0, r)), lin2rec709(Math.max(0, g)), lin2rec709(Math.max(0, b))];
            case 'ACEScg':
                // ACEScg -> Rec.709 (Simple Tonemap)
                // Matrix (AP1 -> Rec.709)
                let rr = r * 1.70485868 - g * 0.62171602 - b * 0.08329937;
                let gg = -r * 0.19644612 + g * 1.26432540 + b * 0.03212072;
                let bb = -r * 0.01776686 - g * 0.00403754 + b * 1.02179971;
                // Simple Tonemap
                rr = rr / (rr + 1); gg = gg / (gg + 1); bb = bb / (bb + 1);
                return [lin2rec709(Math.max(0, rr)), lin2rec709(Math.max(0, gg)), lin2rec709(Math.max(0, bb))];
            default:
                return [r, g, b];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          NPY PARSER
    // ═══════════════════════════════════════════════════════════════════════════

    // v2.2: Async HDR buffer parser (for fetch chain in onExecuted)
    // ── v3.5: Universal HDR/float buffer dispatcher ──────────────────────────
    // Dispatches to the correct parser based on magic bytes or NumPy dtype.
    // Supported: RHDR fp16, OpenEXR (HALF/FLOAT, NONE/ZIPS/ZIP), Radiance RGBE
    // (.hdr), 16-bit/32-bit float TIFF, RF32 raw binary, legacy NumPy .npy.
    async _parseHDRBuffer(buffer) {
        const b4 = new Uint8Array(buffer.slice(0, 4));
        const magic4 = String.fromCharCode(...b4);
        const magic2 = String.fromCharCode(b4[0], b4[1]);
        const u32le = new DataView(buffer).getUint32(0, true);

        // RHDR — proprietary zlib-compressed fp16 sidecar
        if (magic4 === 'RHDR') return await this._parseRHDR(buffer);

        // OpenEXR — magic 0x762f3101 = 20000630 (LE uint32)
        if (u32le === 20000630) return await this._parseEXR(buffer);

        // RF32 — raw float32 binary with 16-byte header
        if (magic4 === 'RF32') return this._parseRaw32(buffer);

        // Radiance RGBE .hdr — starts with "#?RADIANCE" or "#?RGBE" or "#?RG"
        if (magic4.startsWith('#?')) return this._parseRGBE(buffer);

        // TIFF — 'II' (little-endian) or 'MM' (big-endian)
        if (magic2 === 'II' || magic2 === 'MM') return this._parseTIFF(buffer);

        // Legacy NumPy .npy fallback
        return this._parseNumpy(buffer);
    }

    // v2.2: Parse .rhdr (compressed float16) or legacy .npy (float32)
    parseHDRSidecar(buffer) {
        try {
            const view = new DataView(buffer);
            // Check for RHDR magic
            const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));

            if (magic === 'RHDR') {
                return this._parseRHDR(buffer);
            }
            // Fallback: try legacy .npy format
            return this._parseNumpy(buffer);
        } catch (e) {
            console.error("[Radiance] Failed to parse HDR sidecar:", e);
            return null;
        }
    }

    // Parse .rhdr: zlib-compressed float16 with 12-byte header
    async _parseRHDR(buffer) {
        const view = new DataView(buffer);
        const width = view.getUint16(4, true);
        const height = view.getUint16(6, true);
        const channels = view.getUint16(8, true);
        // v4.1: flags field (byte 10-11).
        //   flags = 0 → fp16 payload (legacy, HALF_FLOAT upload)
        //   flags = 1 → fp32 payload (32-bit Float mode, FLOAT upload)
        const flags = view.getUint16(10, true);
        const isFp32 = (flags & 1) !== 0;

        const bytesPerSample = isFp32 ? 4 : 2;
        const expectedSize = width * height * channels * bytesPerSample;
        const compressed = new Uint8Array(buffer, 12);
        const decompressed = await this._zlibInflateAsync(compressed, expectedSize);
        if (!decompressed) return null;

        if (decompressed.byteLength !== expectedSize) {
            console.error(`[Radiance] RHDR integrity failure: expected ${expectedSize} bytes, got ${decompressed.byteLength} (${width}×${height}×${channels}ch, ${isFp32 ? 'fp32' : 'fp16'})`);
            return null;
        }

        if (isFp32) {
            // v4.1: fp32 path — direct Float32Array, no conversion needed.
            // fp16data is null so loadHDRData uses loadFloat32Texture (full precision).
            const fp32 = new Float32Array(
                decompressed.buffer,
                decompressed.byteOffset,
                decompressed.byteLength / 4
            );
            console.log(`[Radiance] RHDR fp32 decoded: ${width}×${height}×${channels}ch (${(decompressed.byteLength / 1048576).toFixed(1)} MB)`);
            return {
                data: fp32,   // Float32Array for CPU reads (probe, scopes)
                fp16data: null,   // null → viewer uses loadFloat32Texture
                shape: [height, width, channels],
                format: 'rhdr_f32'
            };
        }

        // Legacy fp16 path (flags=0)
        // Raw float16 as Uint16Array (for WebGL HALF_FLOAT upload)
        const fp16Raw = new Uint16Array(decompressed.buffer, decompressed.byteOffset, decompressed.byteLength / 2);

        // Also create Float32Array for CPU-side reads (probe, scopes)
        const fp32 = new Float32Array(fp16Raw.length);
        for (let i = 0; i < fp16Raw.length; i++) {
            fp32[i] = this._halfToFloat(fp16Raw[i]);
        }

        return {
            data: fp32,      // Float32Array for CPU reads
            fp16data: fp16Raw,   // Uint16Array for GPU HALF_FLOAT upload
            shape: [height, width, channels],
            format: 'rhdr'
        };
    }

    // Legacy .npy parser (backward compatibility)
    _parseNumpy(buffer) {
        const headerLengthBlock = new DataView(buffer.slice(8, 10)).getUint16(0, true);
        const headerStr = new TextDecoder("ascii").decode(buffer.slice(10, 10 + headerLengthBlock));

        const shapeMatch = headerStr.match(/\'shape\':\s*\(([^)]+)\)/);
        if (!shapeMatch) return null;

        const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        const dtypeMatch = headerStr.match(/\'descr\':\s*\'([^']+)\'/);
        const dtype = dtypeMatch ? dtypeMatch[1] : '<f4';

        const isFloat32 = dtype.includes('f4');
        const offset = 10 + headerLengthBlock;

        if (isFloat32) {
            return {
                data: new Float32Array(buffer.slice(offset)),
                shape: shape,
                format: 'npy'
            };
        }
        return null;
    }

    // ── v3.5: OpenEXR parser ─────────────────────────────────────────────────
    // Supports: single-part scanline, HALF or FLOAT pixels, 1–4 channels.
    // Compression: NONE (0), ZIPS (2, 1 scanline/block), ZIP (3, 16 scanlines/block).
    // Channels are re-ordered from EXR alphabetical (A,B,G,R) to RGBA output.
    async _parseEXR(buffer) {
        const bytes = new Uint8Array(buffer);
        const view = new DataView(buffer);

        // Version flags — only single-part scanline supported
        const flags = view.getUint32(4, true) >> 8;
        if (flags & 0x200) { console.warn('[Radiance EXR] Tiled EXR not supported.'); return null; }

        let p = 8;
        const readNullStr = () => {
            let e = p; while (bytes[e]) e++;
            const s = new TextDecoder('ascii').decode(bytes.subarray(p, e));
            p = e + 1; return s;
        };

        // ── Header ────────────────────────────────────────────────────────────
        let compression = 0;
        let channels = [];
        let dw = { xMin: 0, yMin: 0, xMax: 0, yMax: 0 };

        for (; ;) {
            const name = readNullStr(); if (!name) break;
            const type = readNullStr();                   // attribute type string
            const size = view.getInt32(p, true); p += 4;
            const end = p + size;

            if (name === 'compression') {
                compression = bytes[p];
            } else if (name === 'dataWindow') {
                dw = {
                    xMin: view.getInt32(p, true), yMin: view.getInt32(p + 4, true),
                    xMax: view.getInt32(p + 8, true), yMax: view.getInt32(p + 12, true)
                };
            } else if (name === 'channels') {
                let cp = p;
                while (cp < end) {
                    let ce = cp; while (bytes[ce]) ce++;
                    const chName = new TextDecoder('ascii').decode(bytes.subarray(cp, ce));
                    if (!chName) break;
                    cp = ce + 1;
                    channels.push({
                        name: chName, pixelType: view.getInt32(cp, true),
                        xSampling: view.getInt32(cp + 8, true),
                        ySampling: view.getInt32(cp + 12, true)
                    });
                    cp += 16;
                }
            }
            p = end;
        }

        if (!channels.length) { console.warn('[Radiance EXR] No channels in header.'); return null; }
        if (compression !== 0 && compression !== 2 && compression !== 3) {
            console.warn(`[Radiance EXR] Unsupported compression ${compression} (supported: 0/2/3).`);
            return null;
        }

        const W = dw.xMax - dw.xMin + 1;
        const H = dw.yMax - dw.yMin + 1;
        const nCh = channels.length;
        const pixelType = channels[0].pixelType;   // 1=HALF, 2=FLOAT (assume homogeneous)
        const bytesPerVal = pixelType === 2 ? 4 : 2;
        const linesPerBlock = compression === 3 ? 16 : 1;
        const nBlocks = Math.ceil(H / linesPerBlock);

        // ── Offset table ──────────────────────────────────────────────────────
        // B-10 FIX: OpenEXR spec uses uint64 offsets. Previously only the low 32
        // bits were read (getUint32) which corrupts offsets for files > 4GB.
        // Combine two uint32 reads since DataView lacks getUint64.
        const offsets = [];
        for (let i = 0; i < nBlocks; i++) {
            const lo = view.getUint32(p, true);
            const hi = view.getUint32(p + 4, true);
            offsets.push(lo + hi * 0x100000000);
            p += 8;
        }

        // ── Channel → RGBA slot ───────────────────────────────────────────────
        // EXR stores channels in alphabetical order: A,B,G,R for RGBA; B,G,R for RGB
        const SLOT = { R: 0, G: 1, B: 2, A: 3 };
        const outCh = Math.min(nCh, 4);
        const chSlots = channels.map(ch => {
            const k = ch.name.toUpperCase();
            return (k in SLOT) ? SLOT[k] : (channels.indexOf(ch) < outCh ? channels.indexOf(ch) : 0);
        });

        const out = new Float32Array(W * H * outCh);

        // ── Block inflate (ZIP/ZIPS) — parallel ───────────────────────────────
        let blockDatas = null;
        if (compression !== 0) {
            blockDatas = await Promise.all(offsets.map(async (bOff, b) => {
                const bSize = view.getUint32(bOff + 4, true);
                const linesInBlk = Math.min(linesPerBlock, H - b * linesPerBlock);
                const uncompSize = linesInBlk * nCh * W * bytesPerVal;
                const compData = bytes.subarray(bOff + 8, bOff + 8 + bSize);

                const inflated = await this._zlibInflateAsync(compData, uncompSize);
                if (!inflated) return null;

                // EXR un-prediction: running sum (inverse of delta encode)
                for (let i = 1; i < inflated.length; i++)
                    inflated[i] = (inflated[i] + inflated[i - 1]) & 0xff;

                // EXR reinterleave: [firstHalf | secondHalf] → interleaved bytes
                const half = (uncompSize + 1) >> 1;
                const interleaved = new Uint8Array(uncompSize);
                let k = 0, k1 = 0, k2 = 0;
                while (k < uncompSize) {
                    interleaved[k++] = inflated[k1++];
                    if (k < uncompSize) interleaved[k++] = inflated[half + k2++];
                }
                return interleaved;
            }));
        }

        // ── Decode scanlines ──────────────────────────────────────────────────
        for (let b = 0; b < nBlocks; b++) {
            const bOff = offsets[b];
            const bY = view.getInt32(bOff, true);
            const bSize = view.getUint32(bOff + 4, true);
            const linesInBlk = Math.min(linesPerBlock, H - b * linesPerBlock);

            let bBytes, bView;
            if (compression === 0) {
                bBytes = bytes.subarray(bOff + 8, bOff + 8 + bSize);
                bView = new DataView(bBytes.buffer, bBytes.byteOffset, bBytes.byteLength);
            } else {
                if (!blockDatas[b]) continue;
                bBytes = blockDatas[b];
                bView = new DataView(bBytes.buffer, bBytes.byteOffset, bBytes.byteLength);
            }

            const lineStride = nCh * W * bytesPerVal;

            for (let li = 0; li < linesInBlk; li++) {
                const scanY = (bY - dw.yMin) + li;
                if (scanY < 0 || scanY >= H) continue;

                for (let ci = 0; ci < nCh; ci++) {
                    const slot = chSlots[ci];
                    if (slot >= outCh) continue;
                    const chBase = li * lineStride + ci * W * bytesPerVal;

                    for (let x = 0; x < W; x++) {
                        const vo = chBase + x * bytesPerVal;
                        const val = pixelType === 2
                            ? bView.getFloat32(vo, true)
                            : this._halfToFloat(bView.getUint16(vo, true));
                        out[(scanY * W + x) * outCh + slot] = val;
                    }
                }
            }
        }

        console.log(`[Radiance EXR] Decoded ${W}×${H}×${outCh}ch, comp=${compression}, type=${pixelType === 2 ? 'FLOAT' : 'HALF'}`);
        return { data: out, shape: [H, W, outCh], format: 'exr', isLinear: true };
    }

    // ── v3.5: Radiance RGBE (.hdr) parser ────────────────────────────────────
    // Parses both new RLE (scanline 02 02 whi wlo) and uncompressed RGBE.
    // Output: scene-linear Float32Array, 3 channels (RGB).
    _parseRGBE(buffer) {
        const bytes = new Uint8Array(buffer);
        let p = 0;

        // Read header lines until blank line
        const readLine = () => {
            let e = p;
            while (e < bytes.length && bytes[e] !== 0x0a) e++;
            const line = new TextDecoder('ascii').decode(bytes.subarray(p, e));
            p = e + 1; return line;
        };

        let width = 0, height = 0, foundFormat = false;
        for (let i = 0; i < 200; i++) {          // cap at 200 header lines
            const line = readLine();
            if (!line || line === '\r') break;
            if (line.startsWith('FORMAT=32-bit_rle_rgbe') || line.startsWith('FORMAT=32-bit_rle_xyze'))
                foundFormat = true;
            const m = line.match(/^-Y\s+(\d+)\s+\+X\s+(\d+)/);
            if (m) { height = parseInt(m[1]); width = parseInt(m[2]); break; }
        }
        // Size line may be after the blank-line separator
        if (!width) {
            const sizeLine = readLine();
            const m = sizeLine.match(/-Y\s+(\d+)\s+\+X\s+(\d+)/);
            if (!m) return null;
            height = parseInt(m[1]); width = parseInt(m[2]);
        }
        if (!width || !height) return null;

        const out = new Float32Array(width * height * 3);

        const decodeRGBE = (r, g, b, e, outIdx) => {
            if (e === 0) return;
            const scale = Math.pow(2, e - 128 - 8);
            out[outIdx] = r * scale;
            out[outIdx + 1] = g * scale;
            out[outIdx + 2] = b * scale;
        };

        for (let y = 0; y < height; y++) {
            if (p + 4 > bytes.length) break;

            if (bytes[p] === 2 && bytes[p + 1] === 2) {
                // ── New RLE scanline ──────────────────────────────────────────
                const scanW = (bytes[p + 2] << 8) | bytes[p + 3];
                if (scanW !== width) return null;
                p += 4;

                // Decode 4 channels (R, G, B, E) independently
                const scanline = new Uint8Array(4 * width);
                for (let ch = 0; ch < 4; ch++) {
                    let x = 0;
                    while (x < width) {
                        if (p >= bytes.length) break;
                        const code = bytes[p++];
                        if (code > 128) {
                            const runLen = code - 128, val = bytes[p++];
                            for (let i = 0; i < runLen && x < width; i++, x++)
                                scanline[ch * width + x] = val;
                        } else {
                            for (let i = 0; i < code && x < width; i++, x++)
                                scanline[ch * width + x] = bytes[p++];
                        }
                    }
                }

                for (let x = 0; x < width; x++)
                    decodeRGBE(scanline[x], scanline[width + x], scanline[2 * width + x],
                        scanline[3 * width + x], (y * width + x) * 3);

            } else {
                // ── Uncompressed / old RLE ────────────────────────────────────
                for (let x = 0; x < width; x++) {
                    if (p + 4 > bytes.length) break;
                    // Old RLE repeat token: R==G==B==1 means repeat E times
                    if (bytes[p] === 1 && bytes[p + 1] === 1 && bytes[p + 2] === 1) {
                        const count = bytes[p + 3]; p += 4;
                        const prev = (y * width + x - 1) * 3;
                        const pr = out[prev], pg = out[prev + 1], pb = out[prev + 2];
                        for (let i = 0; i < count && x < width; i++, x++) {
                            out[(y * width + x) * 3] = pr;
                            out[(y * width + x) * 3 + 1] = pg;
                            out[(y * width + x) * 3 + 2] = pb;
                        }
                        x--;  // outer loop increments
                    } else {
                        decodeRGBE(bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3],
                            (y * width + x) * 3);
                        p += 4;
                    }
                }
            }
        }

        console.log(`[Radiance RGBE] Decoded ${width}×${height} HDR`);
        return { data: out, shape: [height, width, 3], format: 'rgbe', isLinear: true };
    }

    // ── v3.5: TIFF parser ─────────────────────────────────────────────────────
    // Supports: uncompressed strips, little-endian and big-endian,
    // 8-bit uint, 16-bit uint, 32-bit uint, and 32-bit float samples.
    // Multi-sample (RGB/RGBA) interleaved or planar (PlanarConfiguration 1/2).
    _parseTIFF(buffer) {
        const view = new DataView(buffer);
        const bytes = new Uint8Array(buffer);
        const order = view.getUint16(0);
        const le = (order === 0x4949);  // II = little-endian

        const g8 = (o) => bytes[o];
        const g16 = (o) => view.getUint16(o, le);
        const g32 = (o) => view.getUint32(o, le);
        const gF32 = (o) => view.getFloat32(o, le);

        if (g16(2) !== 42) { console.warn('[Radiance TIFF] Not a valid TIFF.'); return null; }

        const ifdOff = g32(4);
        const nEntry = g16(ifdOff);
        const tags = {};

        // Type byte-sizes: BYTE=1, ASCII=1, SHORT=2, LONG=4, RATIONAL=8
        const tSz = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8];

        const readVal = (type, off) => {
            if (type === 3) return g16(off);
            if (type === 4) return g32(off);
            return g8(off);
        };

        for (let i = 0; i < nEntry; i++) {
            const eOff = ifdOff + 2 + i * 12;
            const tag = g16(eOff);
            const type = g16(eOff + 2);
            const count = g32(eOff + 4);
            const sz = tSz[type] || 1;
            const dataOff = (count * sz > 4) ? g32(eOff + 8) : (eOff + 8);

            if (count === 1) {
                tags[tag] = readVal(type, dataOff);
            } else {
                const arr = [];
                for (let j = 0; j < count; j++) arr.push(readVal(type, dataOff + j * sz));
                tags[tag] = arr;
            }
        }

        const width = tags[256];
        const height = tags[257];
        const bps = Array.isArray(tags[258]) ? tags[258][0] : (tags[258] || 8);
        const comp = tags[259] || 1;
        const spp = tags[277] || 3;  // SamplesPerPixel
        const sfmt = Array.isArray(tags[339]) ? tags[339][0] : (tags[339] || 1); // SampleFormat
        const planar = tags[284] || 1;  // 1=interleaved, 2=planar
        const rps = tags[278] || height;

        if (!width || !height) return null;
        if (comp !== 1) {
            console.warn(`[Radiance TIFF] Compression ${comp} not supported (uncompressed only).`);
            return null;
        }

        const stripOffsets = Array.isArray(tags[273]) ? tags[273] : [tags[273]];
        const stripByteCounts = Array.isArray(tags[279]) ? tags[279] : [tags[279]];
        const bytesPerSample = bps / 8;
        const isFloat32TIFF = (bps === 32 && sfmt === 3);
        const outCh = Math.min(spp, 4);
        const out = new Float32Array(width * height * outCh);

        let globalRow = 0;
        for (let s = 0; s < stripOffsets.length; s++) {
            const sOff = stripOffsets[s];
            const nRows = Math.min(rps, height - globalRow);

            if (planar === 1) {
                // Interleaved: pixel = [R, G, B, (A)] contiguous
                for (let r = 0; r < nRows; r++) {
                    for (let x = 0; x < width; x++) {
                        const pxOff = sOff + (r * width + x) * spp * bytesPerSample;
                        for (let ch = 0; ch < outCh; ch++) {
                            const cOff = pxOff + ch * bytesPerSample;
                            let val;
                            if (bps === 32 && sfmt === 3) val = gF32(cOff);
                            else if (bps === 32) val = g32(cOff) / 4294967295.0;
                            else if (bps === 16) val = g16(cOff) / 65535.0;
                            else val = g8(cOff) / 255.0;
                            out[((globalRow + r) * width + x) * outCh + ch] = val;
                        }
                    }
                }
            } else {
                // Planar: each channel in separate strip-planes
                // stripOffsets has spp × nStrips entries; this is a simplified pass
                const chStripCount = Math.ceil(height / rps);
                for (let ch = 0; ch < outCh; ch++) {
                    const chSoff = stripOffsets[ch * chStripCount + s] || sOff;
                    for (let r = 0; r < nRows; r++) {
                        for (let x = 0; x < width; x++) {
                            const cOff = chSoff + (r * width + x) * bytesPerSample;
                            let val;
                            if (bps === 32 && sfmt === 3) val = gF32(cOff);
                            else if (bps === 32) val = g32(cOff) / 4294967295.0;
                            else if (bps === 16) val = g16(cOff) / 65535.0;
                            else val = g8(cOff) / 255.0;
                            out[((globalRow + r) * width + x) * outCh + ch] = val;
                        }
                    }
                }
            }
            globalRow += nRows;
        }

        const fmt = isFloat32TIFF ? 'tiff_f32' : (bps === 16 ? 'tiff_u16' : 'tiff_u8');
        console.log(`[Radiance TIFF] Decoded ${width}×${height}×${outCh}ch ${bps}-bit (${fmt})`);
        return {
            data: out, shape: [height, width, outCh], format: fmt,
            isLinear: isFloat32TIFF
        };
    }

    // ── v3.5: RF32 raw float32 binary parser ─────────────────────────────────
    // Minimal binary container for passing full-precision linear float data:
    //   Bytes 0–3  : magic 'RF32'
    //   Bytes 4–7  : width  (uint32 LE)
    //   Bytes 8–11 : height (uint32 LE)
    //   Bytes 12–15: channels (uint32 LE, typically 3 or 4)
    //   Bytes 16+  : raw IEEE 754 float32 values, interleaved RGBARGBA...
    // The Python-side encoder is: struct.pack('<4sIII', b'RF32', W, H, C) + data.tobytes()
    _parseRaw32(buffer) {
        const view = new DataView(buffer);
        const magic = String.fromCharCode(...new Uint8Array(buffer.slice(0, 4)));
        if (magic !== 'RF32') return null;

        const width = view.getUint32(4, true);
        const height = view.getUint32(8, true);
        const channels = view.getUint32(12, true);
        const expected = width * height * channels * 4;

        if (buffer.byteLength < 16 + expected) {
            console.warn(`[Radiance RF32] Buffer too small: need ${16 + expected}, got ${buffer.byteLength}`);
            return null;
        }

        const data = new Float32Array(buffer.slice(16, 16 + expected));
        console.log(`[Radiance RF32] Decoded ${width}×${height}×${channels}ch float32`);
        return { data, shape: [height, width, channels], format: 'rf32', isLinear: true };
    }

    // IEEE 754 half-float (16-bit) → single-precision float (32-bit)
    _halfToFloat(h) {
        const s = (h >> 15) & 0x1;
        const e = (h >> 10) & 0x1f;
        const m = h & 0x3ff;

        if (e === 0) {
            if (m === 0) return s ? -0.0 : 0.0;
            // Subnormal
            let val = m / 1024.0 * Math.pow(2, -14);
            return s ? -val : val;
        }
        if (e === 31) {
            return m ? NaN : (s ? -Infinity : Infinity);
        }
        let val = Math.pow(2, e - 15) * (1 + m / 1024.0);
        return s ? -val : val;
    }

    // v3.2: Robust zlib inflate with error handling and multi-format fallback.
    // Python zlib.compress() outputs zlib-wrapped deflate (RFC 1950).
    // DecompressionStream('deflate') handles zlib wrapper in modern browsers.
    // Fallback chain: deflate → deflate-raw → gzip.
    // expectedBytes: optional hint from RHDR header (width*height*channels*2)
    async _zlibInflateAsync(compressed, expectedBytes) {
        if (typeof DecompressionStream === 'undefined') {
            console.warn('[Radiance] DecompressionStream API not supported. Cannot load compressed RHDR.');
            return null;
        }

        // Use expectedBytes from RHDR header when available, otherwise 256MB hard cap
        const MAX_SIZE = expectedBytes ? Math.max(expectedBytes * 2, 64 * 1024 * 1024) : 256 * 1024 * 1024;

        // Try with zlib wrapper first ('deflate'), then raw deflate, then gzip as last resort
        for (const format of ['deflate', 'deflate-raw', 'gzip']) {
            try {
                const ds = new DecompressionStream(format);
                const writer = ds.writable.getWriter();
                const reader = ds.readable.getReader();

                writer.write(compressed);
                writer.close();

                const chunks = [];
                let totalLen = 0;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    totalLen += value.byteLength;
                    if (totalLen > MAX_SIZE) {
                        // Cancel the reader (writer is already closed)
                        await reader.cancel();
                        console.error(`[Radiance] RHDR decompression limit exceeded (${totalLen} > ${MAX_SIZE} bytes).`);
                        return null;
                    }
                    chunks.push(value);
                }

                if (totalLen === 0) {
                    console.warn(`[Radiance] _zlibInflateAsync(${format}): decompressed to 0 bytes, trying next format`);
                    continue;
                }
                const result = new Uint8Array(totalLen);
                let offset = 0;
                for (const chunk of chunks) {
                    result.set(new Uint8Array(chunk), offset);
                    offset += chunk.byteLength;
                }
                if (format !== 'deflate') {
                    console.log(`[Radiance] RHDR decompressed via fallback format: ${format}`);
                }
                return result;
            } catch (e) {
                if (format === 'gzip') {
                    // All formats failed
                    console.error('[Radiance] _zlibInflateAsync failed (all formats):', e?.message || e);
                    return null;
                }
                // Log and try next format
                console.warn(`[Radiance] _zlibInflateAsync(${format}) failed, trying next:`, e?.message || e);
            }
        }
        return null;
    }

    // Synchronous zlib inflate — not supported. Use _zlibInflateAsync() instead.
    _zlibInflate(compressed) {
        throw new Error('[Radiance] Synchronous zlib inflate is not supported. Use _zlibInflateAsync() instead.');
    }

    _setupComfyListeners() {
        const onProgress = (e) => {
            const { value, max } = e.detail;
            const p = value / max;
            this._lastProgress = p;
            if (this.neuralMonitor) {
                this.neuralMonitor.setProgress(p);
                // Detection for sampling vs decoding based on p jumps
                if (p > 0.01 && p < 1.0) {
                    this.neuralMonitor.setPhase('sampling');
                }
            }
        };

        const onExecuted = () => {
            this._lastProgress = 0;
            if (this.neuralMonitor) {
                this.neuralMonitor.setProgress(1.0);
                this.neuralMonitor.setPhase('stable');
            }
        };

        const onExecuting = (e) => {
            if (this.neuralMonitor) {
                // If we hit a VAE Decode node specifically (heuristic)
                const nodeId = e.detail;
                const node = app.graph.getNodeById(nodeId);
                const type = node?.type?.toLowerCase() || '';

                if (type.includes('sampling') || type.includes('sampler')) {
                    this.neuralMonitor.setPhase('sampling');
                } else if (type.includes('vae') || type.includes('decode')) {
                    this.neuralMonitor.setPhase('decoding');
                } else {
                    this.neuralMonitor.setPhase('init');
                }

                // Visual "Process Log"
                const nodeTitle = node?.title || type || nodeId;
                this._termLog('event', `[PROCESS] Executing: ${nodeTitle}`);
            }
        };

        const onStatus = (e) => {
            const queueCount = e.detail?.status?.exec_info?.queue_remaining || 0;
            if (queueCount > 0 && this.neuralMonitor && this.neuralMonitor.phase === 'stable') {
                this.neuralMonitor.setPhase('init');
            }
        };

        api.addEventListener("progress", onProgress);
        api.addEventListener("executed", onExecuted);
        api.addEventListener("executing", onExecuting);
        api.addEventListener("status", onStatus);
    }

    destroy() {
        RadianceViewer.allInstances.delete(this);
        if (RadianceViewer.activeInstance === this) {
            RadianceViewer.activeInstance = Array.from(RadianceViewer.allInstances)[0] || null;
            // Refresh HUD for the new active instance if it exists
            if (RadianceViewer.activeInstance && RadianceViewer.activeInstance._lastRenderContent) {
                RadianceViewer.activeInstance._lastRenderContent();
            }
        }
        // v2.4: Visibility Fix - Only remove HUD if NO Radiance Viewers remain in workflow
        if (RadianceViewer.allInstances.size === 0 && RadianceViewer.singletonHUD) {
            RadianceViewer.singletonHUD.remove();
            RadianceViewer.singletonHUD = null;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                          NODE REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════════

app.registerExtension({
    name: "FXTD.RadianceViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;
        if (comfyClass !== "FXTD_RadianceViewer" && comfyClass !== "FXTD_RadianceProViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const container = document.createElement('div');
            container.id = `radiance-viewer-${this.id}`;
            this.addDOMWidget("viewer", "viewer", container, { serialize: false, hideOnZoom: false });

            // Force container properties to ensure it expands
            container.style.display = 'flex';
            container.style.width = '100%';
            container.style.height = '100%';

            this.radianceViewer = new RadianceViewer(this, container);

            // Lifecycle hooks for singleton HUD management
            this.onRemoved = () => {
                if (this.radianceViewer) this.radianceViewer.destroy();
            };
            this.onSelected = () => {
                RadianceViewer.activeInstance = this.radianceViewer;
                if (this.radianceViewer && this.radianceViewer._lastRenderContent) {
                    this.radianceViewer._lastRenderContent();
                }
            };
        };
        // ═══════════════════════════════════════════════════════════════════════════════
        //                           CURVE EDITOR MOVED TO END
        // ═══════════════════════════════════════════════════════════════════════════════

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            if (this.radianceViewer) {
                // Debounce resize to prevent thrashing
                if (this._resizeTimer) clearTimeout(this._resizeTimer);
                this._resizeTimer = setTimeout(() => {
                    this.radianceViewer.resize();
                }, 20);
            }
        };

        nodeType.prototype.onExecuted = function (message) {
            // v2.3: Error handling - show backend errors instead of silent failure
            if (message?.error?.length) {
                alert("Radiance Viewer Error:\n" + message.error.join("\n"));
                return;
            }

            if (!message?.radiance_images?.length) return;

            const viewer = this.radianceViewer;
            if (!viewer) return;

            // v3.1: Increment generation ID to invalidate in-flight async loads from previous results
            viewer.generationID++;
            const currentGen = viewer.generationID;

            // Separate main images from compare and zdepth images
            const mainImages = message.radiance_images.filter(img => !img.is_compare && !img.is_zdepth);
            const compareImages = message.radiance_images.filter(img => img.is_compare);
            const zdepthImages = message.radiance_images.filter(img => img.is_zdepth);

            // Reset frame arrays
            viewer.frameImages = [];
            viewer.frameCompareImages = [];
            viewer.frameZdepthImages = [];
            viewer.frameHDRData = [];
            viewer.totalFrames = mainImages.length;
            viewer.currentFrame = 0;
            // FIX: Store the raw message data for robust metadata retrieval (foolproof EXR link)
            viewer.lastResult = message.radiance_images;
            // FIX: Clear stale active HDR data to prevent display of previous run's data
            viewer.hdrData = null;

            // v2.2: HDR-PRIMARY architecture — .rhdr is the display source, PNG is just a placeholder.
            // Like DJV/RV loading EXR: GPU tonemaps float data in real-time.
            mainImages.forEach((imgData, idx) => {
                const hasHDRPrimary = imgData.hdr_sidecar && imgData.hdr_primary;
                const hdrUrl = imgData.hdr_sidecar ?
                    api.apiURL(`/view?filename=${encodeURIComponent(imgData.hdr_sidecar)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`) : null;
                const pngUrl = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);

                // Load tiny PNG thumbnail as placeholder (shows instantly while RHDR loads)
                const img = new Image();
                img.crossOrigin = 'anonymous';
                // Store metadata on image object for fallback access
                img.exr_filename = imgData.exr_filename;
                img.exr_subfolder = imgData.exr_subfolder ?? imgData.subfolder ?? '';
                img.exr_type = imgData.exr_type ?? imgData.type ?? 'temp';
                img.subfolder = imgData.subfolder;
                img.type = imgData.type;

                img.onload = () => {
                    // v3.1: Abort if a newer generation has started
                    if (viewer.generationID !== currentGen) return;

                    viewer.frameImages[idx] = img;
                    // Only set as display if no HDR primary, or as placeholder while HDR loads
                    // If HDR load eventually fails, this will remain as the display image
                    if (idx === 0 && !viewer.frameHDRData[0]) {
                        viewer.image = img;
                        viewer.imageWidth = img.width;
                        viewer.imageHeight = img.height;
                        if (viewer.renderer) viewer.renderer.loadImageTexture(img);
                        viewer.fitToView();
                        viewer.render();
                    }
                    if (viewer._allFramesReady()) viewer.updateFrameDisplay();
                };
                img.onerror = (e) => {
                    console.error("[Radiance] Failed to load thumbnail:", imgData.filename, e);
                };
                img.src = pngUrl;

                // Load .rhdr as PRIMARY display source (like DJV loading EXR)
                if (hdrUrl) {
                    fetch(hdrUrl)
                        .then(r => r.arrayBuffer())
                        .then(async (buffer) => {
                            // v3.1: Abort if a newer generation has started
                            if (viewer.generationID !== currentGen) return;

                            // If parse fails (e.g. no DecompressionStream), it returns null
                            const npy = await viewer._parseHDRBuffer(buffer);

                            if (npy) {
                                npy.height = npy.shape[0];
                                npy.width = npy.shape[1];
                                npy.channels = npy.shape.length > 2 ? npy.shape[2] : 1;

                                // Propagate metadata
                                npy.exr_filename = imgData.exr_filename;
                                npy.exr_subfolder = imgData.exr_subfolder ?? imgData.subfolder ?? '';
                                npy.exr_type = imgData.exr_type ?? imgData.type ?? 'temp';
                                npy.subfolder = imgData.subfolder;
                                npy.type = imgData.type;

                                viewer.frameHDRData[idx] = npy;

                                // Set as PRIMARY display immediately
                                if (idx === viewer.currentFrame) {
                                    viewer.hdrData = npy;
                                    viewer.imageWidth = npy.width;
                                    viewer.imageHeight = npy.height;

                                    if (viewer.renderer) {
                                        let tex;
                                        const frameId = `${imgData.hdr_sidecar}_${idx}`;
                                        try {
                                            if (npy.fp16data) {
                                                // v3.0 #8: Use LRU cache — skip re-upload if already in GPU VRAM
                                                tex = viewer.renderer.loadFloat16TextureCached(
                                                    frameId,
                                                    npy.fp16data, npy.width, npy.height, npy.channels
                                                );
                                            } else {
                                                // B-6 FIX: Use LRU-cached fp32 loader (was loadFloat32Texture
                                                // which bypassed the 8-frame cache, re-uploading every scrub)
                                                tex = viewer.renderer.loadFloat32TextureCached(
                                                    frameId,
                                                    npy.data, npy.width, npy.height, npy.channels
                                                );
                                            }
                                        } catch (e) {
                                            console.warn("[Radiance] HDR Texture creation error:", e);
                                            tex = null;
                                        }

                                        // v3.1: Robust fallback — reload full-res PNG as display source
                                        // This handles case where texture creation fails (e.g. OOM or invalid dimensions)
                                        if (!tex) {
                                            console.warn("[Radiance] HDR texture load failed. Falling back to full-res PNG.");
                                            viewer.hdrData = null;
                                            viewer.frameHDRData[idx] = null;
                                            throw new Error("Texture creation failed"); // Trigger catch block for consistency
                                        }
                                    }

                                    viewer.createPlaceholderImage(npy.width, npy.height);
                                    viewer.fitToView();
                                    viewer.render();
                                    viewer.updateScopes();
                                    viewer.updateInfo();
                                }
                            } else {
                                throw new Error("RHDR parsing failed (returned null)");
                            }
                        })
                        .catch(e => {
                            console.warn("[Radiance] Failed to load RHDR primary:", e);
                            // v3.1: On RHDR fetch/parse failure, ensure PNG is loaded as fallback
                            // FIX: Race condition handled by checking if frameImages[idx] is available.
                            // If available, force-update display to use it if we are on that frame.
                            // If not available yet, img.onload will handle it (since frameHDRData[idx] is unset).

                            // If we already have the PNG and we are on this frame, ensure it's displayed
                            if (idx === viewer.currentFrame && viewer.frameImages[idx]) {
                                viewer.hdrData = null; // Explicitly clear any partial state
                                viewer.image = viewer.frameImages[idx];
                                viewer.imageWidth = viewer.image.width;
                                viewer.imageHeight = viewer.image.height;
                                if (viewer.renderer) viewer.renderer.loadImageTexture(viewer.image);
                                viewer.fitToView();
                                viewer.render();
                                viewer.updateInfo();
                            }
                        });
                }
            });

            // Load compare images
            compareImages.forEach((imgData, idx) => {
                const cmp = new Image();
                cmp.crossOrigin = 'anonymous';
                cmp.onload = () => {
                    if (viewer.generationID !== currentGen) return;
                    viewer.frameCompareImages[idx] = cmp;
                    if (idx === 0) viewer.setCompareImage(cmp);
                };
                cmp.onerror = (e) => console.warn("[Radiance] Failed to load compare image:", imgData.filename);
                cmp.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);
            });

            // Load Z-depth images
            zdepthImages.forEach((imgData, idx) => {
                const zImg = new Image();
                zImg.crossOrigin = 'anonymous';
                zImg.onload = () => {
                    if (viewer.generationID !== currentGen) return;
                    viewer.frameZdepthImages[idx] = zImg;
                    if (idx === viewer.currentFrame) {
                        viewer.zdepthImage = zImg;
                        if (viewer.renderer) viewer.renderer.loadDepthTexture(zImg);
                        // Force re-render if we are already displaying this frame
                        viewer.render();
                    }
                };
                zImg.onerror = (e) => console.warn("[Radiance] Failed to load zdepth image:", imgData.filename);
                zImg.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);
            });

            // Phase 5: Capture Instance ID
            if (message.instance_id && message.instance_id.length > 0) {
                viewer.instanceId = message.instance_id[0];
            }
        };
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
//                    CURVE EDITOR — DaVinci Resolve Style
// ═══════════════════════════════════════════════════════════════════════════════

class RadianceCurveEditor {
    /**
     * DaVinci Resolve–style curve editor with Fritsch–Carlson monotonic interpolation.
     *
     * Visual style matches Resolve's Curves – Custom panel:
     *   - All RGB histograms rendered simultaneously as soft fills
     *   - Clean curve line with round control points
     *   - Minimal grid, dark cinematic background
     *   - Per-channel gain multipliers (Edit intensity 0–200)
     *   - Soft Clip controls (Low, Low Soft, High, High Soft)
     *
     * LUT output: Float32Array(256×4 = 1024) RGBA for full HDR precision.
     * Master RGB curve evaluated first, then per-channel R/G/B on top.
     */
    constructor(width, height, theme, onChange) {
        this.width = width;
        this.height = height;
        this.theme = theme || { mono: 'monospace', textDim: '#888' };
        this.onChange = onChange;
        this.padding = { left: 2, bottom: 2, top: 2, right: 2 };

        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.tabIndex = 1;
        this.canvas.style.outline = 'none';
        this.canvas.style.cursor = 'crosshair';
        this.ctx = this.canvas.getContext('2d');

        this.histograms = { R: null, G: null, B: null, L: null };
        this.snapshots = null;

        // Curve data — endpoints pinned at x=0 and x=1
        this.curves = {
            'RGB': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'R': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'G': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'B': [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            'HueVsHue': [{ x: 0, y: 0.5 }, { x: 1, y: 0.5 }],
            'HueVsSat': [{ x: 0, y: 0.5 }, { x: 1, y: 0.5 }],
            'HueVsLuma': [{ x: 0, y: 0.5 }, { x: 1, y: 0.5 }]
        };

        this.activeChannel = 'RGB';
        this.hoverPoint = null;
        this.draggingPoint = null;
        this.selectedPoints = [];
        this.mousePos = { x: -1, y: -1 };

        this.channelColors = {
            'RGB': '#e0e0e0',
            'R': '#ff4444',
            'G': '#44cc44',
            'B': '#4488ff',
            'HueVsHue': '#ffaaaa',
            'HueVsSat': '#fff144',
            'HueVsLuma': '#44ffaa'
        };

        this.rangeY = 1.0;
        this.levels = { inBlack: 0, inWhite: 255 };

        // Per-channel gain multipliers (DaVinci "Edit" intensity, 0–200, default 100)
        this.channelGain = { Y: 100, R: 100, G: 100, B: 100 };

        // Soft Clip controls (DaVinci Resolve style)
        this.softClipEnabled = false;
        this.softClipChannels = { R: true, G: true, B: true };
        this.softClipParams = {
            low: 0, lowSoft: 0, high: 1.0, highSoft: 0
        };

        // Viewport state
        this.view = {
            zoomX: 1.0, zoomY: 1.0,
            offsetX: 0.0, offsetY: 0.0,
            minZoom: 1.0, maxZoom: 20.0,
            logX: false
        };

        this.setupEvents();
        this.draw();
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        this.draw();
    }

    // ─── Coordinate helpers ────────────────────────────────────
    get plotX() { return this.padding.left; }
    get plotY() { return this.padding.top; }
    get plotW() { return this.width - this.padding.left - this.padding.right; }
    get plotH() { return this.height - this.padding.top - this.padding.bottom; }

    normToCanvas(nx, ny) {
        let lx = nx;
        if (this.view.logX) {
            lx = (nx > 0) ? Math.log10(nx * 9 + 1) : 0;
        }
        const vx = (lx - this.view.offsetX) * this.view.zoomX;
        const vy = (ny - this.view.offsetY) * this.view.zoomY;
        return {
            cx: this.plotX + vx * this.plotW,
            cy: this.plotY + (1 - (vy / this.rangeY)) * this.plotH
        };
    }

    canvasToNorm(cx, cy) {
        const vx = (cx - this.plotX) / this.plotW;
        const vy = (1.0 - (cy - this.plotY) / this.plotH) * this.rangeY;
        let nx = this.view.offsetX + (vx / this.view.zoomX);
        if (this.view.logX) {
            nx = (Math.pow(10, nx) - 1) / 9;
        }
        return {
            x: nx,
            y: this.view.offsetY + (vy / this.view.zoomY)
        };
    }

    // ─── Histogram ─────────────────────────────────────────────
    updateHistogram(img) {
        if (!img) return;
        const scale = Math.min(1.0, 256 / Math.max(img.width, img.height));
        const w = Math.floor(img.width * scale);
        const h = Math.floor(img.height * scale);

        const temp = document.createElement('canvas');
        temp.width = w; temp.height = h;
        const tctx = temp.getContext('2d');
        tctx.drawImage(img, 0, 0, w, h);
        const data = tctx.getImageData(0, 0, w, h).data;

        const buckets = 256;
        const R = new Uint32Array(buckets), G = new Uint32Array(buckets);
        const B = new Uint32Array(buckets), L = new Uint32Array(buckets);

        for (let i = 0; i < data.length; i += 4) {
            R[data[i]]++;
            G[data[i + 1]]++;
            B[data[i + 2]]++;
            const luma = Math.min(255, Math.floor(
                data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722
            ));
            L[luma]++;
        }

        let max = 0;
        for (let i = 1; i < buckets - 1; i++) {
            max = Math.max(max, L[i], R[i], G[i], B[i]);
        }
        if (max === 0) max = 1;

        const norm = (arr) => {
            const r = new Float32Array(buckets);
            for (let i = 0; i < buckets; i++) r[i] = Math.min(1.0, arr[i] / max);
            return r;
        };
        this.histograms = { R: norm(R), G: norm(G), B: norm(B), L: norm(L) };
        this.draw();
    }

    // ─── Channel control ───────────────────────────────────────
    setActiveChannel(ch) {
        this.activeChannel = ch;
        this.draw();
        if (this._onChannelSwitch) this._onChannelSwitch(ch);
    }

    resetActiveChannel() {
        if (this.activeChannel.startsWith('Hue')) {
            this.curves[this.activeChannel] = [{ x: 0, y: 0.5 }, { x: 1, y: 0.5 }];
        } else {
            this.curves[this.activeChannel] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        }
        this.notifyChange();
        this.draw();
    }

    resetAll() {
        for (const ch of ['RGB', 'R', 'G', 'B']) {
            this.curves[ch] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
        }
        for (const ch of ['HueVsHue', 'HueVsSat', 'HueVsLuma']) {
            this.curves[ch] = [{ x: 0, y: 0.5 }, { x: 1, y: 0.5 }];
        }
        this.levels = { inBlack: 0, inWhite: 255 };
        this.channelGain = { Y: 100, R: 100, G: 100, B: 100 };
        this.softClipEnabled = false;
        this.softClipParams = { low: 0, lowSoft: 0, high: 1.0, highSoft: 0 };
        this.notifyChange();
        this.draw();
    }

    resetAllChannels() { this.resetAll(); }

    applyPreset(preset) {
        for (const ch of ['RGB', 'R', 'G', 'B']) {
            if (preset[ch]) {
                this.curves[ch] = preset[ch].map(p => ({ x: p.x, y: p.y }));
            }
        }
        this.notifyChange();
        this.draw();
    }

    applyIndustryPreset(name) {
        const presets = {
            'Punchy': [
                { x: 0, y: 0 }, { x: 0.15, y: 0.08 },
                { x: 0.50, y: 0.52 }, { x: 0.85, y: 0.95 }, { x: 1, y: 1 }
            ],
            'High Contrast': [
                { x: 0, y: 0 }, { x: 0.20, y: 0.05 },
                { x: 0.50, y: 0.50 }, { x: 0.80, y: 0.95 }, { x: 1, y: 1 }
            ],
            'Flat / Log': [
                { x: 0, y: 0.12 }, { x: 0.25, y: 0.30 },
                { x: 0.50, y: 0.50 }, { x: 0.75, y: 0.70 }, { x: 1, y: 0.88 }
            ],
            'Shadow Lift': [
                { x: 0, y: 0.08 }, { x: 0.15, y: 0.20 },
                { x: 0.50, y: 0.52 }, { x: 1, y: 1 }
            ],
            'S-Curve': [
                { x: 0, y: 0 }, { x: 0.25, y: 0.17 },
                { x: 0.50, y: 0.50 }, { x: 0.75, y: 0.83 }, { x: 1, y: 1 }
            ],
            'Film Print': [
                { x: 0, y: 0.02 }, { x: 0.10, y: 0.06 },
                { x: 0.30, y: 0.28 }, { x: 0.60, y: 0.65 },
                { x: 0.85, y: 0.90 }, { x: 1, y: 0.96 }
            ],
            'Bleach Bypass': [
                { x: 0, y: 0 }, { x: 0.15, y: 0.03 },
                { x: 0.40, y: 0.45 }, { x: 0.70, y: 0.82 }, { x: 1, y: 1 }
            ],
            'Cross Process': {
                'R': [{ x: 0, y: 0.05 }, { x: 0.3, y: 0.2 }, { x: 0.7, y: 0.85 }, { x: 1, y: 0.95 }],
                'G': [{ x: 0, y: 0 }, { x: 0.35, y: 0.4 }, { x: 0.65, y: 0.6 }, { x: 1, y: 1 }],
                'B': [{ x: 0, y: 0.1 }, { x: 0.3, y: 0.35 }, { x: 0.7, y: 0.55 }, { x: 1, y: 0.85 }]
            }
        };

        const preset = presets[name];
        if (!preset) return;

        if (Array.isArray(preset)) {
            this.curves[this.activeChannel] = preset.map(p => ({ x: p.x, y: p.y }));
        } else {
            for (const ch of ['RGB', 'R', 'G', 'B']) {
                if (preset[ch]) {
                    this.curves[ch] = preset[ch].map(p => ({ x: p.x, y: p.y }));
                }
            }
        }
        this.notifyChange();
        this.draw();
    }

    copyRGBToAll() {
        const src = this.curves['RGB'].map(p => ({ x: p.x, y: p.y }));
        this.curves['R'] = src.map(p => ({ ...p }));
        this.curves['G'] = src.map(p => ({ ...p }));
        this.curves['B'] = src.map(p => ({ ...p }));
        this.notifyChange();
        this.draw();
    }

    saveSnapshot() {
        this.snapshots = JSON.parse(JSON.stringify(this.curves));
        this.draw();
    }

    clearSnapshot() {
        this.snapshots = null;
        this.draw();
    }

    // ─── Events ────────────────────────────────────────────────
    setupEvents() {
        const cvs = this.canvas;
        const GRAB_RADIUS_PX = 8;

        const hitTest = (e) => {
            const rect = cvs.getBoundingClientRect();
            const px = (e.clientX - rect.left) * (cvs.width / rect.width);
            const py = (e.clientY - rect.top) * (cvs.height / rect.height);
            const norm = this.canvasToNorm(px, py);

            const pts = this.curves[this.activeChannel];
            let best = null, bestDist = GRAB_RADIUS_PX * GRAB_RADIUS_PX;

            for (const p of pts) {
                const c = this.normToCanvas(p.x, p.y);
                const d = (px - c.cx) ** 2 + (py - c.cy) ** 2;
                if (d < bestDist) { bestDist = d; best = p; }
            }
            return { norm, best, px, py };
        };

        cvs.onmousedown = (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            const { norm, best, px, py } = hitTest(e);

            if (best) {
                if (!this.selectedPoints.includes(best)) {
                    this.selectedPoints = [best];
                }
                this.draggingPoint = best;
                this.lastDragNorm = { ...norm };
                cvs.style.cursor = 'grabbing';
            } else if (norm.x > -0.05 && norm.x < 1.05) {
                this.selectedPoints = [];
                const pts = this.curves[this.activeChannel];
                const yMax = (this.activeChannel.startsWith('Hue')) ? 1.0 : this.rangeY;
                const newPt = {
                    x: Math.max(0, Math.min(1.0, norm.x)),
                    y: Math.max(0, Math.min(yMax, norm.y))
                };
                pts.push(newPt);
                pts.sort((a, b) => a.x - b.x);
                this.draggingPoint = newPt;
                this.selectedPoints = [newPt];
                this.lastDragNorm = { ...norm };
                cvs.style.cursor = 'grabbing';
                this.notifyChange();
            } else {
                this.selectedPoints = [];
            }
            this.draw();
        };

        cvs.onmousemove = (e) => {
            e.stopPropagation();
            if (this.draggingPoint && e.buttons !== 1) {
                this.draggingPoint = null;
                cvs.style.cursor = 'crosshair';
                this.draw();
                return;
            }

            const rect = cvs.getBoundingClientRect();
            const px = (e.clientX - rect.left) * (cvs.width / rect.width);
            const py = (e.clientY - rect.top) * (cvs.height / rect.height);
            const norm = this.canvasToNorm(px, py);
            this.mousePos = norm;

            if (this.draggingPoint) {
                const dx = norm.x - this.lastDragNorm.x;
                const dy = norm.y - this.lastDragNorm.y;

                this.selectedPoints.forEach(p => {
                    const pts = this.curves[this.activeChannel];
                    const idx = pts.indexOf(p);
                    const yMax = (this.activeChannel.startsWith('Hue')) ? 1.0 : this.rangeY;

                    if (idx === 0) {
                        const maxX = pts.length > 1 ? (pts[1].x - 0.001) : 1.0;
                        p.x = Math.max(0, Math.min(maxX, p.x + dx));
                        p.y = Math.max(0, Math.min(yMax, p.y + dy));
                    } else if (idx === pts.length - 1) {
                        const minX = pts.length > 1 ? (pts[idx - 1].x + 0.001) : 0.0;
                        p.x = Math.max(minX, Math.min(1.0, p.x + dx));
                        p.y = Math.max(0, Math.min(yMax, p.y + dy));
                    } else {
                        const minX = pts[idx - 1].x + 0.001;
                        const maxX = pts[idx + 1].x - 0.001;
                        p.x = Math.max(minX, Math.min(maxX, p.x + dx));
                        p.y = Math.max(0, Math.min(yMax, p.y + dy));
                    }
                });

                this.lastDragNorm = { ...norm };
                this.notifyChange();
            } else {
                const { best } = hitTest(e);
                const newHover = best || null;
                if (newHover !== this.hoverPoint) {
                    this.hoverPoint = newHover;
                    cvs.style.cursor = newHover ? 'grab' : 'crosshair';
                }
            }
            this.draw();
        };

        const onMouseUp = () => {
            if (this.draggingPoint) {
                this.draggingPoint = null;
                cvs.style.cursor = 'crosshair';
                this.draw();
            }
        };
        window.addEventListener('mouseup', onMouseUp);

        // Double-click: remove interior point
        cvs.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const { best } = hitTest(e);
            if (!best) return;
            const pts = this.curves[this.activeChannel];
            const idx = pts.indexOf(best);
            if (idx > 0 && idx < pts.length - 1) {
                pts.splice(idx, 1);
                this.hoverPoint = null;
                this.notifyChange();
                this.draw();
            }
        };

        // Right-click: also remove
        cvs.oncontextmenu = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const { best } = hitTest(e);
            if (!best) return;
            const pts = this.curves[this.activeChannel];
            const idx = pts.indexOf(best);
            if (idx > 0 && idx < pts.length - 1) {
                pts.splice(idx, 1);
                this.hoverPoint = null;
                this.notifyChange();
                this.draw();
            }
        };

        cvs.onmouseleave = () => {
            this.mousePos = { x: -1, y: -1 };
            if (!this.draggingPoint) this.draw();
        };

        // Zoom
        cvs.onwheel = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const rect = cvs.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) * (cvs.width / rect.width);
            const mouseY = (e.clientY - rect.top) * (cvs.height / rect.height);

            const before = this.canvasToNorm(mouseX, mouseY);
            const delta = -Math.sign(e.deltaY) * 0.15;
            const factor = 1 + delta;

            this.view.zoomX = Math.max(this.view.minZoom, Math.min(this.view.maxZoom, this.view.zoomX * factor));
            this.view.zoomY = Math.max(this.view.minZoom, Math.min(this.view.maxZoom, this.view.zoomY * factor));

            const after = this.canvasToNorm(mouseX, mouseY);
            this.view.offsetX += (before.x - after.x);
            this.view.offsetY += (before.y - after.y);
            this._clampView();
            this.draw();
        };

        // Pan (middle mouse or Alt+drag)
        let isPanning = false;
        let lastPanPos = { x: 0, y: 0 };

        cvs.addEventListener('mousedown', (e) => {
            if (e.button === 1 || (e.button === 0 && e.altKey)) {
                e.preventDefault();
                e.stopPropagation();
                isPanning = true;
                lastPanPos = { x: e.clientX, y: e.clientY };
                cvs.style.cursor = 'move';
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (isPanning) {
                const dx = e.clientX - lastPanPos.x;
                const dy = e.clientY - lastPanPos.y;
                const normDx = (dx / this.plotW) / this.view.zoomX;
                const normDy = (dy / this.plotH) * (this.rangeY / this.view.zoomY);
                this.view.offsetX -= normDx;
                this.view.offsetY += normDy;
                lastPanPos = { x: e.clientX, y: e.clientY };
                this._clampView();
                this.draw();
            }
        });

        window.addEventListener('mouseup', () => {
            if (isPanning) {
                isPanning = false;
                cvs.style.cursor = 'crosshair';
            }
        });

        // Keyboard nudge
        cvs.addEventListener('keydown', (e) => {
            if (this.selectedPoints.length === 0) return;

            const step = e.shiftKey ? 10 / 255 : 1 / 255;
            let dx = 0, dy = 0;

            const key = e.key.toLowerCase();
            if (key === 'arrowup') dy = step;
            else if (key === 'arrowdown') dy = -step;
            else if (key === 'arrowleft') dx = -step;
            else if (key === 'arrowright') dx = step;
            else if (key === 'delete' || key === 'backspace') {
                const pts = this.curves[this.activeChannel];
                this.selectedPoints.forEach(p => {
                    const idx = pts.indexOf(p);
                    if (idx > 0 && idx < pts.length - 1) pts.splice(idx, 1);
                });
                this.selectedPoints = [];
                this.notifyChange();
                this.draw();
                e.preventDefault();
                return;
            } else return;

            e.preventDefault();

            this.selectedPoints.forEach(p => {
                const pts = this.curves[this.activeChannel];
                const idx = pts.indexOf(p);
                const yMax = (this.activeChannel.startsWith('Hue')) ? 1.0 : this.rangeY;

                if (idx === 0 || idx === pts.length - 1) {
                    p.y = Math.max(0, Math.min(yMax, p.y + dy));
                } else {
                    const minX = pts[idx - 1].x + 0.001;
                    const maxX = pts[idx + 1].x - 0.001;
                    p.x = Math.max(minX, Math.min(maxX, p.x + dx));
                    p.y = Math.max(0, Math.min(yMax, p.y + dy));
                }
            });

            this.notifyChange();
            this.draw();
        });
    }

    _clampView() {
        const margin = 0.5 / this.view.zoomX;
        this.view.offsetX = Math.max(-margin, Math.min(1.0 + margin - 1 / this.view.zoomX, this.view.offsetX));
        this.view.offsetY = Math.max(-margin * this.rangeY, Math.min(this.rangeY + margin * this.rangeY - this.rangeY / this.view.zoomY, this.view.offsetY));
    }

    resetView() {
        this.view.zoomX = 1.0;
        this.view.zoomY = 1.0;
        this.view.offsetX = 0.0;
        this.view.offsetY = 0.0;
        this.draw();
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return this.canvasToNorm(e.clientX - rect.left, e.clientY - rect.top);
    }

    findPoint(pos) {
        const pts = this.curves[this.activeChannel];
        const threshX = 0.04, threshY = 0.04;
        return pts.find(p =>
            Math.abs(p.x - pos.x) < threshX && Math.abs(p.y - pos.y) < threshY
        );
    }

    // ─── Monotonic Cubic Hermite (Fritsch–Carlson) ─────────────
    evaluateCurve(points) {
        const n = points.length;
        if (n === 0) return new Float32Array(256).fill(0);
        if (n === 1) return new Float32Array(256).fill(
            Math.max(0, Math.min(1, points[0].y))
        );

        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const lut = new Float32Array(256);

        // Step 1: Secant slopes
        const delta = new Float64Array(n - 1);
        const h = new Float64Array(n - 1);
        for (let i = 0; i < n - 1; i++) {
            h[i] = xs[i + 1] - xs[i];
            delta[i] = (h[i] > 1e-10) ? (ys[i + 1] - ys[i]) / h[i] : 0;
        }

        // Step 2: Initial tangents
        const m = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            if (i === 0) {
                m[i] = delta[0] || 0;
            } else if (i === n - 1) {
                m[i] = delta[n - 2] || 0;
            } else {
                if (delta[i - 1] * delta[i] <= 0) {
                    m[i] = 0;
                } else {
                    const w1 = 2 * h[i] + h[i - 1];
                    const w2 = h[i] + 2 * h[i - 1];
                    m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                }
            }
        }

        // Step 3: Fritsch-Carlson monotonicity preservation
        for (let i = 0; i < n - 1; i++) {
            if (Math.abs(delta[i]) < 1e-10) {
                m[i] = 0;
                m[i + 1] = 0;
            } else {
                const alpha = m[i] / delta[i];
                const beta = m[i + 1] / delta[i];
                const tau = alpha * alpha + beta * beta;
                if (tau > 9) {
                    const s = 3.0 / Math.sqrt(tau);
                    m[i] = s * alpha * delta[i];
                    m[i + 1] = s * beta * delta[i];
                }
            }
        }

        // Step 4: Evaluate cubic Hermite
        for (let i = 0; i < 256; i++) {
            const t = i / 255;

            if (t <= xs[0]) { lut[i] = ys[0]; continue; }
            if (t >= xs[n - 1]) { lut[i] = ys[n - 1]; continue; }

            let lo = 0, hi = n - 2;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (xs[mid + 1] < t) lo = mid + 1; else hi = mid;
            }
            const k = lo;

            const hk = h[k];
            if (hk < 1e-10) { lut[i] = ys[k]; continue; }

            const s = (t - xs[k]) / hk;
            const s2 = s * s;
            const s3 = s2 * s;

            const h00 = 2 * s3 - 3 * s2 + 1;
            const h10 = s3 - 2 * s2 + s;
            const h01 = -2 * s3 + 3 * s2;
            const h11 = s3 - s2;

            lut[i] = h00 * ys[k] + h10 * hk * m[k] +
                h01 * ys[k + 1] + h11 * hk * m[k + 1];
        }

        // Hue curve clamping
        if (points === this.curves['HueVsHue'] || points === this.curves['HueVsSat'] || points === this.curves['HueVsLuma']) {
            for (let i = 0; i < 256; i++) {
                lut[i] = Math.max(0, Math.min(1, lut[i]));
            }
        }

        return lut;
    }

    // Legacy compat
    solveBezierSpline(points) { return this.evaluateCurve(points); }
    solveMonotonicSpline(points) { return this.evaluateCurve(points); }

    // ─── Drawing — DaVinci Resolve Aesthetic ─────────────────────
    draw() {
        const ctx = this.ctx;
        const w = this.width, h = this.height;
        const pX = this.plotX, pY = this.plotY, pW = this.plotW, pH = this.plotH;

        // 1. Background — deep DaVinci dark
        ctx.fillStyle = '#1a1a22';
        ctx.fillRect(0, 0, w, h);

        ctx.save();
        ctx.beginPath();
        ctx.rect(pX, pY, pW, pH);
        ctx.clip();

        // 2. Subtle grid — very faint, DaVinci style (only major grid visible)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 1; i <= 3; i++) {
            const { cx } = this.normToCanvas(i / 4, 0);
            ctx.moveTo(cx, pY); ctx.lineTo(cx, pY + pH);
            const { cy } = this.normToCanvas(0, i / 4);
            ctx.moveTo(pX, cy); ctx.lineTo(pX + pW, cy);
        }
        ctx.stroke();

        // Identity diagonal — very subtle
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (this.activeChannel.startsWith('Hue')) {
            const h0 = this.normToCanvas(-1, 0.5), h1 = this.normToCanvas(2, 0.5);
            ctx.moveTo(h0.cx, h0.cy); ctx.lineTo(h1.cx, h1.cy);
        } else {
            const d0 = this.normToCanvas(0, 0), d1 = this.normToCanvas(1, 1);
            ctx.moveTo(d0.cx, d0.cy); ctx.lineTo(d1.cx, d1.cy);
        }
        ctx.stroke();

        // Rainbow background for Hue curves
        if (this.activeChannel.startsWith('Hue')) {
            const r0 = this.normToCanvas(0, 0).cx;
            const r1 = this.normToCanvas(1, 0).cx;
            const stops = [
                { pos: 0, col: '#ff0000' }, { pos: 1 / 6, col: '#ffff00' },
                { pos: 2 / 6, col: '#00ff00' }, { pos: 3 / 6, col: '#00ffff' },
                { pos: 4 / 6, col: '#0000ff' }, { pos: 5 / 6, col: '#ff00ff' },
                { pos: 1, col: '#ff0000' }
            ];
            ctx.globalAlpha = 0.08;
            const hGrad = ctx.createLinearGradient(r0, 0, r1, 0);
            stops.forEach(s => { if (s.pos >= 0 && s.pos <= 1) hGrad.addColorStop(s.pos, s.col); });
            ctx.fillStyle = hGrad;
            ctx.fillRect(pX, pY, pW, pH);
            ctx.globalAlpha = 1.0;
        }

        // 3. Histograms — all RGB channels shown simultaneously (DaVinci Resolve style)
        if (this.histograms.R) {
            const drawHistFill = (hist, color, alpha) => {
                if (!hist) return;
                ctx.globalAlpha = alpha;
                ctx.beginPath();
                const b0 = this.normToCanvas(0, 0);
                ctx.moveTo(b0.cx, b0.cy);
                for (let i = 0; i < 256; i++) {
                    const val = Math.pow(hist[i], 0.55) * 0.7;
                    const { cx, cy } = this.normToCanvas(i / 255, val);
                    ctx.lineTo(cx, cy);
                }
                const bEnd = this.normToCanvas(1, 0);
                ctx.lineTo(bEnd.cx, bEnd.cy);
                ctx.closePath();
                ctx.fillStyle = color;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            };

            // Always show all 3 RGB histograms simultaneously like DaVinci
            drawHistFill(this.histograms.R, '#cc2222', 0.22);
            drawHistFill(this.histograms.G, '#22aa22', 0.18);
            drawHistFill(this.histograms.B, '#2244cc', 0.22);
        }

        // 4. Ghost curves (inactive channels at low opacity)
        let ghostChannels = [];
        if (this.activeChannel === 'RGB') ghostChannels = ['R', 'G', 'B'];
        else if (['R', 'G', 'B'].includes(this.activeChannel)) ghostChannels = ['RGB'];

        ghostChannels.forEach(ch => {
            const pts = this.curves[ch];
            if (pts.length < 2) return;
            const lut = this.evaluateCurve(pts);
            ctx.globalAlpha = 0.12;
            ctx.strokeStyle = this.channelColors[ch];
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
                if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        });

        // 5. Snapshot ghost
        if (this.snapshots && this.snapshots[this.activeChannel]) {
            const sLut = this.evaluateCurve(this.snapshots[this.activeChannel]);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const { cx, cy } = this.normToCanvas(i / 255, sLut[i]);
                if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // 6. Active curve — DaVinci Resolve style (clean, smooth, with glow)
        const pts = this.curves[this.activeChannel];
        const lut = this.evaluateCurve(pts);
        const color = this.channelColors[this.activeChannel];

        // Glow layer
        ctx.shadowBlur = 8;
        ctx.shadowColor = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        for (let i = 0; i < 256; i++) {
            const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
            if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Main curve line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.0;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        for (let i = 0; i < 256; i++) {
            const { cx, cy } = this.normToCanvas(i / 255, lut[i]);
            if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        ctx.restore(); // Unclip

        // 7. Control points — DaVinci style (clean white circles)
        ctx.save();
        pts.forEach((p, idx) => {
            const { cx, cy } = this.normToCanvas(p.x, p.y);
            const isHover = (p === this.hoverPoint);
            const isDrag = (p === this.draggingPoint);
            const isSelected = this.selectedPoints.includes(p);
            const isEndpoint = (idx === 0 || idx === pts.length - 1);

            const r = (isHover || isDrag) ? 5 : isEndpoint ? 3.5 : 4.5;

            // Outer glow for active points
            if (isHover || isDrag || isSelected) {
                ctx.shadowBlur = 10;
                ctx.shadowColor = color;
            }

            // White filled circle with dark outline — DaVinci style
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = 'rgba(0,0,0,0.5)';
            ctx.lineWidth = 1.2;
            ctx.stroke();

            ctx.shadowBlur = 0;
        });
        ctx.restore();

        // 8. Crosshair on hover
        if (this.mousePos.x >= 0 && this.mousePos.x <= 1 &&
            this.mousePos.y >= 0 && this.mousePos.y <= this.rangeY) {
            const mPos = this.normToCanvas(this.mousePos.x, this.mousePos.y);

            ctx.save();
            ctx.beginPath();
            ctx.rect(pX, pY, pW, pH);
            ctx.clip();

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(mPos.cx, pY); ctx.lineTo(mPos.cx, pY + pH);
            ctx.moveTo(pX, mPos.cy); ctx.lineTo(pX + pW, mPos.cy);
            ctx.stroke();
            ctx.restore();
        }

        // 9. Dragging / hover readout
        if (this.draggingPoint || this.hoverPoint) {
            const target = this.draggingPoint || this.hoverPoint;
            const inVal = Math.round(target.x * 255);
            const outVal = Math.round(target.y * 255);
            const { cx, cy } = this.normToCanvas(target.x, target.y);

            const text = `${inVal} → ${outVal}`;
            ctx.font = `bold 10px ${this.theme.mono}`;
            const tw = ctx.measureText(text).width + 12;
            const tx = Math.min(cx + 14, pX + pW - tw - 4);
            const ty = Math.max(cy - 10, pY + 16);

            // Dark pill background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            const rx = tx, ry = ty - 12, rw = tw, rh = 18, rr = 4;
            ctx.beginPath();
            ctx.moveTo(rx + rr, ry);
            ctx.arcTo(rx + rw, ry, rx + rw, ry + rh, rr);
            ctx.arcTo(rx + rw, ry + rh, rx, ry + rh, rr);
            ctx.arcTo(rx, ry + rh, rx, ry, rr);
            ctx.arcTo(rx, ry, rx + rw, ry, rr);
            ctx.closePath();
            ctx.fill();

            ctx.fillStyle = '#ccc';
            ctx.textAlign = 'left';
            ctx.fillText(text, tx + 6, ty + 1);
        }
    }

    // ─── LUT Generation ─────────────────────────────────────────
    notifyChange() {
        if (!this.onChange) return;

        const master = this.evaluateCurve(this.curves['RGB']);
        const rCurve = this.evaluateCurve(this.curves['R']);
        const gCurve = this.evaluateCurve(this.curves['G']);
        const bCurve = this.evaluateCurve(this.curves['B']);

        const hvhCurve = this.evaluateCurve(this.curves['HueVsHue']);
        const hvsCurve = this.evaluateCurve(this.curves['HueVsSat']);
        const hvlCurve = this.evaluateCurve(this.curves['HueVsLuma']);

        // Levels remapping — operates on a [0, 255] integer grid (8-bit input domain).
        // This is by design: the LUT is 256 entries wide so the levels walk matches
        // it 1-to-1 for SDR. For HDR images, values above 1.0 bypass the levels
        // remap entirely and fall through to the ratio-based HDR extrapolation in
        // the shader (color * topVal). The inBlack/inWhite sliders are therefore
        // SDR-range controls only — they do not affect super-white HDR highlights.
        const inBlack = this.levels.inBlack / 255;
        const inWhite = this.levels.inWhite / 255;
        const inRange = Math.max(0.001, inWhite - inBlack);

        const lookup = (curve, u) => {
            const f = u * 255;
            const k = Math.floor(f);
            const frac = f - k;
            if (k >= 255) return curve[255];
            if (k < 0) return curve[0];
            return curve[k] * (1 - frac) + curve[k + 1] * frac;
        };

        // Channel gain multipliers (Edit intensity, mapped from 0–200 → 0.0–2.0)
        const gY = this.channelGain.Y / 100;
        const gR = this.channelGain.R / 100;
        const gG = this.channelGain.G / 100;
        const gB = this.channelGain.B / 100;

        // Soft Clip parameters
        const sc = this.softClipEnabled ? this.softClipParams : null;

        const lut = new Float32Array(256 * 4);
        const secLut = new Float32Array(256 * 4);

        for (let i = 0; i < 256; i++) {
            const leveled = Math.max(0, Math.min(1, (i / 255 - inBlack) / inRange));

            // Master curve (Y channel gain)
            let mVal = lookup(master, leveled);
            // Apply master gain: lerp between identity and curve output
            mVal = leveled + (mVal - leveled) * gY;

            // Per-channel curves with individual gain
            let r = lookup(rCurve, mVal);
            let g = lookup(gCurve, mVal);
            let b = lookup(bCurve, mVal);

            // Apply per-channel gain (lerp between mVal and curve output)
            r = mVal + (r - mVal) * gR;
            g = mVal + (g - mVal) * gG;
            b = mVal + (b - mVal) * gB;

            // Apply Soft Clip if enabled
            if (sc) {
                const applyClip = (v, ch) => {
                    if (!this.softClipChannels[ch]) return v;
                    // Low clip
                    if (sc.lowSoft > 0 && v < sc.low + sc.lowSoft) {
                        const t = Math.max(0, (v - sc.low) / Math.max(0.001, sc.lowSoft));
                        v = sc.low + (v - sc.low) * t * t * (3 - 2 * t);
                        v = Math.max(sc.low, v);
                    } else {
                        v = Math.max(sc.low, v);
                    }
                    // High clip
                    if (sc.highSoft > 0 && v > sc.high - sc.highSoft) {
                        const t = Math.max(0, (sc.high - v) / Math.max(0.001, sc.highSoft));
                        v = sc.high - (sc.high - v) * t * t * (3 - 2 * t);
                        v = Math.min(sc.high, v);
                    } else {
                        v = Math.min(sc.high, v);
                    }
                    return v;
                };
                r = applyClip(r, 'R');
                g = applyClip(g, 'G');
                b = applyClip(b, 'B');
            }

            lut[i * 4 + 0] = r;
            lut[i * 4 + 1] = g;
            lut[i * 4 + 2] = b;
            lut[i * 4 + 3] = 1.0;

            // Secondary LUT (Hue curves)
            secLut[i * 4 + 0] = Math.max(0, Math.min(1, hvhCurve[i]));
            secLut[i * 4 + 1] = Math.max(0, Math.min(1, hvsCurve[i]));
            secLut[i * 4 + 2] = Math.max(0, Math.min(1, hvlCurve[i]));
            secLut[i * 4 + 3] = 1.0;
        }

        this.onChange(lut, secLut);
    }

    setLevels(inBlack, inWhite) {
        this.levels.inBlack = inBlack;
        this.levels.inWhite = inWhite;
        this.notifyChange();
        this.draw();
    }
}
