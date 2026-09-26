import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────────────────────────────────────
//  Apple Pro Palette Design System
// ─────────────────────────────────────────────────────────────────────────────
const COLORS = {
    BODY:   "#141419",   // Unified Apple Space-Gray Matte Body
    ACCENT: "#ffffff",   // Pure Apple Silver accent
};

// ─────────────────────────────────────────────────────────────────────────────
// LiteGraph shape shim — prevents crashes when shape is set as a string
// ─────────────────────────────────────────────────────────────────────────────
(function patchLiteGraphShape() {
    if (!window.LiteGraph) return;
    if (Object.getOwnPropertyDescriptor(LiteGraph, '_shape')) return;

    const _mapShape = (v) => {
        if (typeof v === 'number' && [1, 2, 4].includes(v)) return v;
        const s = String(v).toLowerCase();
        if (s === 'box')    return 1;
        if (s === 'round')  return 2;
        if (s === 'circle') return 2;
        if (s === 'card')   return 4;
        return typeof v === 'number' ? Math.max(1, v) : 2;
    };

    let _current = _mapShape(LiteGraph.NODE_DEFAULT_SHAPE);
    Object.defineProperty(LiteGraph, 'NODE_DEFAULT_SHAPE', {
        get()  { return _current; },
        set(v) { _current = _mapShape(v); },
        configurable: true,
        enumerable:   true,
    });
})();

// ─────────────────────────────────────────────────────────────────────────────
// Apple Matte Metallic Header Finishes
// ─────────────────────────────────────────────────────────────────────────────
const CATEGORY_COLORS = {
    "CustomTool":{ header: "#0e3452" },   // Matte Ocean Blue / Neon Blue
    "HDR":       { header: "#3c2020" },   // Matte Rose Gold
    "Color":     { header: "#2e3a3a" },   // Matte Alpine Green / Teal
    "Film":      { header: "#221834" },   // Matte Deep Violet
    "VFX":       { header: "#122c1e" },   // Matte Forest Green
    "Gen":       { header: "#182238" },   // Matte Midnight Blue (alias for Generate)
    "Generate":  { header: "#182238" },   // Matte Midnight Blue
    "IO":        { header: "#3e3e4a" },   // Matte Silver / Platinum
    "Pipeline":  { header: "#282830" },   // Matte Space Gray
    "Training":  { header: "#3a1515" },   // Matte Dark Cherry
    "Image":     { header: "#1c223c" },   // Matte Indigo Blue
    "Upscale":   { header: "#1c223c" },   // alias for Image
    "AI":        { header: "#2a153c" },   // Matte Lavender Violet
    "Analysis":  { header: "#202026" },   // Matte Charcoal
    "Monitor":   { header: "#202026" },   // alias for Analysis
    "Utilities": { header: "#202026" },   // alias for Analysis
    "Video":     { header: "#3e3e4a" },   // Matte Silver / Platinum
    "Default":   { header: "#282830" },   // Matte Space Gray
};

/**
 * Resolve the discipline section from a ComfyUI category string, mapping
 * standard third-party categories to coordinated dark headers.
 */
function resolvePalette(categoryStr, cls) {
    if (!categoryStr) return cls ? paletteFromClass(cls) : CATEGORY_COLORS["Default"];
    
    const cat = categoryStr.toLowerCase();
    
    // 1. Resolve Radiance specific namespaces
    if (cat.includes("radiance")) {
        const m = categoryStr.match(/Radiance\/(?:◎\s*)?([^/]+)/i);
        if (m) {
            const key = m[1].trim();
            if (CATEGORY_COLORS[key]) return CATEGORY_COLORS[key];
        }
    }
    
    // 2. Resolve & Skin standard ComfyUI / third-party categories cohesively
    if (cat.includes("loader") || cat.includes("io") || cat.includes("load") || cat.includes("save")) {
        return CATEGORY_COLORS["IO"];
    }
    if (cat.includes("sampling") || cat.includes("sampler") || cat.includes("generate") || cat.includes("model")) {
        return CATEGORY_COLORS["Generate"];
    }
    if (cat.includes("conditioning") || cat.includes("clip") || cat.includes("prompt") || cat.includes("embeddings")) {
        return CATEGORY_COLORS["HDR"];
    }
    if (cat.includes("latent") || cat.includes("noise") || cat.includes("mask") || cat.includes("composite")) {
        return CATEGORY_COLORS["VFX"];
    }
    if (cat.includes("image") || cat.includes("upscale") || cat.includes("postprocess") || cat.includes("transform")) {
        return CATEGORY_COLORS["Image"];
    }
    if (cat.includes("color") || cat.includes("lut") || cat.includes("grading") || cat.includes("ocio") || cat.includes("style")) {
        return CATEGORY_COLORS["Color"];
    }
    if (cat.includes("film") || cat.includes("effect") || cat.includes("filter") || cat.includes("grain")) {
        return CATEGORY_COLORS["Film"];
    }
    if (cat.includes("workflow") || cat.includes("pipeline") || cat.includes("utils") || cat.includes("utility") || cat.includes("tool")) {
        return CATEGORY_COLORS["Pipeline"];
    }
    
    // Default premium charcoal header for everything else
    return CATEGORY_COLORS["Default"];
}

/**
 * Heuristic palette from class name — used while node.category hasn't
 * populated yet (i.e. during the nodeCreated callback).
 */
function paletteFromClass(cls) {
    if (!cls) return CATEGORY_COLORS["Default"];
    if (cls.includes("HDR") || cls.includes("Turbo") || cls.includes("Encoder"))
        return CATEGORY_COLORS["HDR"];
    if (cls.includes("Color") || cls.includes("Grade") || cls.includes("Curves") ||
        cls.includes("ACES")  || cls.includes("WhiteBalance") || cls.includes("OCIO") ||
        cls.includes("ColorSpace") || cls.includes("Scopes") || cls.includes("CDL"))
        return CATEGORY_COLORS["Color"];
    if (cls.includes("Film")  || cls.includes("Grain") || cls.includes("Optic") ||
        cls.includes("Aesthetic") || cls.includes("Camera"))
        return CATEGORY_COLORS["Film"];
    if (cls.includes("Depth") || cls.includes("Overlay") || cls.includes("Composite") ||
        cls.includes("Temporal") || cls.includes("Denoise") || cls.includes("Motion") ||
        cls.includes("Multipass") || cls.includes("Mask"))
        return CATEGORY_COLORS["VFX"];
    if (cls.includes("Loader") || cls.includes("Sampler") || cls.includes("Prompt") ||
        cls.includes("Lora")   || cls.includes("LoRA")    || cls.includes("VAE")    ||
        cls.includes("ControlNet") || cls.includes("Regional") || cls.includes("Studio"))
        return CATEGORY_COLORS["Generate"];
    if (cls.includes("IO")    || cls.includes("EXR")   || cls.includes("Video") ||
        cls.includes("NDI")   || cls.includes("RenderQueue"))
        return CATEGORY_COLORS["IO"];
    if (cls.includes("Nuke")  || cls.includes("Resolve") || cls.includes("Workspace") ||
        cls.includes("Layout") || cls.includes("Queue")  || cls.includes("Mastering") ||
        cls.includes("Metadata"))
        return CATEGORY_COLORS["Pipeline"];
    if (cls.includes("Train") || cls.includes("SDRDeg") || cls.includes("TurboTrain"))
        return CATEGORY_COLORS["Training"];
    if (cls.includes("Image") || cls.includes("Resolution") || cls.includes("Upscale") ||
        cls.includes("Panorama"))
        return CATEGORY_COLORS["Image"];
    if (cls.includes("DNA")   || cls.includes("Engine") || cls.includes("Text") ||
        cls.includes("Vitals") || cls.includes("QC"))
        return CATEGORY_COLORS["Utilities"];
    return CATEGORY_COLORS["Default"];
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension registration
// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "FXTD.Radiance.Style",

    async init() {
        // Apply beautiful Apple Pro dark space-gray styling to window.LiteGraph
        if (window.LiteGraph) {
            LiteGraph.NODE_DEFAULT_BGCOLOR = COLORS.BODY;      // Apple space-gray body
            LiteGraph.NODE_DEFAULT_COLOR   = "#282830";        // Apple default space-gray header
            LiteGraph.NODE_DEFAULT_SHAPE   = 2;                // Perfect rounded corners

            // Sophisticated Apple-style text & border glows
            LiteGraph.NODE_TITLE_COLOR = "#ffffff";            // Pure clean white headers
            LiteGraph.NODE_TEXT_COLOR  = "#a0a0aa";            // Sharp silver-gray labels
            LiteGraph.NODE_SELECTED_BORDER_COLOR = "#ffffff";  // High-contrast clean white border glow

            // Premium Apple Pro Widget colors on canvas
            LiteGraph.WIDGET_BGCOLOR = "#121217";              // Apple Dark Widget background
            LiteGraph.WIDGET_OUTLINE_COLOR = "#2c2c35";        // Subtle silver widget outline
            LiteGraph.WIDGET_TEXT_COLOR = "#e0e0e5";           // Sharp light-gray text
            LiteGraph.WIDGET_SECONDARY_TEXT_COLOR = "#8e8e93"; // Apple System Gray values
        }

        // Apply smooth curved spline noodles (DaVinci/Apple style)
        if (window.LGraphCanvas) {
            LGraphCanvas.link_type = 2; // SPLINE LINK
        }
    },

    async setup() {
        // ── Fix: VHS nodes crash on load when force_size is null ─────────────
        if (window.LiteGraph && window.LGraphNode) {
            const origConfigure = window.LGraphNode.prototype.configure;
            window.LGraphNode.prototype.configure = function (info) {
                if (this.type && this.type.includes("VHS_") &&
                    info && info.widgets_values &&
                    typeof info.widgets_values === 'object' &&
                    !Array.isArray(info.widgets_values) &&
                    info.widgets_values.force_size === null) {
                    info.widgets_values.force_size = "Disabled";
                }
                return origConfigure.apply(this, arguments);
            };
        }

        // ── Inject Premium Apple Frosted Glass UI Styles (macOS/iPadOS style) ─
        const style = document.createElement("style");
        style.innerHTML = `
            /* Main menus and floating panels get high-end macOS frosted glass blur */
            .comfy-menu, 
            .comfy-panel, 
            .comfy-panel-header,
            .radiance-glass-dock,
            #comfy-quick-nodes-div {
                background: rgba(20, 20, 25, 0.72) !important;
                backdrop-filter: blur(20px) saturate(190%) !important;
                -webkit-backdrop-filter: blur(20px) saturate(190%) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 12px !important;
                box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45) !important;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            }

            /* Apple style buttons inside UI panels */
            .comfy-menu button, 
            .comfy-panel button {
                background: rgba(255, 255, 255, 0.04) !important;
                color: #ffffff !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 6px !important;
                font-size: 11px !important;
                font-weight: 500 !important;
                text-shadow: none !important;
                transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }
            .comfy-menu button:hover, 
            .comfy-panel button:hover {
                background: rgba(255, 255, 255, 0.12) !important;
                border-color: rgba(255, 255, 255, 0.22) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            }
            .comfy-menu button:active, 
            .comfy-panel button:active {
                transform: scale(0.97) !important;
                background: rgba(255, 255, 255, 0.06) !important;
            }

            /* Universal Apple Pro range slider inputs */
            input[type="range"] {
                -webkit-appearance: none !important;
                appearance: none !important;
                background: transparent !important;
                cursor: pointer !important;
                height: 16px !important;
                width: 100% !important;
            }
            /* Track */
            input[type="range"]::-webkit-slider-runnable-track {
                background: rgba(255, 255, 255, 0.12) !important;
                border-radius: 4px !important;
                height: 4px !important;
                border: none !important;
            }
            input[type="range"]::-moz-range-track {
                background: rgba(255, 255, 255, 0.12) !important;
                border-radius: 4px !important;
                height: 4px !important;
                border: none !important;
            }
            /* Thumb */
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none !important;
                appearance: none !important;
                background: #ffffff !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                border-radius: 50% !important;
                height: 14px !important;
                width: 14px !important;
                margin-top: -5px !important; /* Center thumb vertically */
                box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
                transition: transform 0.15s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            }
            input[type="range"]::-moz-range-thumb {
                background: #ffffff !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
                border-radius: 50% !important;
                height: 14px !important;
                width: 14px !important;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
                transition: transform 0.15s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            }
            /* Hover interactions */
            input[type="range"]::-webkit-slider-thumb:hover {
                transform: scale(1.22) !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
            }
            input[type="range"]::-moz-range-thumb:hover {
                transform: scale(1.22) !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
            }
        `;
        document.head.appendChild(style);
    },

    async nodeCreated(node) {
        const cls = node.comfyClass || "";

        // ── Skin ALL nodes globally to adopt the Apple Pro finish! ───────────
        const palette = node.category
            ? resolvePalette(node.category, cls)
            : paletteFromClass(cls);

        node.color   = palette.header;  // Coordinated Matte Metallic Header
        node.bgcolor = COLORS.BODY;     // Space-Gray body — completely unified

        if (node.graph) node.setDirtyCanvas(true, true);
    },
});

console.log(" [Radiance Style] Premium Apple Pro Node Theme loaded successfully!");
