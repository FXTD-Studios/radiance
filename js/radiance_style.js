import { app } from "../../scripts/app.js";
import { BODY, DISCIPLINES, disciplineForCategory, OLD_THEME_HEADERS, OLD_THEME_BODIES } from "./radiance_palette.js";

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
// Extension registration
// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "FXTD.Radiance.Style",

    async init() {
        // Apply beautiful Apple Pro dark space-gray styling to window.LiteGraph
        if (window.LiteGraph) {
            // Node colours are not set here: Radiance colours only its own nodes
            // (beforeRegisterNodeDef below); ComfyUI's and other packs' keep theirs.
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

    // Radiance nodes only, from their category (radiance_palette.js). The
    // colour is a default, not data: it is left out of saved workflows, and an
    // old theme colour a workflow saved is replaced by the current one.
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const d = DISCIPLINES[disciplineForCategory(nodeData?.category)];
        if (!d) return;
        const header = d.header;
        const oldHeaders = new Set([...OLD_THEME_HEADERS, ...Object.values(DISCIPLINES).map((x) => x.header)]);
        const oldBodies = new Set(OLD_THEME_BODIES);
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            if (!this.color) this.color = header;
            if (!this.bgcolor) this.bgcolor = BODY;
            return r;
        };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = configure?.apply(this, arguments);
            if (!info?.color || oldHeaders.has(String(info.color).toLowerCase())) this.color = header;
            if (!info?.bgcolor || oldBodies.has(String(info.bgcolor).toLowerCase())) this.bgcolor = BODY;
            return r;
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            const r = serialize?.apply(this, arguments);
            if (o && o.color === header) delete o.color;
            if (o && o.bgcolor === BODY) delete o.bgcolor;
            return r;
        };
    },
});

console.log(" [Radiance Style] Premium Apple Pro Node Theme loaded successfully!");
