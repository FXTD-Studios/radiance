import { app } from "../../scripts/app.js";

/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                         RADIANCE LAYOUT
 *              Advanced Reroute — Compact Pill Rendering
 * ═══════════════════════════════════════════════════════════════════════════════
 */

// Type → Color mapping for auto-coloring
const TYPE_COLORS = {
    "IMAGE": { bg: "#1a3a5a", border: "#3a8adf", accent: "#5ab0ff" },
    "LATENT": { bg: "#3a1a5a", border: "#8a3adf", accent: "#b05aff" },
    "CONDITIONING": { bg: "#5a3a1a", border: "#df8a3a", accent: "#ffb05a" },
    "MODEL": { bg: "#1a5a2a", border: "#3adf5a", accent: "#5affb0" },
    "VAE": { bg: "#5a1a3a", border: "#df3a8a", accent: "#ff5ab0" },
    "CLIP": { bg: "#4a4a1a", border: "#bfbf3a", accent: "#dfdf5a" },
    "MASK": { bg: "#2a2a2a", border: "#8a8a8a", accent: "#c0c0c0" },
    "INT": { bg: "#1a4a3a", border: "#3abf9a", accent: "#5adfba" },
    "FLOAT": { bg: "#1a4a3a", border: "#3abf9a", accent: "#5adfba" },
    "STRING": { bg: "#3a3a1a", border: "#9a9a3a", accent: "#baba5a" },
    "CONTROL_NET": { bg: "#1a3a3a", border: "#3a9a9a", accent: "#5ababa" },
    "CLIP_VISION": { bg: "#3a3a4a", border: "#7a7aaf", accent: "#9a9adf" },
};

const PRESET_COLORS = {
    "Gray": { bg: "#2a2a2e", border: "#6a6a70", accent: "#909098" },
    "Red": { bg: "#4a1a1a", border: "#cf4a4a", accent: "#ff6a6a" },
    "Orange": { bg: "#4a2a0a", border: "#cf7a2a", accent: "#ffa04a" },
    "Yellow": { bg: "#4a4a0a", border: "#cfcf2a", accent: "#ffff5a" },
    "Green": { bg: "#1a4a1a", border: "#4acf4a", accent: "#6aff6a" },
    "Cyan": { bg: "#1a4a4a", border: "#4acfcf", accent: "#6affff" },
    "Blue": { bg: "#1a2a5a", border: "#4a7adf", accent: "#6aa0ff" },
    "Purple": { bg: "#3a1a5a", border: "#8a4adf", accent: "#aa6aff" },
    "Magenta": { bg: "#4a1a4a", border: "#cf4acf", accent: "#ff6aff" },
    "White": { bg: "#3a3a3a", border: "#c0c0c0", accent: "#ffffff" },
};

const DEFAULT_COLORS = { bg: "#232330", border: "#4a4a6a", accent: "#6a9aff" };

// Node dimensions
const REROUTE_WIDTH = 140;
const REROUTE_HEIGHT = 26;
const PILL_RADIUS = 13;

app.registerExtension({
    name: "FXTD.Radiance.Layout",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;
        if (comfyClass !== "RadianceAdvancedReroute") return;

        // Override onNodeCreated for initial setup
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            // Force compact size
            this.size = [REROUTE_WIDTH, REROUTE_HEIGHT];
            this.properties = this.properties || {};
            this.properties._resolvedType = "*";

            // Collapse widgets — they'll be controlled via right-click menu
            this.setSize([REROUTE_WIDTH, REROUTE_HEIGHT]);

            // Hide default title bar
            this.flags = this.flags || {};
            this.flags.no_title = true;

            // Minimize slots visual size
            if (this.inputs && this.inputs[0]) {
                this.inputs[0].label = "";
            }
            if (this.outputs && this.outputs[0]) {
                this.outputs[0].label = "";
            }
        };

        // Override size computation to prevent auto-resize
        nodeType.prototype.computeSize = function () {
            return [REROUTE_WIDTH, REROUTE_HEIGHT];
        };

        // Main rendering override
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (this.flags.collapsed) return;

            const w = this.size[0];
            const h = this.size[1];

            // Resolve colors
            const colors = this._getResolvedColors();

            // 1. Draw Pill Shape
            ctx.save();

            // Shadow
            ctx.shadowColor = "rgba(0,0,0,0.4)";
            ctx.shadowBlur = 6;
            ctx.shadowOffsetY = 2;

            // Pill body
            ctx.beginPath();
            ctx.roundRect(0, 0, w, h, PILL_RADIUS);
            ctx.fillStyle = colors.bg;
            ctx.fill();

            ctx.shadowColor = "transparent";

            // Border
            ctx.strokeStyle = colors.border;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // 2. Connection Dots
            const dotY = h / 2;
            const dotRadius = 4;

            // Input dot (left)
            const hasInput = this.inputs && this.inputs[0] && this.inputs[0].link != null;
            ctx.beginPath();
            ctx.arc(0, dotY, dotRadius, 0, Math.PI * 2);
            ctx.fillStyle = hasInput ? colors.accent : colors.border;
            ctx.fill();
            if (hasInput) {
                ctx.beginPath();
                ctx.arc(0, dotY, dotRadius + 3, 0, Math.PI * 2);
                ctx.strokeStyle = colors.accent;
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.3;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }

            // Output dot (right)
            const hasOutput = this.outputs && this.outputs[0] && this.outputs[0].links && this.outputs[0].links.length > 0;
            ctx.beginPath();
            ctx.arc(w, dotY, dotRadius, 0, Math.PI * 2);
            ctx.fillStyle = hasOutput ? colors.accent : colors.border;
            ctx.fill();
            if (hasOutput) {
                ctx.beginPath();
                ctx.arc(w, dotY, dotRadius + 3, 0, Math.PI * 2);
                ctx.strokeStyle = colors.accent;
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.3;
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }

            // 3. Label Text
            const label = this._getLabel();
            if (label) {
                ctx.font = "bold 11px 'Inter', 'Segoe UI', system-ui, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = colors.accent;

                // Truncate if too long
                let displayLabel = label;
                const maxWidth = w - 24;
                if (ctx.measureText(displayLabel).width > maxWidth) {
                    while (ctx.measureText(displayLabel + "…").width > maxWidth && displayLabel.length > 1) {
                        displayLabel = displayLabel.slice(0, -1);
                    }
                    displayLabel += "…";
                }
                ctx.fillText(displayLabel, w / 2, h / 2);
            } else {
                // No label — draw a small center dot
                ctx.beginPath();
                ctx.arc(w / 2, h / 2, 2, 0, Math.PI * 2);
                ctx.fillStyle = colors.border;
                ctx.fill();
            }

            // 4. Type badge (small text at bottom-right if auto-detected)
            const resolvedType = this._getResolvedType();
            if (resolvedType && resolvedType !== "*" && !label) {
                ctx.font = "9px 'Inter', 'Segoe UI', system-ui, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillStyle = colors.border;
                ctx.globalAlpha = 0.6;

                const shortType = resolvedType.length > 8
                    ? resolvedType.substring(0, 7) + "…"
                    : resolvedType;
                ctx.fillText(shortType, w / 2, h / 2);
                ctx.globalAlpha = 1.0;
            }

            ctx.restore();
        };

        // Hide the background to avoid the default box rendering
        nodeType.prototype.onDrawBackground = function (ctx) {
            // Intentionally empty — pill shape handles all drawing
        };

        // Helper: Get resolved colors
        nodeType.prototype._getResolvedColors = function () {
            // Check widget value for color preset
            const colorWidget = this.widgets?.find(w => w.name === "color");
            const colorPreset = colorWidget?.value || "Auto";

            if (colorPreset !== "Auto") {
                return PRESET_COLORS[colorPreset] || DEFAULT_COLORS;
            }

            // Auto mode: resolve from connected type
            const resolvedType = this._getResolvedType();
            if (resolvedType && TYPE_COLORS[resolvedType]) {
                return TYPE_COLORS[resolvedType];
            }

            return DEFAULT_COLORS;
        };

        // Helper: Get label from widget
        nodeType.prototype._getLabel = function () {
            const labelWidget = this.widgets?.find(w => w.name === "label");
            return labelWidget?.value || "";
        };

        // Helper: Detect connected type
        nodeType.prototype._getResolvedType = function () {
            // Check input link
            if (this.inputs && this.inputs[0] && this.inputs[0].link != null) {
                const linkInfo = this.graph?.links?.[this.inputs[0].link];
                if (linkInfo) {
                    return linkInfo.type || "*";
                }
            }
            // Check output links
            if (this.outputs && this.outputs[0] && this.outputs[0].links?.length > 0) {
                const linkId = this.outputs[0].links[0];
                const linkInfo = this.graph?.links?.[linkId];
                if (linkInfo) {
                    return linkInfo.type || "*";
                }
            }
            return "*";
        };

        // Right-click context menu additions
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            origGetExtraMenuOptions?.apply(this, arguments);

            const node = this;

            // Quick Label Entry
            options.unshift({
                content: "Set Label...",
                callback: () => {
                    const labelWidget = node.widgets?.find(w => w.name === "label");
                    if (!labelWidget) return;
                    const current = labelWidget.value || "";
                    const newLabel = prompt("Enter reroute label:", current);
                    if (newLabel !== null) {
                        labelWidget.value = newLabel;
                        node.setDirtyCanvas(true, true);
                    }
                }
            });

            // Quick Color Picker
            options.unshift({
                content: "Color",
                submenu: {
                    options: Object.keys(PRESET_COLORS).map(colorName => ({
                        content: (node.widgets?.find(w => w.name === "color")?.value === colorName ? "● " : "  ") + colorName,
                        callback: () => {
                            const colorWidget = node.widgets?.find(w => w.name === "color");
                            if (colorWidget) {
                                colorWidget.value = colorName;
                                node.setDirtyCanvas(true, true);
                            }
                        }
                    })).concat([
                        null, // Separator
                        {
                            content: (node.widgets?.find(w => w.name === "color")?.value === "Auto" ? "● " : "  ") + "Auto (detect type)",
                            callback: () => {
                                const colorWidget = node.widgets?.find(w => w.name === "color");
                                if (colorWidget) {
                                    colorWidget.value = "Auto";
                                    node.setDirtyCanvas(true, true);
                                }
                            }
                        }
                    ])
                }
            });
        };
    }
});
