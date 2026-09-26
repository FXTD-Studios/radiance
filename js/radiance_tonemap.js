import { app } from "../../scripts/app.js";

// ALBABIT-FIX: tonemap() (hdr/tonemap.py) used to overwrite these 8 widgets
// unconditionally on every execution, regardless of what they showed (same
// bug class as the Sampler's _apply_presets — see radiance_sampler.js).
// Python now respects the live widget values; this file fills them on
// selection and flags a later edit with a "✎" marker.
const PRESET_CONFIGS = {
    "◎ Cinematic Film": {
        operator: "filmic_aces",
        exposure: 0.0,
        gamma: 2.2,
        white_point: 1.0,
        contrast: 1.1,
        saturation: 0.95,
        highlight_compression: 0.8,
        shadow_lift: 0.02,
    },
    "◎ HDR Display": {
        operator: "agx",
        exposure: 0.3,
        gamma: 2.2,
        white_point: 2.0,
        contrast: 1.0,
        saturation: 1.1,
        highlight_compression: 0.5,
        shadow_lift: 0.0,
    },
    "◎ Web / Social": {
        operator: "filmic_aces",
        exposure: 0.2,
        gamma: 2.2,
        white_point: 1.0,
        contrast: 1.15,
        saturation: 1.1,
        highlight_compression: 0.9,
        shadow_lift: 0.01,
    },
    "◎ Print Ready": {
        operator: "reinhard_luminance",
        exposure: -0.2,
        gamma: 2.2,
        white_point: 1.0,
        contrast: 0.95,
        saturation: 0.9,
        highlight_compression: 0.7,
        shadow_lift: 0.03,
    },
    "◎ Game Engine": {
        operator: "filmic_uncharted2",
        exposure: 0.0,
        gamma: 2.2,
        white_point: 4.0,
        contrast: 1.05,
        saturation: 1.0,
        highlight_compression: 0.6,
        shadow_lift: 0.0,
    },
    "◎ Photography": {
        operator: "reinhard_extended",
        exposure: 0.0,
        gamma: 2.2,
        white_point: 2.0,
        contrast: 1.0,
        saturation: 1.0,
        highlight_compression: 0.5,
        shadow_lift: 0.0,
    },
    "◎ Low Key / Dark": {
        operator: "filmic_aces",
        exposure: -0.5,
        gamma: 2.4,
        white_point: 1.0,
        contrast: 1.2,
        saturation: 0.85,
        highlight_compression: 0.9,
        shadow_lift: 0.0,
    },
    "◎ High Key / Bright": {
        operator: "reinhard",
        exposure: 0.5,
        gamma: 2.0,
        white_point: 1.5,
        contrast: 0.9,
        saturation: 1.05,
        highlight_compression: 0.4,
        shadow_lift: 0.05,
    },
};

const PRESET_MARKER = " ✎";
const TRACKED_FIELDS = [
    "operator", "exposure", "gamma", "white_point",
    "contrast", "saturation", "highlight_compression", "shadow_lift",
];

function getPresetConfig(name) {
    return PRESET_CONFIGS[name] || null;
}

function applyPreset(node, presetName) {
    const config = getPresetConfig(presetName);
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

function valuesEqual(a, b) {
    if (typeof a === "number" || typeof b === "number") {
        const na = Number(a), nb = Number(b);
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return Math.abs(na - nb) < 1e-6;
    }
    return String(a) === String(b);
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = node.widgets.find(w => w.name === "preset");
    const presetVal = presetW ? presetW.value : "None (Custom)";
    const config = presetVal === "None (Custom)" ? null : getPresetConfig(presetVal);

    let changed = false;
    for (const w of node.widgets) {
        if (!w || !w.name || !TRACKED_FIELDS.includes(w.name)) continue;
        const marked = !!config && config[w.name] !== undefined && !valuesEqual(w.value, config[w.name]);
        if (w._radOrigLabel === undefined && !marked) continue;
        if (w._radOrigLabel === undefined) w._radOrigLabel = w.label ?? w.name;
        const wanted = marked ? w._radOrigLabel + PRESET_MARKER : w._radOrigLabel;
        if (w.label !== wanted) {
            w.label = wanted;
            changed = true;
        }
    }
    if (changed) node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "FXTD.RadianceHDRToneMap",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceHDRToneMap") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const self = this;
            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (!presetWidget) return;

            let lastPresetValue = presetWidget.value;

            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);
                if (window.app && window.app.configuringGraph) return;
                if (value !== lastPresetValue && value !== "None (Custom)") {
                    applyPreset(this, value);
                }
                lastPresetValue = value;
            };

            // Poll for manual widget edits, undo/redo — same state-based
            // approach as radiance_sampler.js/radiance_prompt.js.
            this._presetMarkerInterval = setInterval(() => updatePresetDivergenceMarkers(self), 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._presetMarkerInterval) {
                    clearInterval(self._presetMarkerInterval);
                    self._presetMarkerInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            // A freshly-added node's hardcoded widget defaults don't all
            // match its default preset ("◎ Cinematic Film") -- sync once so
            // a brand-new node doesn't start already "diverged". Skipped for
            // nodes restored from a saved workflow (onConfigure sets
            // _configuredByLoad before this timer fires).
            setTimeout(() => {
                if (self._configuredByLoad) return;
                applyPreset(self, presetWidget.value);
            }, 150);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            self._configuredByLoad = true;
            setTimeout(() => updatePresetDivergenceMarkers(self), 150);
            setTimeout(() => updatePresetDivergenceMarkers(self), 600);
        };
    }
});

console.log("[Radiance HDR Tone Map] Extension loaded");
