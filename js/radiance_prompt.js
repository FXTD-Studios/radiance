import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setWidgetVisible } from "./radiance_widget_utils.js";

// ALBABIT-FIX: only known post-execution (resolved_arch depends on the real
// CLIP/model_meta), same convention as radiance_vae_widgets.js's
// overexposure markers (engine.py's "ui" channel via onExecuted).
const WEAK_NEG_MARKER = " ⚠ no CFG, ignored";

function _setLabelMarker(widget, marker) {
    if (!widget) return;
    if (widget._radOrigLabel === undefined && !marker) return;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = marker ? widget._radOrigLabel + marker : widget._radOrigLabel;
    if (widget.label !== wanted) widget.label = wanted;
}

// Mirrors radiance_sampler.js's own copy exactly (each file keeps its own,
// same convention as the small widget helpers above).
function _findModelMetaSourceNode(node) {
    const input = node.inputs?.find(i => i.name === "model_meta");
    if (!input || !input.link) return null;
    const link = app.graph.links[input.link];
    if (!link) return null;
    const originNode = app.graph.getNodeById(link.origin_id);
    if (!originNode || originNode.mode === 2 || originNode.mode === 4) return null;
    return originNode;
}

// ALBABIT-FIX: live read of the connected Loader's preset/model_type. No
// source (unconnected or bypassed) means false, not null, same as
// isSigmaOverrideActive() in radiance_sampler.js. null is reserved for a
// connected, live Loader in Auto-Detect, resolved server-side only.
function _liveMiniMaxState(node) {
    const sourceNode = _findModelMetaSourceNode(node);
    if (!sourceNode) return false;
    const presetW = sourceNode.widgets?.find(w => w.name === "preset");
    if (!presetW) return false;
    if (String(presetW.value).startsWith("MiniMax H3")) return true;
    if (presetW.value !== "Custom") return false;
    const modelTypeW = sourceNode.widgets?.find(w => w.name === "model_type");
    if (!modelTypeW) return false;
    if (modelTypeW.value === "minimax") return true;
    if (modelTypeW.value === "Auto-Detect") return null;
    return false;
}

// ALBABIT-FIX: ComfyUI skips the negative (uncond) pass at cfg 1 for every model,
// so the field is unused when every sampler fed by the "negative" output runs at
// cfg 1. Still read at cfg 1: *_cfg_pp samplers, LTX-AV's audio_cfg, a Self-
// Attention Guidance patch on the model. Any consumer this cannot read (another
// node in between, cfg on an input) keeps the field editable.
const CFG_ONE_SAMPLERS = { RadianceSamplerPro: "sampler", KSampler: "sampler_name", KSamplerAdvanced: "sampler_name" };

function _modelChainHasSag(graph, sampler) {
    let node = sampler;
    for (let depth = 0; node && depth < 32; depth++) {
        if (node.comfyClass === "SelfAttentionGuidance") return true;
        const link = graph.links[node.inputs?.find(i => i.name === "model")?.link];
        node = link ? graph.getNodeById(link.origin_id) : null;
    }
    return false;
}

function _samplerIgnoresNegative(graph, sampler) {
    const samplerField = CFG_ONE_SAMPLERS[sampler.comfyClass];
    if (!samplerField || sampler.inputs?.find(i => i.name === "cfg")?.link != null) return false;
    const value = name => sampler.widgets?.find(w => w.name === name)?.value;
    const audioCfg = sampler.widgets?.find(w => w.name === "audio_cfg");
    return Math.abs(Number(value("cfg")) - 1) < 1e-9  // comfy's math.isclose(cfg, 1.0)
        && !String(value(samplerField)).includes("cfg_pp")
        && !(audioCfg && !audioCfg.hidden && audioCfg.value > 0 && audioCfg.value !== 1)
        && !_modelChainHasSag(graph, sampler);
}

// node.graph, not app.graph: it is the graph holding this node's links, a
// subgraph in the Comfy-Org templates.
function _negativeUnusedDownstream(node) {
    const graph = node.graph;
    const output = name => node.outputs?.find(o => o.name === name);
    if (!graph || output("negative_text")?.links?.length) return false;
    const samplers = (output("negative")?.links ?? [])
        .map(id => graph.links[id])
        .map(link => link && graph.getNodeById(link.target_id))
        .filter(n => n && n.mode !== 2 && n.mode !== 4);
    return samplers.length > 0 && samplers.every(n => _samplerIgnoresNegative(graph, n));
}

// ALBABIT-FIX: same disabled + inputEl + opacity/pointerEvents combination
// already proven in radiance_sampler.js's updateSigmaLocks(). Skips
// reassignment when already correct, this runs from a 250ms poll and would
// otherwise interrupt in-progress typing (same bug class, same fix).
function _applyNegPromptLock(node, locked) {
    const negW = node.widgets?.find(w => w.name === "negative_prompt");
    if (!negW || negW.disabled === locked) return;
    negW.disabled = locked;
    if (negW.inputEl) {
        negW.inputEl.disabled = locked;
        negW.inputEl.style.opacity = locked ? "0.4" : "1.0";
        negW.inputEl.style.pointerEvents = locked ? "none" : "auto";
    }
    node.setDirtyCanvas(true, true);
}

function refreshNodeSize(node) {
    if (!node.computeSize) return;
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    app.graph.setDirtyCanvas(true, true);
}

// ALBABIT-FIX: negative_strength (a combo, no reliable disabled/inputEl
// path the way a text widget has) hides entirely instead, same mechanism
// as every preset-driven widget in radiance_loader.js/radiance_resolution.js.
function _applyNegStrengthLock(node, hidden) {
    const strengthW = node.widgets?.find(w => w.name === "negative_strength");
    if (!strengthW || strengthW.hidden === hidden) return;
    setWidgetVisible(strengthW, !hidden, node, { fallbackType: "combo" });
    refreshNodeSize(node);
}

// ALBABIT-FIX: shared by the poll loop, onConfigure and onExecuted below. A
// Loader in Auto-Detect (liveState null) falls back to the last run's verdict.
function _refreshLiveState(node) {
    updatePresetDivergenceMarkers(node);
    const liveState = _liveMiniMaxState(node);
    const cfgOne = _negativeUnusedDownstream(node);
    const locked = liveState === null ? cfgOne || !!node._radWeakNeg : liveState || cfgOne;
    _applyNegPromptLock(node, locked);
    _applyNegStrengthLock(node, locked);
    const negW = node.widgets?.find(w => w.name === "negative_prompt");
    _setLabelMarker(negW, cfgOne || node._radWeakNeg ? WEAK_NEG_MARKER : null);
}

// ALBABIT-FIX: apply_style_preset() used to overwrite these 7 widgets on
// every execution, not just on selection (same bug class as the Sampler's
// _apply_presets). This file fills them once on selection and flags a later
// edit with a "✎" marker. film_stock/shutter_speed/aspect_ratio have no
// widget here, so Python keeps applying the preset for those.
//
// 3.5.0: the preset table comes from Python (nodes/generate/prompt.py,
// /radiance/prompt/presets) instead of a copy kept here by hand. If the
// request fails, picking a preset leaves the widgets alone and Python applies
// the whole preset at run time (it does so whenever all 7 are still "None").
let PRESET_CONFIGS = {};
const presetsReady = api.fetchApi("/radiance/prompt/presets")
    .then(r => (r.ok ? r.json() : {}))
    .then(data => { PRESET_CONFIGS = data || {}; })
    .catch(err => console.warn("[Radiance Cinematic Encoder] presets not loaded:", err));

const PRESET_MARKER = " ✎";
const TRACKED_FIELDS = [
    "framing", "camera_type", "lens_focal", "aperture_dof",
    "lighting", "style_aesthetic", "color_grading",
];

function getPresetConfig(name) {
    return PRESET_CONFIGS[name] || null;
}

async function applyPreset(node, presetName) {
    await presetsReady;
    const config = getPresetConfig(presetName);
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    // Widgets now match the preset again — clear any "✎" markers.
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

// Every tracked dataset (FRAMING/CAMERAS/LENSES/APERTURES/LIGHTING/STYLES/
// COLOR_GRADING) has "None" as its first, valid entry — switching to
// "None (Custom)" blanks all 7 style widgets back to it, same as selecting
// a named preset overwrites them with that preset's values.
function resetToCustomDefaults(node) {
    const widgets = node.widgets;
    if (!widgets) return;
    for (const widget of widgets) {
        if (TRACKED_FIELDS.includes(widget.name)) {
            widget.value = "None";
        }
    }
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = node.widgets.find(w => w.name === "style_preset");
    const presetVal = presetW ? presetW.value : "None (Custom)";
    const config = presetVal === "None (Custom)" ? null : getPresetConfig(presetVal);

    let changed = false;
    for (const w of node.widgets) {
        if (!w || !w.name || !TRACKED_FIELDS.includes(w.name)) continue;
        let marked = false;
        if (config && config[w.name] !== undefined) {
            marked = String(w.value) !== String(config[w.name]);
        }
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
    name: "FXTD.RadianceCinematicPromptEncoder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceCinematicPromptEncoder") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const self = this;
            const presetWidget = this.widgets?.find(w => w.name === "style_preset");
            if (!presetWidget) return;

            let lastPresetValue = presetWidget.value;

            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);
                if (window.app && window.app.configuringGraph) return;
                if (value !== lastPresetValue) {
                    if (value === "None (Custom)") {
                        resetToCustomDefaults(this);
                    } else {
                        applyPreset(this, value);
                    }
                }
                lastPresetValue = value;
            };

            // Poll for manual widget edits, undo/redo — same state-based
            // approach as js/radiance_sampler.js's marker refresh. Also
            // covers the connected Loader's own preset/model_type changing
            // live, which _liveMiniMaxState reads directly.
            this._presetMarkerInterval = setInterval(() => _refreshLiveState(self), 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._presetMarkerInterval) {
                    clearInterval(self._presetMarkerInterval);
                    self._presetMarkerInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        };

        // Flag pre-existing divergence in a loaded workflow (e.g. a widget
        // edited manually before saving) — mirrors js/radiance_sampler.js.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            setTimeout(() => _refreshLiveState(self), 150);
            setTimeout(() => _refreshLiveState(self), 600);
        };

        // ALBABIT-FIX: the last real run's arch verdict, read by
        // _refreshLiveState for the label marker and the Auto-Detect lock.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            this._radWeakNeg = !!message?.weak_neg_arch?.[0];
            _refreshLiveState(this);
            this.setDirtyCanvas?.(true, true);
        };
    }
});

console.log("[Radiance Cinematic Prompt Encoder] Extension loaded");
