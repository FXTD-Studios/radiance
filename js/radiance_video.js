// radiance_video.js
// ALBABIT-FIX: widget-greying for video nodes when dit_config is connected.
//
// When a RadianceVideoModelInfo node is wired into dit_config, the Python side
// resolves all sampling params from model-specific defaults and ignores the
// manual widgets. This mirrors the sigmas_override greying in radiance_sampler.js.
//
// Applies to: RadianceVideoSampler, RadianceT2VPipeline, RadianceI2VPipeline.
// Not applied to: RadianceVideoLatentNoise (dit_config only provides compression
// ratios — width/height/frames are not overridden) and RadianceVideoBatchDecode
// (dit_config only provides latent_scale — no user-facing widget is overridden).

import { app } from "../../scripts/app.js";

// ALBABIT-FIX: widgets that become inert when an active dit_config is connected.
// Same four params for all three targeted nodes.
const DIT_CONFIG_WIDGETS = ["steps", "cfg", "sampler_name", "scheduler"];

// ALBABIT-FIX: returns true when dit_config has an active (non-muted, non-bypassed) link.
function isDitConfigActive(node) {
    const ditInput = node.inputs?.find(inp => inp.name === "dit_config");
    if (!ditInput || !ditInput.link) return false;
    const link = app.graph.links[ditInput.link];
    if (!link) return false;
    const originNode = app.graph.getNodeById(link.origin_id);
    // mode 2 = Muted, mode 4 = Bypassed — treat as inactive
    return originNode && originNode.mode !== 2 && originNode.mode !== 4;
}

// ALBABIT-FIX: disable/re-enable the widgets that become inert when dit_config is active.
// Uses the same disabled + inputEl styling as updateSigmaLocks() in radiance_sampler.js.
function updateDitConfigLocks(node) {
    if (!node.widgets) return;
    const locked = isDitConfigActive(node);

    node.widgets.forEach(widget => {
        if (!DIT_CONFIG_WIDGETS.includes(widget.name)) return;
        widget.disabled = locked;
        if (widget.inputEl) {
            widget.inputEl.disabled = locked;
            widget.inputEl.style.opacity = locked ? "0.4" : "1.0";
            widget.inputEl.style.pointerEvents = locked ? "none" : "auto";
        }
    });

    node.setDirtyCanvas(true, true);
}

function applyDitConfigGreying(nodeType) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
        const self = this;

        // Hook connection change events so greying updates immediately on connect/disconnect.
        const origConnect = this.onConnectionsChange;
        this.onConnectionsChange = function (...args) {
            if (origConnect) origConnect.apply(this, args);
            updateDitConfigLocks(self);
        };

        // ALBABIT-FIX: poll for upstream mute/bypass changes.
        // onConnectionsChange only fires on this node — it does not fire when
        // the upstream node is muted/bypassed.
        this._ditConfigCheckInterval = setInterval(() => updateDitConfigLocks(self), 250);

        const origOnRemoved = this.onRemoved;
        this.onRemoved = function () {
            if (self._ditConfigCheckInterval) {
                clearInterval(self._ditConfigCheckInterval);
                self._ditConfigCheckInterval = null;
            }
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
        };

        // Apply initial state (covers workflow reload where dit_config was already wired).
        setTimeout(() => updateDitConfigLocks(self), 150);
    };
}

const DIT_CONFIG_NODES = new Set([
    "RadianceVideoSampler",
    "RadianceT2VPipeline",
    "RadianceI2VPipeline",
]);

app.registerExtension({
    name: "Radiance.Video",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!DIT_CONFIG_NODES.has(nodeData.name)) return;
        applyDitConfigGreying(nodeType);
    },
});
