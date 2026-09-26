/**
 * radiance_vae_widgets.js
 * v3.0 — Radiance HDR VAE Decode mode-aware widget sync
 *
 * WHAT THIS FIXES:
 *
 *   decode_mode owns the color contract. Sampler mode hides the HDR-only
 *   controls (RHDR export, decode noise, HDR scale, peak). Direct HDR hides
 *   inverse_tonemap. Auto can take either path at run time, so it shows both
 *   sets. target_space is honoured in every mode. hdr_mode, hdr_output,
 *   display_tonemap and source_space are always hidden and never read: the
 *   mode and the latent's HDR Encode metadata decide them. Workflows saved
 *   before 3.5 carry "Direct HDR / RUDRA"; it is migrated on load.
 *
 *   temporal_overlap is always visible: temporal_size is now an Auto/preset
 *   combo (hdr/vae.py's decode_tiled() integration) where no value statically
 *   guarantees chunking is off, so visibility can no longer be decided from
 *   the widget alone.
 *
 *   Post-execution (read from engine.py's "ui" channel via onExecuted, since
 *   only knowable once decode actually runs):
 *     - hdr_output gets a warning when Compress(Log) ran with no
 *       radiance_meta on the latent (no genuine HDR-encoded source upstream,
 *       stripped by a sampler in between): the log-decompression curve then
 *       inverts values that were never log-encoded, producing genuinely
 *       wrong output. display_tonemap can't fix this, since hdr/vae.py
 *       applies it AFTER the decompression. target_space alone is NOT
 *       flagged: that's a ComfyUI-preview-only quirk, real output is
 *       unaffected. Both confirmed against real decoded pixel values.
 *
 * INSTALL:
 *   Place this file in the same folder as radiance_bootstrap.js (the custom
 *   node's web/js directory). ComfyUI auto-loads all .js files from that dir.
 *   No other changes needed.
 */

import { app } from "../../scripts/app.js";

import {
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
    getWidget,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "number" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    return _setWidgetVisible(widget, visible, node, { fallbackType: "number" });
}

// ALBABIT-FIX: v1.0 matched node.type against the display-name string
// "◎ Radiance HDR VAE Decode" instead of the class key ("RadianceHDRVAEDecode")
// -- confirmed via live browser console that node.type is always the class
// key, so this never matched and the extension was dead code since v3.1.
// Also replaced the canvas ctx.globalAlpha dimming + raw node.element DOM
// badge (LiteGraph-only, inert on Vue) with the widget.label marker
// convention already used on Sampler/Loader/Prompt/Tonemap/Resolution.
const TARGET_NODE = "RadianceHDRVAEDecode";

// Widget name constants
const W_HDR_OUTPUT       = "hdr_output";
const W_DECODE_MODE      = "decode_mode";
const W_DISPLAY_TONEMAP  = "display_tonemap";
const W_HDR_MODE         = "hdr_mode";
const W_EXPORT_RHDR      = "export_rhdr";
const W_INVERSE_TONEMAP  = "inverse_tonemap";
const W_TARGET_STOPS     = "target_stops";
const W_RHDR_PRECISION   = "rhdr_precision";
const W_TARGET_SPACE     = "target_space";
const W_SOURCE_SPACE     = "source_space";
const W_DECODE_NOISE_SCALE = "decode_noise_scale";
const W_HDR_SCALE_FACTOR   = "hdr_scale_factor";
const W_TEMPORAL_OVERLAP   = "temporal_overlap";
const W_HDR_PEAK_NITS      = "hdr_peak_nits";

// ALBABIT-FIX: mirrors nodes/generate/engine.py's _SCENE_REFERRED complement --
// only these 2 of the 12 target_space options are display-ready [0,1] sRGB;
// the other 10 (Linear, ACEScg, ACES 2065-1, Rec.2020 Linear, and the 6 log
// spaces) are scene-referred and will look wrong in ComfyUI's native preview.
const DISPLAY_READY_SPACES = new Set(["sRGB", "Raw"]);

// ALBABIT-FIX: kept short -- a long label suffix widens the whole node and
// squeezes every other widget's value column (Vue sizes the label/value
// split off the widest label in the node). display_tonemap's own tooltip
// already explains the Compress(Log) dependency on hover.
const BLOWOUT_MARKER = " ⚠ overexp risk";

// 3.5: the pre-3.5 name of the direct mode. Migrated to "Direct HDR" on load
// so a saved graph does not land on an invalid combo value.
const LEGACY_DIRECT_MODE = "Direct HDR / RUDRA";
const DIRECT_MODE = "Direct HDR";
const AUTO_MODE = "Auto (Recommended)";

/** Return widget by name from a node, or null. */
// Same convention as radiance_resolution.js's _setLabelMarker: cache the
// original label once, then swap between origLabel and origLabel+marker.
// Renders identically on the legacy LiteGraph canvas and the Vue frontend.
function _setLabelMarker(widget, marker) {
    if (!widget) return;
    if (widget._radOrigLabel === undefined && !marker) return;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = marker ? widget._radOrigLabel + marker : widget._radOrigLabel;
    if (widget.label !== wanted) widget.label = wanted;
}


function refreshNodeSize(node) {
    if (!node.computeSize) return;
    const sz = node.computeSize();
    const newWidth = Math.max(node.size[0], sz[0]);
    const newHeight = sz[1];
    if (node.size[0] === newWidth && node.size[1] === newHeight) return;
    node.setSize([newWidth, newHeight]);
    node.setDirtyCanvas(true, true);
}

/**
 * Sync all dependent widget states based on current hdr_output + hdr_mode.
 */
function syncWidgets(node) {
    const decodeModeW     = getWidget(node, W_DECODE_MODE);
    const hdrOutputW      = getWidget(node, W_HDR_OUTPUT);
    const displayTmW      = getWidget(node, W_DISPLAY_TONEMAP);
    const hdrModeW        = getWidget(node, W_HDR_MODE);
    const exportRhdrW     = getWidget(node, W_EXPORT_RHDR);
    const inverseTmW      = getWidget(node, W_INVERSE_TONEMAP);
    const targetStopsW    = getWidget(node, W_TARGET_STOPS);
    const rhdrPrecisionW  = getWidget(node, W_RHDR_PRECISION);
    const targetSpaceW    = getWidget(node, W_TARGET_SPACE);
    const sourceSpaceW    = getWidget(node, W_SOURCE_SPACE);
    const decodeNoiseW    = getWidget(node, W_DECODE_NOISE_SCALE);
    const hdrScaleW       = getWidget(node, W_HDR_SCALE_FACTOR);
    const temporalOverlapW = getWidget(node, W_TEMPORAL_OVERLAP);
    const hdrPeakW        = getWidget(node, W_HDR_PEAK_NITS);

    if (!decodeModeW) return;

    const hdrMode = hdrModeW?.value ?? "";
    if (decodeModeW.value === LEGACY_DIRECT_MODE) decodeModeW.value = DIRECT_MODE;
    const directHDR = decodeModeW.value === DIRECT_MODE;
    const isCompressLog = directHDR || hdrMode === "Compress (Log)";
    // Overexposure-risk marking happens post-execution in onExecuted below
    // (merged from beta/main): risk depends on radiance_meta, only known once
    // the decode has actually run — not derivable from widget values here.
    const displayTmVal = displayTmW?.value ?? "ACES Filmic";
    const targetSpaceVal = targetSpaceW?.value ?? "sRGB";

    // decode_mode owns these values at execution: sampler mode is Clip/sRGB,
    // direct mode is Compress(Log)/Linear with no display tonemap.
    // MERGE-PORT (beta/main 9a5ac88): gate the node resize on actual
    // visibility transitions so the 250 ms poll stays a no-op at rest.
    // 3.5.0: visibility follows what each mode actually reads. Auto can run
    // either path (Direct HDR only when HDR Encode metadata is live), so it
    // shows the controls of both. Nothing hidden is read by the backend:
    // hdr_mode / hdr_output / display_tonemap / source_space are decided by
    // decode_mode and the latent's own metadata.
    const autoMode = decodeModeW.value === AUTO_MODE;
    const samplerMode = !directHDR && !autoMode;
    let changed = false;
    if (setWidgetVisible(hdrModeW, false, node)) changed = true;
    if (setWidgetVisible(hdrOutputW, false, node)) changed = true;
    if (setWidgetVisible(displayTmW, false, node)) changed = true;
    if (setWidgetVisible(sourceSpaceW, false, node)) changed = true;
    if (setWidgetVisible(targetSpaceW, true, node)) changed = true;
    if (setWidgetVisible(exportRhdrW, !samplerMode, node)) changed = true;
    if (samplerMode && exportRhdrW?.value) exportRhdrW.value = false;
    if (setWidgetVisible(rhdrPrecisionW, !samplerMode && !!exportRhdrW?.value, node)) changed = true;
    if (setWidgetVisible(inverseTmW, !directHDR, node)) changed = true;
    if (setWidgetVisible(targetStopsW, !directHDR && !!inverseTmW?.value, node)) changed = true;
    if (setWidgetVisible(decodeNoiseW, !samplerMode, node)) changed = true;
    if (setWidgetVisible(hdrScaleW, !samplerMode, node)) changed = true;
    if (setWidgetVisible(hdrPeakW, !samplerMode, node)) changed = true;

    // ALBABIT-FIX (VAE tiling integration): temporal_size is now an
    // Auto/preset combo. Auto's chunk size is VRAM-dependent and every
    // manual preset still depends on the clip's actual length at runtime,
    // so no widget value statically guarantees chunking is off. Always show.
    if (setWidgetVisible(temporalOverlapW, true, node)) changed = true;

    if (changed) refreshNodeSize(node);
}

app.registerExtension({
    name: "Radiance.VAEWidgetSync",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            const self = this;

            const hdrOutputW = getWidget(this, W_HDR_OUTPUT);
            const decodeModeW = getWidget(this, W_DECODE_MODE);
            if (!decodeModeW) return;

            // Wire callbacks for instant feedback on the driver widgets.
            [
                hdrOutputW,
                decodeModeW,
                getWidget(this, W_HDR_MODE),
                getWidget(this, W_DISPLAY_TONEMAP),
                getWidget(this, W_INVERSE_TONEMAP),
                getWidget(this, W_EXPORT_RHDR),
                getWidget(this, W_TARGET_SPACE),
            ]
                .forEach(w => {
                    if (!w) return;
                    const origCallback = w.callback;
                    w.callback = function (...args) {
                        const res = origCallback ? origCallback.apply(this, args) : undefined;
                        syncWidgets(self);
                        return res;
                    };
                });

            // ALBABIT-FIX: poll as a state-based fallback (undo/redo, preset
            // import, workflow load) — same pattern as radiance_sampler.js's
            // preset divergence markers, covers mutation paths that bypass
            // the wrapped callbacks above.
            this._vaeSyncInterval = setInterval(() => syncWidgets(self), 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._vaeSyncInterval) {
                    clearInterval(self._vaeSyncInterval);
                    self._vaeSyncInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            setTimeout(() => syncWidgets(self), 150);
        };

        // Re-apply after a saved workflow restores this node — onNodeCreated
        // runs before ComfyUI deserializes widget values.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            setTimeout(() => syncWidgets(self), 150);
            setTimeout(() => syncWidgets(self), 600);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            // "ui" side-channel convention (same as resolution.js's
            // computed_width), for hdr_output.
            // engine.py computes log_overexposure_risk from radiance_meta
            // (only known at decode time), so this only reflects the run that
            // just finished, not the current widget values.
            const overexpRisk = !!message?.log_overexposure_risk?.[0];
            _setLabelMarker(getWidget(this, W_HDR_OUTPUT), overexpRisk ? BLOWOUT_MARKER : null);

            this.setDirtyCanvas?.(true, true);
        };
    },
});
