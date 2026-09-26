import { app } from "../../../scripts/app.js";
import {
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "INT" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    _setWidgetVisible(widget, visible, node, { fallbackType: "INT" });
}

// FIX 5: was "../../scripts/app.js" — extensions in custom_nodes/Radiance/web/
// need three ../ to reach ComfyUI's scripts/ directory.

/**
 * Radiance Resolution — Widget Management (v2.4)
 *
 * FIX 5: Import path corrected (see above).
 * FIX 6: Switched from nodeCreated hook + nested setTimeout to beforeRegisterNodeDef
 *         which is the standard Radiance pattern. nodeCreated fires for every node
 *         in the graph including unrelated ones; beforeRegisterNodeDef targets
 *         exactly our node type before registration, avoiding the fragile 10ms
 *         race condition on widget availability.
 *
 * ALBABIT-FIX v2.5: Updated fix for Nodes 2.0 empty-space bug:
 *   1. setWidgetVisible now takes (widget, visible, node) and unconditionally
 *      forces a remove+reinsert of the widget in node.widgets via
 *      _forceWidgetReinsert — a splice(0,0) no-op alone stops working once a
 *      widget's Vue component has been (re)mounted.
 *   2. computedHeight = 4 (not -4) for hidden widgets — Vue maps this to 0px CSS height.
 *   3. refreshNodeSize uses node.setSize(...) — this is the API Vue's resize
 *      handling actually observes; raw node.size[i] mutation has zero visual effect.
 *   4. Initial toggleFields deferred 100ms — Vue must complete its first layout pass
 *      BEFORE any widget is hidden, otherwise widget.computedHeight is undefined and
 *      the restore path falls back to 32 (wrong height, causes ghost space on re-show).
 *
 * FEATURE: Shows/hides mp_target and mp_aspect_ratio widgets based on whether
 *          mp_target > 0, alongside the existing video/batch toggle.
 */

// ALBABIT-FIX: Mirrors the temporal stride logic in resolution.py's generate()
// (n*stride + 1 frame counts) so video_frames <-> duration_seconds stay in sync
// when the user toggles frame_computation.
// ALBABIT-FIX follow-up: mirrors VIDEO_MODEL_TYPES in resolution.py —
// model_types that emit 5D video latents and should auto-enable "enable_video".
const VIDEO_MODEL_TYPES_JS = new Set(["WAN (16ch)", "WAN TI2V (48ch)", "LTXV (128ch)", "HunyuanVideo (16ch)", "Mochi (12ch)", "Cosmos World (16ch)", "CogVideoX (16ch)", "MiniMax H3 (24ch)"]);

// ALBABIT-FIX follow-up: mirrors SPATIAL_SCALE/_align_up in resolution.py —
// recompute width/height instantly when model_type changes, instead of waiting
// for the next execution's onExecuted sync.
const SPATIAL_SCALE_JS = {
    // ALBABIT-FIX: 32x spatial, confirmed identical for LTX 2.3 and 2.5 --
    // see resolution.py's SPATIAL_SCALE for the full explanation.
    "LTXV (128ch)": 32,
    "Flux.2 / Flux.2 Klein (128ch)": 16,
    // ALBABIT-FIX: WAN 2.2 TI2V-5B's VAE compresses 16x spatially (double
    // standard WAN's 8x) -- real bug fix, see resolution.py's SPATIAL_SCALE.
    "WAN TI2V (48ch)": 16,
    // ALBABIT-FIX: MiniMax H3 mirrors resolution.py's SPATIAL_ALIGN (32px, not
    // its 16x compression): keyframe latents need an even latent size.
    "MiniMax H3 (24ch)": 32,
    // ALBABIT-FIX: "Manual" -> scale=1, _alignUp is a no-op and the +/- step
    // becomes 1, so width/height are fully unconstrained.
    "Manual": 1,
};

function _alignUp(val, scale) {
    return Math.max(scale, Math.ceil(val / scale) * scale);
}

// Aligns widthW/heightW in place to the SPATIAL_SCALE of the current model_type.
// Always realigns from node._resBaseW/_resBaseH (the unaligned base resolution,
// e.g. the raw preset values) rather than the widgets' current value — _alignUp
// only rounds up, so re-aligning an already-aligned value can't recover a smaller
// alignment for a different model_type (e.g. 1088 LTXV -> 1080 Flux).
// Sets the +/- step of a width/height widget so it always lands on a valid
// alignment for the current model_type (32 for LTXV, 16 for Flux.2, 8 default).
function _setWidgetStep(widget, scale) {
    if (!widget) return;
    if (!widget.options) widget.options = {};
    widget.options.step = scale;
    widget.options.step2 = scale;
}

function _syncStepsToModelType(modelTypeW, widthW, heightW) {
    const scale = SPATIAL_SCALE_JS[modelTypeW?.value] || 8;
    _setWidgetStep(widthW, scale);
    _setWidgetStep(heightW, scale);
}

// Video models require video_frames = N*stride + 1 (e.g. WAN: 4k+1 -> 1,5,9,13...;
// LTXV: 8k+1 -> 1,9,17,25...). Snaps to the nearest valid value.
function _alignNk1(val, stride) {
    return Math.max(1, Math.round((val - 1) / stride) * stride + 1);
}

// ALBABIT-FIX: MiniMax H3's frame grid is n%17==5 (5, 22, 39...), not a plain
// stride*k+1 — mirrors resolution.py's _minimax_align_frame_count().
function _alignMiniMaxFrames(val) {
    let n = Math.max(5, Math.round(val));
    while (n % 17 !== 5) n += 1;
    return n;
}

function _isMiniMaxH3(modelType) {
    return modelType === "MiniMax H3 (24ch)";
}

// ALBABIT-FIX: shared by modelTypeW.callback and videoFramesW.callback,
// which used to each inline this same snap. A no-op unless the widget is
// actually showing a value the model_type would reject.
function _snapVideoFramesToGrid(modelTypeW, videoFramesW) {
    if (!videoFramesW || !VIDEO_MODEL_TYPES_JS.has(modelTypeW?.value)) return;
    const val = parseInt(videoFramesW.value, 10);
    const aligned = _isMiniMaxH3(modelTypeW.value)
        ? _alignMiniMaxFrames(val)
        : _alignNk1(val, _frameStride(modelTypeW.value));
    if (aligned !== val) {
        videoFramesW.value = aligned;
        if (videoFramesW.inputEl) videoFramesW.inputEl.value = aligned;
    }
}

// ALBABIT-FIX: shared by modelTypeW.callback and onConfigure's reapply,
// which used to each inline this same check. The hidden widget can't leave
// a stale value (e.g. 90 left over from a different model_type) silently
// feeding into duration_sec/the info string.
function _forceMiniMaxFrameRate(modelTypeW, frameRateW) {
    if (frameRateW && _isMiniMaxH3(modelTypeW?.value) && parseFloat(frameRateW.value) !== 24) {
        frameRateW.value = 24;
        if (frameRateW.inputEl) frameRateW.inputEl.value = 24;
    }
}

// ALBABIT-FIX: single source of truth for "duration_seconds -> aligned frame
// count", shared by the Manual<->Auto sync, the label preview, and
// duration_seconds's own precise-value correction below, so the three can
// never disagree on the answer.
function _autoSecondsFrames(modelType, durationSeconds, frameRateWidgetValue) {
    const isMiniMax = _isMiniMaxH3(modelType);
    const fps = isMiniMax ? 24.0 : (parseFloat(frameRateWidgetValue) || 24.0);
    const stride = _frameStride(modelType);
    const raw = (parseFloat(durationSeconds) || 0) * fps;
    const frames = isMiniMax
        ? _alignMiniMaxFrames(raw)
        : Math.max(1, Math.round(raw / stride) * stride + 1);
    return { frames, fps };
}

function _applyAlignment(node, modelTypeW, widthW, heightW) {
    if (!widthW || !heightW) return;
    const baseW = node._resBaseW ?? parseInt(widthW.value, 10);
    const baseH = node._resBaseH ?? parseInt(heightW.value, 10);
    const scale = SPATIAL_SCALE_JS[modelTypeW?.value] || 8;
    const newW = _alignUp(baseW, scale);
    const newH = _alignUp(baseH, scale);
    widthW.value = newW;
    heightW.value = newH;
    if (widthW.inputEl) widthW.inputEl.value = newW;
    if (heightW.inputEl) heightW.inputEl.value = newH;
}

function _frameStride(modelType) {
    const m = (modelType || "").toLowerCase();
    // ALBABIT-FIX: "Manual" -> stride=1, _alignNk1 is a no-op (any value valid).
    if (m === "manual") return 1;
    if (m.includes("ltx") || m.includes("cosmos")) return 8;
    if (m.includes("mochi")) return 6;
    if (m.includes("wan") || m.includes("hunyuan")) return 4;
    // ALBABIT-FIX: MiniMax H3's grid points (5, 22, 39...) are always exactly
    // 17 apart, so 17 is the correct +/- step even though the grid's formula
    // itself (17k+5) isn't the plain stride*k+1 pattern this function's name
    // implies for every other entry.
    if (m.includes("minimax")) return 17;
    return 4;
}

// ALBABIT-FIX: video_frames declares step=1 in Python, so a "-"/"+" click
// moved by 1 while only stride*k+1 (or MiniMax's 17k+5) values are valid.
// The snap-on-change logic below then corrected every click straight back
// to its starting value, making "-" look completely broken.
function _syncVideoFramesStep(modelTypeW, videoFramesW) {
    _setWidgetStep(videoFramesW, _frameStride(modelTypeW?.value));
}

// ALBABIT-FIX: same class of bug as _syncVideoFramesStep, one level up. Step
// = seconds per grid step (stride/fps), so a click lands cleanly. Also sets
// precision (never explicit, Python only declares step=0.1).
function _syncDurationSecondsStep(modelTypeW, durSecW, frameRateW) {
    if (!durSecW) return;
    const isMiniMax = _isMiniMaxH3(modelTypeW?.value);
    const fps = isMiniMax ? 24.0 : (parseFloat(frameRateW?.value) || 24.0);
    const stepSeconds = _frameStride(modelTypeW?.value) / fps;
    _setWidgetStep(durSecW, stepSeconds);
    if (!durSecW.options) durSecW.options = {};
    durSecW.options.precision = 2;
    // ALBABIT-FIX: options.step/step2 only drive the +/- click delta. The
    // native widget commit path snaps independently via options.round,
    // which stayed at Python's declared 0.1 default and silently knocked an
    // already grid-aligned click result back off-grid on every click,
    // before durSecW.callback's own correction even ran (root cause of the
    // intermittent "-" stall: each click compounded on that corrupted
    // value instead of the aligned one). Syncing round to the same grid
    // step stops the corruption at the source.
    durSecW.options.round = stepSeconds;
}

function refreshNodeSize(node) {
    if (!node.computeSize) return;

    const sz = node.computeSize();
    // ALBABIT-FIX: node.setSize(...) is the API Vue's resize handling actually
    // observes; raw node.size[i] mutation has zero visual effect.
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    app.graph.setDirtyCanvas(true, true);
}

// ALBABIT-FIX: "📐" = scale_factor/mp_target/width/height, either modifier
// active (both apply regardless of preset). "✎" = orientation/latent_channels
// off their neutral default.
function _setLabelMarker(widget, marked, marker) {
    if (!widget) return false;
    if (widget._radOrigLabel === undefined && !marked) return false;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = marked ? widget._radOrigLabel + marker : widget._radOrigLabel;
    if (widget.label === wanted) return false;
    widget.label = wanted;
    return true;
}

function updateResolutionMarkers(node) {
    const scaleFactorW = node.widgets?.find(w => w.name === "scale_factor");
    const mpTargetW    = node.widgets?.find(w => w.name === "mp_target");
    const widthW       = node.widgets?.find(w => w.name === "width");
    const heightW      = node.widgets?.find(w => w.name === "height");
    const orientationW = node.widgets?.find(w => w.name === "orientation");
    const latentChW    = node.widgets?.find(w => w.name === "latent_channels");
    const durSecW      = node.widgets?.find(w => w.name === "duration_seconds");
    const modelTypeW   = node.widgets?.find(w => w.name === "model_type");
    const frameRateW   = node.widgets?.find(w => w.name === "frame_rate");
    const frameModeW   = node.widgets?.find(w => w.name === "frame_computation");

    const scaleActive = scaleFactorW && parseFloat(scaleFactorW.value) !== 1.0;
    const mpActive     = mpTargetW && parseFloat(mpTargetW.value) > 0;

    // ALBABIT-FIX: only redraw when a label actually changed, this runs
    // every 250ms via the polling loop. Unconditional setDirtyCanvas here
    // was the same anti-pattern fixed in the Sampler's updateSigmaLocks().
    let changed = false;
    if (_setLabelMarker(scaleFactorW, scaleActive, " 📐")) changed = true;
    if (_setLabelMarker(mpTargetW, mpActive, " 📐")) changed = true;
    if (_setLabelMarker(widthW, scaleActive || mpActive, " 📐")) changed = true;
    if (_setLabelMarker(heightW, scaleActive || mpActive, " 📐")) changed = true;

    if (_setLabelMarker(orientationW, orientationW && orientationW.value !== "As Preset", " ✎")) changed = true;
    if (_setLabelMarker(latentChW, latentChW && parseInt(latentChW.value, 10) !== 0, " ✎")) changed = true;

    // ALBABIT-FIX: duration_seconds had no live frame-count feedback while
    // typing in Auto mode (video_frames stays hidden then). Previews the
    // model_type-aligned result in the label, for every VIDEO_MODEL_TYPES
    // entry. durSecW.callback (above) corrects the value itself to match.
    const isAutoSecActive = durSecW && frameModeW && frameModeW.value === "Auto (Seconds)"
        && VIDEO_MODEL_TYPES_JS.has(modelTypeW?.value);
    if (isAutoSecActive) {
        const { frames } = _autoSecondsFrames(modelTypeW.value, durSecW.value, frameRateW?.value);
        if (_setLabelMarker(durSecW, true, ` → ${frames}f`)) changed = true;
    } else if (_setLabelMarker(durSecW, false, "")) {
        changed = true;
    }

    if (changed) node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "FXTD.Radiance.Resolution",
    // FIX 6: beforeRegisterNodeDef is the correct hook — targets our node only,
    // widgets exist immediately in onNodeCreated, no timing races.
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceResolution") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            const find = (name) => this.widgets?.find(w => w.name === name);

            const enableVideoW  = find("enable_video");
            const videoFramesW  = find("video_frames");
            const frameRateW    = find("frame_rate");
            const batchSizeW    = find("batch_size");
            const mpTargetW     = find("mp_target");
            const mpAspectW     = find("mp_aspect_ratio");
            // ALBABIT-FIX: Restored from previous radiance version — frame computation
            // mode (Manual frames vs Auto seconds) toggle.
            const frameModeW    = find("frame_computation");
            const durSecW       = find("duration_seconds");
            // ALBABIT-FIX: widgets needed for preset auto-fill
            const presetW       = find("preset");
            const widthW        = find("width");
            const heightW       = find("height");
            const modelTypeW    = find("model_type");
            const scaleFactorW  = find("scale_factor");

            const toggleFields = () => {
                if (!enableVideoW) return;

                // ALBABIT-FIX: include integer 1/0 — Nodes 2.0 may store toggle values as 0/1
                const isVideo = enableVideoW.value === true
                    || enableVideoW.value === 1
                    || enableVideoW.value === "true"
                    || enableVideoW.value === "True";

                // ALBABIT-FIX: Auto (Seconds) hides video_frames in favor of duration_seconds
                const isAutoSec = frameModeW && frameModeW.value === "Auto (Seconds)";

                // ALBABIT-FIX: pass node (this) so setWidgetVisible can splice the widgets array
                // ALBABIT-FIX: MiniMax H3 has no variable-frame-rate support (see
                // _isMiniMaxH3 above). Hidden rather than left showing a value
                // the model can't actually use.
                const isMiniMaxModel = _isMiniMaxH3(modelTypeW?.value);

                setWidgetVisible(frameModeW,   isVideo, this);
                setWidgetVisible(videoFramesW, isVideo && !isAutoSec, this);
                setWidgetVisible(durSecW,      isVideo && isAutoSec, this);
                setWidgetVisible(frameRateW,   isVideo && !isMiniMaxModel, this);
                setWidgetVisible(batchSizeW,   !isVideo, this);

                // FEATURE: mp_aspect_ratio only visible when mp_target > 0
                if (mpTargetW) {
                    const mpActive = parseFloat(mpTargetW.value) > 0;
                    setWidgetVisible(mpAspectW, mpActive, this);
                }

                refreshNodeSize(this);
            };

            // ALBABIT-FIX follow-up: auto-toggle enable_video when model_type
            // switches to/from a video model (mirrors VIDEO_MODEL_TYPES in resolution.py).
            if (modelTypeW && enableVideoW) {
                const orig = modelTypeW.callback;
                modelTypeW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    const isVideoModel = VIDEO_MODEL_TYPES_JS.has(modelTypeW.value);
                    if (enableVideoW.value !== isVideoModel) {
                        enableVideoW.value = isVideoModel;
                        if (enableVideoW.callback) enableVideoW.callback.call(enableVideoW, isVideoModel);
                    }

                    // Match the width/height +/- step to this model_type's alignment
                    // (32 for LTXV, 16 for Flux.2, 8 default) so it always lands on
                    // a valid value, then instantly reflect the alignment this
                    // model_type will enforce, instead of waiting for onExecuted.
                    _syncStepsToModelType(modelTypeW, widthW, heightW);
                    _applyAlignment(node, modelTypeW, widthW, heightW);
                    _syncVideoFramesStep(modelTypeW, videoFramesW);
                    _syncDurationSecondsStep(modelTypeW, durSecW, frameRateW);

                    // Snap video_frames to a valid frame count for the new model_type.
                    _snapVideoFramesToGrid(modelTypeW, videoFramesW);

                    _forceMiniMaxFrameRate(modelTypeW, frameRateW);

                    toggleFields();
                };
            }

            // Wire callbacks
            if (enableVideoW) {
                const orig = enableVideoW.callback;
                enableVideoW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    toggleFields();
                };
            }

            // ALBABIT-FIX: Restored from previous radiance version. Also syncs
            // video_frames <-> duration_seconds when switching modes, so the
            // hidden widget's value reflects what the other mode last computed
            // instead of going stale.
            if (frameModeW) {
                const orig = frameModeW.callback;
                frameModeW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    if (frameModeW.value === "Manual (Frames)" && durSecW && videoFramesW) {
                        const { frames } = _autoSecondsFrames(modelTypeW?.value, durSecW.value, frameRateW?.value);
                        videoFramesW.value = frames;
                    } else if (frameModeW.value === "Auto (Seconds)" && durSecW && videoFramesW) {
                        // ALBABIT-FIX: 2 decimals, not 1. 1 wasn't precise enough to
                        // round-trip stably (e.g. MiniMax's 107f rounded to 4.5s, which
                        // re-aligns to 124f, not 107f). See _autoSecondsFrames above.
                        const { fps } = _autoSecondsFrames(modelTypeW?.value, durSecW.value, frameRateW?.value);
                        durSecW.value = Math.round((videoFramesW.value / fps) * 100) / 100;
                    }

                    toggleFields();
                };
            }

            // ALBABIT-FIX: duration_seconds had no callback at all, so typing a new
            // value while already in Auto (Seconds) mode never showed the PRECISE
            // resulting duration, only the frame count via the label preview above.
            // Corrects the typed value to the exact duration the aligned frame count
            // will actually produce.
            if (durSecW) {
                const orig = durSecW.callback;
                durSecW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    // ALBABIT-FIX: now that _syncDurationSecondsStep keeps
                    // options.round grid-aligned, a +/- click already lands
                    // correctly by the time orig.apply() above returns, so
                    // this rarely has to do anything. Left as a fallback for
                    // values that aren't grid-aligned yet (typed or pasted
                    // directly, or a model_type switch landing mid-grid).
                    if (frameModeW?.value === "Auto (Seconds)" && VIDEO_MODEL_TYPES_JS.has(modelTypeW?.value)) {
                        const { frames, fps } = _autoSecondsFrames(modelTypeW.value, durSecW.value, frameRateW?.value);
                        const precise = Math.round((frames / fps) * 100) / 100;
                        if (videoFramesW) videoFramesW.value = frames;
                        if (parseFloat(durSecW.value) !== precise) {
                            durSecW.value = precise;
                            if (durSecW.inputEl) durSecW.inputEl.value = precise;
                        }
                    }
                };
            }

            // ALBABIT-FIX: instantly snap video_frames to a valid N*stride+1 value for
            // video models (4k+1 for WAN/HunyuanVideo, 8k+1 for LTXV) — mirrors the
            // Python-side WAN warning/suggestion in generate(), but applied live and
            // for all VIDEO_MODEL_TYPES.
            if (videoFramesW) {
                const orig = videoFramesW.callback;
                videoFramesW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    _snapVideoFramesToGrid(modelTypeW, videoFramesW);
                };
            }

            if (mpTargetW) {
                const orig = mpTargetW.callback;
                mpTargetW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    toggleFields();
                };
            }

            // ALBABIT-FIX: presets are now plain Cinema/Social resolutions,
            // model-agnostic. Set width/height as an immediate visual baseline; the
            // final model_type-aligned values are synced back via onExecuted below.
            if (presetW && widthW && heightW) {
                const orig = presetW.callback;

                presetW.callback = function () {
                    if (orig) orig.apply(this, arguments);

                    if (presetW.value === "Custom") return;

                    // Extract dimensions from "... (WxH)"
                    const match = presetW.value.match(/\((\d+)[x×](\d+)\)/);
                    if (match) {
                        widthW.value  = parseInt(match[1], 10);
                        heightW.value = parseInt(match[2], 10);
                        if (widthW.inputEl)  widthW.inputEl.value  = widthW.value;
                        if (heightW.inputEl) heightW.inputEl.value = heightW.value;
                        // New unaligned base resolution — model_type alignment realigns from this.
                        node._resBaseW = parseInt(match[1], 10);
                        node._resBaseH = parseInt(match[2], 10);
                        // Remember the preset's own (pre-alignment) resolution and name, so
                        // manual width/height edits can detect when they drift from it (or
                        // come back to it) and toggle 'preset' accordingly.
                        node._presetRawW = node._resBaseW;
                        node._presetRawH = node._resBaseH;
                        node._lastPresetName = presetW.value;
                    }

                    // Align to the current model_type immediately, so switching presets
                    // after picking a video model_type doesn't briefly show un-aligned values.
                    _applyAlignment(node, modelTypeW, widthW, heightW);

                    // Reset scale_factor to 1.0 — per-model spatial scale is now
                    // handled by SPATIAL_SCALE in resolution.py, not by this widget.
                    if (scaleFactorW) {
                        scaleFactorW.value = 1.0;
                    }

                    toggleFields();
                };
            }

            // Switches 'preset' to "Custom" if the user edits width/height away from
            // what the current preset (aligned for the current model_type) would
            // produce — e.g. typing 1000 while preset=HD 1080p/model_type=LTXV
            // snaps to 992 (still a valid 32px alignment) but no longer matches the
            // preset's own 1920x1088, so it's effectively a custom resolution now.
            const _flagCustomIfNotPreset = (val, rawBase, scale) => {
                if (presetW && presetW.value !== "Custom" && rawBase != null && val !== _alignUp(rawBase, scale)) {
                    presetW.value = "Custom";
                    if (presetW.callback) presetW.callback.call(presetW);
                }
            };

            // Reverse of the above: if the user edits width/height back to exactly
            // what the previously-selected preset (aligned for the current
            // model_type) would produce, switch 'preset' back to that preset.
            const _restorePresetIfMatching = () => {
                if (!presetW || presetW.value !== "Custom" || !node._lastPresetName) return;
                if (node._presetRawW == null || node._presetRawH == null) return;
                const scale = SPATIAL_SCALE_JS[modelTypeW?.value] || 8;
                const w = parseInt(widthW.value, 10);
                const h = parseInt(heightW.value, 10);
                if (w === _alignUp(node._presetRawW, scale) && h === _alignUp(node._presetRawH, scale)) {
                    presetW.value = node._lastPresetName;
                    if (presetW.callback) presetW.callback.call(presetW);
                }
            };

            // Track manual width/height edits (Custom preset) as the new alignment base,
            // so a later model_type change realigns from the user's intended value.
            // The +/- step is synced to the model_type's alignment (_syncStepsToModelType),
            // so +/- clicks always land on valid values; only direct typing can misalign.
            if (widthW) {
                const orig = widthW.callback;
                widthW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    const val = parseInt(widthW.value, 10);
                    node._resBaseW = val;
                    const scale = SPATIAL_SCALE_JS[modelTypeW?.value] || 8;
                    _flagCustomIfNotPreset(val, node._presetRawW, scale);
                    _restorePresetIfMatching();
                };
            }
            if (heightW) {
                const orig = heightW.callback;
                heightW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    const val = parseInt(heightW.value, 10);
                    node._resBaseH = val;
                    const scale = SPATIAL_SCALE_JS[modelTypeW?.value] || 8;
                    _flagCustomIfNotPreset(val, node._presetRawH, scale);
                    _restorePresetIfMatching();
                };
            }

            // Initial alignment base = whatever width/height are at node creation
            // (defaults, or values restored from a saved workflow via onConfigure below).
            node._resBaseW = widthW ? parseInt(widthW.value, 10) : undefined;
            node._resBaseH = heightW ? parseInt(heightW.value, 10) : undefined;
            _syncStepsToModelType(modelTypeW, widthW, heightW);
            _syncVideoFramesStep(modelTypeW, videoFramesW);
            _syncDurationSecondsStep(modelTypeW, durSecW, frameRateW);

            // ALBABIT-FIX: Defer initial toggleFields 100ms so Vue completes its first layout
            // pass before any widget is hidden. If we hide immediately, widget.computedHeight is
            // undefined — the restore path falls back to 32, creating ghost space on re-show.
            setTimeout(() => toggleFields(), 100);

            // Poll for manual edits (onExecuted bypasses callbacks entirely).
            node._resMarkerInterval = setInterval(() => updateResolutionMarkers(node), 250);
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                if (node._resMarkerInterval) clearInterval(node._resMarkerInterval);
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            return r;
        };

        // ALBABIT-FIX: Re-apply toggle state after a saved workflow restores widget values.
        // onNodeCreated fires BEFORE ComfyUI deserialises the JSON, so a node saved with
        // enable_video=true loads stuck-collapsed without this hook.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            const reapply = () => {
                const enableVideoW = self.widgets?.find(w => w.name === "enable_video");
                const mpTargetW    = self.widgets?.find(w => w.name === "mp_target");
                const videoFramesW = self.widgets?.find(w => w.name === "video_frames");
                const frameRateW   = self.widgets?.find(w => w.name === "frame_rate");
                const batchSizeW   = self.widgets?.find(w => w.name === "batch_size");
                const mpAspectW    = self.widgets?.find(w => w.name === "mp_aspect_ratio");
                const frameModeW   = self.widgets?.find(w => w.name === "frame_computation");
                const durSecW      = self.widgets?.find(w => w.name === "duration_seconds");
                const widthW       = self.widgets?.find(w => w.name === "width");
                const heightW      = self.widgets?.find(w => w.name === "height");
                const modelTypeW   = self.widgets?.find(w => w.name === "model_type");
                const presetW      = self.widgets?.find(w => w.name === "preset");
                if (!enableVideoW) return;

                // Re-derive the alignment base from the restored width/height values
                // (onNodeCreated ran before deserialization, so its base was the default).
                if (widthW)  self._resBaseW = parseInt(widthW.value, 10);
                if (heightW) self._resBaseH = parseInt(heightW.value, 10);
                if (presetW && presetW.value !== "Custom") {
                    const match = presetW.value.match(/\((\d+)[x×](\d+)\)/);
                    if (match) {
                        self._presetRawW = parseInt(match[1], 10);
                        self._presetRawH = parseInt(match[2], 10);
                        self._lastPresetName = presetW.value;
                    }
                }
                _syncStepsToModelType(modelTypeW, widthW, heightW);
                _syncVideoFramesStep(modelTypeW, videoFramesW);
                _syncDurationSecondsStep(modelTypeW, durSecW, frameRateW);

                const isVideo = enableVideoW.value === true || enableVideoW.value === 1;
                const mpActive = mpTargetW ? parseFloat(mpTargetW.value) > 0 : false;
                const isAutoSec = frameModeW && frameModeW.value === "Auto (Seconds)";
                const isMiniMaxModel = _isMiniMaxH3(modelTypeW?.value);

                // ALBABIT-FIX: a workflow saved with a stale frame_rate under
                // MiniMax H3 would otherwise load hidden-but-wrong.
                _forceMiniMaxFrameRate(modelTypeW, frameRateW);

                setWidgetVisible(frameModeW,   isVideo, self);
                setWidgetVisible(videoFramesW, isVideo && !isAutoSec, self);
                setWidgetVisible(durSecW,      isVideo && isAutoSec, self);
                setWidgetVisible(frameRateW,   isVideo && !isMiniMaxModel, self);
                setWidgetVisible(batchSizeW,   !isVideo, self);
                setWidgetVisible(mpAspectW,    mpActive, self);
                refreshNodeSize(self);
            };
            requestAnimationFrame(reapply);
            setTimeout(reapply, 250);
        };

        // ALBABIT-FIX: sync width/height widgets to the final
        // model_type-aligned values computed by generate() (always >= preset
        // values, rounded up — see resolution.py _align_up).
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            const newW = message?.computed_width?.[0];
            const newH = message?.computed_height?.[0];
            const widthW  = this.widgets?.find((wg) => wg.name === "width");
            const heightW = this.widgets?.find((wg) => wg.name === "height");

            if (newW != null && widthW) {
                widthW.value = newW;
                if (widthW.inputEl) widthW.inputEl.value = newW;
            }
            if (newH != null && heightW) {
                heightW.value = newH;
                if (heightW.inputEl) heightW.inputEl.value = newH;
            }
            this.setDirtyCanvas?.(true, true);
        };
    }
});
