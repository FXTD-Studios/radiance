import { app } from "../../scripts/app.js";

import {
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "number" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    return _setWidgetVisible(widget, visible, node, { fallbackType: "number" });
}

const PRESET_CONFIGS = {
    "→ Flux txt2img": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Standard Flux text-to-image. Optimal for 1024×1024 images.",
    },
    "→ Flux img2img": {
        steps: 20, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 0.75, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Image-to-image refinement. denoise=0.75 for balanced changes.",
    },
    "→ Flux Inpaint": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "Inpainting with guidance=4.0 for strong detail matching.",
    },
    "→ Flux High-Res Fix": {
        steps: 20, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 0.5, flux_shift: 3.0, flux_guidance: 3.5,
        description: "2× upscale. shift=3.0 enhances high-frequency detail.",
    },
    "→ Flux Fast (12 steps)": {
        steps: 12, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Quick generation for prompt testing and iteration.",
    },
    "→ Flux Quality (28 steps)": {
        steps: 28, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "Maximum quality final outputs. Higher guidance for adherence.",
    },
    "→ Flux Cinematic (30 steps)": {
        steps: 30, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "Cinema-grade 30-step Flux run with strong guidance.",
    },
    "→ Flux Schnell (4 steps)": {
        steps: 4, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 0.0,
        description: "Flux Schnell distilled model — 4 steps, no guidance needed.",
    },
    "→ SD3.5 Turbo (4 steps)": {
        steps: 4, cfg: 1.6, sampler: "euler", scheduler: "sgm_uniform",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 0.0,
        description: "SD3.5 Turbo — 4-step distilled, CFG=1.6.",
    },
    "→ Flux Ultra Fast (8 steps)": {
        steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 2.0,
        description: "8-step Flux with reduced guidance for fast drafts.",
    },
    "▶ WAN txt2vid (30 steps)": {
        steps: 30, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 0.0,
        description: "WAN text-to-video. Uses the model's native schedule without an extra shift.",
    },
    "▶ WAN img2vid (20 steps)": {
        steps: 20, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 0.75, flux_shift: 1.0, flux_guidance: 0.0,
        description: "WAN image-to-video. 20 steps at denoise=0.75.",
    },
    "▶ LTX-Video (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 2.37, flux_guidance: 0.0,
        description: "LTX-V standard — shift=2.37 per spec.",
    },
    "▶ LTX 2.3 LowRes (20 steps)": {
        steps: 20, start_step: 0, end_step: 0, cfg: 3.0, sampler: "euler",
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "simple", // ALBABIT-FIX: simple matches LTXVScheduler linspace base; beta was incorrect
        scheduler_mode: "Manual", denoise: 1.0, flux_shift: 3.0,
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true,
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed",
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false,
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian",
        conditioning_clip_target: "Auto",
        tile_mode: false, refiner_start_step: 0, latent_format: "",
        terminal_sigma_to_zero: true, force_exact_steps: true,
        description: "LTX 2.3 LowRes. Optimal settings for 720p base generation.",
    },
    "▶ LTX 2.3 HighRes (40 steps)": {
        // ALBABIT-FIX: cfg=1 matches old Radiance and skips the negative-prompt forward pass
        steps: 40, start_step: 0, end_step: 0, cfg: 1.0, sampler: "euler",
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "simple", // ALBABIT-FIX: idem
        scheduler_mode: "Manual", denoise: 0.45, flux_shift: 6.0,
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true,
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed",
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false,
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian",
        conditioning_clip_target: "Auto",
        tile_mode: false, refiner_start_step: 0, latent_format: "",
        terminal_sigma_to_zero: true, force_exact_steps: true,
        description: "High-Res upscale. Uses Euler by default. If using a LoRA, adjust denoise as needed.",
    },
    // ALBABIT-FIX: same steps/cfg/denoise/flux_shift as the LTX 2.3 pair --
    // Dev has no official Comfy-Org template to source from. sampler is the
    // one field NOT mirrored: verified against the official 2.5 T2V template
    // (KSamplerSelect nodes), which uses euler_ancestral for both stages.
    "▶ LTX 2.5 LowRes (20 steps)": {
        steps: 20, start_step: 0, end_step: 0, cfg: 3.0, audio_cfg: 0.0, sampler: "euler_ancestral",
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "simple",
        scheduler_mode: "Manual", denoise: 1.0, flux_shift: 3.0,
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true,
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed",
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false,
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian",
        conditioning_clip_target: "Auto",
        tile_mode: false, refiner_start_step: 0, latent_format: "",
        terminal_sigma_to_zero: true, force_exact_steps: true,
        description: "LTX 2.5 (Dev) LowRes. Same base settings as LTX 2.3 LowRes — 20-step base generation.",
    },
    "▶ LTX 2.5 HighRes (40 steps)": {
        steps: 40, start_step: 0, end_step: 0, cfg: 1.0, audio_cfg: 0.0, sampler: "euler_ancestral",
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "simple",
        scheduler_mode: "Manual", denoise: 0.45, flux_shift: 6.0,
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true,
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed",
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false,
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian",
        conditioning_clip_target: "Auto",
        tile_mode: false, refiner_start_step: 0, latent_format: "",
        terminal_sigma_to_zero: true, force_exact_steps: true,
        description: "LTX 2.5 (Dev) HighRes upscale. Same base settings as LTX 2.3 HighRes — will likely move to Manual Sigmas later.",
    },
    "▶ HunyuanVideo (30 steps)": {
        steps: 30, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 7.0, flux_guidance: 0.0,
        description: "HunyuanVideo — shift=7, CFG=6.",
    },
    // ALBABIT-FIX: matches the official T2V template's KSamplerSelect
    // (res_multistep) + BasicScheduler (simple, 20 steps). cfg=1.0 is inert
    // (BasicGuider has no cfg input at all). The widget itself hides for
    // this model_type, see resolveModelType()/hiddenNames below.
    "▶ MiniMax H3 T2V (20 steps)": {
        steps: 20, cfg: 1.0, sampler: "res_multistep", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 0.0,
        description: "MiniMax H3 text-to-video. BasicGuider pipeline, cfg has no effect.",
    },
    "◈ Draft (4-step / AYS)": {
        steps: 4, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Ultra-fast AYS draft. Great for rough composition checks.",
    },
    "◈ Fast (8-step / AYS)": {
        steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "AYS 8-step — good quality/speed balance.",
    },
    "◈ Balanced (20-step)": {
        steps: 20, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 3.5,
        description: "Default 20-step production run.",
    },
    "◈ Quality (35-step)": {
        steps: 35, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.0, flux_guidance: 4.0,
        description: "High-quality 35-step with Phase-Shift SGM in the backend.",
    },
    "◈ Cinema (60-step)": {
        steps: 60, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 1.5, flux_guidance: 4.5,
        description: "Cinema-grade 60-step with Phase-Shift SGM. Maximum fidelity.",
    },
    // ALBABIT-FIX: cfg 1.0->4.0, sampler euler->res_multistep -- same
    // official-workflow evidence as MODEL_TYPE_SAMPLING_DEFAULTS.z_image
    // above (this manual preset was equally stale, not just the
    // auto-detected defaults).
    "◈ z_image (25 steps)": {
        steps: 25, cfg: 4.0, sampler: "res_multistep", scheduler: "simple",
        denoise: 1.0, flux_shift: 3.0, flux_guidance: 3.5,
        description: "z_image / Lumina variant — shift=3, CFG=4.",
    },
    "◈ Lumina2 (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 6.0, flux_guidance: 3.5,
        description: "Lumina2 — shift=6, guidance=3.5.",
    },
};

const LTX_PRESETS = [
    "▶ LTX 2.3 LowRes (20 steps)",
    "▶ LTX 2.3 HighRes (40 steps)",
    "▶ LTX 2.5 LowRes (20 steps)",
    "▶ LTX 2.5 HighRes (40 steps)"
];

// Model taxonomy — mirrors sampler_utils.py so the UI folds the same way the
// backend resolves models. GUIDANCE_EMBED models use flux_guidance; CFG_GUIDED
// models drive denoising with plain CFG and ignore the guidance-embed widgets.
// ALBABIT-FIX: "sd35" renamed "sd3.5". flux2/flux2-klein embed guidance like
// flux; lumina2 and z_image run a plain KSampler cfg in their official templates.
// tests/test_sampler_js_mirror.py keeps these sets equal to sampler_utils.py's.
const GUIDANCE_EMBED_MODELS = new Set(["flux", "flux2", "flux2-klein", "ltxv", "longcat_image"]);
// ALBABIT-FIX: "sd15" renamed "sd1.5". lumina2, z_image and wan_ti2v run a real
// KSampler cfg with no guidance-embed node in their official workflows.
const CFG_GUIDED_MODELS = new Set([
    "wan", "wan_ti2v", "hunyuan_video", "sdxl", "sd1.5", "sd3", "sd3.5",
    "ltxav", "cogvideox", "lumina2", "z_image", "mochi",
    "qwen_image", "krea2", "hunyuan_image", "hunyuan_video_15",
    "hidream", "omnigen2", "kandinsky5", "kandinsky5_image", "qwen_image21",
]);
const LTX_MODEL_TYPES = new Set(["ltxv", "ltxav"]);

// ALBABIT-FIX: mirrors sampler_utils.py's VIDEO_MODEL_TYPES -- used to
// filter the "Phase-Shift" sampler_mode options (see PHASE_SHIFT_MODES
// below), which nodes_sampler.py silently falls back to Standard for.
const VIDEO_MODEL_TYPES = new Set([
    "wan", "wan_ti2v", "ltxv", "ltxav", "hunyuan_video", "cosmos", "cogvideox", "mochi",
    "minimax", "hunyuan_video_15", "kandinsky5",
]);

// ALBABIT-FIX: mirrors sampler_utils.py's SamplerMode names, to filter the
// sampler_mode combo (3.5f in applyFolding): Phase-Shift is a no-op on video
// models, CFG++ at cfg 1 (it interpolates cfg toward 1.0).
const PHASE_SHIFT_MODES = new Set(["Phase-Shift (Euler >> DPM)", "Phase-Shift (Euler >> SGM)"]);
const CFG_PLUS_PLUS_MODE = "CFG++ (Perpendicular)";

// ALBABIT-FIX: mirrors sampler_utils.py's AYS_ANCHORS coverage (5 direct
// entries + aliases) -- everything NOT in this set silently falls through
// to the standard sigma computation when ays_schedule=True, no warning.
const AYS_SUPPORTED_MODELS = new Set([
    "sdxl", "sd1.5", "flux", "sd3", "wan", "ltxv",
    "sd3.5", "chroma", "hunyuan_video", "lumina2", "z_image",
]);

// ALBABIT-FIX: PAG hooks the "middle_block" attention layer (sampler_utils.py's
// pag_attention_patch checks block_type=="middle") -- a U-Net-only concept,
// no effect at all for DiT architectures (verified in apply_pag_to_model()).
const PAG_SUPPORTED_MODELS = new Set(["sdxl", "sd1.5"]);

// Infer the effective model type when model_type is left on "auto" by reading
// the chosen preset name. Returns "auto" when nothing matches (treated as a
// generic flux-style flow model — guidance widgets stay visible).
// The dropdown values come from the backend (WORKFLOW_PRESETS) as "[F] …",
// "[V] …", "[Q] …", but the PRESET_CONFIGS table above is keyed with "→/▶/◈"
// markers. Match by the label after the marker so either naming resolves to the
// same config — without this, selecting a preset silently applies nothing.
function normPresetKey(name) {
    return String(name || "")
        .replace(/^\s*\[[A-Za-z]\]\s*/, "")
        .replace(/^\s*[^A-Za-z0-9]+\s*/, "")
        .trim()
        .toLowerCase();
}
function getPresetConfig(name) {
    if (!name) return null;
    if (PRESET_CONFIGS[name]) return PRESET_CONFIGS[name];
    const target = normPresetKey(name);
    for (const k in PRESET_CONFIGS) {
        if (normPresetKey(k) === target) return PRESET_CONFIGS[k];
    }
    return null;
}

function resolveModelType(presetVal, modelTypeVal) {
    // ALBABIT-FIX: prioritise preset name FIRST — it is always more reliable than
    // modelTypeVal, which may carry a stale backend default ("ltxav") from an earlier
    // LTX workflow, causing Flux/WAN presets to be falsely classified as isLTX and
    // hiding flux_guidance / tile widgets even for non-LTX presets.
    const p = (presetVal || "").toLowerCase();
    if (LTX_PRESETS.includes(presetVal) || p.includes("ltx 2.3") || p.includes("ltx 2.5")) return "ltxav";
    if (p.includes("ltx"))      return "ltxv";
    if (p.includes("wan"))      return "wan";
    if (p.includes("hunyuan"))  return "hunyuan_video";
    if (p.includes("minimax"))  return "minimax";
    if (p.includes("z_image"))  return "z_image";
    if (p.includes("lumina"))   return "lumina2";
    // ALBABIT-FIX: return "sd3.5" (canonical form, matches Loader/detect.py)
    if (p.includes("sd3.5") || p.includes("sd35")) return "sd3.5";
    if (p.includes("flux") || p.includes("schnell") || p.includes("draft") ||
        p.includes("fast")    || p.includes("balanced") || p.includes("quality") ||
        p.includes("cinema")  || p.includes("txt2img")  || p.includes("img2img") ||
        p.includes("inpaint") || p.includes("high-res")) return "flux";
    // Preset name didn't match a known model family: fall back to the widget value
    if (modelTypeVal && modelTypeVal !== "auto") return modelTypeVal;
    return "auto";
}

const LTX_INCOMPATIBLE_WIDGETS = [
    "flux_guidance",
    "flux_guidance_profile",
    "preview_method",
    "tile_mode",
    "tile_size",
    "tile_overlap",
    "tile_stride",
    "tile_blend",
    // ALBABIT-FIX: hide widgets that have no effect under LTX — single unified encoder,
    // no AYS table, no PAG self-attention. guidance_rescale_phi was here too but that
    // predated the CFG-function contract fix; live-tested afterward and confirmed
    // working (consistent saturation drop, no artifacts) -- removed from this list.
    "conditioning_clip_target",
    "ays_schedule",
    "pag_scale"
];

// ALBABIT-FIX: widgets that become inert when an active sigmas_override is connected.
// start_step/end_step are intentionally excluded — in v3 they still control the
// sigmas_remaining slice window even when the override is active.
const SIGMA_OVERRIDE_WIDGETS = [
    "steps", "denoise", "scheduler", "scheduler_mode", "flux_shift",
    "terminal_sigma_to_zero", "ays_schedule", "custom_ays_anchors", "force_exact_steps",
];


// ── 2. Resize and redraw helper ──
function refreshNodeSize(node) {
    if (!node.computeSize) return;

    const sz = node.computeSize();
    const newWidth = Math.max(node.size[0], sz[0]);
    const newHeight = sz[1];
    // ALBABIT-FIX: skip if unchanged -- applyFolding() calls this on every
    // poll tick now, and reassigning size even when identical was another
    // contributor to the typing-interruption bug (see applyFolding).
    if (node.size[0] === newWidth && node.size[1] === newHeight) return;
    // ALBABIT-FIX: node.setSize(...) is the API Vue's resize handling actually
    // observes; raw node.size[i] mutation has zero visual effect.
    node.setSize([newWidth, newHeight]);
    app.graph.setDirtyCanvas(true, true);
}

// ── 3. Dynamic folding logic ──
// Self-heal + diagnostic. A widget marked visible (widget.hidden === false) but
// still typed "hidden" means a restore was missed by the frontend — force it back
// so parameters can never silently vanish. Set window.__RADIANCE_SAMPLER_DEBUG = true
// in the browser console to log the folded state on every toggle.
function auditSamplerWidgets(node) {
    if (!node.widgets) return;
    const stuck = node.widgets
        .filter(w => w && w.name && w.hidden !== true && w.type === "hidden")
        .map(w => w.name);
    if (stuck.length) {
        stuck.forEach(name => {
            const w = node.widgets.find(x => x.name === name);
            setWidgetVisible(w, true, node);
        });
        refreshNodeSize(node);
        console.warn("[Radiance Sampler] self-healed stranded widgets:", stuck.join(", "));
    }
    if (window.__RADIANCE_SAMPLER_DEBUG) {
        const hidden = node.widgets.filter(w => w && (w.hidden || w.type === "hidden")).map(w => w.name);
        const pv = (node.widgets.find(w => w.name === "preset") || {}).value;
        console.debug("[Radiance Sampler] preset=%s | hidden=[%s]", pv, hidden.join(", "));
    }
}

function toggleFields(node) {
    if (!node.widgets) return;
    // ALBABIT-FIX: must run before applyFolding() -- it writes the resolved
    // model_type that applyFolding()'s architecture-aware folding reads, so
    // folding first left every model-dependent show/hide one cycle stale.
    updateModelMetaDefaults(node);
    applyFolding(node);
    auditSamplerWidgets(node);
    // ALBABIT-FIX: after Vue processes this render cycle, clear the synthetic
    // computedHeight = 32 we set so Vue can recompute the real per-widget height.
    // The 32 was needed as a safe initial value for the _forceWidgetReinsert mount;
    // after Vue's first layout pass the value would be wrong if the real height ≠ 32.
    setTimeout(() => {
        if (!node.widgets) return;
        let changed = false;
        node.widgets.forEach(w => {
            if (!w.options?.hidden && !w.hidden && w.type !== "hidden") {
                if (w.computedHeight === 32) {
                    delete w.computedHeight;
                    changed = true;
                }
            }
        });
        if (changed) node.widgets.splice(0, 0);
    }, 50);
}

function applyFolding(node) {
    if (!node.widgets) return;

    const find = (name) => node.widgets.find(w => w.name === name);

    // Get key widget references
    const presetW = find("preset");
    const presetVal = presetW ? presetW.value : "Auto";
    const isCustom = presetVal === "Custom";

    // Dummy compatibility absorbers that are always hidden
    const dummyWidgets = ["_js_export_btn", "_js_import_btn", "_js_preset_info"];

    // ALBABIT-FIX: "Auto" (formerly "None") no longer hides everything -- it
    // falls through to the same smart-fold branch as named presets below,
    // just without hardcoded values (those come live from
    // updateModelMetaDefaults instead).

    // ALBABIT-FIX: build the hidden set first and apply it once. Showing all then
    // re-hiding remounted each folded widget's Vue component on every poll tick,
    // which interrupted typing; the steady state now changes nothing.
    const hiddenNames = new Set(dummyWidgets);

    // ── Custom: full manual control → everything visible, only tile sub-options follow tile_mode ──
    if (isCustom) {
        hiddenNames.add("preset_info");
        const tileModeW = find("tile_mode");
        const isTiled = tileModeW && tileModeW.value === true;
        if (!isTiled) {
            hiddenNames.add("tile_size");
            hiddenNames.add("tile_overlap");
            hiddenNames.add("tile_blend");
        }
        // "control_after_generate" is never added to hiddenNames, so this
        // loop alone already leaves it visible.
        let visChanged = false;
        node.widgets.forEach(w => {
            if (setWidgetVisible(w, !hiddenNames.has(w.name), node)) visChanged = true;
        });
        // ALBABIT-FIX: restore the full sampler_mode option list -- Custom
        // means no restrictions, even if Auto previously filtered it down
        // for a video model / cfg==1.0 (see 3.5f below).
        const samplerModeW = find("sampler_mode");
        if (samplerModeW?._radOrigOptions) {
            const currentValues = samplerModeW.options?.values || [];
            if (currentValues.length !== samplerModeW._radOrigOptions.length) {
                samplerModeW.options.values = samplerModeW._radOrigOptions.slice();
                _forceWidgetReinsert(samplerModeW, node);
            }
        }
        // ALBABIT-FIX: only resize on an actual visibility transition --
        // resizing every poll tick disrupted in-progress typing.
        if (visChanged) refreshNodeSize(node);
        return;
    }

    // ── Preset/Auto: compute the smart, model-aware hidden set ──
    const tileModeW = find("tile_mode");
    const restartCountW = find("restart_count");
    const aysScheduleW = find("ays_schedule");
    const modelTypeW = find("model_type");
    const samplerModeW = find("sampler_mode");

    // Check optional link states using node.inputs
    const hasRefinerModel = node.inputs && node.inputs.some(i => i.name === "refiner_model" && i.link !== null);
    const hasSdrReference = node.inputs && node.inputs.some(i => i.name === "sdr_reference" && i.link !== null);

    const modelType = modelTypeW ? modelTypeW.value : "auto";
    const samplerMode = samplerModeW ? samplerModeW.value : "Standard";

    // Resolve the effective model so folding matches what the backend will run.
    const effectiveModel = resolveModelType(presetVal, modelType);
    const isLTX = LTX_MODEL_TYPES.has(effectiveModel) || LTX_PRESETS.includes(presetVal);
    const sdTurboActive = _isSdTurboActive(node);

    // 3.1. Refiner: visible if refiner_model input port is wired up
    if (!hasRefinerModel) hiddenNames.add("refiner_start_step");

    // 3.1b. SDR reference conditioning: visible if sdr_reference input port is
    // wired up (sdr_blend/inject_steps/decay are entirely gated on
    // sdr_reference+sdr_vae in nodes_sampler.py, inert without it).
    if (!hasSdrReference) {
        hiddenNames.add("sdr_blend");
        hiddenNames.add("sdr_inject_steps");
        hiddenNames.add("sdr_decay");
    }

    // 3.2. Tiled latent sampling: visible if tile_mode is checked
    const isTiled = tileModeW && tileModeW.value === true;
    if (!isTiled) {
        hiddenNames.add("tile_size");
        hiddenNames.add("tile_overlap");
        hiddenNames.add("tile_blend");
    }

    // 3.3. Restart schedules: visible if restart_count > 0
    const hasRestartCount = restartCountW && parseInt(restartCountW.value, 10) > 0;
    if (!hasRestartCount) {
        hiddenNames.add("noise_alpha_start");
        hiddenNames.add("noise_alpha_end");
    }

    // 3.4. Sigma blend steps: visible if Phase-Shift sampler_mode OR ays_schedule
    // is active. phase_split: only meaningful in a Phase-Shift sampler_mode.
    const isPhaseShift = samplerMode.includes("Phase-Shift");
    const isAys = aysScheduleW && aysScheduleW.value === true;
    if (!(isPhaseShift || isAys)) hiddenNames.add("sigma_blend_steps");
    if (!isPhaseShift) hiddenNames.add("phase_split");

    // 3.5. Model-aware shift/guidance folding.
    //  - flux_shift (flow-match shift) is a no-op for SDXL/SD1.5 (ddpm noise
    //    schedule, no flow-matching) -- hidden for those, shown otherwise
    //    (including "auto"/unresolved, same show-by-default bias as guidance below).
    //  - flux_guidance / profile only apply to guidance-embed models; CFG-guided
    //    models (WAN, Hunyuan, SDXL, SD1.5/3/3.5, LTX-AV, CogVideoX) ignore them.
    const usesGuidanceEmbed =
        GUIDANCE_EMBED_MODELS.has(effectiveModel) ||
        (effectiveModel === "auto" && !CFG_GUIDED_MODELS.has(effectiveModel));
    const usesFlowShift = effectiveModel !== "sdxl" && effectiveModel !== "sd1.5";
    if (!usesFlowShift) hiddenNames.add("flux_shift");
    if (!usesGuidanceEmbed) {
        hiddenNames.add("flux_guidance");
        hiddenNames.add("flux_guidance_profile");
    }

    // 3.5b. PAG / AYS: narrow architecture support (verified against
    // sampler_utils.py's actual hook conditions, not just naming) -- hidden
    // unless the loaded model is confirmed to support them.
    if (!PAG_SUPPORTED_MODELS.has(effectiveModel)) hiddenNames.add("pag_scale");
    if (!AYS_SUPPORTED_MODELS.has(effectiveModel)) hiddenNames.add("ays_schedule");

    // 3.5c. Guidance rescale only has an effect when cfg > 1.0 (nodes_sampler.py
    // gates it on that exact condition) -- moot for guidance-embed models,
    // whose cfg is pinned at 1.0 by design.
    // ALBABIT-FIX: gated on the live cfg too: LongCat embeds guidance and still
    // runs a real cfg (4), where the rescale does apply.
    if (usesGuidanceEmbed && Number(find("cfg")?.value) <= 1) hiddenNames.add("guidance_rescale_phi");

    // 3.5c-2. MiniMax H3's reference pipeline uses BasicGuider, which has no
    // cfg input at all. Unlike guidance-embed models (flux_guidance stands
    // in for it), there's no alternate widget either, so cfg just hides.
    if (effectiveModel === "minimax") {
        hiddenNames.add("cfg");
        hiddenNames.add("guidance_rescale_phi");
    }

    // 3.5d. SDXL Turbo's discrete schedule (get_sd_turbo_sigmas) ignores
    // scheduler/scheduler_mode/terminal_sigma_to_zero/force_exact_steps
    // entirely, and its cfg is pinned at 1.0 so guidance_rescale_phi's own
    // "cfg > 1.0" gate never fires -- hide all five rather than showing
    // values that silently do nothing.
    if (sdTurboActive) {
        hiddenNames.add("scheduler");
        hiddenNames.add("scheduler_mode");
        hiddenNames.add("terminal_sigma_to_zero");
        hiddenNames.add("force_exact_steps");
        hiddenNames.add("guidance_rescale_phi");
    }

    // 3.5e. LTX models can't use the flux-style guidance / tiling / preview widgets.
    if (isLTX) {
        LTX_INCOMPATIBLE_WIDGETS.forEach(name => hiddenNames.add(name));
    }

    // 3.5f. audio_cfg (LTX 2.5 dual-CFG) is default-hidden -- shown for the
    // two dedicated LTX 2.5 presets, or under "Auto" when model_meta detects
    // an LTX 2.5 filename. LTX 2.3 keeps plain cfg (no separate audio scale).
    const showAudioCfg = presetVal.toLowerCase().includes("ltx 2.5")
        || (presetVal === "Auto" && _isAutoDetectedLtx25(node));
    if (!showAudioCfg) hiddenNames.add("audio_cfg");

    // ── Apply the final state in one pass (preset_info / control_after_generate
    // are never added to hiddenNames, so they stay visible automatically) ──
    let visChanged = false;
    node.widgets.forEach(w => {
        if (setWidgetVisible(w, !hiddenNames.has(w.name), node)) visChanged = true;
    });

    // 3.5g. sampler_mode combo: filter out individual choices that are dead
    // for the current state, rather than hiding the whole widget (Standard
    // and the Phase-Shift options remain meaningful for most models).
    // Mutates the combo's own option list -- a different mechanism from
    // setWidgetVisible, needed because these are choices inside one dropdown.
    if (samplerModeW) {
        if (!samplerModeW._radOrigOptions) {
            samplerModeW._radOrigOptions = (samplerModeW.options?.values || []).slice();
        }
        const cfgIsOne = Number(find("cfg")?.value) === 1;
        const isVideoModel = VIDEO_MODEL_TYPES.has(effectiveModel);
        const allowedModes = samplerModeW._radOrigOptions.filter(m => {
            if (PHASE_SHIFT_MODES.has(m) && isVideoModel) return false;
            if (m === CFG_PLUS_PLUS_MODE && cfgIsOne) return false;
            return true;
        });
        const currentValues = samplerModeW.options?.values || [];
        const listChanged = currentValues.length !== allowedModes.length ||
            currentValues.some((v, i) => v !== allowedModes[i]);
        if (listChanged) {
            if (!samplerModeW.options) samplerModeW.options = {};
            samplerModeW.options.values = allowedModes;
            if (!allowedModes.includes(samplerModeW.value)) {
                samplerModeW.value = "Standard";
            }
            _forceWidgetReinsert(samplerModeW, node);
        }
    }

    // ALBABIT-FIX: update disabled state for sigmas_override-dependent widgets.
    updateSigmaLocks(node);
    // ALBABIT-FIX: only resize on an actual visibility transition -- resizing
    // every poll tick disrupted in-progress typing.
    if (visChanged) refreshNodeSize(node);
}

function updateUILocks(node, presetName) {
    if (!node.widgets) return;
    // ALBABIT-FIX: LTX_PRESETS holds the unicode-marker keys ("▶ LTX 2.3 …"),
    // but presetName here is the raw backend combo value ("[V] LTX 2.3 …"),
    // so this literal-equality check was always false -- dead code, no LTX
    // widget was ever actually locked. resolveModelType() already does the
    // real (substring-based) match; reuse it instead of a second stale check.
    const isLTX = resolveModelType(presetName, "auto") === "ltxav";
    const isCustom = presetName === "Auto" || presetName === "Custom";

    node.widgets.forEach((widget) => {
        if (widget.name === "preset" || widget.name === "preset_info") return;

        const wName = widget.name ? widget.name.toLowerCase() : "";
        const isTargetWidget = LTX_INCOMPATIBLE_WIDGETS.some(t => t.toLowerCase() === wName);

        if (isTargetWidget) {
            if (!isCustom && isLTX) {
                widget.disabled = true;
                if (widget.inputEl) {
                    widget.inputEl.disabled = true;
                    widget.inputEl.style.opacity = "0.4";
                    widget.inputEl.style.pointerEvents = "none";
                }
            } else {
                widget.disabled = false;
                if (widget.inputEl) {
                    widget.inputEl.disabled = false;
                    widget.inputEl.style.opacity = "1.0";
                    widget.inputEl.style.pointerEvents = "auto";
                }
            }
        }
    });

    node.setDirtyCanvas(true, true);
}

// ALBABIT-FIX: returns true when sigmas_override has an active (non-muted, non-bypassed) link.
function isSigmaOverrideActive(node) {
    const sigmasInput = node.inputs?.find(inp => inp.name === "sigmas_override");
    if (!sigmasInput || !sigmasInput.link) return false;
    const link = app.graph.links[sigmasInput.link];
    if (!link) return false;
    const originNode = app.graph.getNodeById(link.origin_id);
    // mode 2 = Muted, mode 4 = Bypassed — treat as inactive
    return originNode && originNode.mode !== 2 && originNode.mode !== 4;
}

// ALBABIT-FIX: disable/re-enable the widgets that become inert when sigmas_override is active.
// Uses the same disabled + inputEl styling as updateUILocks().
// start_step/end_step are NOT in the list — they still slice the override to produce sigmas_remaining.
function updateSigmaLocks(node) {
    if (!node.widgets) return;
    const locked = isSigmaOverrideActive(node);

    // ALBABIT-FIX: same bug class as setWidgetVisible -- this runs every 250ms
    // via the polling loop, and reassigning widget.disabled/inputEl styling
    // even when "locked" hasn't changed was enough to interrupt in-progress
    // typing in "steps"/"denoise"/"scheduler"/etc. (SIGMA_OVERRIDE_WIDGETS).
    // Skip entirely per-widget when already in the desired state.
    let changed = false;
    node.widgets.forEach(widget => {
        if (!SIGMA_OVERRIDE_WIDGETS.includes(widget.name)) return;
        if (widget.disabled === locked) return;
        widget.disabled = locked;
        changed = true;
        if (widget.inputEl) {
            widget.inputEl.disabled = locked;
            widget.inputEl.style.opacity = locked ? "0.4" : "1.0";
            widget.inputEl.style.pointerEvents = locked ? "none" : "auto";
        }
    });

    if (changed) node.setDirtyCanvas(true, true);
}

function applyPreset(node, presetName) {
    if (presetName === "Auto" || presetName === "Custom") return;

    const config = getPresetConfig(presetName);
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    // Apply values silently without triggering loops
    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    // ALBABIT-FIX: model_type is left untouched here now -- resolveModelType()
    // already derives the effective family from the preset name for widget
    // folding, so forcing it (old inferModelTypeForPreset) only mislabeled
    // every non-Flux.1 guidance-embedded model as literally "flux".

    // ALBABIT-FIX: widgets now match the preset again — clear any "✎" markers.
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

// ── Preset divergence markers ──
// ALBABIT-FIX: Python no longer forces preset values, so a widget whose value
// differs from the selected preset gets a "✎" on its label instead. Driven by
// the 250ms poll, it covers edits, undo/redo, imports and workflow loads.
const PRESET_MARKER_EXCLUDED = new Set([
    "seed", "control_after_generate", "description", "preset", "preset_info",
]);
const PRESET_MARKER = " ✎";

function presetValuesEqual(a, b) {
    if (typeof a === "number" || typeof b === "number") {
        const na = Number(a), nb = Number(b);
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return Math.abs(na - nb) < 1e-6;
    }
    return String(a) === String(b);
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = node.widgets.find(w => w.name === "preset");
    const presetVal = presetW ? presetW.value : "Auto";
    const config = (presetVal === "Auto" || presetVal === "Custom")
        ? null
        : getPresetConfig(presetVal);

    let changed = false;
    for (const w of node.widgets) {
        if (!w || !w.name) continue;
        // ALBABIT-FIX: skip widgets currently owned by the model_meta system
        // (🧲/its own ✎) -- see _markLinkedWidget(). Magnet takes priority.
        if (w._radMetaLinked) continue;
        let marked = false;
        if (config && config[w.name] !== undefined && !PRESET_MARKER_EXCLUDED.has(w.name)) {
            marked = !presetValuesEqual(w.value, config[w.name]);
        }
        // Never touch the label of a widget that was never marked, so the
        // default label (undefined → name is displayed) stays untouched.
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

// ALBABIT-FIX: Flux.2 Klein Base (~50 steps, guidance 4.0) and distilled (4
// steps, ~1.0) share one architecture; only the file name tells them apart.
// With model_meta wired to a Radiance Loader, its unet_name is read live, and
// the derived widgets get "🧲" (the Loader's convention), not "✎".
const LINKED_MARKER = " 🧲";

function _findModelMetaSourceNode(node) {
    const input = node.inputs?.find(i => i.name === "model_meta");
    if (!input || !input.link) return null;
    const link = app.graph.links[input.link];
    if (!link) return null;
    const originNode = app.graph.getNodeById(link.origin_id);
    if (!originNode || originNode.mode === 2 || originNode.mode === 4) return null;
    return originNode;
}

// ALBABIT-FIX: LTXVConcatAVLatent sits directly in front of latent_image on
// BOTH pipeline stages, not just HighRes, so a one-hop check always found it
// first and misclassified every stage as HighRes. LTXVLatentUpsampler is the
// only unambiguous HighRes signal; walk back a bounded number of hops through
// the video_latent chain looking for it (never audio_latent).
const LTX_AV_UPSCALE_STAGE_MAX_HOPS = 6;
function _nextLatentInputName(node) {
    if (!node.inputs) return null;
    if (node.inputs.some(i => i.name === "video_latent")) return "video_latent";
    const latentInput = node.inputs.find(i => i.type === "LATENT");
    return latentInput ? latentInput.name : null;
}
function _isLtxAvHighResStage(node) {
    let current = node;
    let inputName = "latent_image";
    for (let hop = 0; hop < LTX_AV_UPSCALE_STAGE_MAX_HOPS; hop++) {
        const input = current.inputs?.find(i => i.name === inputName);
        if (!input || !input.link) return false;
        const link = app.graph.links[input.link];
        if (!link) return false;
        const originNode = app.graph.getNodeById(link.origin_id);
        if (!originNode) return false;
        if (originNode.type === "LTXVLatentUpsampler") return true;
        current = originNode;
        inputName = _nextLatentInputName(current);
        if (!inputName) return false;
    }
    return false;
}

// ALBABIT-FIX: some checkpoints need settings their model_type's default lacks;
// only the file name tells them apart (checked against the model cards). "turbo"
// also needs detectedType (SDXL and SD3.5 Turbo differ). LTX 2.3 Dev/Distilled
// is left to the LTX 2.3 LowRes/HighRes presets: community values disagree.
function _deriveDistillationOverride(filename, detectedType) {
    if (!filename) return null;
    const f = filename.toLowerCase();
    if (f.includes("klein")) {
        return f.includes("base") ? { flux_guidance: 4.0, steps: 50 } : { flux_guidance: 1.0, steps: 4 };
    }
    if (f.includes("schnell")) return { flux_guidance: 0.0, steps: 4 };
    if (f.includes("krea")) return { flux_guidance: 4.5 };
    // ALBABIT-FIX: sampler verified against ComfyUI's own official SDXL Turbo
    // workflow (sdxlturbo_example.png) -- scheduler there is "SDTurboScheduler",
    // a dedicated node with no standard-scheduler equivalent, left unset.
    if (detectedType === "sdxl" && f.includes("turbo")) return { cfg: 1.0, steps: 1, sampler: "euler_ancestral" };
    // ALBABIT-FIX: cfg=1.6 (not the "pure" diffusers guidance_scale=0.0
    // translation) to match the Sampler's own pre-existing "[F] SD3.5 Turbo
    // (4 steps)" preset, already tuned in practice.
    if (detectedType === "sd3.5" && f.includes("turbo")) return { cfg: 1.6, steps: 4 };
    // ALBABIT-FIX: verified against Comfy-Org's official Z-Image Turbo
    // workflow template -- KSampler cfg=1/steps=8 (sampler stays
    // "res_multistep", inherited unchanged from MODEL_TYPE_SAMPLING_DEFAULTS
    // .z_image above, same for both Base and Turbo).
    if (detectedType === "z_image" && f.includes("turbo")) return { cfg: 1.0, steps: 8 };
    return null;
}

// ALBABIT-FIX: reuses the already-vetted "LTX 2.3/2.5 LowRes/HighRes" preset
// objects as the Auto-mode default, applied per-stage (see
// _isLtxAvHighResStage above) instead of a filename-based override.
function _resolveLtxAvStageDefaults(unetName, isHighRes) {
    if (!unetName) return null;
    const f = unetName.toLowerCase();
    const version = f.includes("2.5") ? "2.5" : f.includes("2.3") ? "2.3" : null;
    if (!version) return null;
    const key = isHighRes
        ? `▶ LTX ${version} HighRes (40 steps)`
        : `▶ LTX ${version} LowRes (20 steps)`;
    return PRESET_CONFIGS[key] ?? null;
}

// ALBABIT-FIX: mirrors config/model_map.py's CHECKPOINT_PRESETS[...]["model_type"]
// -- lets the Sampler resolve the Loader's architecture from its preset name
// alone, no execution needed. Must be kept in sync by hand (same pattern
// already used for GUIDANCE_EMBED_MODELS/CFG_GUIDED_MODELS above).
const LOADER_PRESET_MODEL_TYPE = {
    "Flux.1": "flux", "Flux.1 (Low VRAM)": "flux",
    "Chroma": "chroma",
    "SD3.5": "sd3.5",
    "SDXL": "sdxl", "SD 1.5": "sd1.5",
    "HunyuanVideo": "hunyuan_video",
    "Wan 2.1": "wan", "Wan 2.1 (Low VRAM)": "wan",
    "Wan 2.2": "wan", "Wan 2.2 (Low VRAM)": "wan", "Wan 2.2 TI2V": "wan_ti2v",
    "LTX Video": "ltxv", "LTX Video (Low VRAM)": "ltxv",
    "LTX Video 2.3": "ltxav", "LTX Video 2.3 (Low VRAM)": "ltxav",
    "LTX Video 2.5": "ltxav", "LTX Video 2.5 (Low VRAM)": "ltxav",
    "Cosmos World": "cosmos", "CogVideoX": "cogvideox", "Mochi": "mochi",
    "PixArt Sigma": "pixart", "AuraFlow": "aura_flow",
    "Lumina2": "lumina2", "Z-Image": "z_image",
    "MiniMax H3": "minimax", "MiniMax H3 (Low VRAM)": "minimax",
    "Qwen-Image 2.1": "qwen_image21", "Qwen-Image 2.1 (Low VRAM)": "qwen_image21",
};

// ALBABIT-FIX: mirrors sampler_utils.py's MODEL_DEFAULTS (test_sampler_js_mirror.py).
// "guidance" is the architecture fallback; a _deriveDistillationOverride() file
// name match (Klein, Schnell) takes priority, as on the Python side.
// Native shifts belong to the loaded model, not the extra-shift widget.
const MODEL_TYPE_SAMPLING_DEFAULTS = {
    // ALBABIT-FIX: steps=20 added to flux/flux2/flux2-klein, verified
    // against Comfy-Org's official Flux.1 Dev/Flux.2 Dev/Flux.2 Klein
    // workflow templates.
    flux:          { cfg: 1.0, sampler: "euler",    scheduler: "simple",      guidance: 3.5, steps: 20 },
    flux2:         { cfg: 1.0, sampler: "euler",    scheduler: "simple",      guidance: 4.0, steps: 20 },
    "flux2-klein": { cfg: 1.0, sampler: "euler",    scheduler: "simple",      guidance: 4.0, steps: 20 },
    // ALBABIT-FIX: cfg/scheduler/steps verified against lodestones' own
    // official Chroma1-HD ComfyUI workflow (cfg was 1.0, scheduler "simple" --
    // both wrong). "steps" is a generic fallback, new for this architecture.
    chroma:        { cfg: 3.8, sampler: "euler",    scheduler: "beta",        guidance: 0.0, steps: 26 },
    // ALBABIT-FIX: cfg 4.5->5.45, sampler dpmpp_2m->euler, steps=30 -- all
    // verified directly against the official SD3 Medium example workflow's
    // embedded JSON (sd3_simple_example.png, comfyanonymous/ComfyUI_examples).
    sd3:           { cfg: 5.45, sampler: "euler",   scheduler: "sgm_uniform", guidance: 0.0, steps: 30 },
    // ALBABIT-FIX: sampler euler (was dpmpp_2m) and steps 20 per Comfy-Org's
    // official SD3.5 Large workflow; cfg 4.0 checked in a real ComfyUI workflow.
    "sd3.5":       { cfg: 4.0, sampler: "euler",    scheduler: "sgm_uniform", guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: cfg 7.0->8.0, sampler dpmpp_2m->euler, scheduler
    // karras->normal, matching ComfyUI's own official SDXL example workflow.
    // steps=20 added, same file (base stage runs 0-20 of a nominal 25-step
    // schedule with the optional refiner stage disabled by default).
    sdxl:          { cfg: 8.0, sampler: "euler",    scheduler: "normal",      guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: cfg 7.0->8.0, sampler dpmpp_2m->euler, steps=20 -- all
    // verified against ComfyUI's own default startup workflow (the graph
    // shown on first launch). Key renamed "sd15" -> "sd1.5" -- same
    // rationale as "sd35" -> "sd3.5" (converges on the Loader form).
    "sd1.5":       { cfg: 8.0, sampler: "euler",    scheduler: "normal",      guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: euler -> uni_pc, confirmed by 2 official Comfy-Org
    // workflows (Wan 2.1 1.3B T2V and Wan 2.1 14B I2V 720P). steps=20
    // added, from the same 14B I2V workflow.
    wan:           { cfg: 6.0, sampler: "uni_pc",  scheduler: "simple",      guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: WAN 2.2 TI2V-5B -- verified against Comfy-Org's official
    // bundled "video_wan2_2_5B_ti2v.json" workflow. Same as "wan" except cfg
    // (5 vs 6, genuinely different).
    wan_ti2v:      { cfg: 5.0, sampler: "uni_pc",  scheduler: "simple",      guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: steps=30, upgraded to high confidence -- confirmed by
    // ComfyUI's own official LTX Video example workflow.
    ltxv:          { cfg: 1.0, sampler: "euler",    scheduler: "simple",      guidance: 3.5, steps: 30 },
    ltxav:         { cfg: 3.0, sampler: "euler",    scheduler: "beta",        guidance: 0.0 },
    // ALBABIT-FIX: steps=20, from the same official ComfyUI HunyuanVideo
    // workflow already used for shift/sampler/scheduler (Tencent's own CLI
    // README recommends 50 -- a divergence, not resolved here).
    hunyuan_video: { cfg: 6.0, sampler: "euler",    scheduler: "simple",      guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: official example workflow shows plain KSampler cfg=4, no
    // guidance-embed node -- cfg 1.0->4.0, sampler euler->res_multistep,
    // steps=25 added (matches the workflow; its own Note claims "36 steps"
    // as official but the saved workflow itself uses 25).
    lumina2:       { cfg: 4.0, sampler: "res_multistep", scheduler: "simple", guidance: 0.0, steps: 25 },
    // ALBABIT-FIX: per Comfy-Org's official Z-Image (Base) template: plain
    // KSampler cfg 4, res_multistep, 25 steps, no guidance-embed node, like
    // Lumina2. Turbo's 8 steps come from _deriveDistillationOverride().
    z_image:       { cfg: 4.0, sampler: "res_multistep", scheduler: "simple", guidance: 0.0, steps: 25 },
    // ALBABIT-FIX: steps=20, verified against ComfyUI's own official
    // Cosmos-1.0 7B example workflow.
    cosmos:        { cfg: 7.0, sampler: "euler",    scheduler: "simple",      guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: steps=50, verified against THUDM's official CogVideoX-5b
    // model card (cfg was already exact).
    cogvideox:     { cfg: 6.0, sampler: "euler",    scheduler: "simple",      guidance: 0.0, steps: 50 },
    // ALBABIT-FIX: steps=64, verified against Genmo's official Mochi 1
    // model card (cfg was already exact).
    mochi:         { cfg: 4.5, sampler: "euler",    scheduler: "simple",      guidance: 0.0, steps: 64 },
    // ALBABIT-FIX: previously fell back to "sd1.5" (cfg=7.0/dpmpp_2m/normal) --
    // verified against AuraFlow's own official ComfyUI workflow, which
    // contradicts all three. No shift node present (unlike Lumina2, which
    // reuses the same ModelSamplingAuraFlow node but at shift=6.0 -- confirmed
    // NOT applicable to AuraFlow's own workflow, checked directly).
    aura_flow:     { cfg: 3.48, sampler: "euler",   scheduler: "sgm_uniform", guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: used to fall back to "sd1.5". No official ComfyUI workflow
    // exists: cfg/sampler from community sources, steps 20 from the diffusers
    // pipeline default, scheduler left at the sd1.5 value.
    pixart:        { cfg: 4.5,  sampler: "dpmpp_2m", scheduler: "normal",     guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: mirrors sampler_utils.py's MODEL_DEFAULTS["minimax"].
    // Matches the official T2V template (KSamplerSelect=res_multistep,
    // BasicScheduler=simple/20 steps). cfg=1.0 is inert (BasicGuider has no
    // cfg input at all); the widget itself hides regardless, see applyFolding.
    minimax:       { cfg: 1.0,  sampler: "res_multistep", scheduler: "simple", guidance: 0.0, steps: 20 },
    // ALBABIT-FIX: mirrors sampler_utils.py's MODEL_DEFAULTS["qwen_image21"].
    qwen_image21:  { cfg: 1.0,  sampler: "euler",    scheduler: "simple",      guidance: 0.0, steps: 25 },
    // ALBABIT-FIX: the 3.5 families, same values as sampler_utils.py's MODEL_DEFAULTS.
    qwen_image:         { cfg: 2.5, sampler: "euler",           scheduler: "simple", guidance: 0.0, steps: 20 },
    krea2:              { cfg: 1.0, sampler: "euler",           scheduler: "simple", guidance: 0.0, steps: 8 },
    hunyuan_image:      { cfg: 3.5, sampler: "euler",           scheduler: "simple", guidance: 0.0, steps: 50 },
    hunyuan_video_15:   { cfg: 6.0, sampler: "euler",           scheduler: "simple", guidance: 0.0, steps: 20 },
    hidream:            { cfg: 5.0, sampler: "uni_pc",          scheduler: "simple", guidance: 0.0, steps: 50 },
    omnigen2:           { cfg: 5.0, sampler: "euler",           scheduler: "simple", guidance: 0.0, steps: 20 },
    longcat_image:      { cfg: 4.0, sampler: "euler",           scheduler: "simple", guidance: 4.0, steps: 20 },
    kandinsky5:         { cfg: 5.0, sampler: "euler_ancestral", scheduler: "beta",   guidance: 0.0, steps: 50 },
    kandinsky5_image:   { cfg: 3.5, sampler: "euler",           scheduler: "simple", guidance: 0.0, steps: 50 },
};

function _resolveLoaderModelType(loaderNode) {
    if (!loaderNode) return null;
    const presetVal = loaderNode.widgets?.find(w => w.name === "preset")?.value;
    // ALBABIT-FIX: "Flux.2"/"Flux.2 (Low VRAM)" cover Dev and Klein in one
    // preset (Auto-Detect tells them apart at execution time) -- resolve
    // here the same way, from the Loader's own unet_name, since the preset
    // name alone can't.
    if (presetVal === "Flux.2" || presetVal === "Flux.2 (Low VRAM)") {
        const unetName = loaderNode.widgets?.find(w => w.name === "unet_name")?.value || "";
        return unetName.toLowerCase().includes("klein") ? "flux2-klein" : "flux2";
    }
    if (presetVal && presetVal !== "Custom" && LOADER_PRESET_MODEL_TYPE[presetVal]) {
        return LOADER_PRESET_MODEL_TYPE[presetVal];
    }
    const modelType = loaderNode.widgets?.find(w => w.name === "model_type")?.value;
    return (modelType && modelType !== "Auto-Detect") ? modelType : null;
}

// ALBABIT-FIX: shared by updateModelMetaDefaults() (value sync) and
// applyFolding() (Auto visibility) -- mirrors nodes_sampler.py's
// use_sd_turbo_schedule. Re-resolves the Loader link/unet_name itself
// rather than caching -- cheap, and avoids relying on call order between
// the two functions (toggleFields() calls updateModelMetaDefaults() first).
function _isSdTurboActive(node) {
    const sourceNode = _findModelMetaSourceNode(node);
    const unetName = sourceNode?.widgets?.find(w => w.name === "unet_name")?.value ?? "";
    const detectedType = _resolveLoaderModelType(sourceNode);
    return detectedType === "sdxl" && unetName.toLowerCase().includes("turbo");
}

// ALBABIT-FIX: same re-resolve-don't-cache pattern as _isSdTurboActive above.
// Used to auto-show audio_cfg under "Auto" (no named preset) when model_meta
// is wired to a Loader with an LTX 2.5 filename -- named presets already
// handle their own visibility via presetVal, this only covers Auto.
function _isAutoDetectedLtx25(node) {
    const sourceNode = _findModelMetaSourceNode(node);
    const unetName = sourceNode?.widgets?.find(w => w.name === "unet_name")?.value ?? "";
    return unetName.toLowerCase().includes("2.5");
}

// ALBABIT-FIX: can't just check "is the widget still at its generic default"
// -- after the first auto-write the value IS the derived one, so a later
// Loader change would never re-apply. _radAutoValue tracks what WE last
// wrote instead; no prior tracking (fresh, or right after a named preset)
// is never "user touched", so it applies unconditionally.
function _syncAutoValue(widget, newValue) {
    if (!widget || newValue === undefined) {
        if (widget) widget._radAutoValue = undefined;
        return false;
    }
    const userTouched = widget._radAutoValue !== undefined && widget.value !== widget._radAutoValue;
    widget._radAutoValue = newValue;
    if (userTouched || widget.value === newValue) return false;
    widget.value = newValue;
    return true;
}

// ALBABIT-FIX: _radMetaLinked marks this widget as owned by the model_meta
// system (🧲 or its own ✎ divergence) so updatePresetDivergenceMarkers()
// leaves its label alone -- both systems write the same widget.label, and
// without an explicit flag, whichever ran last silently won regardless of
// which one was actually supposed to be authoritative.
function _markLinkedWidget(widget, linked, inSync) {
    if (!widget) return false;
    widget._radMetaLinked = linked;
    const markerText = linked ? (inSync ? LINKED_MARKER : PRESET_MARKER) : null;
    if (widget._radOrigLabel === undefined && !markerText) return false;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = markerText ? widget._radOrigLabel + markerText : widget._radOrigLabel;
    if (widget.label === wanted) return false;
    widget.label = wanted;
    return true;
}

// ALBABIT-FIX: extends the guidance/steps sync (above) to model_type/cfg/
// sampler/scheduler/denoise (plus, for LTX-AV, which Sampler
// stage this node is -- see _isLtxAvHighResStage). Gated on preset
// (Auto/Custom) only -- the per-field checks in _syncAutoValue() already
// protect any field the user deliberately set.
function updateModelMetaDefaults(node) {
    if (!node.widgets) return;
    const presetW = node.widgets.find(w => w.name === "preset");
    const presetVal = presetW ? presetW.value : "Auto";
    const eligible = presetVal === "Auto" || presetVal === "Custom";

    const sourceNode = eligible ? _findModelMetaSourceNode(node) : null;
    const unetName = sourceNode?.widgets?.find(w => w.name === "unet_name")?.value ?? null;
    const detectedType = _resolveLoaderModelType(sourceNode);
    const override = _deriveDistillationOverride(unetName, detectedType);
    const modelDefaults = MODEL_TYPE_SAMPLING_DEFAULTS[detectedType] ?? null;
    // ALBABIT-FIX: stage-aware LTX-AV defaults (see _isLtxAvHighResStage /
    // _resolveLtxAvStageDefaults above) take priority over the generic
    // ltxav MODEL_TYPE_SAMPLING_DEFAULTS entry when resolvable.
    const ltxavStage = detectedType === "ltxav"
        ? _resolveLtxAvStageDefaults(unetName, _isLtxAvHighResStage(node))
        : null;
    // ALBABIT-FIX: the scheduler widget's actual value is ignored server-side
    // for this case (a dedicated discrete schedule is used instead, see
    // get_sd_turbo_sigmas), so there's no specific value to sync it to, just
    // a link to flag (and, under Auto, a widget to hide -- see applyFolding).
    const sdTurboActive = eligible && _isSdTurboActive(node);

    // ALBABIT-FIX: only write model_type when the combo offers it: pixart and
    // aura_flow have defaults here but are not in sampler_utils.py's MODEL_TYPES,
    // and execution rejects a value that is "not in list".
    const modelTypeW = node.widgets.find(w => w.name === "model_type");
    const validModelType = (detectedType && modelTypeW?.options?.values?.includes(detectedType))
        ? detectedType : undefined;

    const pairs = [
        [modelTypeW, validModelType],
        [node.widgets.find(w => w.name === "flux_guidance"), override?.flux_guidance ?? ltxavStage?.flux_guidance ?? modelDefaults?.guidance],
        [node.widgets.find(w => w.name === "steps"), override?.steps ?? ltxavStage?.steps ?? modelDefaults?.steps],
        [node.widgets.find(w => w.name === "cfg"), override?.cfg ?? ltxavStage?.cfg ?? modelDefaults?.cfg],
        [node.widgets.find(w => w.name === "sampler"), override?.sampler ?? ltxavStage?.sampler ?? modelDefaults?.sampler],
        // ALBABIT-FIX: denoise isn't part of the generic per-architecture
        // tables (only meaningful for LTX-AV's two-stage LowRes/HighRes
        // split so far) -- undefined everywhere else, same as any other
        // unresolved field above.
        [node.widgets.find(w => w.name === "denoise"), ltxavStage?.denoise],
    ];

    let changed = false;
    for (const [widget, derivedVal] of pairs) {
        if (_syncAutoValue(widget, derivedVal)) changed = true;
        const linked = derivedVal !== undefined;
        const inSync = linked && widget && widget.value === derivedVal;
        if (_markLinkedWidget(widget, linked, inSync)) changed = true;
    }

    const schedulerW = node.widgets.find(w => w.name === "scheduler");
    if (sdTurboActive) {
        _syncAutoValue(schedulerW, undefined); // no value to track/force -- link only
        if (_markLinkedWidget(schedulerW, true, true)) changed = true;
    } else {
        const schedulerDefault = ltxavStage?.scheduler ?? modelDefaults?.scheduler;
        if (_syncAutoValue(schedulerW, schedulerDefault)) changed = true;
        const linked = schedulerDefault !== undefined;
        const inSync = linked && schedulerW && schedulerW.value === schedulerDefault;
        if (_markLinkedWidget(schedulerW, linked, inSync)) changed = true;
    }

    if (changed) node.setDirtyCanvas(true, true);
}

// Safely extract tracking values
function getTrackedState(node) {
    const state = {};
    if (!node.widgets) return state;
    const trackedFields = [
        "steps", "cfg", "sampler", "scheduler", "denoise",
        "flux_shift", "flux_guidance", "force_exact_steps",
        "terminal_sigma_to_zero"
    ];
    for (const w of node.widgets) {
        if (trackedFields.includes(w.name)) {
            state[w.name] = w.value;
        }
    }
    return state;
}

async function exportPreset(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    const exportFields = ["steps", "cfg", "sampler", "scheduler", "denoise", "flux_shift", "flux_guidance"];
    const settings = {};
    for (const widget of widgets) {
        if (exportFields.includes(widget.name)) {
            settings[widget.name] = widget.value;
        }
    }

    const presetName = await promptSamplerAction("Sampler Preset", "Enter a name for this sampler preset.", "My Sampler Preset", "Export");
    if (!presetName) return;

    const preset = {
        name: presetName,
        created: new Date().toISOString(),
        version: "1.0",
        settings,
    };

    const jsonStr = JSON.stringify(preset, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${presetName.replace(/[^a-z0-9]/gi, "_")}_sampler.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function showSamplerToast(message, tone = "info") {
    const toast = document.createElement("div");
    const toneColor = tone === "error" ? "#ff6b6b" : tone === "success" ? "#4cd964" : "#00a8ff";
    Object.assign(toast.style, {
        position: "fixed",
        left: "50%",
        bottom: "24px",
        zIndex: "10000",
        transform: "translateX(-50%) translateY(12px)",
        opacity: "0",
        maxWidth: "420px",
        padding: "10px 14px",
        color: "#f5f5f7",
        background: "rgba(18, 18, 24, 0.94)",
        border: `1px solid ${toneColor}55`,
        borderRadius: "8px",
        boxShadow: "0 12px 36px rgba(0,0,0,0.45)",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        fontSize: "12px",
        lineHeight: "1.35",
        pointerEvents: "none",
        transition: "opacity 160ms ease, transform 160ms ease",
    });
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        toast.style.opacity = "1";
        toast.style.transform = "translateX(-50%) translateY(0)";
    });

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(-50%) translateY(12px)";
        setTimeout(() => toast.remove(), 180);
    }, 3200);
}

function promptSamplerAction(titleText, message, defaultValue = "", confirmLabel = "Continue") {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        Object.assign(overlay.style, {
            position: "fixed",
            inset: "0",
            zIndex: "10001",
            display: "grid",
            placeItems: "center",
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(8px)",
        });

        const dialog = document.createElement("div");
        Object.assign(dialog.style, {
            width: "min(420px, calc(100vw - 32px))",
            padding: "18px",
            color: "#f5f5f7",
            background: "rgba(18,18,24,0.96)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "8px",
            boxShadow: "0 18px 60px rgba(0,0,0,0.65)",
            fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            fontSize: "13px",
        });

        const title = document.createElement("div");
        title.textContent = titleText;
        title.style.cssText = "font-weight:700;font-size:15px;margin-bottom:8px;";

        const copy = document.createElement("div");
        copy.textContent = message;
        copy.style.cssText = "color:#b8c0cc;margin-bottom:12px;line-height:1.45;";

        const input = document.createElement("input");
        input.type = "text";
        input.value = defaultValue;
        input.style.cssText = "width:100%;box-sizing:border-box;height:36px;margin-bottom:16px;border-radius:8px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);color:#f5f5f7;padding:0 10px;outline:none;";

        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;gap:10px;justify-content:flex-end;";

        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.textContent = "Cancel";
        cancel.style.cssText = "height:32px;padding:0 12px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.05);color:#f5f5f7;cursor:pointer;";

        const confirm = document.createElement("button");
        confirm.type = "button";
        confirm.textContent = confirmLabel;
        confirm.style.cssText = "height:32px;padding:0 12px;border-radius:6px;border:1px solid rgba(0,168,255,0.45);background:rgba(0,168,255,0.16);color:#9fdcff;cursor:pointer;font-weight:700;";

        const close = (value) => {
            overlay.remove();
            resolve(value);
        };

        cancel.addEventListener("click", () => close(null));
        confirm.addEventListener("click", () => close(input.value.trim()));
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") close(input.value.trim());
            if (event.key === "Escape") close(null);
        });
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(null);
        });

        actions.append(cancel, confirm);
        dialog.append(title, copy, input, actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        input.focus();
        input.select();
    });
}

function importPreset(node) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.onchange = (e) => {
        document.body.removeChild(fileInput);

        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const preset = JSON.parse(event.target.result);

                if (!preset.settings || typeof preset.settings !== "object") {
                    showSamplerToast("Invalid preset file: missing settings object.", "error");
                    return;
                }

                const widgets = node.widgets;
                let appliedCount = 0;

                for (const widget of widgets) {
                    if (preset.settings[widget.name] !== undefined) {
                        widget.value = preset.settings[widget.name];
                        appliedCount++;
                    }
                }

                const presetWidget = widgets.find(w => w.name === "preset");
                if (presetWidget) {
                    presetWidget.value = "Custom";
                    updateUILocks(node, "Custom");
                }

                node.setDirtyCanvas(true);
                showSamplerToast(`Preset "${preset.name || "Unnamed"}" imported. ${appliedCount} settings applied.`, "success");

            } catch (error) {
                console.error("[Radiance Sampler] Failed to import preset:", error);
                showSamplerToast(`Failed to import preset: ${error.message}`, "error");
            }
        };

        reader.readAsText(file);
    };

    fileInput.click();
}

app.registerExtension({
    name: "FXTD.RadianceSampler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceSamplerPro") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const self = this;

            this.addWidget("button", "› Export Preset", null, () => exportPreset(this), { serialize: false });
            this.addWidget("button", "› Import Preset", null, () => importPreset(this), { serialize: false });

            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (!presetWidget) return;

            let descWidget = this.widgets?.find(w => w.name === "preset_info");
            if (!descWidget) {
                descWidget = this.addWidget("text", "preset_info", "", () => { }, {
                    multiline: true,
                    serialize: false
                });
            }

            setTimeout(() => {
                if (descWidget && descWidget.inputEl) {
                    descWidget.inputEl.style.fontFamily = "monospace";
                    descWidget.inputEl.style.fontSize = "11px";
                    descWidget.inputEl.style.color = "#888";
                    descWidget.inputEl.style.fontStyle = "italic";
                    descWidget.inputEl.style.height = "60px";
                    descWidget.inputEl.readOnly = true;
                }
            }, 100);

            const updateDescription = (presetName) => {
                const config = getPresetConfig(presetName);
                let text = "Manual / Custom Mode. All widgets are unlocked.";
                if (presetName === "Auto") {
                    text = "Auto mode. Only the parameters that actually apply to the loaded model are shown, auto-filled from the Loader (🧲). Select 'Custom' to unlock every widget.";
                } else if (config && config.description) {
                    text = config.description;
                }

                if (descWidget) {
                    descWidget.value = text;
                    if (descWidget.inputEl) {
                        descWidget.inputEl.value = text;
                    }
                }
            };

            let lastPresetValue = presetWidget.value;

            // Handle Preset changes explicitly
            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);

                if (window.app && window.app.configuringGraph) return;

                if (value !== lastPresetValue && value !== "Auto" && value !== "Custom") {
                    lastPresetValue = value;
                    applyPreset(this, value);
                    updateUILocks(this, value);
                    updateDescription(value);
                    toggleFields(this);
                } else if (value !== lastPresetValue) {
                    lastPresetValue = value;
                    // ALBABIT-FIX: no longer resets model_type -- named presets
                    // don't force it anymore (see applyPreset), so there's
                    // nothing to undo when switching to Auto/Custom.
                    updateUILocks(this, value);
                    updateDescription(value);
                    toggleFields(this);
                }
            };

            // ALBABIT-FIX: no auto-switch to "Custom" on a widget edit: it hung on
            // onPropertyChanged, which LiteGraph never fires for widgets, and Custom
            // unfolds every widget. The "✎" markers show divergence instead.

            // Wire up callbacks for dynamic folding on change
            const foldTriggers = ["preset", "tile_mode", "restart_count", "ays_schedule", "model_type", "sampler_mode"];
            foldTriggers.forEach(name => {
                const w = self.widgets?.find(x => x.name === name);
                if (w) {
                    const origCallback = w.callback;
                    w.callback = function(...args) {
                        const res = origCallback ? origCallback.apply(this, args) : undefined;
                        toggleFields(self);
                        return res;
                    };
                }
            });

            // Hook connection change events (optional ports linked/unlinked)
            const origConnect = this.onConnectionsChange;
            this.onConnectionsChange = function (...args) {
                if (origConnect) origConnect.apply(this, args);
                // ALBABIT-FIX: deferred one tick -- calling toggleFields()
                // synchronously here (mid LiteGraph link-drag completion) could
                // leave its drag-state stuck ("Already dragging links." on the
                // next attempt). The 250ms poll below re-syncs regardless.
                setTimeout(() => toggleFields(this), 0);
            };

            // ALBABIT-FIX: polled, since onConnectionsChange misses upstream mute or
            // bypass and Loader-side edits (a new unet_name). Refreshes the "✎"
            // markers, the model_meta values and the folding.
            this._sigmaCheckInterval = setInterval(() => {
                updateSigmaLocks(self);
                updatePresetDivergenceMarkers(self);
                updateModelMetaDefaults(self);
                applyFolding(self);
                auditSamplerWidgets(self);
            }, 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._sigmaCheckInterval) {
                    clearInterval(self._sigmaCheckInterval);
                    self._sigmaCheckInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };

            // Initialize UI state immediately (safe — no widget visibility changes)
            const val = presetWidget.value;
            if (val) {
                lastPresetValue = val;
                updateUILocks(this, val);
                updateDescription(val);
            }

            // ALBABIT-FIX: _configuredByLoad is set in onConfigure (loaded workflow).
            // For a loaded node, onConfigure fires right after onNodeCreated with correct values;
            // its 150ms timer handles initial folding so we skip this one to avoid a race where
            // this fires while configure() is still running (large workflows take > 100ms).
            // For a freshly added node, onConfigure never fires, so this timer is the only one.
            const nodeRef = this;
            setTimeout(() => {
                if (nodeRef._configuredByLoad) return;
                const val = presetWidget.value;
                if (val) {
                    lastPresetValue = val;
                    updateUILocks(nodeRef, val);
                    updateDescription(val);
                }
                toggleFields(nodeRef);
            }, 150);
        };

        // Re-apply folding after a saved workflow restores this node. onNodeCreated
        // runs BEFORE ComfyUI deserializes widget values, so the preset value isn't
        // known at creation time; without this hook a node saved with a non-Auto
        // preset (or in Custom mode) could load stuck-collapsed.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            const self = this;
            // ALBABIT-FIX: signal to the onNodeCreated timer that onConfigure ran,
            // so the timer skips and avoids a potential race with in-progress configuration.
            self._configuredByLoad = true;
            const reapply = () => {
                const presetW = self.widgets?.find(w => w.name === "preset");
                if (presetW) updateUILocks(self, presetW.value);
                toggleFields(self);
                // ALBABIT-FIX: flag widgets already diverging from the preset in
                // the loaded workflow (e.g. cfg edited before saving).
                updatePresetDivergenceMarkers(self);
            };
            // ALBABIT-FIX: 150ms for Vue's first layout pass; 600ms safety net for
            // heavy workflows where graph.configure() stalls the main thread > 100ms.
            setTimeout(reapply, 150);
            setTimeout(reapply, 600);
        };

        // ALBABIT-FIX: show the values sample() actually used (MODEL_DEFAULTS or the
        // Flux.2 Klein refinement can adjust them), via the "ui" dict like
        // radiance_resolution.js's computed_width/height. A no-op when nothing changed.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            const sync = (name, msgKey) => {
                const val = message?.[msgKey]?.[0];
                const w = this.widgets?.find(wg => wg.name === name);
                if (val != null && w && w.value !== val) {
                    w.value = val;
                    if (w.inputEl) w.inputEl.value = val;
                }
            };
            sync("cfg", "resolved_cfg");
            sync("flux_guidance", "resolved_flux_guidance");
            sync("flux_shift", "resolved_flux_shift");
            sync("sampler", "resolved_sampler");
            sync("steps", "resolved_steps");
            this.setDirtyCanvas?.(true, true);
        };
    }
});

console.log("[Radiance Sampler] Extension loaded");
