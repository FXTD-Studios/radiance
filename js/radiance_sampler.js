import { app } from "../../../scripts/app.js";

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
        denoise: 1.0, flux_shift: 8.0, flux_guidance: 0.0,
        description: "WAN text-to-video. shift=8 is critical for correct temporal dynamics.",
    },
    "▶ WAN img2vid (20 steps)": {
        steps: 20, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 0.75, flux_shift: 8.0, flux_guidance: 0.0,
        description: "WAN image-to-video. 20 steps at denoise=0.75.",
    },
    "▶ LTX-Video (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 2.37, flux_guidance: 0.0,
        description: "LTX-V standard — shift=2.37 per spec.",
    },
    "▶ LTX 2.3 LowRes (32 steps)": {
        steps: 32, start_step: 0, end_step: 0, cfg: 3.0, sampler: "euler", 
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "beta", 
        scheduler_mode: "Manual", denoise: 1.0, flux_shift: 3.0, 
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true, 
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed", 
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false, 
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian", 
        multi_cond_mode: "Off", cond_weight_b: 0.0, conditioning_clip_target: "Auto", 
        tile_mode: false, refiner_start_step: 0, latent_format: "", 
        force_full_denoise_steps: true, force_exact_steps: true, terminal_sigma: 0.0,
        description: "LTX 2.3 LowRes.",
    },
    "▶ LTX 2.3 HighRes (40 steps)": {
        steps: 40, start_step: 0, end_step: 0, cfg: 3.0, sampler: "euler", 
        sampler_mode: "Standard", phase_split: 0.0, scheduler: "beta", 
        scheduler_mode: "Manual", denoise: 0.45, flux_shift: 6.0, 
        flux_guidance: 0.0, flux_guidance_profile: "Static", add_noise: true, 
        return_with_leftover_noise: false, seed: 0, control_after_generate: "fixed", 
        pag_scale: 0.0, model_type: "ltxav", sigma_blend_steps: 0, ays_schedule: false, 
        guidance_rescale_phi: 0.0, preview_method: "None", noise_type: "Gaussian", 
        multi_cond_mode: "Off", cond_weight_b: 0.0, conditioning_clip_target: "Auto", 
        tile_mode: false, refiner_start_step: 0, latent_format: "", 
        force_full_denoise_steps: true, force_exact_steps: true, terminal_sigma: 0.0,
        description: "High-Res upscale without LoRA. Uses Euler by default. If you're using a LoRA, you can plug in “sigmas_override” and adjust your settings accordingly.",
    },
    "▶ HunyuanVideo (30 steps)": {
        steps: 30, cfg: 6.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 7.0, flux_guidance: 0.0,
        description: "HunyuanVideo — shift=7, CFG=6.",
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
    "◈ z_image (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 3.0, flux_guidance: 3.5,
        description: "z_image / Lumina variant — shift=3.",
    },
    "◈ Lumina2 (25 steps)": {
        steps: 25, cfg: 1.0, sampler: "euler", scheduler: "simple",
        denoise: 1.0, flux_shift: 6.0, flux_guidance: 3.5,
        description: "Lumina2 — shift=6, guidance=3.5.",
    },
};

const LTX_PRESETS = [
    "▶ LTX 2.3 LowRes (32 steps)",
    "▶ LTX 2.3 HighRes (40 steps)"
];

const LTX_INCOMPATIBLE_WIDGETS = [
    "flux_guidance",
    "flux_guidance_profile",
    "preview_method",
    "tile_mode",
    "tile_size",
    "tile_overlap",
    "tile_stride",
    "tile_blend"
];

function updateUILocks(node, presetName) {
    if (!node.widgets) return;
    const isLTX = LTX_PRESETS.includes(presetName);
    const isCustom = presetName === "None (Custom)";

    node.widgets.forEach((widget) => {
        if (widget.name === "preset" || widget.name === "preset_info" || widget.name === "cond_weight_b") return;

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

    const multiCondWidget = node.widgets.find(w => w.name === "multi_cond_mode");
    if (multiCondWidget && multiCondWidget.callback) {
        multiCondWidget.callback(multiCondWidget.value);
    }

    node.setDirtyCanvas(true, true);
}

function applyPreset(node, presetName) {
    if (presetName === "None (Custom)") return;

    const config = PRESET_CONFIGS[presetName];
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
            if (widget.callback) widget.callback(config[widget.name]);
        }
    }

    node.setDirtyCanvas(true);
}

function exportPreset(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    const exportFields = ["steps", "cfg", "sampler", "scheduler", "denoise", "flux_shift", "flux_guidance"];
    const settings = {};
    for (const widget of widgets) {
        if (exportFields.includes(widget.name)) {
            settings[widget.name] = widget.value;
        }
    }

    const presetName = prompt("Enter preset name:", "My Sampler Preset");
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
                    alert("Invalid preset file: Missing settings object");
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
                    presetWidget.value = "None (Custom)";
                    updateUILocks(node, "None (Custom)");
                }

                node.setDirtyCanvas(true);
                alert(`Preset "${preset.name || "Unnamed"}" imported!\n${appliedCount} settings applied.`);

            } catch (error) {
                console.error("[Radiance Sampler] Failed to import preset:", error);
                alert("Failed to import preset: " + error.message);
            }
        };

        reader.readAsText(file);
    };

    fileInput.click();
}

app.registerExtension({
    name: "FXTD.RadianceSampler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;
        if (comfyClass !== "RadianceSamplerPro") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            this.addWidget("button", "› Export Preset", null, () => exportPreset(this));
            this.addWidget("button", "› Import Preset", null, () => importPreset(this));

            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (!presetWidget) return;

            const multiCondWidget = this.widgets?.find(w => w.name === "multi_cond_mode");
            const weightBWidget = this.widgets?.find(w => w.name === "cond_weight_b");

            if (multiCondWidget && weightBWidget) {
                const origMultiCb = multiCondWidget.callback;
                multiCondWidget.callback = function(val) {
                    if (origMultiCb) origMultiCb.apply(this, arguments);
                    
                    const disableWeight = (val === "Off");
                    weightBWidget.disabled = disableWeight;
                    
                    if (weightBWidget.inputEl) {
                        weightBWidget.inputEl.disabled = disableWeight;
                        weightBWidget.inputEl.style.opacity = disableWeight ? "0.4" : "1.0";
                        weightBWidget.inputEl.style.pointerEvents = disableWeight ? "none" : "auto";
                    }

                    if (disableWeight) {
                        if (window.app && !window.app.configuringGraph) weightBWidget.value = 0.0;
                    } else {
                        if (window.app && !window.app.configuringGraph && weightBWidget.value === 0.0) weightBWidget.value = 0.5;
                    }
                };
                setTimeout(() => multiCondWidget.callback(multiCondWidget.value), 150);
            }

            // ALBABIT-FIX: Replaced the floating div logic with a native embedded text widget
            // Added serialize: false to stop the bug where it gets duplicated on workflow reload
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
                const config = PRESET_CONFIGS[presetName];
                const text = (config && config.description) ? config.description : "Manual / Custom Mode. All widgets are unlocked.";

                if (descWidget) {
                    descWidget.value = text;
                    if (descWidget.inputEl) {
                        descWidget.inputEl.value = text;
                    }
                }
            };

            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);
                
                if (window.app && window.app.configuringGraph) {
                    updateUILocks(this, value);
                    updateDescription(value);
                    return;
                }

                setTimeout(() => {
                    applyPreset(this, value);
                    updateUILocks(this, value);
                    updateDescription(value);
                }, 10);
            };

            setTimeout(() => {
                const val = presetWidget.value;
                if (val) {
                    if (val !== "None (Custom)" && !(window.app && window.app.configuringGraph)) {
                        applyPreset(this, val);
                    }
                    updateUILocks(this, val);
                    updateDescription(val);
                }
            }, 100);
        };
    }
});

console.log("[Radiance Sampler] Extension loaded");
