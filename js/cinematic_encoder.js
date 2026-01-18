/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                    FXTD CINEMATIC ENCODER WIDGET
 *              Prompt Preview + Randomize Button for 10/10 UX
 *                        FXTD Studios © 2024-2026
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";

// Style presets list
const STYLE_PRESETS = [
    "None (Custom)",
    "🎬 Classic Hollywood",
    "🌃 Film Noir",
    "🚀 Sci-Fi Cinematic",
    "🌆 Cyberpunk",
    "🎭 Drama / Emotional",
    "🏔️ Epic Landscape",
    "👤 Portrait Pro",
    "📰 Documentary",
    "🎨 Artistic / Painterly",
    "📼 Retro VHS",
    "🌅 Golden Hour Magic",
    "🌙 Moody Night",
    "⚡ Action / Dynamic",
    "🎪 Wes Anderson",
    "🎞️ 1970s New Hollywood",
    "📼 1980s Retro Action",
    "📺 1990s Music Video",
    "📹 2000s Digital Look",
];

// Preset configurations (Synced with nodes_prompt.py)
const PRESET_CONFIGS = {
    "🎬 Classic Hollywood": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Panavision Panaflex Gold II (35mm)",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Paramount Lighting",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Technicolor (Vintage)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "🌃 Film Noir": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Film Noir Lighting",
        "style_aesthetic": "Monochrome Noir",
        "film_stock": "Kodak Tri-X 400 (B&W)",
        "color_grading": "Bleach Bypass (Gritty)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "🚀 Sci-Fi Cinematic": {
        "framing": "Extreme Wide Shot (EWS)",
        "camera_type": "ARRI Alexa 65 (IMAX)",
        "lens_focal": "ARRI Master Anamorphic",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Cinematic Haze / Volumetric Fog",
        "style_aesthetic": "Blade Runner Atmosphere",
        "film_stock": "None",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "🌆 Cyberpunk": {
        "framing": "Dutch Angle (Canted)",
        "camera_type": "Sony Venice 2",
        "lens_focal": "Anamorphic Lens",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Neon Cyberpunk Lighting",
        "style_aesthetic": "Cyberpunk 2077 Aesthetic",
        "film_stock": "Cinestill 800T",
        "color_grading": "Cyberpunk Neon Grading",
        "aspect_ratio": "21:9 (Ultrawide)",
    },
    "🎭 Drama / Emotional": {
        "framing": "Close-Up (CU)",
        "camera_type": "ARRI Alexa Mini LF",
        "lens_focal": "85mm Portrait Prime",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Rembrandt Lighting",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Desaturated (Muted)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "🏔️ Epic Landscape": {
        "framing": "Extreme Wide Shot (EWS)",
        "camera_type": "ARRI Alexa 65 (IMAX)",
        "lens_focal": "14mm Ultra-Wide Angle",
        "aperture_dof": "f/11 (Landscape Sharpness)",
        "lighting": "Golden Hour (Magic Hour)",
        "style_aesthetic": "National Geographic Style",
        "film_stock": "Fujifilm Velvia 50",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "21:9 (Ultrawide)",
    },
    "👤 Portrait Pro": {
        "framing": "Medium Close-Up (MCU)",
        "camera_type": "Sony A7S III",
        "lens_focal": "85mm Portrait Prime",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Editorial Photography",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "📰 Documentary": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Canon C700 FF",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Practical Lighting",
        "style_aesthetic": "Documentary Texture",
        "film_stock": "None",
        "color_grading": "Desaturated (Muted)",
        "aspect_ratio": "16:9 (Widescreen)",
    },
    "🎨 Artistic / Painterly": {
        "framing": "Medium Shot (MS)",
        "camera_type": "None",
        "lens_focal": "Petzval 85mm (Classic Swirl)",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Oil Painting (Classic)",
        "film_stock": "None",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "📼 Retro VHS": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Super 8mm Camera",
        "lens_focal": "50mm Standard Prime",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Practical Lighting",
        "style_aesthetic": "Vintage 1990s VHS",
        "film_stock": "Polaroid 600",
        "color_grading": "Cross Processed",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "🌅 Golden Hour Magic": {
        "framing": "Full Body Shot (Wide)",
        "camera_type": "Sony Venice 2",
        "lens_focal": "85mm Portrait Prime",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Golden Hour (Magic Hour)",
        "style_aesthetic": "Photorealistic (Raw)",
        "film_stock": "Kodak Ektar 100",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "16:9 (Widescreen)",
    },
    "🌙 Moody Night": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Sony A7S III",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/1.2 (Dreamy Bokeh)",
        "lighting": "Moonlight",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Cinestill 800T",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "⚡ Action / Dynamic": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "RED V-Raptor XL",
        "lens_focal": "24mm Wide Angle",
        "aperture_dof": "f/5.6 (Sharp Subject)",
        "lighting": "Harsh Sunlight",
        "style_aesthetic": "Hyper-Realism",
        "film_stock": "None",
        "shutter_speed": "1/1000th sec (Frozen Action)",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "🎪 Wes Anderson": {
        "framing": "Symmetrical Composition",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/8.0 (Deep Focus)",
        "lighting": "Soft Window Light",
        "style_aesthetic": "Wes Anderson Symmetric",
        "film_stock": "Kodak Portra 400",
        "color_grading": "Pastel Soft Tones",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "🎞️ 1970s New Hollywood": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Panavision Panaflex Gold II (35mm)",
        "lens_focal": "35mm Classic Wide",
        "aperture_dof": "f/2.8 (Cinematic Separation)",
        "lighting": "Practical Lighting",
        "style_aesthetic": "Cinematic Movie Still",
        "film_stock": "Kodak Vision3 500T",
        "color_grading": "Technicolor (Vintage)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "📼 1980s Retro Action": {
        "framing": "Low Angle (Hero Shot)",
        "camera_type": "ARRI Alexa 35",
        "lens_focal": "Anamorphic Lens",
        "aperture_dof": "f/4.0 (Balanced)",
        "lighting": "Cinematic Haze / Volumetric Fog",
        "style_aesthetic": "Hyper-Realism",
        "film_stock": "None",
        "color_grading": "Teal and Orange (Blockbuster)",
        "aspect_ratio": "2.39:1 (Anamorphic Scope)",
    },
    "📺 1990s Music Video": {
        "framing": "Extreme Close-Up (ECU)",
        "camera_type": "Super 8mm Camera",
        "lens_focal": "Fish-Eye Lens",
        "aperture_dof": "f/1.8 (Soft Background)",
        "lighting": "Neon Cyberpunk Lighting",
        "style_aesthetic": "Vintage 1990s VHS",
        "film_stock": "Polaroid 600",
        "color_grading": "Cross Processed",
        "aspect_ratio": "4:3 (Academy Ratio)",
    },
    "📹 2000s Digital Look": {
        "framing": "Medium Shot (MS)",
        "camera_type": "Sony A7S III",
        "lens_focal": "24mm Wide Angle",
        "aperture_dof": "f/5.6 (Sharp Subject)",
        "lighting": "Harsh Sunlight",
        "style_aesthetic": "Editorial Photography",
        "film_stock": "None",
        "color_grading": "Vibrant High Contrast",
        "aspect_ratio": "16:9 (Widescreen)",
    },
};

// Styles for Dynamic UI Feedback (Theme Color / Border)
// Format: [backgroundColor (transparent), borderColor, textColor]
const PRESET_STYLES = {
    "None (Custom)": ["rgba(0,0,0,0)", "rgba(100,100,100,0.3)", "#888"],
    "🎬 Classic Hollywood": ["rgba(255, 215, 0, 0.05)", "rgba(255, 215, 0, 0.4)", "#eecfa1"], // Gold
    "🌃 Film Noir": ["rgba(0, 0, 0, 0.2)", "rgba(180, 180, 180, 0.5)", "#cdcdcd"], // Mono
    "🚀 Sci-Fi Cinematic": ["rgba(0, 255, 255, 0.05)", "rgba(0, 255, 255, 0.4)", "#aaffff"], // Cyan
    "🌆 Cyberpunk": ["rgba(255, 0, 255, 0.05)", "rgba(255, 0, 255, 0.5)", "#ff77ff"], // Neon Pink
    "🎭 Drama / Emotional": ["rgba(100, 149, 237, 0.05)", "rgba(100, 149, 237, 0.4)", "#aaccff"], // Cornflower
    "🏔️ Epic Landscape": ["rgba(34, 139, 34, 0.05)", "rgba(50, 205, 50, 0.4)", "#90ee90"], // Green
    "👤 Portrait Pro": ["rgba(255, 182, 193, 0.05)", "rgba(255, 182, 193, 0.4)", "#ffb6c1"], // Pink
    "📰 Documentary": ["rgba(240, 230, 140, 0.05)", "rgba(218, 165, 32, 0.4)", "#f0e68c"], // Khaki
    "🎨 Artistic / Painterly": ["rgba(255, 99, 71, 0.05)", "rgba(255, 99, 71, 0.4)", "#ffa07a"], // Tomato
    "📼 Retro VHS": ["rgba(70, 130, 180, 0.05)", "rgba(138, 43, 226, 0.4)", "#dda0dd"], // Blue/Purple
    "🌅 Golden Hour Magic": ["rgba(255, 165, 0, 0.08)", "rgba(255, 140, 0, 0.5)", "#ffd700"], // Orange/Gold
    "🌙 Moody Night": ["rgba(25, 25, 112, 0.2)", "rgba(72, 61, 139, 0.5)", "#b0c4de"], // Dark Blue
    "⚡ Action / Dynamic": ["rgba(220, 20, 60, 0.1)", "rgba(255, 69, 0, 0.6)", "#ff6347"], // Red/Orange
    "🎪 Wes Anderson": ["rgba(255, 255, 224, 0.1)", "rgba(240, 128, 128, 0.5)", "#f08080"], // Pastel Yellow/Red
    "🎞️ 1970s New Hollywood": ["rgba(210, 180, 140, 0.1)", "rgba(139, 69, 19, 0.4)", "#deb887"], // Tan
    "📼 1980s Retro Action": ["rgba(0, 0, 139, 0.1)", "rgba(255, 0, 255, 0.4)", "#00ffff"], // Synthwave
    "📺 1990s Music Video": ["rgba(50, 205, 50, 0.1)", "rgba(255, 20, 147, 0.5)", "#7fff00"], // Lime/Pink
    "📹 2000s Digital Look": ["rgba(240, 248, 255, 0.1)", "rgba(70, 130, 180, 0.5)", "#f0f8ff"], // Digital Blue
};

const FRAMING = [
    "None", "Extreme Close-Up (ECU)", "Close-Up (CU)", "Medium Close-Up (MCU)",
    "Medium Shot (MS)", "Cowboy Shot (American Shot)", "Full Body Shot (Wide)",
    "Extreme Wide Shot (EWS)", "Establishing Shot", "Over-The-Shoulder (OTS)",
    "Point of View (POV)", "Low Angle (Hero Shot)", "High Angle (Vulnerability)",
    "Bird's Eye View (Overhead)", "Worm's Eye View", "Dutch Angle (Canted)",
    "Symmetrical Composition", "Rule of Thirds"
];

const CAMERAS = [
    "None", "ARRI Alexa 65 (IMAX)", "ARRI Alexa Mini LF", "ARRI Alexa 35",
    "Sony Venice 2", "Sony FX9", "Sony A7S III",
    "RED V-Raptor XL", "RED Komodo", "RED Monstro 8K VV",
    "Panavision Millennium DXL2", "Panavision Panaflex Gold II (35mm)",
    "IMAX 15/70mm Film Camera", "Bolex H16 (16mm Film)", "Super 8mm Camera"
];

const LIGHTING = [
    "None", "Rembrandt Lighting", "Chiaroscuro (High Contrast)", "Film Noir Lighting",
    "Split Lighting", "Butterfly Lighting", "Paramount Lighting",
    "Soft Window Light", "Golden Hour (Magic Hour)", "Blue Hour",
    "Cinematic Haze / Volumetric Fog", "God Rays (Crepuscular Rays)",
    "Neon Cyberpunk Lighting", "Practical Lighting", "Bioluminescence"
];

app.registerExtension({
    name: "FXTD.CinematicEncoder",

    async nodeCreated(node) {
        if (node.comfyClass !== "FXTDCinematicPromptEncoder") return;

        // Add custom widget for prompt preview
        const previewWidget = node.addWidget("text", "prompt_preview", "", () => { }, {
            multiline: true,
            inputEl: null,
        });
        previewWidget.computeSize = () => [node.size[0] - 20, 60];
        previewWidget.serializeValue = () => undefined; // Don't save this

        // Style the preview
        if (previewWidget.inputEl) {
            previewWidget.inputEl.readOnly = true;
            previewWidget.inputEl.style.background = "rgba(0, 168, 255, 0.1)";
            previewWidget.inputEl.style.border = "1px solid rgba(0, 168, 255, 0.3)";
            previewWidget.inputEl.style.color = "#aaccff";
            previewWidget.inputEl.style.fontSize = "10px";
            previewWidget.inputEl.style.fontFamily = "monospace";
        }

        // Randomize button
        node.addWidget("button", "🎲 Randomize Style", null, () => {
            randomizeSettings(node);
        });

        // Update preview when inputs change
        const originalOnPropertyChanged = node.onPropertyChanged;
        node.onPropertyChanged = function (property, value) {
            if (originalOnPropertyChanged) originalOnPropertyChanged.call(this, property, value);
            updatePreview(node, previewWidget);
        };

        // Also update on widget change
        const originalOnWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, old_value, widget) {
            if (originalOnWidgetChanged) originalOnWidgetChanged.call(this, name, value, old_value, widget);

            if (name === "style_preset") {
                // Apply preset values
                applyPreset(node, value);
            } else if (name !== "prompt_preview" && name !== "style_preset") {
                // Check if user manually changed something
                checkCustomOverride(node, name, value);
            }

            updatePreview(node, previewWidget);
        };

        // Initial update
        setTimeout(() => updatePreview(node, previewWidget), 100);
    }
});

// Global flag to prevent recursive updates
let IS_APPLYING_PRESET = false;

function applyPreset(node, presetName) {
    if (IS_APPLYING_PRESET) return;
    if (presetName === "None (Custom)" || !PRESET_CONFIGS[presetName]) return;

    IS_APPLYING_PRESET = true;

    const config = PRESET_CONFIGS[presetName];
    const widgets = node.widgets;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    IS_APPLYING_PRESET = false;
    node.setDirtyCanvas(true);
}

function checkCustomOverride(node, changedWidgetName, newValue) {
    if (IS_APPLYING_PRESET) return;

    const presetWidget = node.widgets.find(w => w.name === "style_preset");
    if (!presetWidget || presetWidget.value === "None (Custom)") return;

    // Check if the new value contradicts the current preset
    const currentPreset = presetWidget.value;
    const config = PRESET_CONFIGS[currentPreset];

    if (config && config[changedWidgetName] !== undefined) {
        if (config[changedWidgetName] !== newValue) {
            // User manually changed something that contradicts the preset
            // Switch preset to Custom
            presetWidget.value = "None (Custom)";
            node.setDirtyCanvas(true);
        }
    }
}

function randomizeSettings(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    // Smart Randomizer Logic
    // 50% chance to pick a curated style (coherent)
    // 50% chance to mix, but with some logic? For now, let's just make it purely style-based or pure random.
    // User requested "Smart". Best smart thing is to mostly pick presets.

    // Let's implement weighted random for Style Preset, and then auto-apply it.
    // This is much better than randomizing framing/camera independently which leads to "iPhone 1800s".

    // Find style preset widget
    const styleWidget = widgets.find(w => w.name === "style_preset");
    if (!styleWidget) return;

    const usePreset = Math.random() > 0.3; // 70% chance to pick a cohesive preset

    if (usePreset) {
        // Pick a random preset (not "None")
        const randomPreset = STYLE_PRESETS[Math.floor(Math.random() * (STYLE_PRESETS.length - 1)) + 1];
        styleWidget.value = randomPreset;
        // This will trigger onWidgetChanged -> applyPreset automatically!
    } else {
        // Mix and match (Legacy chaos mode, maybe good for inspiration)
        styleWidget.value = "None (Custom)";

        for (const widget of widgets) {
            if (widget.name === "style_preset") continue;

            // Skip text/numbers
            if (widget.type !== "COMBO" && !Array.isArray(widget.options?.values)) continue;

            // Get options
            let options = widget.options?.values || [];
            if (!options.length) {
                // Look for global constants if matched by name (hacky but works for this specifically)
                if (widget.name === "framing") options = FRAMING;
                else if (widget.name === "camera_type") options = CAMERAS;
                else if (widget.name === "lighting") options = LIGHTING;
            }

            if (options.length > 1) {
                widget.value = options[Math.floor(Math.random() * (options.length - 1)) + 1];
            }
        }
    }

    // Trigger update
    node.setDirtyCanvas(true);

    // Find and update preview
    const previewWidget = widgets.find(w => w.name === "prompt_preview");
    if (previewWidget) {
        updatePreview(node, previewWidget);
    }
}

function updatePreview(node, previewWidget) {
    if (!previewWidget) return;

    const widgets = node.widgets;
    if (!widgets) return;

    // Get current values
    let basePrompt = "", framing = "", camera = "", lighting = "", style = "";
    let lens = "", aperture = "", film = "";

    for (const widget of widgets) {
        switch (widget.name) {
            case "base_prompt": basePrompt = widget.value || "A cinematic scene..."; break;
            case "framing": framing = widget.value; break;
            case "camera_type": camera = widget.value; break;
            case "lighting": lighting = widget.value; break;
            case "style_aesthetic": style = widget.value; break;
            case "lens_focal": lens = widget.value; break;
            case "aperture_dof": aperture = widget.value; break;
            case "film_stock": film = widget.value; break;
        }
    }

    // Build preview
    let parts = [];

    // Framing + Subject
    if (framing && framing !== "None") {
        parts.push(`${framing} of ${basePrompt}`);
    } else {
        parts.push(basePrompt);
    }

    // Camera Tech
    let tech = [];
    if (camera && camera !== "None") tech.push(`Shot on ${camera}`);
    if (lens && lens !== "None") tech.push(`with ${lens}`);
    if (aperture && aperture !== "None") tech.push(`at ${aperture}`);

    if (tech.length > 0) parts.push(tech.join(" "));

    // Lighting/Style
    if (lighting && lighting !== "None") parts.push(`Lighting is ${lighting}`);
    if (style && style !== "None") parts.push(style);
    if (film && film !== "None") parts.push(`on ${film}`);

    const preview = parts.join(". ").substring(0, 300) + (parts.join(". ").length > 300 ? "..." : "");

    previewWidget.value = preview;

    // Dynamic Styling
    if (previewWidget.inputEl) {
        // Find style preset used (or custom)
        const presetUsed = widgets.find(w => w.name === "style_preset")?.value || "None (Custom)";
        const styles = PRESET_STYLES[presetUsed] || PRESET_STYLES["None (Custom)"];

        if (styles) {
            previewWidget.inputEl.style.backgroundColor = styles[0];
            previewWidget.inputEl.style.borderColor = styles[1];
            previewWidget.inputEl.style.color = styles[2];

            // Add subtle glow if fancy
            if (presetUsed.includes("Cyberpunk") || presetUsed.includes("Sci-Fi") || presetUsed.includes("Neon")) {
                previewWidget.inputEl.style.boxShadow = `0 0 10px ${styles[1]}`;
            } else {
                previewWidget.inputEl.style.boxShadow = "none";
            }
        }

        previewWidget.inputEl.value = preview;
    }

    node.setDirtyCanvas(true);
}
