"""
◎ Radiance Cinematic Prompt Encoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade cinematic prompt builder with direct CLIP/T5 encoding.
Auto-selects prose for Flux/T5 and structured keywords for SD1.5/SDXL.

v2.4.0 Changelog (from audit):
───────────────────────────────────────
CRITICAL FIXES:
  [BUG-C4] _build_prose_prompt: NeuralGrammar.scientificize() was applied
           AFTER _apply_subject_weight(), so the subject was already wrapped in
           "(text:1.25)" when scientificize tried to parse it. The parens were
           stripped, replacement words landed inside the brackets, and the weight
           syntax was corrupted. Fix: apply scientificize on the raw subject
           BEFORE weight wrapping.

IMPORTANT FIXES:
  [BUG-I9]  _detect_arch_from_clip: returned "flux" for ALL T5-based architectures
            (SD3, SD3.5, Wan, PixArt, LTX all mapped to "flux"). This caused
            negative prompt warnings to fire for every T5 model. Fix: use
            multi-key fingerprinting — SD3 has t5xxl + g + l simultaneously;
            Flux has t5xxl + l only; LTX/Wan have t5xxl alone. Added LTX
            detection via "gemma" / "ltxv" key patterns, and "ltxav" to PROSE_ARCHS.
  [BUG-I10] Auto context-window upgrade: if detected arch is T5/LLM-based and user
            left context_window at "Standard (CLIP 77)", silently upgrade to
            Medium (256) with a one-time warning. CLIP 77 silently truncates
            prose prompts, causing scrambled output that looks like model failure.
  [BUG-I11] Arch-aware negative auto-tune: architectures where negatives have
            near-zero effect (Flux, Wan, LTX, HunyuanVideo) now auto-downgrade
            negative_strength to "Soft" when user selects Standard/Aggressive,
            with a clear log warning. Previously the negatives were encoded at
            full strength, wasting token budget with no effect.

MINOR FIXES:
  [BUG-M7] enhance_prompt_grammar: dedup check "masterpiece" not in p.lower()
           fired for prose architectures even though prose injects completely
           different creative tags. Any prompt containing "masterpiece of X"
           would silently skip prose quality tag injection. Fix: only check
           "masterpiece" for non-prose archs; prose arch uses its own dedup key.
  [BUG-M8] _build_prose_prompt: parenthetical descriptors like
           "f/2.8 (Cinematic Separation)" and "50mm Standard Prime" read
           awkwardly in natural-language prose. New _clean_for_prose() strips
           or converts parentheticals to flowing language before T5 encoding.
  [BUG-M9] _build_prose_prompt: "Cinematic technique: X." pattern used a
           colon-prefixed header that reads stilted in natural prose. Replaced
           with architecture-natural "Photographed on X, through a Y, at Z."
           sentence structure that T5/LLM encoders process more naturally.

v2.3.3 Changelog (from review audit):
───────────────────────────────────────
CRITICAL FIXES:
  [BUG-C1] _encode() fallback path returned raw (cond, pooled) tuple instead of
           ComfyUI conditioning format [[cond, {"pooled_output": pooled}]].
           Sampler crashes on older ComfyUI installs (pre encode_from_tokens_scheduled).
  [BUG-C2] Structured path: "Est. Year {year_era}." had trailing period, then
           ", ".join(finish) + "." appended another → double period "..".
  [BUG-C3] Structured path: art_direction and lora_keywords both inserted at index 1,
           reversing their intended order (lora ended up before art_direction).

IMPORTANT FIXES:
  [BUG-I1] `subject_first` weight mode offered in INPUT_TYPES but had zero distinct
           handling — fell through to `balanced` silently. Now leads with subject
           at elevated weight before any technical descriptors.
  [BUG-I2] `sd3.5` missing from target_arch dropdown despite being in PROSE_ARCHS.
           Also added `hunyuan_video` which was in PROSE_ARCHS but missing from dropdown.
  [BUG-I3] Redundant `import torch as _torch` inside _real_token_count — torch is
           already imported at module level.
  [BUG-I4] DEFAULT_YEAR = 2024 was stale. Updated to 2025.
  [BUG-I5] use_break parameter type mismatch — string "On"/"Off" in encode_cinematic
           vs boolean False passed to build_cinematic_prompt_v3. Unified to boolean.
  [BUG-I6] Version string in DESCRIPTION said "v3.0" — updated to v2.3.3.
  [BUG-I7] clip.clone() not guarded — some custom CLIP wrappers lack .clone().
           Added try/except fallback.
  [BUG-I8] _validate_presets() only logged errors silently. Added strict mode
           that raises on validation failure when RADIANCE_STRICT env var is set.

MINOR FIXES:
  [BUG-M1] NeuralGrammar.scientificize could produce irregular spacing when
           punctuation-bearing words were replaced with multi-word phrases.
           Added post-cleanup pass.
  [BUG-M2] estimate_tokens didn't account for BREAK token overhead. Added
           BREAK_TOKEN_OVERHEAD constant used in insert_break_points.
  [BUG-M3] Structured path: tech block could duplicate camera when
           prompt_weight_mode was "subject_first" (new mode). Gated properly.
  [BUG-M4] _real_token_count: explicit tuple unpacking validation added.
  [BUG-M5] enhance_prompt_grammar: creative tags injected even when they were
           already present in the prompt — causing duplication on re-encodes.
           Added dedup check.
  [BUG-M6] Negative prompt for Anime style was missing "3d render" exclusion.
"""
import datetime
import functools
import json
import logging
import os
import re
import weakref
import torch
from typing import Optional

from comfy_api.latest import io
from comfy_extras import nodes_edit_model, nodes_flux, nodes_post_processing, nodes_qwen

logger = logging.getLogger("radiance.prompt")

from radiance.config.constants import VERSION as __version__  # noqa: E402
from radiance.nodes.branding import schema_branding  # noqa: E402


class CinematicDatasets:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    """Shared datasets for all cinematic prompt nodes. Single source of truth."""

    CAMERAS = [
        "None",
        "ARRI Alexa 65 (IMAX)",
        "ARRI Alexa Mini LF",
        "ARRI Alexa 35",
        "Sony Venice 2",
        "Sony FX9",
        "Sony A7S III",
        "RED V-Raptor XL",
        "RED Komodo",
        "RED Monstro 8K VV",
        "Panavision Millennium DXL2",
        "Panavision Panaflex Gold II (35mm)",
        "IMAX 15/70mm Film Camera",
        "Bolex H16 (16mm Film)",
        "Super 8mm Camera",
        "Canon C700 FF",
        "Canon C300 Mark III",
        "Blackmagic URSA Mini Pro 12K",
        "GoPro Hero 12",
        "iPhone 15 Pro Max",
        "Polaroid SX-70",
        "Vintage Daguerreotype Camera",
    ]

    LENSES = [
        "None",
        # Prime Focal Lengths
        "14mm Ultra-Wide Angle",
        "16mm Ultra-Wide Angle",
        "24mm Wide Angle",
        "28mm Wide Angle",
        "35mm Classic Wide",
        "40mm Semi-Wide",
        "50mm Standard Prime",
        "85mm Portrait Prime",
        "105mm Macro",
        "135mm Medium Telephoto",
        "200mm Telephoto Compression",
        "600mm Super Telephoto",
        # Specialty Lenses
        "Anamorphic Lens",
        "Fish-Eye Lens",
        "Tilt-Shift Lens",
        # Cinema Primes (High-End)
        "Cooke S7/i Full Frame",
        "Cooke Speed Panchro Vintage",
        "Zeiss Master Prime",
        "Zeiss Supreme Prime",
        "ARRI/Zeiss Signature Prime",
        "ARRI Master Anamorphic",
        "Panavision Primo 70",
        "Panavision C-Series Anamorphic",
        # Vintage & Character Lenses
        "Canon K35 Vintage",
        "Canon FD 50mm L",
        "Helios 44-2 58mm (Swirly Bokeh)",
        "Lensbaby Velvet (Soft Focus)",
        "Petzval 85mm (Classic Swirl)",
        # Modern Professional
        "Leica Summilux 50mm f/1.4",
        "Leica Summicron 35mm",
        "Sigma Art 35mm f/1.4",
        "Sigma Art 85mm f/1.4",
        "Sony G Master 24-70mm",
        "Sony G Master 85mm",
        # Specialty & Macro
        "Laowa Probe Lens (Macro)",
        "Freefly Wave (High-Speed)",
        "Angenieux Optimo Zoom",
        "Fujinon Premista Zoom",
    ]

    APERTURES = [
        "None",
        "f/0.95 (Razor Thin DoF)",
        "f/1.2 (Dreamy Bokeh)",
        "f/1.8 (Soft Background)",
        "f/2.0 (Shallow Cinematic)",
        "f/2.8 (Cinematic Separation)",
        "f/4.0 (Balanced)",
        "f/5.6 (Sharp Subject)",
        "f/8.0 (Deep Focus)",
        "f/11 (Landscape Sharpness)",
        "f/16 (Everything in Focus)",
        "f/22 (Diffraction Starbursts)",
    ]

    FRAMING = [
        "None",
        "Extreme Close-Up (ECU)",
        "Close-Up (CU)",
        "Medium Close-Up (MCU)",
        "Medium Shot (MS)",
        "Medium Wide (MW)",
        "Medium Full Shot (MFS)",
        "Cowboy Shot (American Shot)",
        "Wide Shot (WS)",
        "Full Body Shot (Wide)",
        "Extreme Wide Shot (EWS)",
        "Establishing Shot",
        "Over-The-Shoulder (OTS)",
        "Point of View (POV)",
        "Low Angle (Hero Shot)",
        "High Angle (Vulnerability)",
        "Bird's Eye View (Overhead)",
        "Worm's Eye View",
        "Dutch Angle (Canted)",
        "Symmetrical Composition",
        "Rule of Thirds",
    ]

    LIGHTING = [
        "None",
        "Rembrandt Lighting",
        "Chiaroscuro (High Contrast)",
        "Film Noir Lighting",
        "Split Lighting",
        "Butterfly Lighting",
        "Paramount Lighting",
        "Soft Window Light",
        "Golden Hour (Magic Hour)",
        "Blue Hour",
        "Cinematic Haze / Volumetric Fog",
        "God Rays (Crepuscular Rays)",
        "Neon Cyberpunk Lighting",
        "Practical Lighting",
        "Bioluminescence",
        "Studio Strobe 3-Point Setup",
        "Ring Light",
        "Candlelight",
        "Moonlight",
        "Overcast Soft Light",
        "Harsh Sunlight",
        # v2.3.3: Added for director-style presets
        "Natural Ambient Light",
        "High-Key Lighting",
        "Naturalistic Interior Light",
        "Interview 3-Point Setup",
        "Volumetric Fog / Atmospheric Haze",
    ]

    STYLES = [
        "None",
        "Photorealistic (Raw)",
        "Cinematic Movie Still",
        "Hyper-Realism",
        "Editorial Photography",
        "National Geographic Style",
        "Documentary Texture",
        "Vintage 1990s VHS",
        "Analog Film (Kodak Portra 400)",
        "Fujifilm Velvia 50",
        "Black and White (Ilford HP5)",
        "Monochrome Noir",
        "CGI 3D Render (Octane)",
        "Unreal Engine 5",
        "Pixar Animation Style",
        "Anime (Makoto Shinkai)",
        "Oil Painting (Classic)",
        "Concept Art",
        "Cyberpunk 2077 Aesthetic",
        "Wes Anderson Symmetric",
        "Tarantino Violence",
        "Kubrick One-Point Perspective",
        "Blade Runner Atmosphere",
        # v2.3.3: Added for director-style presets
        "Film Noir Aesthetic",
        "Dreamy Soft Focus",
        "Atmospheric Cinematic",
        "Vintage Kodachrome Look",
        "Cinematic TV Drama",
        "Documentary Realism",
        "Neo-Noir Modern",
        "Magical Realism",
    ]

    FILM_STOCKS = [
        "None",
        "Kodak Vision3 500T",
        "Kodak Vision3 250D",
        "Kodak Portra 400",
        "Kodak Ektar 100",
        "Kodak Tri-X 400 (B&W)",
        "Fujifilm Pro 400H",
        "Cinestill 800T",
        "Fujifilm Velvia 50",
        "Ilford Delta 3200",
        "Polaroid 600",
        "Wet Plate Collodion",
        # v2.3.3: Added for director-style presets
        "IMAX 15/70mm Film Stock",
        "Eastman Kodak 5254 (Vintage)",
    ]

    SHUTTER_SPEEDS = [
        "None",
        "1/50th sec (Standard Motion Blur)",
        "1/1000th sec (Frozen Action)",
        "Long Exposure (Light Trails)",
        "Slow Shutter (Dreamy Blur)",
    ]

    COLOR_GRADING = [
        "None",
        "Teal and Orange (Blockbuster)",
        "Bleach Bypass (Gritty)",
        "Technicolor (Vintage)",
        "Cross Processed",
        "Desaturated (Muted)",
        "Vibrant High Contrast",
        "Sepia Tone",
        "Monochrome High Key",
        "Cyberpunk Neon Grading",
        "Pastel Soft Tones",
        # v2.3.3: Added for director-style presets
        "Neutral ACES Workflow",
        "Desaturated Cool Tones",
        "Moody Shadows",
        "Orange and Teal (Hollywood)",
    ]

    ASPECT_RATIOS = [
        "None",
        "16:9 (Widescreen)",
        "2.39:1 (Anamorphic Scope)",
        "4:3 (Academy Ratio)",
        "1:1 (Square)",
        "9:16 (Social Vertical)",
        "21:9 (Ultrawide)",
        # v2.3.3: Added for director-style presets
        "1.43:1 (IMAX)",
        "1.85:1 (Standard Widescreen)",
        "2.00:1 (DCI Flat)",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    #                         ONE-CLICK STYLE PRESETS
    # ═══════════════════════════════════════════════════════════════════════════

    STYLE_PRESETS = [
        "None (Custom)",
        "→ Classic Hollywood",
        "→ Film Noir",
        "→ Sci-Fi Cinematic",
        "→ Cyberpunk",
        "→ Drama / Emotional",
        "→ Epic Landscape",
        "→ Portrait",
        "→ Documentary",
        "→ Artistic / Painterly",
        "→ Retro VHS",
        "→ Golden Hour Magic",
        "→ Moody Night",
        "→ Action / Dynamic",
        "→ Wes Anderson",
        "→ 1970s New Hollywood",
        "→ 1980s Retro Action",
        "→ 1990s Music Video",
        "→ 2000s Digital Look",
        # Director & Genre Presets
        "→ Horror / Thriller",
        "→ Romance / Soft Focus",
        "→ Christopher Nolan",
        "→ Denis Villeneuve",
        "→ Quentin Tarantino",
        "→ Prestige Drama (HBO)",
        "→ True Crime Documentary",
        "→ Western",
        "→ Mystery / Detective",
        "→ Fantasy / Magical",
    ]

    # Preset configurations: {preset_name: {setting: value, ...}}
    # v2.3.3: ALL values validated against their respective dataset lists
    PRESET_CONFIGS = {
        "→ Classic Hollywood": {
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
        "→ Film Noir": {
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
        "→ Sci-Fi Cinematic": {
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
        "→ Cyberpunk": {
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
        "→ Drama / Emotional": {
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
        "→ Epic Landscape": {
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
        "→ Portrait": {
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
        "→ Documentary": {
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
        "→ Artistic / Painterly": {
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
        "→ Retro VHS": {
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
        "→ Golden Hour Magic": {
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
        "→ Moody Night": {
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
        "→ Action / Dynamic": {
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
        "→ Wes Anderson": {
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
        "→ 1970s New Hollywood": {
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
        "→ 1980s Retro Action": {
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
        "→ 1990s Music Video": {
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
        "→ 2000s Digital Look": {
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
        # ───────────────────────────────────────────────────────────
        # Director & Genre Presets (v2.3.3: validated against datasets)
        # ───────────────────────────────────────────────────────────
        "→ Horror / Thriller": {
            "framing": "Low Angle (Hero Shot)",
            "camera_type": "ARRI Alexa Mini LF",
            "lens_focal": "16mm Ultra-Wide Angle",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Chiaroscuro (High Contrast)",
            "style_aesthetic": "Film Noir Aesthetic",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Bleach Bypass (Gritty)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Romance / Soft Focus": {
            "framing": "Medium Close-Up (MCU)",
            "camera_type": "Sony A7S III",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/1.2 (Dreamy Bokeh)",
            "lighting": "Soft Window Light",
            "style_aesthetic": "Dreamy Soft Focus",
            "film_stock": "Kodak Portra 400",
            "color_grading": "Pastel Soft Tones",
            "aspect_ratio": "1.85:1 (Standard Widescreen)",
        },
        "→ Christopher Nolan": {
            "framing": "Wide Shot (WS)",
            "camera_type": "IMAX 15/70mm Film Camera",
            "lens_focal": "28mm Wide Angle",
            "aperture_dof": "f/8.0 (Deep Focus)",
            "lighting": "Natural Ambient Light",
            "style_aesthetic": "Photorealistic (Raw)",
            "film_stock": "IMAX 15/70mm Film Stock",
            "color_grading": "Neutral ACES Workflow",
            "aspect_ratio": "1.43:1 (IMAX)",
        },
        "→ Denis Villeneuve": {
            "framing": "Extreme Wide Shot (EWS)",
            "camera_type": "ARRI Alexa 65 (IMAX)",
            "lens_focal": "24mm Wide Angle",
            "aperture_dof": "f/4.0 (Balanced)",
            "lighting": "Volumetric Fog / Atmospheric Haze",
            "style_aesthetic": "Atmospheric Cinematic",
            "film_stock": "None",
            "color_grading": "Desaturated Cool Tones",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Quentin Tarantino": {
            "framing": "Medium Shot (MS)",
            "camera_type": "Panavision Panaflex Gold II (35mm)",
            "lens_focal": "40mm Semi-Wide",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "High-Key Lighting",
            "style_aesthetic": "Vintage Kodachrome Look",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Vibrant High Contrast",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Prestige Drama (HBO)": {
            "framing": "Medium Shot (MS)",
            "camera_type": "ARRI Alexa Mini LF",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/2.0 (Shallow Cinematic)",
            "lighting": "Naturalistic Interior Light",
            "style_aesthetic": "Cinematic TV Drama",
            "film_stock": "None",
            "color_grading": "Moody Shadows",
            "aspect_ratio": "2.00:1 (DCI Flat)",
        },
        "→ True Crime Documentary": {
            "framing": "Medium Close-Up (MCU)",
            "camera_type": "Canon C300 Mark III",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Interview 3-Point Setup",
            "style_aesthetic": "Documentary Realism",
            "film_stock": "None",
            "color_grading": "Desaturated Cool Tones",
            "aspect_ratio": "16:9 (Widescreen)",
        },
        "→ Western": {
            "framing": "Medium Full Shot (MFS)",
            "camera_type": "Panavision Panaflex Gold II (35mm)",
            "lens_focal": "35mm Classic Wide",
            "aperture_dof": "f/5.6 (Sharp Subject)",
            "lighting": "Harsh Sunlight",
            "style_aesthetic": "Vintage Kodachrome Look",
            "film_stock": "Eastman Kodak 5254 (Vintage)",
            "color_grading": "Orange and Teal (Hollywood)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Mystery / Detective": {
            "framing": "Over-The-Shoulder (OTS)",
            "camera_type": "ARRI Alexa 35",
            "lens_focal": "50mm Standard Prime",
            "aperture_dof": "f/2.8 (Cinematic Separation)",
            "lighting": "Film Noir Lighting",
            "style_aesthetic": "Neo-Noir Modern",
            "film_stock": "Kodak Vision3 500T",
            "color_grading": "Desaturated Cool Tones",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
        "→ Fantasy / Magical": {
            "framing": "Medium Wide (MW)",
            "camera_type": "ARRI Alexa 65 (IMAX)",
            "lens_focal": "Anamorphic Lens",
            "aperture_dof": "f/2.0 (Shallow Cinematic)",
            "lighting": "God Rays (Crepuscular Rays)",
            "style_aesthetic": "Magical Realism",
            "film_stock": "Kodak Vision3 250D",
            "color_grading": "Technicolor (Vintage)",
            "aspect_ratio": "2.39:1 (Anamorphic Scope)",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                         NEURAL GRAMMAR ENGINE (v3.5)
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralGrammar:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    """
    Advanced logic to "Neuralize" prompts, making them more precise,
    scientific, and architecturally aware for T5/Diffusion models.
    """

    SCIENTIFIC_REPLACEMENTS = {
        # v3.1: Toned down from jargon to natural cinematic language.
        # Diffusion models respond better to concrete visual descriptors
        # than compound technical terms.
        "visualize": "render with volumetric detail",
        "accurate": "high-fidelity",
        "precise": "pixel-perfect",
        "design": "architectural composition",
        "process": "sequential transformation",
        "generation": "procedural reconstruction",
    }

    @staticmethod
    def scientificize(text: str) -> str:
        """Replace common words with high-precision cinematic terms.

        BUG-1 FIX (prior): Preserved case on split.
        BUG-2 FIX (prior): Added parens to strip set.
        BUG-M1 FIX (v2.3.3): Post-cleanup pass to fix irregular spacing
        caused by multi-word replacements adjacent to punctuation.
        """
        words = text.split()           # preserve original case
        new_words = []
        modified = False
        for word in words:
            clean_word = word.strip(".,!?;:()")   # also strip parens
            if clean_word.lower() in NeuralGrammar.SCIENTIFIC_REPLACEMENTS:
                replacement = NeuralGrammar.SCIENTIFIC_REPLACEMENTS[clean_word.lower()]
                # [BUG-M1] Preserve trailing punctuation from the original word
                trailing = ""
                for ch in reversed(word):
                    if ch in ".,!?;:()":
                        trailing = ch + trailing
                    else:
                        break
                new_words.append(replacement + trailing)
                modified = True
            else:
                new_words.append(word)  # preserve original capitalisation
        if modified:
            logger.debug("[NeuralGrammar] Applied cinematic vocabulary enhancement.")
        # [BUG-M1] Clean up any double-spaces introduced by multi-word replacements
        result = " ".join(new_words)
        return re.sub(r" {2,}", " ", result)

    @staticmethod
    def enhance_syntax(prompt_parts: list) -> list:
        """Improve grammar and technical structure of prompt parts."""
        enhanced = []
        for part in prompt_parts:
            if part.startswith("Shot on"):
                # v3.1 FIX: Preserve original camera name.
                camera_name = part.replace("Shot on ", "", 1)
                part = f"Captured with high-precision {camera_name}"
            enhanced.append(part)
        return enhanced


# ═══════════════════════════════════════════════════════════════════════════════
#                   PRESET VALIDATION (v2.3.3 — runs at import time)
# ═══════════════════════════════════════════════════════════════════════════════

_FIELD_TO_DATASET = {
    "framing": CinematicDatasets.FRAMING,
    "camera_type": CinematicDatasets.CAMERAS,
    "lens_focal": CinematicDatasets.LENSES,
    "aperture_dof": CinematicDatasets.APERTURES,
    "lighting": CinematicDatasets.LIGHTING,
    "style_aesthetic": CinematicDatasets.STYLES,
    "film_stock": CinematicDatasets.FILM_STOCKS,
    "shutter_speed": CinematicDatasets.SHUTTER_SPEEDS,
    "color_grading": CinematicDatasets.COLOR_GRADING,
    "aspect_ratio": CinematicDatasets.ASPECT_RATIOS,
}


def _validate_presets():
    """Validate all preset config values exist in their respective dataset lists.

    v2.3.3 [BUG-I8]: In strict mode (RADIANCE_STRICT=1 env var), raises
    RuntimeError on validation failure instead of silently logging.
    """
    errors = []
    for preset_name, config in CinematicDatasets.PRESET_CONFIGS.items():
        for field, value in config.items():
            if field not in _FIELD_TO_DATASET:
                errors.append(f"  [{preset_name}] Unknown field: '{field}'")
                continue
            dataset = _FIELD_TO_DATASET[field]
            if value not in dataset:
                errors.append(f"  [{preset_name}] {field}='{value}' not in dataset")
    if errors:
        msg = f"◎ PRESET VALIDATION FAILED ({len(errors)} issues):\n" + "\n".join(errors)
        logger.error(msg)
        # [BUG-I8] Strict mode: raise so broken presets are caught in CI/testing
        if os.environ.get("RADIANCE_STRICT", "0") == "1":
            raise RuntimeError(msg)
    else:
        logger.debug("✓ All preset configs validated against datasets")


# Run validation at import time so mismatches are caught immediately
_validate_presets()


# ═══════════════════════════════════════════════════════════════════════════════
#                         TOKEN UTILITIES (v2.3.3)
# ═══════════════════════════════════════════════════════════════════════════════

CLIP_MAX_TOKENS = 77
BREAK_TOKEN = "BREAK"  # nosec B105
BREAK_TOKEN_OVERHEAD = 2  # [BUG-M2] BREAK token + surrounding spaces ≈ 2 tokens
# [FIX-8] Runtime year — never goes stale across calendar years.
# RADIANCE_DEFAULT_YEAR env var allows override for reproducible test fixtures.
DEFAULT_YEAR: int = int(os.environ.get("RADIANCE_DEFAULT_YEAR",
                                       datetime.date.today().year))

# ═══════════════════════════════════════════════════════════════════════════════
#                  ARCHITECTURE-AWARE CONSTANTS  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# These architectures use T5/LLM encoders that prefer natural language prose.
# Comma-separated keyword chains perform significantly worse on them.
PROSE_ARCHS = {"flux", "sd3", "sd3.5", "wan", "ltxv", "ltxav", "pixart",  # ALBABIT-FIX: "ltx" → "ltxv"
               "hunyuan_video", "aura_flow", "minimax",
               # 3.5.0: every T5 / LLM encoder the loader can report. They used
               # to fall through to CLIP-style tag prompts with a 77-token cap.
               "flux2", "flux2-klein", "z_image", "lumina2", "chroma", "qwen_image",
               "hidream", "cosmos", "cogvideox", "mochi", "wan_ti2v", "krea2",
               "hunyuan_image", "hunyuan_video_15", "kandinsky5", "kandinsky5_image",
               "longcat_image", "omnigen2", "qwen3_llm", "qwen25_llm", "llm"}

# Only these read their prompt through CLIP alone (77-token chunks, weighted
# tags). Anything else, including an arch this file has never heard of, is a
# T5 / LLM encoder and gets prose: a new model is far more likely to be
# LLM-conditioned than CLIP-only.
CLIP_ONLY_ARCHS = {"sd1.5", "sd2", "sdxl"}


def _is_prose_arch(arch: str) -> bool:
    return arch not in CLIP_ONLY_ARCHS

# Guidance-distilled models: the sampler runs them at CFG 1, where ComfyUI
# never evaluates the negative (CFGGuider skips the uncond pass at cfg 1.0),
# and MiniMax H3's reference graph has no negative at all. Encoding the
# automatic negative for them costs a full text-encoder pass for nothing.
# Deliberately narrow: Wan, LTX and HunyuanVideo 1.5 run real CFG.
_GUIDANCE_DISTILLED_ARCHS = {"flux", "flux2", "minimax"}

# Architectures where the negative prompt has no effect: the guidance-distilled
# ones, sampled at CFG 1 (MiniMax H3's reference graph uses BasicGuider and has
# no negative at all). 3.5.0: Wan, LTX and HunyuanVideo left this set. They run
# real CFG and their reference workflows ship a long negative, yet the encoder
# cut theirs to "Soft" and the UI marked the field "no CFG, ignored".
_WEAK_NEG_ARCHS = set(_GUIDANCE_DISTILLED_ARCHS)


@functools.lru_cache(maxsize=4)
def _parse_model_meta_cached(model_meta: str) -> str:
    """
    Parse model_meta JSON and return the arch string.
    Cached: the loader only changes arch when the user loads a new model,
    so the same model_meta string always yields the same arch.
    Returns "" if parsing fails or arch is unknown.
    """
    try:
        meta = json.loads(model_meta)
        arch = meta.get("arch", "")
        if arch and arch != "unknown":
            return arch.lower()
    except Exception as e:
        logger.debug(f"[Encoder] model_meta parse failed: {e}")
    return ""


@functools.lru_cache(maxsize=4)
def _fingerprint_clip_keys(clip_id: int) -> frozenset:
    """
    Tokenize "test" and return the frozenset of token-dict keys.
    Cached per clip object identity — the CLIP model's tokenizer vocab
    doesn't change between runs, so this call only needs to happen once
    per loaded clip.

    NOTE: cache is keyed on id(clip), not clip itself, to avoid holding
    a strong reference to large CLIP objects. The LRU maxsize=4 means we
    keep at most 4 distinct CLIP objects' fingerprints in memory at once.
    Call ``_fingerprint_clip_keys.cache_clear()`` when a new clip is loaded
    if precise eviction is required.
    """
    # This function is called from _detect_arch_from_clip with id(clip).
    # The actual clip object must be passed separately (see _detect_arch_from_clip).
    raise RuntimeError("_fingerprint_clip_keys should not be called directly")


def _detect_arch_from_clip(clip, target_arch: str,
                           model_meta: Optional[str] = None) -> str:
    """
    Resolve architecture string for prompt-building decisions.

    Priority order (most → least stable):
      1. Explicit ``target_arch`` set by the user (not "Auto") — always honoured.
      2. ``model_meta`` JSON string from RadianceUnifiedLoader's model_meta output.
         The loader already ran a rigorous safetensors key scan; we trust its
         "arch" field unconditionally.  This is the correct architectural contract:
         the loader knows the arch, the encoder consumes it.  [FIX-1]
      3. CLIP tokenizer key fingerprinting (heuristic, fragile — last resort only).
         Cached per clip object to avoid redundant tokenize("test") calls.

    v2.4.0 FIX-1: Previously the encoder always re-detected arch from tokenizer
    keys, which are implementation details of ComfyUI CLIP wrappers and change
    without notice.  Now the loader's model_meta output is the primary source.

    v2.4.0 BUG-I9: Granular multi-key fingerprinting replaces the old single-key
    "t5xxl → flux" mapping that collapsed all T5 archs into one.

    v3.1.1: model_meta JSON parsing is now cached (same string → same arch).
    Tokenizer fingerprinting is cached per clip object id to eliminate the
    redundant clip.tokenize("test") call on every workflow execution.
    """
    # ── Priority 1: explicit user override ──────────────────────────────────
    if target_arch and target_arch != "Auto":
        return target_arch.lower()

    # ── Priority 2: loader model_meta JSON (stable contract, cached) ─────────
    if model_meta:
        arch = _parse_model_meta_cached(model_meta)
        if arch:
            logger.debug(f"[Encoder] Arch from model_meta: '{arch}'")
            return arch

    # ── Priority 3: tokenizer key fingerprinting (cached per clip object) ────
    # ALBABIT-FIX: was keyed on id(clip) in a plain dict, so a collected
    # clip's entry stuck around and a later object reusing that freed
    # address inherited its stale fingerprint (real, reproducible under
    # pytest's object churn). WeakKeyDictionary keyed on the object itself
    # evicts on real collection, so a reused address can't inherit it.
    try:
        keys = _detect_arch_from_clip._key_cache.get(clip)
    except TypeError:
        keys = None  # clip doesn't support weak references, don't cache
    if keys is None:
        try:
            test_tokens = clip.tokenize("test")
            keys = frozenset(test_tokens.keys())
        except Exception as e:
            logger.debug(f"[Encoder] Arch detection failed: {e}, defaulting to sdxl")
            keys = frozenset()
        try:
            _detect_arch_from_clip._key_cache[clip] = keys
        except TypeError:
            pass

    # ALBABIT-FIX: LTXAVGemmaTokenizer registers as "gemma3_12b", exact key
    # match (the old substring check for "gemma" never matched it, falling
    # through to "sdxl"). LTX 2.5's Gemma4Tokenizer registers under "gemma4"
    # instead, same fallback gap, its own check.
    if "gemma3_12b" in keys or "gemma4" in keys:
        return "ltxav"
    # MiniMax H3's Qwen3-VL-32B encoder registers as "qwen3vl_32b" (name=
    # in comfy/text_encoders/minimax.py's MiniMaxH3TEModel/Tokenizer).
    if "qwen3vl_32b" in keys:
        return "minimax"
    # LTX-V (pre-2.3, T5-based) — still matched by key fragments
    if any(k in keys for k in ("ltxv", "ltx")):
        return "ltxv"
    # SD3/SD3.5 (and HiDream, which adds an LLM): all three encoders
    if "t5xxl" in keys and "g" in keys and "l" in keys:
        return "sd3"
    # Flux: T5 + CLIP-L only (no CLIP-G)
    if "t5xxl" in keys and "l" in keys:
        return "flux"
    # 3.5.0: tokenizer names from comfy/text_encoders (ComfyUI 0.32). Every
    # one of these used to fall through to "sdxl".
    if "umt5xxl" in keys:
        return "wan"
    if "mistral3_24b" in keys:
        return "flux2"
    if "qwen3_4b" in keys or "qwen3_8b" in keys:
        return "qwen3_llm"          # Z-Image or Flux.2 Klein: same encoder
    if "gemma2_2b" in keys:
        return "lumina2"
    if "qwen25_7b" in keys:
        return "qwen25_llm"         # Qwen-Image or HunyuanVideo 1.5
    if "pile_t5xl" in keys:
        return "aura_flow"
    # Wan / PixArt / Chroma / other T5-only
    if "t5xxl" in keys:
        return "wan"
    # Generic LLM (HunyuanVideo, older wrappers)
    if "llm" in keys:
        return "hunyuan_video"
    # SDXL: dual CLIP
    if "g" in keys and "l" in keys:
        return "sdxl"
    # SD1.5: CLIP-L only
    if "l" in keys:
        return "sd1.5"
    # Any other encoder key is a T5 / LLM this list does not know yet. Prose
    # is the right default for it; only a clip with no keys at all (tokenize
    # failed) falls back to the CLIP-style path.
    if keys - {"h"}:
        return "llm"

    return "sdxl"  # Safe fallback — structured format for CLIP-only


# Per-clip fingerprint cache. WeakKeyDictionary, not a plain dict: entries
# are removed automatically when the clip object itself is collected.
_detect_arch_from_clip._key_cache = weakref.WeakKeyDictionary()

# ═══════════════════════════════════════════════════════════════════════════════
#                  SCENE MOOD VOCABULARY  (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

SCENE_MOODS = [
    "None",
    "Tense", "Melancholic", "Joyful", "Ominous", "Nostalgic",
    "Awe-Inspiring", "Intimate", "Chaotic", "Peaceful",
    "Surreal", "Gritty", "Ethereal", "Foreboding", "Euphoric",
]

# Vocabulary injected per mood — chosen to steer T5 and CLIP equally well
_MOOD_VOCAB = {
    "Tense":         "heightened tension, breath held, claustrophobic atmosphere, nervous energy",
    "Melancholic":   "melancholic, quiet sorrow, fading light, wistful silence",
    "Joyful":        "vibrant joy, warm energy, infectious optimism, radiant smile",
    "Ominous":       "ominous foreboding, unseen threat, dread building beneath the surface",
    "Nostalgic":     "nostalgic warmth, memory of simpler times, sepia-tinted emotion",
    "Awe-Inspiring": "breathtaking grandeur, overwhelming scale, reverential silence",
    "Intimate":      "quiet intimacy, close proximity, whispered emotion, soft connection",
    "Chaotic":       "frantic energy, motion blur everywhere, sensory overload, disorientation",
    "Peaceful":      "serene tranquility, unhurried pace, meditative stillness",
    "Surreal":       "dreamlike unreality, logic dissolved, impossible beauty",
    "Gritty":        "raw grit, unpolished truth, weathered texture, unflinching honesty",
    "Ethereal":      "otherworldly ethereal light, gossamer beauty, translucent and floating",
    "Foreboding":    "creeping dread, something wrong just out of sight, shadows advance",
    "Euphoric":      "ecstatic joy, transcendent moment, colours oversaturated with feeling",
}


# Per-encoder pad token id, keyed by the tokenizer dict key (the pad_token each
# tokenizer passes in comfy/text_encoders). 3.5.0: the LLM encoders were
# missing, so a Wan prompt counted its 512-token padding. Unlisted keys use 0,
# the T5 / SentencePiece pad.
_PAD_IDS = {
    "l": 49407, "g": 49407, "h": 0,
    "t5xxl": 0, "umt5xxl": 0, "pile_t5xl": 1, "llm": 0, "llama": 128258,
    "mistral3_24b": 11,
    "qwen3_4b": 151643, "qwen3_8b": 151643, "qwen25_7b": 151643,
    "qwen3vl_8b": 151643, "qwen3vl_32b": 151643, "qwen35_2b": 248044,
    "gemma2_2b": 0, "gemma3_4b": 0, "gemma3_12b": 0, "gemma4": 0,
}
_CLIP_KEYS = ("l", "g", "h")


def _count_key(tokens: dict):
    """The encoder whose tokens the model reads the prompt through: the T5 /
    LLM one when there is one, else CLIP."""
    for key, val in tokens.items():
        if key not in _CLIP_KEYS and val:
            return key
    for key in _CLIP_KEYS:
        if tokens.get(key):
            return key
    return None


def _real_token_count(clip, text: str, tokens: dict = None) -> int:
    """
    Get actual token count using the connected CLIP tokenizer.
    Falls back to estimate_tokens() if tokenizer API is unavailable.

    v3.1 FIX: For T5/LLM encoders, tokens are padded to a fixed length
    (e.g., 256 or 512). shape[-1] returns the padded length, not the
    actual token count. We count non-padding tokens where possible.
    T5/generic LLM use pad_token_id=0; CLIP uses pad_token_id=49407;
    MiniMax H3's Qwen3-VL-32B ("qwen3vl_32b") uses 151643 (its own
    special_tokens config in comfy/text_encoders/minimax.py).

    v2.3.3 [BUG-I3]: Removed redundant `import torch as _torch` — torch
    is already imported at module level.
    """
    if tokens is None:
        try:
            tokens = clip.tokenize(text)
        except Exception:
            return estimate_tokens(text)

    try:
        key = _count_key(tokens)
        if key is not None:
            tok_data = tokens[key][0]
            if hasattr(tok_data, "shape"):
                # Tensor path (T5 / newer CLIP wrappers) — single tensor,
                # no multi-chunk structure. Count non-pad tokens directly.
                # [BUG-I3] Use module-level torch directly
                if torch.is_tensor(tok_data):
                    pad_id = _PAD_IDS.get(key, 0)
                    non_pad = (tok_data != pad_id).sum().item()
                    if non_pad > 0:
                        return non_pad
                return int(tok_data.shape[-1])
            elif isinstance(tok_data, list) and tok_data:
                # BUG-5 FIX: ComfyUI CLIP tokenizer returns a list of chunks:
                #   tokens[key] = [chunk0, chunk1, ...]
                # where each chunk is [(token_id, weight), ...] padded to 77.
                #
                # BUG-6 FIX: Previous code only examined tokens[key][0] —
                # the first chunk. For prompts > 77 tokens (BREAK or SDXL
                # long prompts), there are multiple chunks and the count was
                # severely undercounted. Fix: sum across ALL chunks.
                #
                # [BUG-M4] v2.3.3: Explicit tuple validation
                if isinstance(tok_data[0], (tuple, list)) and len(tok_data[0]) >= 1:
                    pad_id = _PAD_IDS.get(key, 0)
                    total_non_pad = 0
                    total_len = 0
                    for chunk in tokens[key]:   # iterate ALL chunks
                        for item in chunk:
                            tok_id = item[0] if isinstance(item, (tuple, list)) else item
                            if tok_id != pad_id:
                                total_non_pad += 1
                        total_len += len(chunk)
                    return total_non_pad if total_non_pad > 0 else total_len
                # Non-tuple list: sum lengths across all chunks
                return sum(len(chunk) for chunk in tokens[key])
        # Fallback: count all token entries if shape not accessible
        for val in tokens.values():
            if val and hasattr(val[0], "shape"):
                return int(val[0].shape[-1])
            elif val and hasattr(val[0], "__len__"):
                return len(val[0])
    except Exception as exc:
        logger.warning("[nodes_prompt]: %s", exc)
    return estimate_tokens(text)


# ═══════════════════════════════════════════════════════════════════════════════
#                  MENU LABEL -> PROMPT TEXT  (3.5.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The menus read like "Close-Up (CU)" or "f/2.8 (Cinematic Separation)". The
# brackets are for the person choosing; ComfyUI's tokenizer reads "(...)" as
# emphasis, so encoding a label as-is gave "CU" or "Cinematic Separation" 1.1x
# weight, on SDXL for every bracketed entry and on T5 / LLM models for the
# style and film stock labels. Every label now goes through _label(), which
# never returns a bracket. The user's own text is never touched: brackets
# there are deliberate weights.

_LABEL_TEXT = {
    # cameras
    "ARRI Alexa 65 (IMAX)": "ARRI Alexa 65 IMAX",
    "Panavision Panaflex Gold II (35mm)": "Panavision Panaflex Gold II 35mm film camera",
    "Bolex H16 (16mm Film)": "Bolex H16 16mm film camera",
    # lenses
    "Helios 44-2 58mm (Swirly Bokeh)": "Helios 44-2 58mm lens with swirly bokeh",
    "Lensbaby Velvet (Soft Focus)": "Lensbaby Velvet soft-focus lens",
    "Petzval 85mm (Classic Swirl)": "Petzval 85mm lens with classic swirl",
    "Laowa Probe Lens (Macro)": "Laowa probe macro lens",
    "Freefly Wave (High-Speed)": "Freefly Wave high-speed camera",
    # apertures
    "f/0.95 (Razor Thin DoF)": "f/0.95 for a razor-thin depth of field",
    "f/1.2 (Dreamy Bokeh)": "f/1.2 for dreamy bokeh",
    "f/1.8 (Soft Background)": "f/1.8 for a soft background",
    "f/2.0 (Shallow Cinematic)": "f/2.0 for a shallow cinematic depth of field",
    "f/2.8 (Cinematic Separation)": "f/2.8 for cinematic subject separation",
    "f/4.0 (Balanced)": "f/4.0 for balanced depth of field",
    "f/5.6 (Sharp Subject)": "f/5.6 for a sharp subject",
    "f/8.0 (Deep Focus)": "f/8.0 for deep focus",
    "f/11 (Landscape Sharpness)": "f/11 for landscape sharpness",
    "f/16 (Everything in Focus)": "f/16 with everything in focus",
    "f/22 (Diffraction Starbursts)": "f/22 with diffraction starbursts",
    # framing (tag form; prose uses _FRAMING_PHRASE)
    "Extreme Close-Up (ECU)": "extreme close-up",
    "Close-Up (CU)": "close-up",
    "Medium Close-Up (MCU)": "medium close-up",
    "Medium Shot (MS)": "medium shot",
    "Medium Wide (MW)": "medium wide shot",
    "Medium Full Shot (MFS)": "medium full shot",
    "Cowboy Shot (American Shot)": "cowboy shot",
    "Wide Shot (WS)": "wide shot",
    "Full Body Shot (Wide)": "full-body wide shot",
    "Extreme Wide Shot (EWS)": "extreme wide shot",
    "Over-The-Shoulder (OTS)": "over-the-shoulder shot",
    "Point of View (POV)": "point-of-view shot",
    "Low Angle (Hero Shot)": "low-angle hero shot",
    "High Angle (Vulnerability)": "high-angle shot",
    "Bird's Eye View (Overhead)": "overhead bird's-eye view",
    "Dutch Angle (Canted)": "canted Dutch angle",
    # lighting
    "Chiaroscuro (High Contrast)": "high-contrast chiaroscuro lighting",
    "Golden Hour (Magic Hour)": "golden hour light",
    "God Rays (Crepuscular Rays)": "god rays",
    # styles
    "Photorealistic (Raw)": "raw photorealistic",
    "Analog Film (Kodak Portra 400)": "analog film, Kodak Portra 400",
    "Black and White (Ilford HP5)": "black and white, Ilford HP5",
    "CGI 3D Render (Octane)": "CGI 3D render, Octane",
    "Anime (Makoto Shinkai)": "anime in the style of Makoto Shinkai",
    "Oil Painting (Classic)": "classic oil painting",
    # film stocks
    "Kodak Tri-X 400 (B&W)": "Kodak Tri-X 400 black and white",
    "Eastman Kodak 5254 (Vintage)": "vintage Eastman Kodak 5254",
    # shutter
    "1/50th sec (Standard Motion Blur)": "1/50th second shutter with natural motion blur",
    "1/1000th sec (Frozen Action)": "1/1000th second shutter freezing the action",
    "Long Exposure (Light Trails)": "long exposure with light trails",
    "Slow Shutter (Dreamy Blur)": "slow shutter with dreamy blur",
    # grading
    "Teal and Orange (Blockbuster)": "blockbuster teal and orange",
    "Bleach Bypass (Gritty)": "gritty bleach bypass",
    "Technicolor (Vintage)": "vintage Technicolor",
    "Desaturated (Muted)": "muted, desaturated",
    "Orange and Teal (Hollywood)": "Hollywood orange and teal",
    # aspect ratios
    "16:9 (Widescreen)": "16:9 widescreen",
    "2.39:1 (Anamorphic Scope)": "2.39:1 anamorphic scope",
    "4:3 (Academy Ratio)": "4:3 academy ratio",
    "1:1 (Square)": "1:1 square",
    "9:16 (Social Vertical)": "9:16 vertical",
    "21:9 (Ultrawide)": "21:9 ultrawide",
    "1.43:1 (IMAX)": "1.43:1 IMAX",
    "1.85:1 (Standard Widescreen)": "1.85:1 widescreen",
    "2.00:1 (DCI Flat)": "2.00:1 DCI flat",
}

# How each framing opens a sentence: "{} " + subject.
_FRAMING_PHRASE = {
    "Extreme Close-Up (ECU)": "An extreme close-up of",
    "Close-Up (CU)": "A close-up of",
    "Medium Close-Up (MCU)": "A medium close-up of",
    "Medium Shot (MS)": "A medium shot of",
    "Medium Wide (MW)": "A medium wide shot of",
    "Medium Full Shot (MFS)": "A medium full shot of",
    "Cowboy Shot (American Shot)": "A cowboy shot, framed from mid-thigh up, of",
    "Wide Shot (WS)": "A wide shot of",
    "Full Body Shot (Wide)": "A full-body wide shot of",
    "Extreme Wide Shot (EWS)": "An extreme wide shot of",
    "Establishing Shot": "An establishing shot of",
    "Over-The-Shoulder (OTS)": "An over-the-shoulder shot of",
    "Point of View (POV)": "A point-of-view shot of",
    "Low Angle (Hero Shot)": "A low-angle hero shot of",
    "High Angle (Vulnerability)": "A high-angle shot looking down on",
    "Bird's Eye View (Overhead)": "An overhead bird's-eye view of",
    "Worm's Eye View": "A worm's-eye view looking up at",
    "Dutch Angle (Canted)": "A canted Dutch-angle shot of",
    "Symmetrical Composition": "A symmetrically composed shot of",
    "Rule of Thirds": "A rule-of-thirds composition of",
}

# What the scene is "lit by".
_LIGHTING_PHRASE = {
    "Rembrandt Lighting": "Rembrandt lighting",
    "Chiaroscuro (High Contrast)": "high-contrast chiaroscuro lighting",
    "Film Noir Lighting": "hard film noir lighting",
    "Split Lighting": "split lighting",
    "Butterfly Lighting": "butterfly lighting",
    "Paramount Lighting": "Paramount lighting",
    "Soft Window Light": "soft window light",
    "Golden Hour (Magic Hour)": "warm golden-hour sunlight",
    "Blue Hour": "cool blue-hour twilight",
    "Cinematic Haze / Volumetric Fog": "light through volumetric haze and fog",
    "God Rays (Crepuscular Rays)": "god rays streaming through the air",
    "Neon Cyberpunk Lighting": "neon cyberpunk light",
    "Practical Lighting": "practical lights within the scene",
    "Bioluminescence": "a bioluminescent glow",
    "Studio Strobe 3-Point Setup": "a three-point studio strobe setup",
    "Ring Light": "a ring light",
    "Candlelight": "candlelight",
    "Moonlight": "moonlight",
    "Overcast Soft Light": "soft overcast daylight",
    "Harsh Sunlight": "harsh direct sunlight",
    "Natural Ambient Light": "natural ambient light",
    "High-Key Lighting": "bright high-key lighting",
    "Naturalistic Interior Light": "naturalistic interior light",
    "Interview 3-Point Setup": "a three-point interview setup",
    "Volumetric Fog / Atmospheric Haze": "light through volumetric fog and atmospheric haze",
}


def _label(value) -> str:
    """Prompt text for a menu entry: "" for None, never a bracket."""
    if value in ("None", None, ""):
        return ""
    text = _LABEL_TEXT.get(value)
    if text is None:
        # Not in the table (a label added later): keep the words, drop the
        # brackets so they cannot turn into a weight.
        text = re.sub(r"\s*\(([^)]*)\)", r" \1", str(value))
        text = re.sub(r" {2,}", " ", text.replace("(", "").replace(")", "")).strip()
    return text


def _a(text: str) -> str:
    """'a' or 'an' in front of text ("an 85mm lens", "an ARRI", "a Cooke")."""
    first = text.lstrip()[:2].lower()
    vowel = first[:1] in "aeio" or first in ("8", "80", "81", "82", "83", "84", "85",
                                              "86", "87", "88", "89", "11", "18")
    return ("an " if vowel else "a ") + text


def _framing_opener(framing) -> str:
    if framing in ("None", None, ""):
        return ""
    return _FRAMING_PHRASE.get(framing) or f"A {_label(framing)} of"


def _lighting_phrase(lighting) -> str:
    if lighting in ("None", None, ""):
        return ""
    return _LIGHTING_PHRASE.get(lighting) or _label(lighting)


def _lower_article(subject: str) -> str:
    """'A man' -> 'a man' after an opener like 'A close-up of'."""
    if re.match(r"(A|An|The)\s", subject):
        return subject[0].lower() + subject[1:]
    return subject


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join_and(items) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _apply_subject_weight(text: str, weight: float) -> str:
    """Wrap text in attention weight syntax if weight != 1.0."""
    if abs(weight - 1.0) < 0.01:
        return text
    return f"({text}:{weight:.2f})"


def _clean_for_prose(text: str) -> str:
    """
    v2.4.0 [BUG-M8]: Convert cinematography jargon parentheticals into
    natural language readable by T5/LLM encoders.

    Examples:
      "f/2.8 (Cinematic Separation)"  →  "f/2.8 for cinematic separation"
      "f/11 (Landscape Sharpness)"    →  "f/11 for landscape sharpness"
      "ARRI Alexa 65 (IMAX)"          →  "ARRI Alexa 65 IMAX"
      "50mm Standard Prime"           →  unchanged (no parens)
    """
    # Convert "(description)" → "for description" where preceded by f/number or similar
    text = re.sub(
        r'(f/[\d.]+)\s*\(([^)]+)\)',
        lambda m: f"{m.group(1)} for {m.group(2).lower()}",
        text
    )
    # For all other parentheticals: just integrate the content (drop the parens)
    text = re.sub(r'\s*\(([^)]+)\)', lambda m: f" {m.group(1)}", text)
    # Clean up double spaces
    return re.sub(r' {2,}', ' ', text).strip()


# ALBABIT-FIX: the prompt goes out as typed; it used to always get a period,
# which changed the conditioning of an edit instruction. It takes one only to
# end a sentence before the ones the builder adds after it.
def _end_subject_sentence(parts):
    if len(parts) > 1 and not parts[0].endswith((".", "!", "?")):
        parts[0] += "."


def _build_prose_prompt(
    base_prompt, framing, camera, lens, aperture, lighting,
    style, film_stock, shutter, color_grading, aspect_ratio,
    custom_details, year_era, lora_keywords, art_direction,
    scene_mood, subject_weight, weight_mode,
) -> str:
    """
    Build a natural-language prose prompt for T5/LLM-based architectures
    (Flux, SD3, PixArt, Wan, LTX, HunyuanVideo).
    These encoders respond much better to flowing sentences than comma chains.

    v2.4.0 [BUG-C4]: scientificize() now applied BEFORE weight wrapping.
    v2.4.0 [BUG-M8]: parenthetical jargon cleaned via _clean_for_prose().
    v2.4.0 [BUG-M9]: tech block rewritten as natural sentence, not "Cinematic technique: X".
    """
    def c(v): return "" if v in ("None", None, "") else v

    parts = []

    # 1. Subject
    subject = base_prompt.strip()

    # 3.5.0: the subject is never rewritten. NeuralGrammar.scientificize used
    # to run whenever it contained "accurate", "precise" or "visualize", and
    # silently swapped "design", "process", "generation" and others for
    # unrelated phrases ("the design of a gear" -> "the architectural
    # composition of a gear").

    # Weight wrapping comes AFTER all text transforms
    if subject_weight != 1.0:
        subject = _apply_subject_weight(subject, subject_weight)

    # [BUG-I1] Weight modes: technique_first, subject_first, balanced
    bare_subject = False
    if weight_mode == "technique_first" and c(camera):
        parts.append(f"Photographed on {_label(camera)}, {subject}.")
    elif weight_mode == "subject_first":
        parts.append(f"{subject}, the central focus of this scene.")
    elif c(framing):
        parts.append(f"{_framing_opener(framing)} {_lower_article(subject)}.")
    else:
        parts.append(subject)
        bare_subject = True

    # 2. Art direction (right after subject — closest semantic anchor)
    if art_direction and art_direction.strip():
        parts.append(art_direction.strip())

    # 3. Scene mood — always a separate sentence, never merged into art_direction.
    # [FIX-6] Previously the mood vocabulary was embedded into the art_direction
    # string ("art — mood"), which broke re-encode idempotency: running the same
    # inputs twice produced different output because art_direction was mutated on
    # the first pass.  Keeping them separate guarantees same inputs → same output.
    mood = c(scene_mood)
    if mood and mood in _MOOD_VOCAB:
        parts.append(_MOOD_VOCAB[mood])

    # 3. LoRA trigger words
    if lora_keywords and lora_keywords.strip():
        parts.append(lora_keywords.strip())

    # 4. Cinematic technique — [BUG-M9] natural sentence, not "Cinematic technique: X."
    # 3.5.0: every menu value goes through _label()/the phrase tables, so no
    # bracket reaches the encoder and the sentences read as English.
    tech_parts = []
    if c(camera) and weight_mode != "technique_first":
        tech_parts.append(f"shot on {_a(_label(camera))}")
    if c(lens):
        lens_text = _label(lens)
        if not re.search(r"\b(lens|camera|prime|zoom|anamorphic)\b", lens_text, re.I):
            lens_text += " lens"
        tech_parts.append(f"through {_a(lens_text)}")
    if c(aperture):
        tech_parts.append(f"at {_label(aperture)}")
    if c(shutter):
        tech_parts.append(f"with {_a(_label(shutter))}")
    if tech_parts:
        parts.append(f"The image was {_join_and(tech_parts)}.")

    # 5. Lighting
    if c(lighting):
        parts.append(f"Lit by {_lighting_phrase(lighting)}.")

    # 6. Look: style, grade, film stock, one sentence
    look = []
    if c(style):         look.append(f"{_label(style)} style")
    if c(color_grading): look.append(f"{_label(color_grading)} color grade")
    if c(film_stock):    look.append(f"the texture of {_label(film_stock)} film")
    if look:
        parts.append(_cap(_join_and(look) + "."))

    # 7. Era
    if year_era != DEFAULT_YEAR:
        parts.append(f"Set in the year {year_era}.")

    # 8. Aspect ratio
    if c(aspect_ratio):
        parts.append(f"Composed for a {_label(aspect_ratio)} frame.")

    # 9. Framing context for subject_first (add after technique, not at top)
    if weight_mode == "subject_first" and c(framing):
        parts.append(f"Framed as {_a(_label(framing))}.")

    # 10. Custom details last
    if c(custom_details):
        parts.append(custom_details.strip())

    if bare_subject:
        _end_subject_sentence(parts)
    return " ".join(parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate CLIP tokens for a text string.
    CLIP's BPE tokenizer averages ~1.3 tokens per whitespace-delimited word
    due to subword splitting. This is a rough heuristic; actual count depends
    on the specific vocabulary and text content.
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)


def insert_break_points(prompt: str, max_tokens: int = 70,
                        clip=None) -> str:
    """
    Insert BREAK tokens at logical sentence boundaries to chunk long prompts.
    Helps with CLIP's 77-token limit by creating separate encoding chunks.

    v2.3.3: Improved sentence splitting on ". " to avoid breaking abbreviations.
    v2.3.3 [BUG-M2]: Account for BREAK token overhead in chunk budget.
    v2.5.0 [FIX-4]: Accepts optional ``clip`` tokenizer for accurate per-sentence
    token counts.  Previously used the ~1.3 tokens/word heuristic, which
    placed BREAK points inaccurately for vocab-heavy cinematography text
    (e.g. "Panavision C-Series Anamorphic" tokenises as ~8 tokens, not ~3).
    Falls back to estimate_tokens() when clip is unavailable.
    """
    def _count(text: str) -> int:
        if clip is not None:
            try:
                return _real_token_count(clip, text)
            except Exception as exc:
                logger.warning("[nodes_prompt] _count: %s", exc)
        return estimate_tokens(text)

    if _count(prompt) <= max_tokens:
        return prompt

    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    if len(sentences) <= 1:
        return prompt

    effective_limit = max_tokens - BREAK_TOKEN_OVERHEAD
    result = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _count(sentence)
        if current_tokens + sent_tokens > effective_limit and current_chunk:
            result.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sent_tokens

    if current_chunk:
        result.append(" ".join(current_chunk))

    return f" {BREAK_TOKEN} ".join(result)


def enhance_prompt_grammar(prompt: str, level: str, arch: str = "sdxl") -> str:
    """
    Fix common grammar/formatting issues and optionally inject creative tags.

    v3.1: arch parameter controls which creative tags to inject.
    Danbooru-style tags (masterpiece, best quality) only help SD1.5/SDXL.
    Prose architectures get natural-language quality descriptors instead.

    v2.3.3 [BUG-M5]: Check for existing creative tags before injection
    to prevent duplication on re-encodes.
    """
    if level == "Off" or not prompt:
        return prompt

    p = prompt
    # Remove redundant spaces
    p = re.sub(r' {2,}', ' ', p)
    # Fix stray/duplicate commas
    p = re.sub(r'\s+,', ',', p)
    p = re.sub(r',+', ',', p)
    p = re.sub(r',\s*,', ',', p)
    # Clean up double periods (but preserve ellipses ...)
    p = re.sub(r'(?<!\.)\.\.(?!\.)', '.', p)

    if level == "Creative Enhancement":
        if arch in PROSE_ARCHS:
            creative_tags = (
                "Exceptionally detailed with stunning visual clarity "
                "and ultra-high resolution rendering"
            )
            # [BUG-M7] Prose dedup key is distinct from SDXL "masterpiece" check.
            # Checking "masterpiece" for prose archs would false-positive on any
            # prompt containing "masterpiece of engineering" etc.
            already_present = creative_tags in p or "ultra-high resolution rendering" in p
        else:
            creative_tags = "masterpiece, best quality, highly detailed, stunning, ultra-high resolution"
            already_present = creative_tags in p or "masterpiece" in p.lower()

        if not already_present:
            if p.endswith('.'):
                p = p[:-1] + ", " + creative_tags + "."
            else:
                p += ", " + creative_tags

    return p.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#                         SHARED PROMPT BUILDER  (v2.3.3)
# ═══════════════════════════════════════════════════════════════════════════════


def build_cinematic_prompt_v3(
    base_prompt,
    framing,
    camera_type,
    lens_focal,
    aperture_dof,
    lighting,
    style_aesthetic,
    film_stock="None",
    shutter_speed="None",
    color_grading="None",
    aspect_ratio="None",
    custom_details="",
    year_era=DEFAULT_YEAR,
    negative_strength="Standard",
    negative_custom="",
    lora_keywords="",
    use_break=False,
    target_arch="sdxl",
    scene_mood="None",
    subject_weight=1.0,
    art_direction="",
    prompt_weight_mode="balanced",
    base_prompt_b="",
    active_prompt="A",
):
    """
    v2.5.0 universal prompt builder.

    ``target_arch`` MUST be a pre-resolved architecture string (e.g. "flux",
    "sdxl", "wan").  Passing "Auto" is a programming error — the caller
    (``RadianceCinematicPromptEncoder.execute``) is responsible for resolving arch via
    ``_detect_arch_from_clip()`` before calling this function.  Passing "Auto"
    here now raises ValueError so misuse is caught immediately rather than
    silently falling back to "sdxl" and producing wrong output.  [FIX-2]

    Returns: (final_prompt, negative_prompt, estimated_token_count).
    """
    # [FIX-2] Guard against the previously-silent Auto fallback.
    if target_arch == "Auto":
        raise ValueError(
            "build_cinematic_prompt_v3: target_arch='Auto' is not allowed. "
            "Resolve arch via _detect_arch_from_clip() before calling this function."
        )

    def c(v): return "" if v in ("None", None, "") else v

    # A/B prompt mode
    if active_prompt == "B" and base_prompt_b.strip():
        effective_base = base_prompt_b.strip()
    else:
        effective_base = base_prompt.strip()
        if active_prompt == "B":
            logger.info("[Encoder] Prompt B is empty — falling back to Prompt A.")

    # v3.1 FIX: Subject weight is applied INSIDE each path (prose/structured),
    # not here. Previously it was applied at both levels, causing double-weight.

    # [FIX-2] target_arch is pre-resolved by the caller — no "Auto" path here.
    resolved_arch = target_arch
    use_prose = _is_prose_arch(resolved_arch)

    if use_prose:
        final_prompt = _build_prose_prompt(
            base_prompt=effective_base, framing=framing, camera=camera_type,
            lens=lens_focal, aperture=aperture_dof, lighting=lighting,
            style=style_aesthetic, film_stock=film_stock, shutter=shutter_speed,
            color_grading=color_grading, aspect_ratio=aspect_ratio,
            custom_details=custom_details, year_era=year_era,
            lora_keywords=lora_keywords, art_direction=art_direction,
            scene_mood=scene_mood, subject_weight=subject_weight,
            weight_mode=prompt_weight_mode,
        )
    else:
        # Structured keyword format (SD1.5 / SDXL)
        parts = []
        style = c(style_aesthetic)

        # Apply subject weight once in the structured path
        weighted_base = _apply_subject_weight(effective_base, subject_weight)

        # [BUG-I1] subject_first: lead with weighted subject, no camera prefix
        # 3.5.0: menu values through _label() -- no "(CU)" or "(Gritty)"
        # reaching CLIP, where brackets are a 1.1x weight.
        bare_subject = False
        if prompt_weight_mode == "subject_first":
            parts.append(f"{weighted_base}.")
            if c(framing):
                parts.append(f"Framed as {_a(_label(framing))}.")
        elif prompt_weight_mode == "technique_first" and c(camera_type):
            parts.append(f"Shot on {_label(camera_type)}.")
            if c(framing):
                parts.append(f"{_framing_opener(framing)} {_lower_article(weighted_base)}.")
            else:
                parts.append(f"{weighted_base}.")
        elif c(framing):
            parts.append(f"{_framing_opener(framing)} {_lower_article(weighted_base)}.")
        else:
            parts.append(weighted_base)
            bare_subject = True

        # [BUG-C3] FIX: Insert art_direction THEN lora_keywords at sequential
        # indices so art_direction stays closer to subject than lora_keywords.
        # Previous code inserted both at index 1, reversing the intended order.
        insert_idx = len(parts)  # Insert after subject/framing block
        if art_direction and art_direction.strip():
            parts.insert(insert_idx, art_direction.strip())
            insert_idx += 1
        if lora_keywords and lora_keywords.strip():
            parts.insert(insert_idx, lora_keywords.strip())

        if scene_mood and scene_mood != "None" and scene_mood in _MOOD_VOCAB:
            parts.append(_MOOD_VOCAB[scene_mood])

        tech = []
        # [BUG-M3] Gate camera in tech block for both technique_first AND subject_first
        if c(camera_type) and prompt_weight_mode not in ("technique_first",):
            tech.append(f"Shot on {_label(camera_type)}")
        if c(lens_focal):    tech.append(f"with {_label(lens_focal)}")
        if c(aperture_dof):  tech.append(f"at {_label(aperture_dof)}")
        if c(shutter_speed): tech.append(f"{_label(shutter_speed)}")
        if tech: parts.append(", ".join(tech) + ".")

        if c(lighting):      parts.append(f"Lit by {_lighting_phrase(lighting)}.")
        if c(color_grading): parts.append(_cap(f"{_label(color_grading)} color grade."))

        finish = []
        if style:             finish.append(_label(style))
        if c(film_stock):     finish.append(f"on {_label(film_stock)}")
        # [BUG-C2] FIX: Removed trailing period from year_era fragment.
        # The join + "." at the end of the finish block already adds one.
        if year_era != DEFAULT_YEAR: finish.append(f"Est. Year {year_era}")
        if finish: parts.append(_cap(", ".join(finish) + "."))

        if c(aspect_ratio):   parts.append(f"{_label(aspect_ratio)} format.")
        if c(custom_details): parts.append(custom_details.strip())

        if bare_subject:
            _end_subject_sentence(parts)
        final_prompt = " ".join(p for p in parts if p).strip()

    # [BUG-I5] use_break is now always boolean from the caller
    if use_break:
        final_prompt = insert_break_points(final_prompt)

    token_count = estimate_tokens(final_prompt)

    # ── Auto-negative (arch-aware) ──────────────────────────────────────────
    style_val = c(style_aesthetic)
    effective_negative_strength = negative_strength
    if resolved_arch in _WEAK_NEG_ARCHS:
        if negative_strength in ("Standard", "Aggressive"):
            logger.info(
                "[Encoder] target_arch='%s': negative prompts have limited effect. "
                "Auto-tuning %s negatives to Soft.",
                resolved_arch,
                negative_strength,
            )
            effective_negative_strength = "Soft"

    negative_prompt = ""
    if effective_negative_strength != "Off":
        neg: list[str] = []
        if effective_negative_strength in ("Soft", "Standard", "Aggressive"):
            neg.extend(["blur", "low quality", "watermark"])
            # 3.5.0: "text" only where the model cannot draw lettering anyway.
            # On Flux, SD3, Qwen-Image, HiDream and the other T5 / LLM models
            # it suppressed signs, titles and captions the prompt asked for.
            if resolved_arch in CLIP_ONLY_ARCHS:
                neg.append("text")
        if effective_negative_strength in ("Standard", "Aggressive"):
            neg.extend(["deformed", "ugly", "duplicate", "disfigured", "bad anatomy"])
            if any(kw in style_val for kw in ("Photorealistic", "Cinematic", "Documentary", "Realism")):
                neg.extend(["cartoon", "anime", "illustration", "painting", "cgi",
                             "3d render", "drawing", "sketch"])
            elif "Anime" in style_val:
                # [BUG-M6] Added "3d render" which was missing from Anime exclusions
                neg.extend(["photograph", "realistic", "photo", "photorealistic",
                             "3d", "3d render"])
            elif any(kw in style_val for kw in ("Painting", "Oil", "Painterly")):
                neg.extend(["photograph", "realistic", "photo", "digital", "3d render"])
            elif any(kw in style_val for kw in ("CGI", "Unreal", "3D", "Octane")):
                neg.extend(["photograph", "realistic", "2d", "flat", "hand drawn"])
        if effective_negative_strength == "Aggressive":
            neg.extend(["mutated", "extra limbs", "missing limbs", "floating limbs",
                         "disconnected limbs", "pixelated", "noise", "grainy",
                         "cropped", "out of frame", "worst quality", "lowres"])
        negative_prompt = ", ".join(neg)

    # Merge user custom negatives
    if negative_custom and negative_custom.strip():
        negative_prompt = (f"{negative_prompt}, {negative_custom.strip()}"
                           if negative_prompt else negative_custom.strip())

    return (final_prompt, negative_prompt, token_count)


# ALBABIT-FIX: fields with a live node widget. apply_style_preset() used to
# overwrite these unconditionally on every execution -- the docstring's
# "partial overrides... intentional" claim never held, since every preset
# defines all 9 fields. Widgets are now respected; js/radiance_prompt.js
# fills them on selection and flags later edits with a "●" marker.
_WIDGET_BACKED_PRESET_FIELDS = frozenset({
    "framing", "camera_type", "lens_focal", "aperture_dof",
    "lighting", "style_aesthetic", "color_grading",
})


def apply_style_preset(preset_name, current_settings):
    """
    Apply a style preset to the current settings. Returns a new dict.

    film_stock, shutter_speed and aspect_ratio have no widget and always come
    from the preset. The 7 widget fields are filled by js/radiance_prompt.js
    when the preset is picked, and edits after that are kept.

    3.5.0: a workflow queued through the API, or built without the frontend,
    never ran that JS, so the preset only set the 3 hidden fields. When every
    widget field is still "None" the preset was clearly never applied, and it
    is applied here instead. A graph with any widget set is left as the user
    left it.
    """
    if (
        preset_name == "None (Custom)"
        or preset_name not in CinematicDatasets.PRESET_CONFIGS
    ):
        return current_settings

    preset = CinematicDatasets.PRESET_CONFIGS[preset_name]
    updated = current_settings.copy()
    never_applied = all(
        current_settings.get(k, "None") in ("None", None, "")
        for k in _WIDGET_BACKED_PRESET_FIELDS
    )
    for key, value in preset.items():
        if key not in _WIDGET_BACKED_PRESET_FIELDS or never_applied:
            updated[key] = value
    return updated


def preset_widget_values() -> dict:
    """The widget half of every preset, for js/radiance_prompt.js.

    Served at /radiance/prompt/presets so the frontend no longer keeps its
    own copy of PRESET_CONFIGS (the two had to be edited in step)."""
    return {
        name: {k: v for k, v in cfg.items() if k in _WIDGET_BACKED_PRESET_FIELDS}
        for name, cfg in CinematicDatasets.PRESET_CONFIGS.items()
    }


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return  # tests, or ComfyUI without a server
    server = getattr(PromptServer, "instance", None)
    if server is None or getattr(server, "_radiance_prompt_routes_registered", False):
        return
    server._radiance_prompt_routes_registered = True

    @server.routes.get("/radiance/prompt/presets")
    async def prompt_presets(request):
        return web.json_response(preset_widget_values())


_register_routes()


# Inference only. .eval() does not clear requires_grad on parameters, so an
# unguarded forward still builds and retains an autograd graph.
@torch.no_grad()
def _encode_tokens(clip, tokens):
    """
    Encode CLIP tokens into conditioning.
    Gracefully falls back to encode_from_tokens and formats the output
    in the proper ComfyUI conditioning list structure [[cond, {"pooled_output": pooled}]]
    to prevent sampler crashes on older ComfyUI installations (addresses [BUG-C1]).
    """
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)
    
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]



def _zero_conditioning(cond):
    """Zeroed copy of a conditioning list, as ComfyUI's ConditioningZeroOut.

    Same shapes and keys as the positive, so every sampler accepts it; costs
    no text-encoder pass.
    """
    out = []
    for t in cond:
        d = dict(t[1]) if len(t) > 1 and isinstance(t[1], dict) else {}
        for key in ("pooled_output", "conditioning_lyrics"):
            if torch.is_tensor(d.get(key)):
                d[key] = torch.zeros_like(d[key])
        base = t[0]
        out.append([torch.zeros_like(base) if torch.is_tensor(base) else base, d])
    return out


# ALBABIT-FIX: the architectures whose reference images the Prompt passes on
# (flux2 is Flux.2 Dev, flux2-klein the Klein models).
_REFERENCE_ARCHS = ("qwen_image21", "flux2", "flux2-klein")


# ALBABIT-FIX: Flux.2 reference images, wired as the official Dev and Klein edit
# templates: each scaled to about 1 megapixel and VAE-encoded, the empty latent
# sized on image_1.
def _flux2_references(vae, references):
    latents, empty = [], None
    for name in sorted(references, key=lambda n: int(n.rsplit("_", 1)[-1])):
        scaled = nodes_post_processing.ImageScaleToTotalPixels.execute(references[name], "lanczos", 1.0, 1).args[0]
        latents.append(vae.encode(scaled[:, :, :, :3]))
        if empty is None:
            empty = nodes_flux.EmptyFlux2LatentImage.execute(width=scaled.shape[2], height=scaled.shape[1]).args[0]
    return latents, empty


def _with_reference_latents(cond, latents):
    for latent in latents:
        cond = nodes_edit_model.ReferenceLatent.execute(cond, {"samples": latent}).args[0]
    return cond


# Prompt length each family was trained on. Longer prompts still reach the
# encoder whole (ComfyUI's T5/LLM tokenizers take any length and CLIP is
# chunked by 77); past this the model starts to ignore the tail, so it is
# logged, never cut.
_TRAINED_TOKEN_WINDOW = {"flux": 512, "wan": 512, "sd3": 256, "sd3.5": 256}


class RadianceCinematicPromptEncoder(io.ComfyNode):
    """
    v3.2 — Ultra-clean, simplified cinematic prompt builder with direct CLIP encoding.
    No canvas clutter: inputs reduced from 29 to 10, outputs from 7 to 3.
    Auto-detects model architecture and optimizes prompting internally.
    """

    # Reference shared datasets
    CAMERAS = CinematicDatasets.CAMERAS
    LENSES = CinematicDatasets.LENSES
    APERTURES = CinematicDatasets.APERTURES
    FRAMING = CinematicDatasets.FRAMING
    LIGHTING = CinematicDatasets.LIGHTING
    STYLES = CinematicDatasets.STYLES
    COLOR_GRADING = CinematicDatasets.COLOR_GRADING
    STYLE_PRESETS = CinematicDatasets.STYLE_PRESETS

    # ALBABIT-FIX: V3 schema, for Autogrow inputs. Inputs keep the V1 names and
    # order, so saved workflows load their widget values unchanged.
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RadianceCinematicPromptEncoder",
            **schema_branding("RadianceCinematicPromptEncoder",
                              NODE_DISPLAY_NAME_MAPPINGS["RadianceCinematicPromptEncoder"]),
            description=(
                "Professional cinematic encoder. Detects the text encoder (CLIP, T5 or LLM) "
                "and writes the prompt in the form it reads best; skips the negative encode "
                "on guidance-distilled models."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP model for encoding."),
                # 3.5.0: empty with a placeholder. The old default text was a
                # real value, so an untouched node encoded "A cinematic scene...".
                io.String.Input(
                    "base_prompt", multiline=True, default="", optional=True,
                    placeholder="Describe the subject and the scene",
                    tooltip="Primary subject/scene description.",
                ),
                # ALBABIT-FIX: default to "None (Custom)" so a freshly-added node starts
                # blank rather than silently pre-loaded with "→ Classic Hollywood"'s
                # look. The 7 style widget defaults below are set to "None" to match --
                # js/radiance_prompt.js's resetToCustomDefaults() keeps them in sync
                # whenever "None (Custom)" is (re)selected.
                io.Combo.Input("style_preset", options=cls.STYLE_PRESETS, default="None (Custom)",
                               optional=True, tooltip="One-click style preset."),
                io.Combo.Input("framing", options=cls.FRAMING, default="None",
                               optional=True, tooltip="Shot framing type."),
                io.Combo.Input("camera_type", options=cls.CAMERAS, default="None",
                               optional=True, tooltip="Camera body."),
                io.Combo.Input("lens_focal", options=cls.LENSES, default="None",
                               optional=True, tooltip="Lens + focal length."),
                io.Combo.Input("aperture_dof", options=cls.APERTURES, default="None",
                               optional=True, tooltip="Depth of field."),
                io.Combo.Input("lighting", options=cls.LIGHTING, default="None",
                               optional=True, tooltip="Lighting style."),
                io.Combo.Input("style_aesthetic", options=cls.STYLES, default="None",
                               optional=True, tooltip="Visual aesthetic."),
                io.Combo.Input("color_grading", options=cls.COLOR_GRADING, default="None",
                               optional=True, tooltip="Color grading look."),
                io.Combo.Input(
                    "negative_strength", options=["Off", "Soft", "Standard", "Aggressive"],
                    default="Standard", optional=True,
                    tooltip="Auto-negative strength. 'Soft' is recommended for Flux.",
                ),
                io.String.Input(
                    "negative_prompt", multiline=True, default="", optional=True,
                    tooltip="Custom negative prompt. Appended after auto-negatives.",
                ),
                io.Combo.Input(
                    "negative_mode", options=["Auto", "Always encode", "Zero (skip encode)"],
                    default="Auto", optional=True,
                    tooltip="Auto: on guidance-distilled models (Flux, Flux.2, MiniMax H3) "
                            "with no custom negative, return a zeroed negative instead of "
                            "encoding one: ComfyUI never reads the negative at CFG 1, so this "
                            "saves a full text-encoder pass. Always encode: pick this if you "
                            "run those models above CFG 1 with a negative. Zero: never encode "
                            "the negative.",
                ),
                # ALBABIT-FIX: reference images for Qwen-Image 2.1 and Flux.2 editing,
                # wired as the native nodes. image_1 is the image to edit.
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="Encodes the reference images into the latents the model edits "
                            "from. Flux.2 needs it; Qwen-Image 2.1 without it reads the "
                            "images through the text encoder only.",
                ),
                # ALBABIT-FIX: forceInput -- this is always a wired value from the
                # Loader, never hand-typed; matches the other model_meta inputs
                # added to RUDRA-capable nodes (engine.py, uplift_universal.py).
                io.String.Input(
                    "model_meta", default="", force_input=True, optional=True,
                    tooltip="Optional JSON metadata from Radiance Read Models. "
                            "When connected, architecture detection uses this before "
                            "tokenizer heuristics.",
                ),
                io.Int.Input(
                    "resolution", default=1024, min=0, max=4096, step=32, optional=True,
                    tooltip="Qwen-Image 2.1: reference images are resized to about resolution "
                            "x resolution pixels, at multiples of 32, keeping their aspect "
                            "ratio. 0 keeps each at its own size, rounded to a multiple of 32. "
                            "Flux.2 scales them to about 1 megapixel.",
                ),
                io.Autogrow.Input(
                    "images", optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 17)], min=0),
                    tooltip="Reference images for Qwen-Image 2.1 and Flux.2 editing. image_1 "
                            "is the image to edit, the others are references. With Qwen-Image "
                            "2.1, cite them in the prompt as <image1>, <image2>...",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive",
                                       tooltip="Positive conditioning for the sampler."),
                io.Conditioning.Output(display_name="negative",
                                       tooltip="Negative conditioning for the sampler."),
                io.String.Output(display_name="positive_text",
                                 tooltip="Final positive prompt text that was encoded."),
                io.String.Output(display_name="negative_text",
                                 tooltip="Final negative prompt text that was encoded."),
                io.String.Output(display_name="resolved_arch",
                                 tooltip="Detected architecture used to choose prose vs CLIP-style prompting."),
                io.Int.Output(display_name="token_count",
                              tooltip="Tokenizer-derived positive prompt token count after safety handling."),
                io.Latent.Output(display_name="latent",
                                 tooltip="Empty latent at image_1's size after the resize. Sample on "
                                         "it to edit: any other size shifts the edit. None without "
                                         "reference images."),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        base_prompt="",
        style_preset="None (Custom)",
        framing="None",
        camera_type="None",
        lens_focal="None",
        aperture_dof="None",
        lighting="None",
        style_aesthetic="None",
        color_grading="None",
        negative_strength="Standard",
        negative_prompt="",
        model_meta="",
        negative_mode="Auto",
        vae=None,
        resolution=1024,
        images=None,
    ):
        # ── Validation ──────────────────────────────────────────────────────
        if clip is None:
            raise RuntimeError("CLIP input is None. Connect a valid CLIP model.")
        if not base_prompt or not base_prompt.strip():
            raise ValueError("Cinematic Encoder: the prompt is empty. Describe the "
                             "subject and the scene in base_prompt.")
        negative_prompt_in = negative_prompt

        # ── Resolve architecture automatically ──────────────────────────────
        resolved_arch = _detect_arch_from_clip(clip, "Auto", model_meta)

        # ALBABIT-FIX: only Qwen-Image 2.1 and Flux.2 read reference images here;
        # any other encoder would drop them without a word.
        references = {name: image for name, image in (images or {}).items() if image is not None}
        if references and resolved_arch not in _REFERENCE_ARCHS:
            raise ValueError(
                f"Prompt: reference images are read by Qwen-Image 2.1 and Flux.2 only, and "
                f"the text encoder resolves to '{resolved_arch}'. Connect model_meta from "
                f"the Loader, or disconnect the images.")
        if references and resolved_arch != "qwen_image21" and vae is None:
            raise ValueError("Prompt: Flux.2 reads reference images as VAE latents. "
                             "Connect the vae input.")

        # ── Apply style preset ──────────────────────────────────────────────
        settings = {
            "framing": framing, "camera_type": camera_type,
            "lens_focal": lens_focal, "aperture_dof": aperture_dof,
            "lighting": lighting, "style_aesthetic": style_aesthetic,
            "film_stock": "None", "shutter_speed": "None",
            "color_grading": color_grading, "aspect_ratio": "None",
        }
        if style_preset != "None (Custom)":
            settings = apply_style_preset(style_preset, settings)

        # ── Build prompt ────────────────────────────────────────────────────
        # ALBABIT-FIX: ltxav uses the same _build_prose_prompt path as Flux/WAN.
        # Gemma3-12B's audio head learns near-zero influence from purely
        # visual gear terms, no corruption. Note: quoted dialogue in
        # base_prompt (e.g. 'says "Hello!"') WILL generate audible speech,
        # intended LTX-AV behaviour, not a bug.
        final_prompt, negative_prompt, _ = build_cinematic_prompt_v3(
            base_prompt=base_prompt,
            base_prompt_b="",
            active_prompt="A",
            framing=settings["framing"],
            camera_type=settings["camera_type"],
            lens_focal=settings["lens_focal"],
            aperture_dof=settings["aperture_dof"],
            lighting=settings["lighting"],
            style_aesthetic=settings["style_aesthetic"],
            film_stock=settings["film_stock"],
            shutter_speed=settings["shutter_speed"],
            color_grading=settings["color_grading"],
            aspect_ratio=settings["aspect_ratio"],
            custom_details="",
            year_era=DEFAULT_YEAR,
            negative_strength=negative_strength,
            negative_custom=negative_prompt,
            lora_keywords="",
            use_break=False,  # Handled below
            target_arch=resolved_arch,
            scene_mood="None",
            subject_weight=1.0,
            art_direction="",
            prompt_weight_mode="balanced",
        )

        # ── Formatting clean-up (spacing, stray commas) ───────────────────────
        # Auto-fix punctuation/spacing. Prose archs don't need danbooru tag enhancers.
        final_prompt = enhance_prompt_grammar(final_prompt, "Grammar & Formatting", arch=resolved_arch)

        # ── Tokenize, count, encode ─────────────────────────────────────────
        # 3.5.0: no BREAK insertion and no truncation. ComfyUI has no BREAK
        # syntax (that is A1111), so the word "break" was encoded into the
        # prompt; and prompts over 77 CLIP tokens were cut to the first chunk,
        # which dropped camera, lens and lighting (they come after the
        # subject). ComfyUI already chunks long CLIP prompts and T5 / LLM
        # tokenizers take any length.
        pos_tokens = clip.tokenize(final_prompt)
        real_count = _real_token_count(clip, final_prompt, tokens=pos_tokens)
        window = _TRAINED_TOKEN_WINDOW.get(resolved_arch)
        if window and real_count > window:
            logger.warning(
                "[Encoder] %d-token prompt on %s, trained on %d: the tail may carry "
                "little weight. Nothing was cut.", real_count, resolved_arch, window)

        # ALBABIT-FIX: with reference images the native TextEncodeQwenImage21 encodes
        # both prompts: it resizes the images, shows them to the vision tower,
        # splices their VAE latents in and sizes the latent on image_1. Flux.2
        # encodes the text as usual and adds the references to both prompts.
        flux2_refs = None
        if references and resolved_arch == "qwen_image21":
            positive_cond, native_negative, latent = nodes_qwen.TextEncodeQwenImage21.execute(
                clip=clip, prompt=final_prompt, negative_prompt=negative_prompt, vae=vae,
                resolution=resolution, images=references).args
        else:
            positive_cond, latent = _encode_tokens(clip, pos_tokens), None
            if references:
                flux2_refs, latent = _flux2_references(vae, references)
                positive_cond = _with_reference_latents(positive_cond, flux2_refs)

        user_negative = bool(negative_prompt_in and negative_prompt_in.strip())
        skip_negative = (
            negative_mode == "Zero (skip encode)"
            or (negative_mode == "Auto" and not user_negative
                and resolved_arch in _GUIDANCE_DISTILLED_ARCHS)
        )
        if skip_negative:
            negative_cond = _zero_conditioning(positive_cond)   # keeps reference latents
            negative_prompt = ""        # nothing was encoded; say so
        elif references and resolved_arch == "qwen_image21":
            negative_cond = native_negative
        else:
            safe_negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            negative_cond = _encode_tokens(clip, clip.tokenize(safe_negative))
            if flux2_refs:
                negative_cond = _with_reference_latents(negative_cond, flux2_refs)

        # ALBABIT-FIX: weak_neg_arch only known post-execution (resolved_arch
        # depends on the real CLIP/model_meta), so js/radiance_prompt.js flags
        # negative_prompt with a label marker via onExecuted, same convention
        # as engine.py's rudra_fallback/log_overexposure_risk.
        return {
            "ui": {"weak_neg_arch": [resolved_arch in _WEAK_NEG_ARCHS],
                   "negative_skipped": [bool(skip_negative)]},
            "result": (
                positive_cond,
                negative_cond,
                final_prompt,
                negative_prompt,
                resolved_arch,
                int(real_count),
                latent,
            ),
        }

# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceCinematicPromptEncoder": RadianceCinematicPromptEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceCinematicPromptEncoder": "◎ Cinematic Prompt Encoder",
}
