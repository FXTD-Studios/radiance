"""
FXTD STUDIO - PROMPT ENGINEERING NODES
--------------------------------------
Professional tools for constructing cinematic, high-fidelity prompts for Flux and other diffusion models.
Includes "Prompt Machine" for text output and "Cinematic Encoder" for direct CLIP conditioning.

Features:
- Industry-standard cinematic terminology
- One-click style presets (Film Noir, Cyberpunk, etc.)
- Direct CLIP encoding with clip skip support
- Smart auto-negative generation

Example Usage (Cinematic Encoder):
    Connect CLIP model → Set base prompt → Select style preset or customize → 
    Connect positive/negative outputs directly to sampler
"""

import random


# ═══════════════════════════════════════════════════════════════════════════════
#                         SHARED CINEMATIC DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

class CinematicDatasets:
    """Shared datasets for all cinematic prompt nodes. Single source of truth."""
    
    CAMERAS = [
        "None",
        "ARRI Alexa 65 (IMAX)", "ARRI Alexa Mini LF", "ARRI Alexa 35",
        "Sony Venice 2", "Sony FX9", "Sony A7S III",
        "RED V-Raptor XL", "RED Komodo", "RED Monstro 8K VV",
        "Panavision Millennium DXL2", "Panavision Panaflex Gold II (35mm)",
        "IMAX 15/70mm Film Camera", "Bolex H16 (16mm Film)", "Super 8mm Camera",
        "Canon C700 FF", "Blackmagic URSA Mini Pro 12K",
        "GoPro Hero 12", "iPhone 15 Pro Max", "Polaroid SX-70",
        "Vintage Daguerreotype Camera"
    ]
    
    LENSES = [
        "None",
        # Prime Focal Lengths
        "14mm Ultra-Wide Angle", "24mm Wide Angle", "35mm Classic Wide",
        "50mm Standard Prime", "85mm Portrait Prime", "105mm Macro",
        "135mm Medium Telephoto", "200mm Telephoto Compression", "600mm Super Telephoto",
        # Specialty Lenses
        "Anamorphic Lens", "Fish-Eye Lens", "Tilt-Shift Lens",
        # Cinema Primes (High-End)
        "Cooke S7/i Full Frame", "Cooke Speed Panchro Vintage",
        "Zeiss Master Prime", "Zeiss Supreme Prime",
        "ARRI/Zeiss Signature Prime", "ARRI Master Anamorphic",
        "Panavision Primo 70", "Panavision C-Series Anamorphic",
        # Vintage & Character Lenses
        "Canon K35 Vintage", "Canon FD 50mm L", 
        "Helios 44-2 58mm (Swirly Bokeh)", "Lensbaby Velvet (Soft Focus)",
        "Petzval 85mm (Classic Swirl)",
        # Modern Professional
        "Leica Summilux 50mm f/1.4", "Leica Summicron 35mm",
        "Sigma Art 35mm f/1.4", "Sigma Art 85mm f/1.4",
        "Sony G Master 24-70mm", "Sony G Master 85mm",
        # Specialty & Macro
        "Laowa Probe Lens (Macro)", "Freefly Wave (High-Speed)",
        "Angenieux Optimo Zoom", "Fujinon Premista Zoom"
    ]
    
    APERTURES = [
        "None",
        "f/0.95 (Razor Thin DoF)", "f/1.2 (Dreamy Bokeh)", "f/1.8 (Soft Background)",
        "f/2.8 (Cinematic Separation)", "f/4.0 (Balanced)", "f/5.6 (Sharp Subject)",
        "f/8.0 (Deep Focus)", "f/11 (Landscape Sharpness)", "f/16 (Everything in Focus)",
        "f/22 (Diffraction Starbursts)"
    ]
    
    FRAMING = [
        "None",
        "Extreme Close-Up (ECU)", "Close-Up (CU)", "Medium Close-Up (MCU)",
        "Medium Shot (MS)", "Cowboy Shot (American Shot)", "Full Body Shot (Wide)",
        "Extreme Wide Shot (EWS)", "Establishing Shot", "Over-The-Shoulder (OTS)",
        "Point of View (POV)", "Low Angle (Hero Shot)", "High Angle (Vulnerability)",
        "Bird's Eye View (Overhead)", "Worm's Eye View", "Dutch Angle (Canted)",
        "Symmetrical Composition", "Rule of Thirds"
    ]
    
    LIGHTING = [
        "None",
        "Rembrandt Lighting", "Chiaroscuro (High Contrast)", "Film Noir Lighting",
        "Split Lighting", "Butterfly Lighting", "Paramount Lighting",
        "Soft Window Light", "Golden Hour (Magic Hour)", "Blue Hour",
        "Cinematic Haze / Volumetric Fog", "God Rays (Crepuscular Rays)",
        "Neon Cyberpunk Lighting", "Practical Lighting", "Bioluminescence",
        "Studio Strobe 3-Point Setup", "Ring Light", "Candlelight",
        "Moonlight", "Overcast Soft Light", "Harsh Sunlight"
    ]
    
    STYLES = [
        "None",
        "Photorealistic (Raw)", "Cinematic Movie Still", "Hyper-Realism",
        "Editorial Photography", "National Geographic Style", "Documentary Texture",
        "Vintage 1990s VHS", "Analog Film (Kodak Portra 400)", "Fujifilm Velvia 50",
        "Black and White (Ilford HP5)", "Monochrome Noir",
        "CGI 3D Render (Octane)", "Unreal Engine 5", "Pixar Animation Style",
        "Anime (Makoto Shinkai)", "Oil Painting (Classic)", "Concept Art",
        "Cyberpunk 2077 Aesthetic", "Wes Anderson Symmetric", "Tarantino Violence",
        "Kubrick One-Point Perspective", "Blade Runner Atmosphere"
    ]
    
    FILM_STOCKS = [
        "None",
        "Kodak Vision3 500T", "Kodak Portra 400", "Kodak Ektar 100",
        "Kodak Tri-X 400 (B&W)", "Fujifilm Pro 400H", "Cinestill 800T",
        "Ilford Delta 3200", "Polaroid 600", "Wet Plate Collodion"
    ]

    SHUTTER_SPEEDS = [
        "None",
        "1/50th sec (Standard Motion Blur)", "1/1000th sec (Frozen Action)", 
        "Long Exposure (Light Trails)", "Slow Shutter (Dreamy Blur)"
    ]

    COLOR_GRADING = [
        "None",
        "Teal and Orange (Blockbuster)", "Bleach Bypass (Gritty)", 
        "Technicolor (Vintage)", "Cross Processed", "Desaturated (Muted)", 
        "Vibrant High Contrast", "Sepia Tone", "Monochrome High Key",
        "Cyberpunk Neon Grading", "Pastel Soft Tones"
    ]
    
    ASPECT_RATIOS = [
        "None",
        "16:9 (Widescreen)", "2.39:1 (Anamorphic Scope)", 
        "4:3 (Academy Ratio)", "1:1 (Square)", "9:16 (Social Vertical)",
        "21:9 (Ultrawide)"
    ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    #                         ONE-CLICK STYLE PRESETS
    # ═══════════════════════════════════════════════════════════════════════════
    
    STYLE_PRESETS = [
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
    ]
    
    # Preset configurations: {preset_name: {setting: value, ...}}
    PRESET_CONFIGS = {
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
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                         SHARED PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_cinematic_prompt(base_prompt, framing, camera_type, lens_focal, aperture_dof, 
                           lighting, style_aesthetic, film_stock="None", shutter_speed="None", 
                           color_grading="None", aspect_ratio="None", custom_details="", 
                           year_era=2024, negative_strength="Standard"):
    """
    Shared prompt builder function used by all cinematic prompt nodes.
    Returns tuple of (final_prompt, negative_prompt).
    negative_strength options: "Off", "Soft", "Standard", "Aggressive"
    """
    
    def clean(val):
        """Remove 'None' values and return empty string."""
        if val == "None": 
            return ""
        return val

    parts = []
    
    # 1. Style & Era (Prefix context)
    style = clean(style_aesthetic)
    
    # 2. Framing & Subject
    frame = clean(framing)
    if frame:
        parts.append(f"{frame} of {base_prompt}.")
    else:
        parts.append(f"{base_prompt}.")
        
    # 3. Technical Camera Specs
    cam = clean(camera_type)
    lens = clean(lens_focal)
    ap = clean(aperture_dof)
    shut = clean(shutter_speed)
    
    tech_stack = []
    if cam: 
        tech_stack.append(f"Shot on {cam}")
    if lens: 
        tech_stack.append(f"with {lens}")
    if ap: 
        tech_stack.append(f"at {ap}")
    if shut: 
        tech_stack.append(f"shutter speed {shut}")
    
    if tech_stack:
        parts.append(" ".join(tech_stack) + ".")
        
    # 4. Lighting & Color
    light = clean(lighting)
    grade = clean(color_grading)
    
    if light: 
        parts.append(f"Lighting is {light}.")
    if grade: 
        parts.append(f"Color graded in {grade}.")
        
    # 5. Film Stock / Finish / Era
    stock = clean(film_stock)
    current_year_context = ""
    
    if year_era != 2024 or "Vintage" in style or "Classic" in style:
        current_year_context = f"Est. Year {year_era}."
    
    finish_stack = []
    if style: 
        finish_stack.append(style)
    if stock: 
        finish_stack.append(f"on {stock}")
    if current_year_context: 
        finish_stack.append(current_year_context)
    
    if finish_stack:
        parts.append(", ".join(finish_stack) + ".")
        
    # 6. Aspect Ratio Token
    ar = clean(aspect_ratio)
    if ar:
        parts.append(f"{ar} format.")
        
    # 7. Custom Details
    if custom_details.strip():
        parts.append(custom_details)
        
    # Join final prompt
    final_prompt = " ".join([p for p in parts if p]).strip()
    
    # --- AUTO NEGATIVE GENERATION ---
    # --- AUTO NEGATIVE GENERATION ---
    negative_prompt = ""
    
    if negative_strength != "Off":
        neg_terms = []
        
        # Soft: Minimal cleanup
        if negative_strength in ["Soft", "Standard", "Aggressive"]:
            neg_terms.extend(["blur", "low quality", "watermark", "text"])
            
        # Standard: Add deformities and duplicates
        if negative_strength in ["Standard", "Aggressive"]:
            neg_terms.extend(["deformed", "ugly", "duplicate", "disfigured", "bad anatomy"])
            
            # Context-aware negatives (Standard+)
            if "Photorealistic" in style or "Cinematic" in style or "Documentary" in style:
                neg_terms.extend(["cartoon", "anime", "illustration", "painting", "cgi", "3d render", "drawing", "sketch"])
            elif "Anime" in style:
                neg_terms.extend(["photograph", "realistic", "photo", "photorealistic", "3d"])
            elif "Painting" in style or "Oil" in style:
                neg_terms.extend(["photograph", "realistic", "photo", "digital", "3d render"])
            elif "CGI" in style or "Unreal" in style or "3D" in style:
                neg_terms.extend(["photograph", "realistic", "2d", "flat", "hand drawn"])
        
        # Aggressive: Add everything specific
        if negative_strength == "Aggressive":
             neg_terms.extend(["mutated", "extra limbs", "missing limbs", "floating limbs", "disconnected limbs",
                               "pixelated", "noise", "grainy", "cropped", "out of frame", "worst quality", "lowres"])

        negative_prompt = ", ".join(neg_terms)
    
    return (final_prompt, negative_prompt)


def apply_style_preset(preset_name, current_settings):
    """
    Apply a style preset to the current settings.
    Returns updated settings dict.
    """
    if preset_name == "None (Custom)" or preset_name not in CinematicDatasets.PRESET_CONFIGS:
        return current_settings
    
    preset = CinematicDatasets.PRESET_CONFIGS[preset_name]
    updated = current_settings.copy()
    updated.update(preset)
    return updated


# ═══════════════════════════════════════════════════════════════════════════════
#                     CINEMATIC ENCODER (CONDITIONING OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════════

class FXTDCinematicPromptEncoder:
    """
    All-in-One Cinematic Prompt Encoder.
    Combines the Cinematic Prompt Machine with CLIP encoding to output CONDITIONING directly.
    Eliminates the need for a separate CLIP Text Encode node.
    
    Features:
    - One-click style presets for instant cinematic looks
    - CLIP skip support for different encoding depths
    - Direct CONDITIONING output ready for samplers
    - Smart auto-negative generation
    
    Example:
        CLIP → Encoder → positive/negative → Sampler
        (Replaces: Prompt Machine → CLIP Encode × 2 → Sampler)
    """
    
    # Reference shared datasets
    CAMERAS = CinematicDatasets.CAMERAS
    LENSES = CinematicDatasets.LENSES
    APERTURES = CinematicDatasets.APERTURES
    FRAMING = CinematicDatasets.FRAMING
    LIGHTING = CinematicDatasets.LIGHTING
    STYLES = CinematicDatasets.STYLES
    FILM_STOCKS = CinematicDatasets.FILM_STOCKS
    SHUTTER_SPEEDS = CinematicDatasets.SHUTTER_SPEEDS
    COLOR_GRADING = CinematicDatasets.COLOR_GRADING
    ASPECT_RATIOS = CinematicDatasets.ASPECT_RATIOS
    STYLE_PRESETS = CinematicDatasets.STYLE_PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the prompt into conditioning."}),
            },
            "optional": {
                "base_prompt": ("STRING", {"multiline": True, "default": "A cinematic scene...", "tooltip": "Your core subject/scene description."}),
                "style_preset": (cls.STYLE_PRESETS, {"default": "🎬 Classic Hollywood", "tooltip": "One-click style preset for instant cinematic look."}),
                "framing": (cls.FRAMING, {"default": "Medium Shot (MS)", "tooltip": "Camera framing and shot type."}),
                "camera_type": (cls.CAMERAS, {"default": "ARRI Alexa 35", "tooltip": "Camera body type for realism cues."}),
                "lens_focal": (cls.LENSES, {"default": "50mm Standard Prime", "tooltip": "Lens choice affects bokeh, distortion, and feel."}),
                "aperture_dof": (cls.APERTURES, {"default": "f/2.8 (Cinematic Separation)", "tooltip": "Depth of field control."}),
                "lighting": (cls.LIGHTING, {"default": "Cinematic Haze / Volumetric Fog", "tooltip": "Lighting style and atmosphere."}),
                "style_aesthetic": (cls.STYLES, {"default": "Photorealistic (Raw)", "tooltip": "Overall visual style and aesthetic."}),
                "film_stock": (cls.FILM_STOCKS, {"default": "None", "tooltip": "Film stock emulation for color/grain."}),
                "shutter_speed": (cls.SHUTTER_SPEEDS, {"default": "None", "tooltip": "Motion blur characteristics."}),
                "color_grading": (cls.COLOR_GRADING, {"default": "None", "tooltip": "Color grading look."}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "None", "tooltip": "Frame aspect ratio."}),
                "custom_details": ("STRING", {"multiline": False, "default": "", "tooltip": "Additional custom prompt details."}),
                "year_era": ("INT", {"default": 2024, "min": 1800, "max": 2100, "step": 1, "tooltip": "Time period for era-specific looks."}),
                "negative_strength": (["Off", "Soft", "Standard", "Aggressive"], {"default": "Standard", "tooltip": "Strength of auto-generated negative prompt."}),
                "clip_skip": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1, "tooltip": "Number of CLIP layers to skip (0 = use all layers)."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning for the sampler (encoded cinematic prompt).",
        "Negative conditioning for the sampler (auto-generated or empty).",
    )
    FUNCTION = "encode_cinematic"
    CATEGORY = "FXTD Studios/Radiance/Utilities"
    DESCRIPTION = "All-in-one cinematic prompt builder with direct CLIP encoding. Clean interface: clip in → CONDITIONING out."

    def encode_cinematic(self, clip, base_prompt="A cinematic scene...", style_preset="🎬 Classic Hollywood", 
                         framing="Medium Shot (MS)", camera_type="ARRI Alexa 35", 
                         lens_focal="50mm Standard Prime", aperture_dof="f/2.8 (Cinematic Separation)", 
                         lighting="Cinematic Haze / Volumetric Fog", style_aesthetic="Photorealistic (Raw)",
                         film_stock="None", shutter_speed="None", color_grading="None", aspect_ratio="None",
                         custom_details="", year_era=2024, negative_strength="Standard", clip_skip=0):
        
        # Validate CLIP input
        if clip is None:
            raise RuntimeError("ERROR: CLIP input is invalid (None). Please connect a valid CLIP model.")
        
        # Build current settings dict
        settings = {
            "framing": framing,
            "camera_type": camera_type,
            "lens_focal": lens_focal,
            "aperture_dof": aperture_dof,
            "lighting": lighting,
            "style_aesthetic": style_aesthetic,
            "film_stock": film_stock,
            "shutter_speed": shutter_speed,
            "color_grading": color_grading,
            "aspect_ratio": aspect_ratio,
        }
        
        # Apply preset if selected
        if style_preset != "None (Custom)":
            settings = apply_style_preset(style_preset, settings)
        
        # Generate prompts using shared builder
        final_prompt, negative_prompt = build_cinematic_prompt(
            base_prompt=base_prompt,
            framing=settings["framing"],
            camera_type=settings["camera_type"],
            lens_focal=settings["lens_focal"],
            aperture_dof=settings["aperture_dof"],
            lighting=settings["lighting"],
            style_aesthetic=settings["style_aesthetic"],
            film_stock=settings.get("film_stock", "None"),
            shutter_speed=settings.get("shutter_speed", "None"),
            color_grading=settings.get("color_grading", "None"),
            aspect_ratio=settings.get("aspect_ratio", "None"),
            custom_details=custom_details,
            year_era=year_era,
            negative_strength=negative_strength
        )
        
        # --- CLIP ENCODING WITH CLIP SKIP ---
        # Apply clip skip if specified
        if clip_skip > 0:
            clip = clip.clone()
            clip.clip_layer(-clip_skip)
        
        # Encode positive prompt (includes cinematic details)
        pos_tokens = clip.tokenize(final_prompt)
        positive_cond = clip.encode_from_tokens_scheduled(pos_tokens)
        
        # Encode negative prompt
        if negative_prompt:
            neg_tokens = clip.tokenize(negative_prompt)
            negative_cond = clip.encode_from_tokens_scheduled(neg_tokens)
        else:
            neg_tokens = clip.tokenize("")
            negative_cond = clip.encode_from_tokens_scheduled(neg_tokens)
        
        return (positive_cond, negative_cond)
