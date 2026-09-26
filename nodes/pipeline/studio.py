from radiance.film.camera_profiles import (
    CAMERAS,
    LENSES,
    APERTURES,
    SHUTTER_ANGLES,
    ISO_SETTINGS,
)


class RadianceCinemaStudio:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Generate authentic cinematic prompts using real-world camera and lens profiles.
    Returns both the prompt text and technical data for downstream use.
    """

    # Pre-calculate flattened lists for dropdowns
    CAMERA_LIST = ["None"] + sorted(
        [
            f"{brand} {model}"
            for brand, models in CAMERAS.items()
            for model in models.keys()
        ]
    )

    LENS_LIST = ["None"] + sorted(
        [
            f"{brand} {series}"
            for brand, series_dict in LENSES.items()
            for series in series_dict.keys()
        ]
    )

    # Collect all unique focal lengths for a generic list, or provide a comprehensive one
    # For simplicity, we'll use a standard list of focal lengths, but the node logic
    # will try to match them to the selected lens series if possible, or just append them.
    FOCAL_LENGTHS = [
        "Variable / Zoom",
        "8mm Fisheye",
        "12mm Ultra-Wide",
        "14mm Ultra-Wide",
        "16mm Wide",
        "18mm Wide",
        "21mm Wide",
        "24mm Wide",
        "28mm Wide",
        "32mm Wide",
        "35mm Classic",
        "40mm Natural",
        "50mm Standard",
        "65mm Portrait",
        "75mm Portrait",
        "85mm Portrait",
        "100mm Telephoto",
        "135mm Telephoto",
        "150mm Telephoto",
        "180mm Telephoto",
        "200mm Long Lens",
        "300mm Super Tele",
    ]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cinematic shot of...",
                        "dynamicPrompts": False,
                        "tooltip": "Subject and scene description. It opens the prompt (after the shot type, if set) and gets a closing full stop if it has none.",
                    },
                ),
                "camera": (
                    cls.CAMERA_LIST,
                    {
                        "default": (
                            cls.CAMERA_LIST[1] if len(cls.CAMERA_LIST) > 1 else "None"
                        ),
                        "tooltip": "Camera body named in the prompt as 'Shot on <camera>'; its sensor is listed in the technical report. Text only, no image processing. None omits it.",
                    },
                ),
                "lens_series": (
                    cls.LENS_LIST,
                    {"default": cls.LENS_LIST[1] if len(cls.LENS_LIST) > 1 else "None",
                     "tooltip": "Lens set named in the prompt as 'using <lens>'. None omits it."},
                ),
                "focal_length": (cls.FOCAL_LENGTHS, {"default": "50mm Standard", "tooltip": "Focal length added as 'at <N>mm' (the descriptive word is dropped). Variable / Zoom omits it."}),
                "aperture": (APERTURES, {"default": "T2.0 (Cinematic Separation)", "tooltip": "T-stop written into the prompt as 'aperture T<N>', a depth-of-field cue for the model. Always included."}),
                "shutter": (
                    SHUTTER_ANGLES,
                    {"default": "180° (Standard Motion - 1/48s)",
                     "tooltip": "Shutter angle written as 'shutter <angle>°', a motion-blur cue for the model. Always included."},
                ),
                "iso": (ISO_SETTINGS, {"default": "800 ISO (Native Digital)", "tooltip": "ISO written as 'ISO <N>', a noise and grain cue for the model. Always included."}),
            },
            "optional": {
                "shot_type": (
                    [
                        "None",
                        "Extreme Wide Shot",
                        "Wide Shot",
                        "Full Shot",
                        "Medium Wide Shot",
                        "Medium Shot",
                        "Medium Close-Up",
                        "Close-Up",
                        "Extreme Close-Up",
                        "Macro Detail",
                    ],
                    {"default": "Medium Shot",
                     "tooltip": "Framing prefixed to the prompt as '<shot type> of <base prompt>'. None omits it."},
                ),
                "camera_movement": (
                    [
                        "None",
                        "Static Tripod",
                        "Handheld Shake",
                        "Steadicam Smooth",
                        "Dolly In",
                        "Dolly Out",
                        "Truck Left",
                        "Truck Right",
                        "Crane Up",
                        "Crane Down",
                        "Dutch Angle",
                        "Whip Pan",
                    ],
                    {"default": "None",
                     "tooltip": "Camera move added as its own sentence after the subject. None omits it."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "technical_data_str")
    FUNCTION = "generate_cinema_prompt"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Build a text prompt from camera, lens, exposure and framing choices using real-world camera and lens names. Outputs the prompt plus a plain-text technical report; it does not process images."

    def generate_cinema_prompt(
        self,
        base_prompt,
        camera,
        lens_series,
        focal_length,
        aperture,
        shutter,
        iso,
        shot_type="None",
        camera_movement="None",
    ):

        # Build the technical description
        tech_parts = []

        # 1. Shot Type & Movement
        if shot_type != "None":
            tech_parts.append(shot_type)
        if camera_movement != "None":
            tech_parts.append(camera_movement)

        # 2. Camera Choice
        if camera != "None":
            tech_parts.append(f"Shot on {camera}")

        # 3. Lens & Focal Length
        if lens_series != "None":
            # Extract just the name parts if needed, but the full string is usually descriptive enough
            # "ARRI Signature Primes" -> "ARRI Signature Primes"
            tech_parts.append(f"using {lens_series}")

        if focal_length != "Variable / Zoom":
            # Clean up focal length string: "50mm Standard" -> "50mm"
            focal_val = focal_length.split(" ")[0]
            tech_parts.append(f"at {focal_val}")

        # 4. Settings
        # Clean aperture: "T2.0 (Cinematic Separation)" -> "T2.0"
        if aperture:
            ap_val = aperture.split(" ")[0]
            tech_parts.append(f"aperture {ap_val}")

        # Clean shutter: "180° (Standard Motion - 1/48s)" -> "180° shutter"
        if shutter:
            shut_val = shutter.split(" ")[0]
            tech_parts.append(f"shutter {shut_val}")

        # Clean ISO: "800 ISO (Native Digital)" -> "ISO 800"
        if iso:
            iso_val = iso.split(" ")[0]
            tech_parts.append(f"ISO {iso_val}")

        # Assemble final prompt
        # "[Shot Type] of [Base Prompt]. Shot on [Camera] using [Lens] at [Focal]..."

        # Start with base prompt context if it's not a sentence fragment
        prompt_parts = []

        # If we have a shot type, prefix it: "Medium Shot of..."
        prefix = ""
        if shot_type != "None":
            prefix = f"{shot_type} of "
            # Remove shot_type from tech_parts to avoid duplication if we use it here
            # But kept in tech_parts logic above for consistent list building,
            # let's just make the sentence flow naturally.

        # Construct the core phrase
        core_sentence = f"{prefix}{base_prompt}"
        if not core_sentence.endswith((".", "!", "?")):
            core_sentence += "."
        prompt_parts.append(core_sentence)

        # Append technical details
        # Remove shot_type and movement from tech string if we integrated them?
        # Actually, "Medium Shot of [subject]. Static Tripod. Shot on..." works well.

        # Let's reconstruct the technical sentence completely for better flow
        tech_sentence_parts = []

        if camera_movement != "None":
            tech_sentence_parts.append(camera_movement + ".")

        # "Shot on ARRI Alexa 35 using ARRI Signature Primes at 50mm, aperture T2.0, shutter 180°, ISO 800."
        cam_lens_part = []
        if camera != "None":
            cam_lens_part.append(f"Shot on {camera}")

        if lens_series != "None":
            cam_lens_part.append(f"using {lens_series}")

        if focal_length != "Variable / Zoom":
            focal_val = focal_length.split(" ")[0]
            cam_lens_part.append(f"at {focal_val}")

        if cam_lens_part:
            tech_sentence_parts.append(" ".join(cam_lens_part))

        # Settings part
        settings_part = []
        if aperture:
            settings_part.append(f"aperture {aperture.split(' ')[0]}")
        if shutter:
            settings_part.append(f"shutter {shutter.split(' ')[0]}")
        if iso:
            settings_part.append(f"ISO {iso.split(' ')[0]}")

        if settings_part:
            # Join with commas and add to the sentence
            if cam_lens_part:
                # "Shot on ... at 50mm, aperture T2.0..."
                tech_sentence_parts[-1] += ", " + ", ".join(settings_part) + "."
            else:
                # "Aperture T2.0, shutter 180°..."
                tech_sentence_parts.append(", ".join(settings_part).capitalize() + ".")

        # Join everything
        final_prompt = " ".join(prompt_parts + tech_sentence_parts)

        # Generate technical data report
        tech_data = {
            "camera": camera,
            "lens": lens_series,
            "focal_length": focal_length,
            "aperture": aperture,
            "shutter": shutter,
            "iso": iso,
            "shot_type": shot_type,
            "movement": camera_movement,
        }

        # Search for sensor details
        sensor_info = "Unknown Sensor"
        for brand, models in CAMERAS.items():
            for model, specs in models.items():
                if f"{brand} {model}" == camera:
                    sensor_info = f"{specs['sensor']} ({specs['size']})"
                    break

        tech_report = (
            f"CAMERA: {camera}\n"
            f"SENSOR: {sensor_info}\n"
            f"LENS: {lens_series} @ {focal_length}\n"
            f"SETTINGS: {aperture} | {shutter} | {iso}\n"
            f"SHOT: {shot_type} | {camera_movement}"
        )

        return (final_prompt, tech_report)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {"RadianceCinemaStudio": RadianceCinemaStudio}

NODE_DISPLAY_NAME_MAPPINGS = {"RadianceCinemaStudio": "◎ Radiance Manager"}
