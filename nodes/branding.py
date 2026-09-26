"""Public node naming and menu taxonomy for Radiance.

ComfyUI keeps node type keys as workflow compatibility identifiers. This module
centralizes the user-facing brand and category names without renaming those keys.

Naming scheme (Option C — compositor-friendly hybrid):
  * Compositor-native nodes (Color, HDR finishing, VFX, Load & Save, Review,
    Upscale, Pipeline) use bare, industry-standard names — the menu tab carries
    the brand, exactly like Nuke shows ``Grade`` under Color.
  * Generation nodes (Core dashboards, Generate, Video, Developer) keep a plain
    ``Radiance`` prefix so the diffusion layer stays clearly badged.
The ``◎`` glyph and marketing words (Pro/Smart/Ultra/Cinematic) are dropped.
Search aliases keep every node discoverable by typing "radiance".
"""
from __future__ import annotations

import logging
import re
from typing import Any, MutableMapping

BRAND = "Radiance"
DISPLAY_PREFIX = "◎ Radiance"  # retained for backward-compatible imports
MENU_ROOT = "FXTD STUDIOS/Radiance"

logger = logging.getLogger("radiance.branding")

MENU_STRUCTURE = {
    "Core": "Manager, workspace, resolution, and everyday Radiance utilities.",
    "Load & Save": "Readers, writers, EXR, image, video, and file exchange.",
    "Generate": "Samplers, loaders, prompts, LoRA, regional conditioning, and denoise.",
    "Color": "White balance, CDL, curves, LUTs, OCIO, grades, and color checks.",
    "HDR": "HDR uplift, ACES, tone mapping, analysis, encoding, and delivery.",
    "VFX": "Plate prep, masks, inpaint, depth, optics, motion, multipass, and scene cuts.",
    "Video": "Text-to-video, image-to-video, temporal generation, and video HDR.",
    "Upscale": "AI upscaling, repair, enhancement, and restoration.",
    "Review": "Viewer, scopes, preview servers, contact sheets, and QC.",
    "Pipeline": "DCC bridges, metadata, audio, project handoff, and studio integration.",
    "Developer": "Training, diagnostics, and developer-only utilities.",
}

DEFAULT_MENU_SECTION = "Core"

# Sections whose nodes keep the "Radiance" prefix (the generation/AI layer).
# Empty = every node uses a bare, Nuke-style name (no "Radiance " prefix).
# Nodes stay findable by typing "radiance" via the injected search aliases.
GENERATION_SECTIONS: set = set()

# Marketing words stripped from any node label.
_MARKETING = re.compile(r"\b(Pro|Smart|Ultra|Cinematic)\b", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
#  Declared menu section, per node.
#
#  This table is the answer; `classify_menu_section`'s keyword rules are only
#  the fallback for a node that is not listed here.
#
#  The keyword classifier was the primary mechanism until 2026-08. It reads the
#  node key, display name, module path and category as one string and returns on
#  the first rule that matches, which makes placement depend on rule ORDER
#  rather than on anything about the node. "mask" is tested for VFX several
#  rules before "conditioning" is tested for Generate, so RadianceEnergyMask —
#  which produces CONDITIONING for the sampler — landed one menu away from the
#  only node that reads it. Every such case had to be discovered by eye and
#  patched into SECTION_OVERRIDES afterwards.
#
#  Declaring the section removes the guess. Adding a node to this table is one
#  line, and `tests/test_menu_sections.py` fails if a registered node is
#  missing from it, so the omission is caught at commit time rather than by a
#  user wondering where their node went.
# ─────────────────────────────────────────────────────────────────────────────

NODE_SECTIONS = {
    # ── Color ─────────────────────────────────────────────────────────────
    "RadianceApplyGradeInfo": "Color",
    "RadianceCDLExport": "Color",
    "RadianceCDLImport": "Color",
    "RadianceCDLTransform": "Color",
    "RadianceColorSpaceConvert": "Color",
    "RadianceCurves": "Color",
    "RadianceGrade": "Color",
    "RadianceGradeApply": "Color",
    "RadianceGradeMatch": "Color",
    "RadianceHueCurves": "Color",
    "RadianceLUTApply": "Color",
    "RadianceLUTBlend": "Color",
    "RadianceOCIOContext": "Color",
    "RadianceWhiteBalance": "Color",
    # ── Core ──────────────────────────────────────────────────────────────
    "RadianceProjectManager": "Core",
    # ── Generate ──────────────────────────────────────────────────────────
    "RadianceCinematicPromptEncoder": "Generate",
    "RadianceControlNetApply": "Generate",
    "RadianceDenoise": "Generate",
    "RadianceEnergyMask": "Generate",
    "RadianceHDRLatentEncoder": "Generate",
    "RadianceHDRVAEDecode": "Generate",
    "RadianceHDRVAEEncode": "Generate",
    "RadianceLoraStack": "Generate",
    "RadianceRegionalGrid": "Generate",
    "RadianceRegionalPrompt": "Generate",
    "RadianceResolution": "Generate",
    "RadianceSamplerPro": "Generate",
    "RadianceUnifiedLoader": "Generate",
    # ── HDR ───────────────────────────────────────────────────────────────
    "RadianceACES2OutputTransformFull": "HDR",
    "RadianceACES2ReachGamutCompress": "HDR",
    "RadianceACES2Tonescale": "HDR",
    "RadianceACESTransform": "HDR",
    "RadianceClipDetector": "HDR",
    "RadianceHDRAutoLogSelect": "HDR",
    "RadianceHDRColorPipeline": "HDR",
    "RadianceHDRDiagnostics": "HDR",
    "RadianceHDREncode": "HDR",
    "RadianceHDRExpandDynamicRange": "HDR",
    "RadianceHDRHighlightComposite": "HDR",
    "RadianceHDRLoRAApply": "HDR",
    "RadianceHDRLoRALoader": "HDR",
    "RadianceHDRSynthesisEngine": "HDR",
    "RadianceHDRToneMap": "HDR",
    "RadianceSDRToHDRPrepare": "HDR",
    "RadianceSDRToHDRRecover": "HDR",
    "RadianceSDRToHDRUniversal": "HDR",
    "RadianceSDRtoHDRExpand": "HDR",
    # ── Load & Save ───────────────────────────────────────────────────────
    "RadianceDigitalCinemaRead": "Load & Save",
    "RadianceDigitalCinemaWrite": "Load & Save",
    "RadianceEXRMultiPart": "Load & Save",
    "RadianceRead": "Load & Save",
    "RadianceWrite": "Load & Save",
    # ── Pipeline ──────────────────────────────────────────────────────────
    "RadianceDaVinciSend": "Pipeline",
    "RadianceMCP": "Pipeline",
    "RadianceNukeSend": "Pipeline",
    # ── Review ────────────────────────────────────────────────────────────
    "RadianceContactSheet": "Review",
    "RadianceFlipbookGIF": "Review",
    "RadianceFocusPeaking": "Review",
    "RadianceFrameStamp": "Review",
    "RadianceHDRMonitor": "Review",
    "RadiancePolicyGuard": "Review",
    "RadiancePreviewServer": "Review",
    "RadianceQC": "Review",
    "RadianceViewer": "Review",
    # ── Upscale ───────────────────────────────────────────────────────────
    "RadianceUpscaleFaceRestore": "Upscale",
    "RadianceUpscaleImage": "Upscale",
    "RadianceUpscaleTiler": "Upscale",
    "RadianceUpscaleVideo": "Upscale",
    # ── VFX ───────────────────────────────────────────────────────────────
    "RadianceAnamorphicStreaks": "VFX",
    "RadianceBitDepthDegrade": "VFX",
    "RadianceBlendComposite": "VFX",
    "RadianceChromaticAberration": "VFX",
    "RadianceDepthMapGenerator": "VFX",
    "RadianceEXRPassesWriter": "VFX",
    "RadianceFilmGrain": "VFX",
    "RadianceHDRCrop": "VFX",
    "RadianceHDRGrainMatcher": "VFX",
    "RadianceHDRStitch": "VFX",
    "RadianceLensDistortion": "VFX",
    "RadianceLinearMatting": "VFX",
    "RadianceLoadImageMask": "VFX",
    "RadianceMotionBlur": "VFX",
    "RadianceMultiMaskVisualPicker": "VFX",
    "RadianceMultipassAOVReader": "VFX",
    "RadianceMultipassComposite": "VFX",
    "RadianceMultipassMaster": "VFX",
    "RadianceMultipassEstimate": "VFX",
    "RadianceMultipassRelight": "VFX",
    "RadianceOpticalFlow": "VFX",
    "RadianceRelightEngine": "VFX",
    "RadianceSAMGenerator": "VFX",
    "RadianceSAMModelLoader": "VFX",
    "RadianceSceneCutDetect": "VFX",
    "RadianceSceneCutSplit": "VFX",
    "RadianceSubpixelStabilizer": "VFX",
    "RadianceTemporalStitchStabilizer": "VFX",
    "RadianceVignette": "VFX",
    # ── Video ─────────────────────────────────────────────────────────────
    "RadianceI2VPipeline": "Video",
    "RadianceT2VPipeline": "Video",
    "RadianceVideoAssembler": "Video",
    "RadianceVideoBatchDecode": "Video",
    "RadianceVideoCondMerge": "Video",
    "RadianceVideoExport": "Video",
    "RadianceVideoFrameRouter": "Video",
    "RadianceVideoHDRConditioner": "Video",
    "RadianceVideoHDRDecode": "Video",
    "RadianceVideoLatentNoise": "Video",
    "RadianceVideoLoader": "Video",
    "RadianceVideoMaskPropagator": "Video",
    "RadianceVideoModelInfo": "Video",
    "RadianceVideoSampler": "Video",
    # ── Published by the aggregate sweep, 2026-08 ─────────────────────────
    # Twenty finished nodes that no group's __init__.py ever listed, so
    # ComfyUI never saw them. See nodes/aggregate.py for how that happened and
    # why it cannot happen again silently.
    "RadianceACES2Compliance": "HDR",
    "RadianceACESMetadataFile": "HDR",
    "RadianceColorSpaceInfo": "HDR",
    "RadianceHDRAnalysis": "HDR",
    "RadianceHDRBlendValidator": "HDR",
    "RadianceHDRPerChannelDenorm": "HDR",
    "RadianceHDRPerChannelNorm": "HDR",
    "RadianceHDRTurboEncoder": "HDR",
    "RadianceLuminanceGuidance": "HDR",
    "RadianceAudioCut": "Pipeline",
    "RadianceAudioTranscribe": "Pipeline",
    # Named "◎ Radiance Manager" in its own module, which strips to a bare
    # "Manager" — indistinguishable from Project Manager in the menu. It is
    # neither: it builds prompt text from real camera and lens profiles, so it
    # belongs beside the other prompt nodes with a name that says so.
    "RadianceCinemaStudio": "Generate",
    "RadianceLinearCheck": "Pipeline",
    "RadianceNDISender": "Pipeline",
    "RadianceCameraSync": "VFX",
    "RadianceShotGradeRouter": "VFX",
    "RadianceVideoPromptBuilder": "Video",
    # Alias keys: the same class as a node already in this table, kept so that
    # workflows saved against the older key still load. They are marked
    # DEPRECATED at the class level so ComfyUI keeps loading them but leaves
    # them out of the node search, which is why they do not double up the menu.
    "RadianceControlApply": "Generate",
    "RadianceImageLoader": "Generate",
    "RadianceWorkspace": "Core",
    # ── Published by the registration layer, 2026-09-18 ────────────────────
    # Twenty-six more finished nodes that ComfyUI never saw. Twenty-three of
    # them sat in the top-level implementation packages (radiance.image,
    # radiance.hdr, radiance.film), which declare a full NODE_CLASS_MAPPINGS
    # that nothing in the load chain reads -- they are not node groups, and the
    # aggregate sweep cannot cross a package boundary. Three more were missing
    # from their own leaf module's mapping. See nodes/catalog.py.
    #
    # Sections are chosen from what each node does, not from the package it
    # happens to live in, which is why several radiance.hdr nodes are filed
    # under Color. Twenty-five of the twenty-six are here: RadianceHDRHistogram
    # emits a 5-D IMAGE and is withheld until that is fixed, so declaring a
    # section for it would only trip the "names a node that is gone" check.
    #
    # radiance.image.upscale, resampling and precision, the module's own
    # declared CATEGORY for all of them.
    "RadianceProUpscale": "Upscale",
    "RadianceUpscaleBySize": "Upscale",
    "RadianceDownscale32bit": "Upscale",
    "RadianceBitDepthConvert": "Upscale",
    "RadianceAIUpscale": "Upscale",
    # nodes/upscale/upscale.py, analyses a frame and recommends the tier for
    # the upscalers it sits in front of.
    "RadianceUpscaleRouter": "Upscale",
    # radiance.hdr, dynamic range, ACES and OCIO.
    "RadianceFloat32Convert": "HDR",
    "RadianceACES2OutputTransform": "HDR",
    "RadianceHDRExposureBlend": "HDR",
    "RadianceHDRShadowHighlight": "HDR",
    "RadianceGPUTensorOps": "HDR",
    # Builds equirectangular HDRI environment maps. Its module declares an
    # Upscale category, which is where it was written, not what it does.
    "RadianceHDR360Generate": "HDR",
    "RadianceHighlightSynthesis": "HDR",
    # The OCIO trio stays together: the config manager and the colorspace
    # lister exist to feed the transform, and all three carry ACES config
    # handling, which MENU_STRUCTURE files under HDR.
    "RadianceACESConfigManager": "HDR",
    "RadianceHDROCIOTransform": "HDR",
    "RadianceOCIOListColorspaces": "HDR",
    # radiance.hdr, but colour work: gamut and grading controls that belong
    # beside the other colour nodes rather than behind an HDR tab.
    "RadianceFloat32ColorCorrect": "Color",
    "RadianceHDRColorConvert": "Color",
    "RadianceDaVinciWideGamut": "Color",
    "RadianceARRIWideGamut4": "Color",
    # nodes/monitor/realtime.py, on-set exposure and A/B comparison scopes.
    "RadianceFalseColorMonitor": "Review",
    "RadianceSplitView": "Review",
    # radiance.film.camera, in-camera artefacts.
    "RadianceDepthOfField": "VFX",
    "RadianceRollingShutter": "VFX",
    "RadianceCompressionArtifacts": "VFX",
}


# Per-node section overrides for nodes the keyword classifier mis-files.
SECTION_OVERRIDES = {
    # Kept only for nodes that exist in the source but are NOT registered, so
    # their placement is already decided if they are ever published. Everything
    # registered is declared in NODE_SECTIONS above instead.
    #
    # Empty, and that is the point. Both entries this map ever held —
    # RadianceGradeApply and RadianceNDISender — were nodes somebody had
    # written, given a menu home, and then never published; the override was
    # the only trace that they existed at all. Both ship now, and
    # `nodes/aggregate.py` makes publishing the default, so a third one cannot
    # quietly accumulate here.
}

# Exact label overrides keyed by node class id. The value is the BASE label
# (the section rule then adds the "Radiance" prefix for generation sections).
# Comp nodes are mapped to the vocabulary a Nuke/Flame compositor expects.
TERM_OVERRIDES = {
    # ── Generate ──────────────────────────────────────────────────────────
    # Its module calls it "◎ Radiance Manager", which strips to "Manager" and
    # collides with Project Manager. It builds prompts from camera and lens
    # profiles.
    "RadianceCinemaStudio": "Camera & Lens Prompt",
    # ── Color ─────────────────────────────────────────────────────────────
    "RadianceCurves": "ColorLookup",
    "RadianceHueCurves": "HueCorrect",
    "RadianceColorSpaceConvert": "OCIO ColorSpace",
    "RadianceColorSpaceInfo": "ColorSpace Info",
    "RadianceCDLTransform": "CDL",
    "RadianceFloat32ColorCorrect": "ColorCorrect",
    "RadianceLUTApply": "LUT",
    # ── HDR / finishing ───────────────────────────────────────────────────
    "RadianceHDRToneMap": "Tonemap",
    "RadianceHDROCIOTransform": "OCIO Transform",
    # ── VFX ───────────────────────────────────────────────────────────────
    "RadianceDepthOfField": "Defocus",
    "RadianceFilmGrain": "Grain",
    "RadianceMotionBlur": "MotionBlur",
    "RadianceChromaticAberration": "Aberration",
    "RadianceLensDistortion": "LensDistortion",
    "RadianceSubpixelStabilizer": "Stabilize",
    "RadianceDepthMapGenerator": "Depth",
    "RadianceOpticalFlow": "MotionVectors",
    "RadianceMultipassAOVReader": "Multipass AOV Reader",
    "RadianceMultipassMaster": "Multipass Extract (Legacy)",
    "RadianceMultipassEstimate": "Multipass Estimate",
    "RadianceMultipassComposite": "Multipass Composite",
    "RadianceMultipassRelight": "Multipass Relight",
    "RadianceSAMGenerator": "Keyer (SAM, not shipped)",
    "RadianceSAMModelLoader": "SAM Loader (not shipped)",
    "RadianceLinearMatting": "Matte",
    # ── Load & Save ───────────────────────────────────────────────────────
    "RadianceImageLoader": "Read Image",
    "RadianceLoadImageMask": "Read Mask",
    "RadianceDigitalCinemaRead": "DPX Read",
    "RadianceDigitalCinemaWrite": "DPX Write",
    "RadianceEXRMultiPart": "Write EXR",
    "RadianceEXRPassesWriter": "Write EXR Passes",
    # ── Review ────────────────────────────────────────────────────────────
    "RadianceViewer": "Viewer",
    "RadianceFrameStamp": "Burn-In",
    # ── Pipeline / DCC ────────────────────────────────────────────────────
    "RadianceNukeSend": "Export to Nuke",
    "RadianceDaVinciSend": "Export to Resolve",
    # ── Generation (prefix added by the section rule) ─────────────────────
    "RadianceSamplerPro": "Sampler",
    "RadianceUnifiedLoader": "Loader",
    "RadianceCinematicPromptEncoder": "Prompt",
    "RadianceHDRVAEDecode": "VAE Decode (HDR)",
    "RadianceHDRVAEEncode": "VAE Encode (HDR)",
    # Its decoder was retired in 3.5.0; kept registered for saved workflows.
    "RadianceHDRLatentEncoder": "HDR Latent Encoder (Legacy)",
    "RadianceHDRTurboEncoder": "HDR Turbo Encoder (Legacy)",
    "RadianceControlNetApply": "ControlNet",
    "RadianceControlApply": "Control",
}


def category_path(section: str) -> str:
    """Return the full ComfyUI category path for a menu section."""

    return f"{MENU_ROOT}/{section}"


def apply_radiance_branding(
    class_mappings: MutableMapping[str, Any],
    display_name_mappings: MutableMapping[str, Any],
) -> None:
    """Normalize visible node names and categories after discovery.

    Workflow node keys stay unchanged so existing graphs continue to load.
    """

    for node_key, node_class in class_mappings.items():
        raw = display_name_mappings.get(node_key, node_key)
        section = classify_menu_section(node_key, node_class, raw)
        final = compose_display_name(node_key, raw, section)
        display_name_mappings[node_key] = final
        _set_node_category(node_class, section)
        _inject_search_aliases(node_class, node_key, raw)


def compose_display_name(node_key: str, display_name: Any, section: str) -> str:
    """Build the Option C display name: bare for comp sections, prefixed for gen."""

    if node_key in TERM_OVERRIDES:
        label = TERM_OVERRIDES[node_key]
    else:
        label = _base_label(node_key, display_name)
    if section in GENERATION_SECTIONS:
        return f"{BRAND} {label}".strip()
    return label


def normalize_display_name(node_key: str, display_name: Any) -> str:
    """Backward-compatible helper: always returns the legacy '◎ Radiance' form."""

    return f"{DISPLAY_PREFIX} {_base_label(node_key, display_name)}".strip()


def _base_label(node_key: str, display_name: Any) -> str:
    """Strip glyphs, the Radiance/FXTD prefix, and marketing words to a bare label."""

    label = _clean_display_label(str(display_name or ""))
    if not label:
        label = _camel_to_words(node_key)
    label = _strip_known_prefixes(label)
    if not label:
        label = _camel_to_words(node_key)
    label = _MARKETING.sub("", label)
    label = re.sub(r"\s{2,}", " ", label).strip(" -—·")
    if not label:
        label = _camel_to_words(node_key)
    return label


def _inject_search_aliases(node_class: Any, node_key: str, raw: Any) -> None:
    """Keep nodes findable by 'radiance' even when the display name is bare."""

    try:
        existing = list(getattr(node_class, "SEARCH_ALIASES", []) or [])
        words = _searchable_text(_base_label(node_key, raw))
        extras = ["radiance", f"radiance {words}".strip(), words]
        merged = []
        for a in existing + extras:
            a = str(a).strip()
            if a and a.lower() not in {m.lower() for m in merged}:
                merged.append(a)
        setattr(node_class, "SEARCH_ALIASES", merged)
    except Exception:
        return


def classify_menu_section(node_key: str, node_class: Any, display_name: str) -> str:
    """The menu section for a node.

    Declared placement wins. The keyword rules below are a fallback for nodes
    that have not been added to NODE_SECTIONS, and they warn when they run,
    because a guessed placement is how RadianceEnergyMask ended up in VFX.
    """

    if node_key in SECTION_OVERRIDES:
        return SECTION_OVERRIDES[node_key]

    declared = NODE_SECTIONS.get(node_key)
    if declared is not None:
        return declared

    logger.warning(
        "[Radiance] '%s' has no declared menu section; falling back to keyword "
        "matching, which guesses. Add it to NODE_SECTIONS in nodes/branding.py.",
        node_key,
    )

    module_name = getattr(node_class, "__module__", "") or ""
    current_category = str(getattr(node_class, "CATEGORY", "") or "")
    text = _searchable_text(" ".join([node_key, display_name, module_name, current_category]))


    if _has_any(text, "training", "sdr degradation", "turbo train", "developer", "diagnostic"):
        return "Developer"

    if _has_any(text, "manager", "workspace", "utility", "utilities"):
        return "Core"

    if _has_any(text, "upscale", "restore", "restoration", "supir", "esrgan", "swinir", "hat", "face restore"):
        return "Upscale"

    if _has_any(text, "viewer", "scope", "preview", "monitor", "qc", "policy", "contact sheet", "frame stamp"):
        return "Review"

    if _has_any(text, "video", "t2v", "i2v", "dit", "character", "wan", "ltx"):
        return "Video"

    if _has_any(
        text,
        "vfx", "scene cut", "shot", "edit", "router",
        "multipass", "relight", "aov", "cryptomatte", "normal pass",
        "material pass", "lighting pass", "mask", "matte", "roto",
        "alpha", "segmentation", "plate", "stabil", "grain matcher",
        "inpaint", "crop", "stitch", "depth", "geometry", "normal",
        "disparity", "lens", "optic", "camera", "chromatic", "anamorphic",
        "vignette", "film grain", "motion", "flow", "temporal",
    ):
        return "VFX"

    if _has_any(text, "hdr", "aces", "tonemap", "tone map", "luminance", "nit", "exr delivery", "vae"):
        return "HDR"

    if _has_any(text, "color", "colour", "grade", "curve", "ocio", "lut", "cdl", "white balance", "color space", "bit depth"):
        return "Color"

    if _has_any(text, "sampler", "loader", "lora", "prompt", "generate", "conditioning", "regional", "resolution", "denoise", "control net"):
        return "Generate"

    if _has_any(text, "read", "write", "io", "image", "sequence", "file", "exr", "png", "jpeg", "load image", "save image"):
        return "Load & Save"

    if _has_any(text, "project", "dcc", "nuke", "resolve", "mcp", "bridge", "metadata", "audio", "pipeline"):
        return "Pipeline"

    return DEFAULT_MENU_SECTION


def _set_node_category(node_class: Any, section: str) -> None:
    try:
        setattr(node_class, "CATEGORY", category_path(section))
    except Exception:
        return


def _clean_display_label(display_name: str) -> str:
    label = display_name.strip()
    label = re.sub(r"^[◎○●◉▷▶›>\-\s]+", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def _strip_known_prefixes(label: str) -> str:
    patterns = (r"^radiance\s+", r"^fxtd\s+studios?\s+", r"^fxtd\s+")
    result = label.strip()
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result


def _camel_to_words(value: str) -> str:
    value = re.sub(r"^Radiance", "", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return value.replace("_", " ").strip()


def _searchable_text(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _has_any(text: str, *tokens: str) -> bool:
    padded = f" {text} "
    return any(f" {_searchable_text(token)} " in padded for token in tokens)


__all__ = [
    "BRAND",
    "DISPLAY_PREFIX",
    "MENU_ROOT",
    "MENU_STRUCTURE",
    "GENERATION_SECTIONS",
    "TERM_OVERRIDES",
    "apply_radiance_branding",
    "category_path",
    "classify_menu_section",
    "compose_display_name",
    "normalize_display_name",
]
