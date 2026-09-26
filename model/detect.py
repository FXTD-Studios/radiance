"""Model architecture detection from safetensors key heuristics."""
from __future__ import annotations

import os

import logging

import comfy.sd
import folder_paths

logger = logging.getLogger("radiance.model.detect")


def _tensor_dim(f, ks: list[str], substr: str, dim: int = 0) -> int | None:
    """Return shape[dim] of the first key containing `substr`, or None.

    ALBABIT-FIX: used to distinguish architectures that share identical key
    names but differ by tensor width (e.g. Lumina2 vs Z-Image, dim=0) or by
    input-channel count (e.g. WAN 14B/1.3B vs WAN 2.2 TI2V-5B's patch_embedding,
    dim=1 -- verified against real checkpoints: [1536/5120, 16, 1, 2, 2] for
    T2V/I2V vs [3072, 48, 1, 2, 2] for TI2V-5B).
    """
    for k in ks:
        if substr in k:
            try:
                return f.get_slice(k).get_shape()[dim]
            except Exception:
                return None
    return None


def _block_count(ks: list[str], prefix: str) -> int:
    """Count distinct numbered blocks for a key prefix (e.g. "single_blocks.").

    ALBABIT-FIX: used to distinguish architectures that share identical key
    names but differ by depth (e.g. Flux.2 Dev vs Flux.2 Klein).
    """
    indices = set()
    for k in ks:
        if k.startswith(prefix):
            idx = k[len(prefix):].split(".", 1)[0]
            if idx.isdigit():
                indices.add(idx)
    return len(indices)


# 3.5: families ComfyUI 0.32 supports that this table did not know. Signatures
# are the same keys comfy/model_detection.py branches on, so a checkpoint the
# native loader accepts lands on the same family here. Ordered before the
# older, looser matchers they would otherwise fall into (hunyuan_video's
# "img_in"+"proj", flux's "double_blocks").
_NEW_FAMILY_HEURISTICS = [
    # HunyuanVideo 1.5: HunyuanVideo keys plus the vision_in projector.
    (lambda ks, f: any("txt_in.individual_token_refiner" in k for k in ks)
     and any(k.startswith("vision_in.proj.") for k in ks), "hunyuan_video_15"),
    # HunyuanImage 2.1: HunyuanVideo keys, no vector_in (vec_in_dim None).
    (lambda ks, f: any("txt_in.individual_token_refiner" in k for k in ks)
     and not any(k.startswith("vector_in.") for k in ks)
     and not any(k.startswith("vision_in.") for k in ks), "hunyuan_image"),
    # LongCat-Image: a Flux transformer with no vector_in and a 3584-wide
    # (Qwen2.5-VL) context. Chroma is matched later by its guidance layer,
    # so it is excluded here explicitly.
    (lambda ks, f: any("double_blocks" in k for k in ks)
     and not any(k.startswith("vector_in.") for k in ks)
     and not any("distilled_guidance_layer" in k for k in ks)
     and _tensor_dim(f, ks, "txt_in.weight", dim=1) == 3584, "longcat_image"),
    # Kandinsky 5: video (model_dim 4096 pro / 1792 lite) or image (2560).
    (lambda ks, f: any("visual_transformer_blocks.0.cross_attention.key_norm.weight" in k for k in ks)
     and _tensor_dim(f, ks, "visual_embeddings.in_layer.bias") == 2560, "kandinsky5_image"),
    (lambda ks, f: any("visual_transformer_blocks.0.cross_attention.key_norm.weight" in k for k in ks), "kandinsky5"),
    (lambda ks, f: any("caption_projection.0.linear.weight" in k for k in ks), "hidream"),
    (lambda ks, f: any("time_caption_embed.timestep_embedder.linear_1.bias" in k for k in ks)
     and not any("img_instruct_attn" in k for k in ks), "omnigen2"),
    (lambda ks, f: any("txtfusion.projector.weight" in k for k in ks), "krea2"),
    # Qwen-Image: txt_norm 3584 wide (Mage-Flow shares the key at 2560).
    (lambda ks, f: any(k == "txt_norm.weight" for k in ks)
     and _tensor_dim(f, ks, "txt_norm.weight") == 3584, "qwen_image"),
]

_ARCH_HEURISTICS = _NEW_FAMILY_HEURISTICS + [
    # ALBABIT-FIX: Chroma (and Chroma Radiance) share Flux's double_blocks/
    # img_in keys but add a distilled_guidance_layer — must be checked before
    # "flux" (different CLIP slots: chroma=["t5xxl"] vs flux=["clip_l","t5xxl"]).
    # Chroma Radiance (pixel-space, nerf_blocks.*) is excluded — unsupported,
    # falls through like before rather than being mislabeled "chroma" (16ch VAE).
    (lambda ks, f: any("double_blocks" in k for k in ks)
     and any("distilled_guidance_layer" in k for k in ks)
     and not any("nerf_blocks" in k for k in ks), "chroma"),
    # ALBABIT-FIX: Flux.2 Klein shares Flux.2 Dev's double_stream_modulation_img
    # key but has far fewer single_blocks (measured: Dev=48, Klein 9B=24,
    # Klein Base 4B=20) — must be checked before "flux2" (different CLIP:
    # mistral_3_small vs qwen3). Both share Flux's double_blocks/img_in keys
    # too, so this whole group must be checked before "flux".
    (lambda ks, f: any("double_stream_modulation_img" in k for k in ks)
     and _block_count(ks, "single_blocks.") < 40, "flux2-klein"),
    (lambda ks, f: any("double_stream_modulation_img" in k for k in ks), "flux2"),
    (lambda ks, f: any("double_blocks" in k for k in ks), "flux"),
    (lambda ks, f: any("joint_blocks" in k for k in ks), "sd3"),
    (lambda ks, f: any("img_in" in k and "proj" in k for k in ks), "hunyuan_video"),
    # ALBABIT-FIX: Lumina2 and Z-Image share the same NextDiT architecture and
    # key names (cap_embedder.1.weight + noise_refiner.0.attention.k_norm.weight,
    # per comfy/model_detection.py); they differ only by the output dim of
    # cap_embedder.1.weight (2304 = Lumina2, 3840 = Z-Image). The previous
    # "cap_v_projection.weight" heuristic never matched any real checkpoint.
    (lambda ks, f: any("cap_embedder.1.weight" in k for k in ks)
     and any("noise_refiner.0.attention.k_norm.weight" in k for k in ks)
     and _tensor_dim(f, ks, "cap_embedder.1.weight") == 3840, "z_image"),
    (lambda ks, f: any("cap_embedder.1.weight" in k for k in ks)
     and any("noise_refiner.0.attention.k_norm.weight" in k for k in ks), "lumina2"),
    (lambda ks, f: any("auraflow" in k.lower() for k in ks), "aura_flow"),
    # ALBABIT-FIX: Mochi (Genmo preview) UNET.
    (lambda ks, f: any("t5_yproj.weight" in k for k in ks), "mochi"),
    # ALBABIT-FIX: Cosmos World (T2V/I2V, stride 8) UNET.
    (lambda ks, f: any("blocks.block0.blocks.0.block.attn.to_q.0.weight" in k for k in ks), "cosmos"),
    # ALBABIT-FIX: CogVideoX UNET.
    (lambda ks, f: any("blocks.0.norm1.linear.weight" in k for k in ks), "cogvideox"),
    # ALBABIT-FIX: TI2V-5B shares patch_embedding keys with every WAN variant
    # but takes 48 input channels not 16 ([3072,48,1,2,2] vs [*,16,1,2,2] on
    # real checkpoints). Must be checked before the generic "wan" entry, a
    # 16ch empty latent fed to a 48ch UNET is a real crash risk.
    (lambda ks, f: any("patch_embedding" in k for k in ks)
     and any("time_embedding" in k for k in ks)
     and not any("joint_blocks" in k for k in ks)
     and _tensor_dim(f, ks, "patch_embedding.weight", dim=1) == 48, "wan_ti2v"),
    (lambda ks, f: any("patch_embedding" in k for k in ks)
     and any("time_embedding" in k for k in ks)
     and not any("joint_blocks" in k for k in ks), "wan"),
    # ALBABIT-FIX: MiniMax H3, a joint video+audio DiT. Verified directly
    # against both real checkpoint variants (bf16 and pruned_int8_convrot):
    # both keys are present in each, with no equivalent in any other
    # supported architecture.
    (lambda ks, f: any("audio_patch_proj" in k for k in ks)
     and any("video_patch_proj" in k for k in ks), "minimax"),
    # ALBABIT-FIX: return "ltxv" (not "ltx") — matches sampler_utils.py vocabulary
    (lambda ks, f: any("patchify_proj" in k for k in ks), "ltxv"),
    (lambda ks, f: any("patch_embedding" in k for k in ks)
     and any("adaln_single" in k for k in ks)
     and not any("time_embedding" in k for k in ks), "ltxv"),
    (lambda ks, f: any("adaln_single" in k for k in ks)
     and not any("patchify_proj" in k for k in ks), "pixart"),
    (lambda ks, f: any("down_blocks.0" in k for k in ks)
     and any("add_embedding" in k for k in ks), "sdxl"),
    (lambda ks, f: any("input_blocks.0" in k for k in ks), "sd1.5"),
    (lambda ks, f: any("down_blocks.0" in k for k in ks), "sdxl"),
]

LATENT_CHANNELS = {
    # ALBABIT-FIX: LTX-Video VAE (incl. LTX 2.3) uses 128 latent channels, not 16
    "flux": 16, "sd3": 16, "sd3.5": 16, "ltxv": 128, "ltxav": 128,  # ALBABIT-FIX: "ltx" → "ltxv"
    "hunyuan_video": 16, "wan": 16, "lumina2": 16, "z_image": 16,
    "sdxl": 4, "sd1.5": 4, "pixart": 4, "aura_flow": 4,
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi latent channels
    "cosmos": 16, "cogvideox": 16, "mochi": 12,
    # ALBABIT-FIX: Chroma (distilled Flux, 16ch) and Flux.2 / Flux.2 Klein (128ch)
    "chroma": 16, "flux2": 128, "flux2-klein": 128,
    # ALBABIT-FIX: WAN 2.2 TI2V-5B's VAE is a distinct architecture (comfy's own
    # latent_formats.Wan22) -- 48 latent channels, not 16. Verified directly on
    # a real wan2.2_ti2v_5B_fp16.safetensors checkpoint's patch_embedding.weight
    # shape ([3072, 48, 1, 2, 2]) and against comfy/latent_formats.py.
    "wan_ti2v": 48,
    # ALBABIT-FIX: MiniMax H3's video VAE is 24 latent channels. Its native
    # audio stream (32ch, comfy.ldm.minimax.audio_vae) isn't tracked here,
    # since this table is UNET/video-latent-only, matching every other entry.
    "minimax": 24,
    # 3.5: ComfyUI 0.32 families. Channel counts from comfy/latent_formats.py:
    # Qwen-Image, Krea 2 and Kandinsky 5 (video) ride the Wan 2.1 / Hunyuan
    # 16ch VAEs; HiDream, OmniGen2, LongCat and Kandinsky 5 image use the
    # Flux 16ch VAE; HunyuanImage 2.1 is 64ch (32px), HunyuanVideo 1.5 is
    # 32ch (16px, 4 frame).
    "qwen_image": 16, "krea2": 16, "kandinsky5": 16, "kandinsky5_image": 16,
    "hidream": 16, "omnigen2": 16, "longcat_image": 16,
    "hunyuan_image": 64, "hunyuan_video_15": 32,
}

# Spatial (and temporal) VAE factors that differ from the 8px default, for
# the families whose latent is not the SD/Flux 8x8 grid. Read by Resolution
# and the HDR VAE decode tile maths.
VAE_SPATIAL_FACTOR = {
    "ltxv": 32, "ltxav": 32, "flux2": 16, "flux2-klein": 16,
    "wan_ti2v": 16, "minimax": 16,
    "hunyuan_image": 32, "hunyuan_video_15": 16,
}
VAE_TEMPORAL_FACTOR = {
    "wan": 4, "wan_ti2v": 4, "hunyuan_video": 4, "hunyuan_video_15": 4,
    "kandinsky5": 4, "ltxv": 8, "ltxav": 8, "cosmos": 8, "cogvideox": 4,
    "mochi": 6, "minimax": 4,
}

_FORMAT_MAP = {
    "flux": "flux_16ch", "sd3": "sd3_16ch", "sd3.5": "sd3_16ch",
    "ltxv": "ltx_128ch", "ltxav": "ltx_128ch",  # ALBABIT-FIX: "ltx" → "ltxv"; match 128ch VAE
    "hunyuan_video": "hunyuan_16ch", "wan": "wan_16ch",
    "lumina2": "lumina_16ch", "z_image": "z_image_16ch",
    "sdxl": "sd_4ch", "sd1.5": "sd_4ch", "pixart": "sd_4ch",
    "aura_flow": "sd_4ch",
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi latent formats
    "cosmos": "cosmos_16ch", "cogvideox": "cogvideox_16ch", "mochi": "mochi_12ch",
    # ALBABIT-FIX: Chroma and Flux.2 / Flux.2 Klein (share the 128ch VAE)
    "chroma": "chroma_16ch", "flux2": "flux2_128ch", "flux2-klein": "flux2_128ch",
    "wan_ti2v": "wan_ti2v_48ch",
    "minimax": "minimax_24ch",
    "qwen_image": "wan_16ch", "krea2": "wan_16ch",
    "kandinsky5": "hunyuan_16ch", "kandinsky5_image": "flux_16ch",
    "hidream": "flux_16ch", "omnigen2": "flux_16ch", "longcat_image": "flux_16ch",
    "hunyuan_image": "hunyuan_image_64ch", "hunyuan_video_15": "hunyuan_video_15_32ch",
}

CLIP_SLOT_ORDER = {
    "flux": ["clip_l", "t5xxl"],
    "sd3": ["clip_l", "clip_g", "t5xxl"],
    "sd3.5": ["clip_l", "clip_g", "t5xxl"],
    "sdxl": ["clip_l", "clip_g"],
    "sd1.5": ["clip_l"],
    "hunyuan_video": ["llm_encoder", "clip_l"],
    "wan": ["t5xxl"],
    # ALBABIT-FIX: "ltxv" (pre-2.3) has no text_projection slot -- its CLIP
    # loading (comfy/text_encoders/lt.py's ltxv_te) is T5-only, 1 file. Adding
    # a 2nd file would route comfy.sd's CLIPType.LTXV branch into ltxav_te
    # (Gemma-based, LTX 2.3 only) instead -- wrong encoder, not just unused.
    "ltxv": ["llm_encoder"],  # ALBABIT-FIX: "ltx" → "ltxv"
    "ltxav": ["llm_encoder", "text_projection"],
    # ALBABIT-FIX: Lumina2 (Gemma-2 2B) and Z-Image (Qwen3-4B) are routed to
    # llm_encoder by their presets, not t5xxl — fixes "No CLIP encoders
    # provided" error when only llm_encoder is filled.
    "lumina2": ["llm_encoder"],
    "z_image": ["llm_encoder"],
    "pixart": ["t5xxl"],
    # ALBABIT-FIX: AuraFlow's real encoder is a T5 variant (comfy.text_encoders.
    # aura_t5.AuraT5Model, TEModel.T5_XL) -- "clip_l" was wrong, no file matching
    # that slot's naming convention exists for AuraFlow anywhere.
    "aura_flow": ["t5xxl"],
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi all use a single T5XXL text encoder
    "cosmos": ["t5xxl"], "cogvideox": ["t5xxl"], "mochi": ["t5xxl"],
    # ALBABIT-FIX: Chroma — distilled Flux, single T5XXL (no clip_l). Flux.2 /
    # Flux.2 Klein — single LLM encoder (Mistral-3 24B / Qwen3-4B).
    "chroma": ["t5xxl"], "flux2": ["llm_encoder"], "flux2-klein": ["llm_encoder"],
    # ALBABIT-FIX: WAN 2.2 TI2V-5B uses the same umt5-xxl text encoder as every
    # other WAN variant -- confirmed via the official TI2V-5B workflow's
    # CLIPLoader (umt5_xxl_fp8_e4m3fn_scaled.safetensors, type "wan").
    "wan_ti2v": ["t5xxl"],
    # ALBABIT-FIX: MiniMax H3's conditioning encoder is a single Qwen3-VL-32B
    # checkpoint (truncated to 50 layers). One llm_encoder slot, no
    # companion clip_l/text_projection file, same shape as flux2/z_image.
    "minimax": ["llm_encoder"],
    # 3.5 families. Slot order follows the file order comfy.sd.load_clip
    # expects for each CLIPType (comfy/sd.py): Qwen-Image, OmniGen2 and Krea 2
    # take one Qwen2.5-VL file; HunyuanImage 2.1 takes Qwen2.5-VL plus ByT5;
    # HunyuanVideo 1.5 takes Qwen2.5-VL plus ByT5 (byt5 optional); HiDream
    # takes clip_l, clip_g, t5xxl and llama; Kandinsky 5 takes Qwen2.5-VL plus
    # clip_l; LongCat takes one Qwen2.5-VL.
    "qwen_image": ["llm_encoder"], "krea2": ["llm_encoder"], "omnigen2": ["llm_encoder"],
    "longcat_image": ["llm_encoder"],
    "hunyuan_image": ["llm_encoder", "text_projection"],
    "hunyuan_video_15": ["llm_encoder", "text_projection"],
    "hidream": ["clip_l", "clip_g", "t5xxl", "llm_encoder"],
    "kandinsky5": ["llm_encoder", "clip_l"], "kandinsky5_image": ["llm_encoder", "clip_l"],
}

#: Slots an architecture cannot run without. comfy.sd.load_clip picks the text
#: encoder class from the SET of files it is given, so a missing slot does not
#: fail, it silently selects a different encoder: Flux given only t5xxl loads
#: Mochi's T5 encoder (no clip_l pooled vector), SDXL given only clip_l loads
#: SD1.5's. Seen in the 3.5 live log: a Flux run on "MochiTEModel_".
CLIP_SLOTS_REQUIRED = {
    "flux": ["clip_l", "t5xxl"],
    "sdxl": ["clip_l", "clip_g"],
    "hunyuan_video": ["llm_encoder", "clip_l"],
    "hidream": ["clip_l", "clip_g", "t5xxl", "llm_encoder"],
    "kandinsky5": ["llm_encoder", "clip_l"],
    "kandinsky5_image": ["llm_encoder", "clip_l"],
}

#: Filename patterns that identify a text-encoder file for a slot, most
#: specific first. Used only to fill a REQUIRED slot left empty.
CLIP_SLOT_PATTERNS = {
    "clip_l": ("clip_l",),
    "clip_g": ("clip_g",),
    "t5xxl": ("t5xxl", "t5_xxl"),
}


def autofill_required_clip_slots(arch: str, available: list, **slots) -> tuple:
    """Fill required-but-empty slots from the text_encoders list.

    Returns (slots, filled, missing): ``filled`` maps slot -> file chosen,
    ``missing`` lists slots that could not be filled unambiguously. A slot is
    filled only when exactly one file matches its pattern (after preferring
    exact ``<slot>.safetensors``); two candidates is a choice for the user.
    """
    slots = dict(slots)
    filled, missing = {}, []
    for slot in CLIP_SLOTS_REQUIRED.get(arch, ()):
        if slots.get(slot) not in (None, "", "None"):
            continue
        pats = CLIP_SLOT_PATTERNS.get(slot)
        if not pats:
            missing.append(slot)
            continue
        cands = [f for f in available if any(p in f.lower() for p in pats)]
        exact = [f for f in cands if os.path.basename(f).lower() in (f"{slot}.safetensors", f"{slot}.sft")]
        pick = exact[0] if len(exact) == 1 else (cands[0] if len(cands) == 1 else None)
        if pick:
            slots[slot] = pick
            filled[slot] = pick
        else:
            missing.append(slot)
    return slots, filled, missing


_CLIP_TYPE_VARIANTS = {
    "kandinsky5_image": ["KANDINSKY5_IMAGE", "KANDINSKY5"],
    "hunyuan_video_15": ["HUNYUAN_VIDEO_15", "HUNYUAN_VIDEO"],
    "longcat_image": ["LONGCAT_IMAGE", "FLUX"],
    "ltxv": ["LTX_VIDEO", "LTXV", "LTX"],   # ALBABIT-FIX: "ltx" → "ltxv"
    "ltxav": ["LTX_VIDEO", "LTXV", "LTX"],
    "hunyuan_video": ["HUNYUAN_VIDEO", "HUNYUANVIDEO"],
    "wan": ["WAN", "WAN2", "WAN_VIDEO"],
    # ALBABIT-FIX: no dedicated CLIPType.WAN_TI2V exists in ComfyUI -- TI2V-5B
    # reuses the same CLIPType.WAN as every other WAN variant.
    "wan_ti2v": ["WAN", "WAN2", "WAN_VIDEO"],
    "aura_flow": ["AURA_FLOW", "AURAFLOW"],
    # ALBABIT-FIX: Flux.2 Klein shares CLIPType.FLUX2 with Flux.2 Dev (the
    # auto-generated "FLUX2-KLEIN" enum name doesn't exist).
    "flux2-klein": ["FLUX2"],
    # ALBABIT-FIX: "minimax" deliberately has no entry here. comfy.sd.CLIPType.
    # MINIMAX already exists (verified directly), and get_clip_type_enum()'s
    # own generic fallback (model_type.upper()) already produces "MINIMAX"
    # unaided, so no override candidates are needed.
}

_BASE_CLIP_VRAM = {
    "flux": 4.5, "sd3": 3.0, "sd3.5": 3.5, "sdxl": 1.5, "sd1.5": 0.8,
    "hunyuan_video": 4.5, "wan": 3.0, "ltxv": 2.5, "ltxav": 8.0,  # ALBABIT-FIX: "ltx" → "ltxv"
    "pixart": 2.0, "aura_flow": 2.0, "lumina2": 3.0, "z_image": 3.0,
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi — single T5XXL encoder, similar to Wan
    "cosmos": 3.0, "cogvideox": 3.0, "mochi": 3.0,
    # ALBABIT-FIX: Chroma — single T5XXL (no clip_l). Flux.2 Dev — Mistral-3 24B
    # encoder (much heavier). Flux.2 Klein — Qwen3-4B, same as Z-Image.
    "chroma": 3.5, "flux2": 8.0, "flux2-klein": 3.0,
    # ALBABIT-FIX: same umt5-xxl CLIP as "wan" -- identical VRAM cost.
    "wan_ti2v": 3.0,
    # ALBABIT-FIX: MiniMax H3's Qwen3-VL-32B encoder (truncated to 50 layers)
    # is heavier than Flux.2's Mistral-3 24B (8.0). Rough parameter-count
    # scaling, not a sourced benchmark like ltxav's own correction below.
    "minimax": 10.0,
    # 3.5 families (bf16 text encoders): Qwen2.5-VL 7B ~15 GB, 3B ~6 GB,
    # HiDream's llama-3.1-8B + three CLIPs ~20 GB, ByT5 small.
    "qwen_image": 15.0, "krea2": 15.0, "omnigen2": 6.0, "longcat_image": 15.0,
    "hunyuan_image": 16.0, "hunyuan_video_15": 16.0, "hidream": 20.0,
    "kandinsky5": 16.0, "kandinsky5_image": 16.0,
}

_DTYPE_MULT = {
    "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
    "fp8_e4m3fn": 0.6, "fp8_e5m2": 0.6,
}

_CLIP_DTYPE_MULT = {
    "fp32": 2.0, "fp16": 1.0, "bf16": 1.0,
    "fp8_e4m3fn": 0.55, "fp8_e5m2": 0.55,
}

_BASE_VRAM = {
    "flux": 12.0, "sd3": 10.0, "sd3.5": 12.0,
    "sdxl": 6.5, "sd1.5": 3.5,
    "hunyuan_video": 20.0, "wan": 14.0, "ltxv": 11.0,
    # ALBABIT-FIX: "ltx" -> "ltxv". "ltxav" corrected 15.0->39.0 -- ComfyUI's
    # own staging log reported 40048MB live (LTX 2.5 Dev bf16, 22B), not the
    # ~15-23GB this table implied.
    "ltxav": 39.0,
    "pixart": 6.0, "aura_flow": 8.0,
    "lumina2": 12.0, "z_image": 14.0,
    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi base VRAM estimates
    "cosmos": 14.0, "cogvideox": 12.0, "mochi": 16.0,
    # ALBABIT-FIX: Chroma (~8.9B distilled Flux) and Flux.2 / Flux.2 Klein (128ch)
    "chroma": 10.0, "flux2": 20.0, "flux2-klein": 14.0,
    # ALBABIT-FIX: WAN 2.2 TI2V-5B is a ~5B DiT (vs "wan"'s 14B) -- rough
    # estimate, not sourced from a specific benchmark like the others above.
    "wan_ti2v": 7.0,
    # ALBABIT-FIX: matches the real fl2va bf16 checkpoint's file size (66.3GB).
    # The pruned_int8_convrot preset is a genuinely smaller checkpoint, not a
    # dtype cast of this one, so it over-estimates there. Same accepted
    # imprecision as every other architecture sharing one number across
    # differently-sized presets (e.g. "wan"'s 1.3B vs 14B variants).
    "minimax": 66.0,
    # 3.5 families, bf16 transformer weights.
    "qwen_image": 40.0, "krea2": 28.0, "omnigen2": 8.0, "longcat_image": 12.0,
    "hunyuan_image": 34.0, "hunyuan_video_15": 17.0, "hidream": 34.0,
    "kandinsky5": 40.0, "kandinsky5_image": 12.0,
}

# ALBABIT-FIX: per-architecture key remap before handing an audio-VAE state
# dict to comfy.sd.VAE(). Absent means no remap needed (comfy.sd.VAE() auto-
# detects MiniMax H3's audio VAE natively). LTX-AV namespaces its tensors
# under audio_vae./vocoder., which needs stripping first.
AUDIO_VAE_KEY_REMAP = {
    "ltxav": {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
}


# comfy model_config class name -> Radiance arch, for the native fallback.
# Only families with a distinct Radiance table entry are listed; a match on
# an unlisted class is reported by name so the operator can pick manually.
_COMFY_CONFIG_TO_ARCH = {
    "SD15": "sd1.5", "SD20": "sd1.5", "SDXL": "sdxl", "SDXLRefiner": "sdxl",
    "SSD1B": "sdxl", "Segmind_Vega": "sdxl", "SD3": "sd3",
    "AuraFlow": "aura_flow", "PixArtAlpha": "pixart", "PixArtSigma": "pixart",
    "Flux": "flux", "FluxInpaint": "flux", "FluxSchnell": "flux",
    "Flux2": "flux2", "Chroma": "chroma", "GenmoMochi": "mochi",
    "LTXV": "ltxv", "LTXAV": "ltxav", "MiniMaxH3": "minimax",
    "HunyuanVideo": "hunyuan_video", "HunyuanVideoI2V": "hunyuan_video",
    "HunyuanVideoSkyreelsI2V": "hunyuan_video",
    "HunyuanImage21": "hunyuan_image", "HunyuanVideo15": "hunyuan_video_15",
    "HunyuanVideo15_SR_Distilled": "hunyuan_video_15",
    "CosmosT2V": "cosmos", "CosmosI2V": "cosmos",
    "Lumina2": "lumina2", "ZImage": "z_image",
    "WAN22_T2V": "wan_ti2v", "CogVideoX_T2V": "cogvideox", "CogVideoX_I2V": "cogvideox",
    "CogVideoX_Inpaint": "cogvideox",
    "HiDream": "hidream", "Omnigen2": "omnigen2", "Krea2": "krea2",
    "QwenImage": "qwen_image", "LongCatImage": "longcat_image",
    "Kandinsky5": "kandinsky5", "Kandinsky5Image": "kandinsky5_image",
}


class _ShapeOnlyStateDict(dict):
    """A state dict that reads a tensor from safetensors only when asked.

    comfy.model_detection.detect_unet_config() branches on key presence and
    on .shape of a handful of tensors; loading a 20 GB checkpoint to answer
    that is not acceptable at graph-build time. This maps every key to a
    lazily-fetched slice, so only the tensors comfy actually touches are read.
    """

    def __init__(self, f, keys):
        super().__init__((k, None) for k in keys)
        self._f = f

    _MATERIALISE_BELOW = 1 << 20   # elements; norms and biases, not weights

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if val is None:
            sl = self._f.get_slice(key)
            shape = tuple(sl.get_shape())
            n = 1
            for d in shape:
                n *= int(d)
            if n <= self._MATERIALISE_BELOW:
                # Small tensors (norm weights, biases) are read for real so
                # anything comfy computes on them (torch.std for Lumina's
                # fp16 check) works. Weights stay as shape-only proxies.
                val = sl[...]
            else:
                class _S:
                    __slots__ = ("shape", "_sl")

                    def __init__(self, sl, shape):
                        self._sl = sl
                        self.shape = shape

                    def __getitem__(self, idx):
                        return self._sl[idx]
                val = _S(sl, shape)
            dict.__setitem__(self, key, val)
        return val

    def get(self, key, default=None):
        return self[key] if key in self else default


def _detect_via_comfy(unet_path: str, f, keys: list[str]) -> str | None:
    """Ask ComfyUI's own detector, from shapes only.

    Runs after Radiance's heuristics have all missed. Returns a Radiance arch
    when the family has a table entry here, otherwise logs the comfy class so
    the operator knows what the file is, and returns None.
    """
    try:
        import comfy.model_detection as _md
    except Exception:  # noqa: BLE001 - not running under ComfyUI
        return None
    try:
        sd = _ShapeOnlyStateDict(f, keys)
        prefix = ""
        for k in keys:
            if k.startswith("model.diffusion_model."):
                prefix = "model.diffusion_model."
                break
        metadata = None
        try:
            metadata = f.metadata()
        except Exception:  # noqa: BLE001
            pass
        cfg = _md.model_config_from_unet(sd, prefix, metadata=metadata)
        if cfg is None:
            return None
        name = type(cfg).__name__
        arch = _COMFY_CONFIG_TO_ARCH.get(name)
        if arch:
            logger.info("Auto-detected architecture via ComfyUI: %s (%s) from %s",
                        arch, name, unet_path.rsplit("/", 1)[-1])
            return arch
        lf = getattr(cfg, "latent_format", None)
        ch = getattr(lf, "latent_channels", None)
        logger.warning(
            "ComfyUI recognises %s as %s (latent %s ch) but Radiance has no "
            "table for it; select the model type manually.",
            unet_path.rsplit("/", 1)[-1], name, ch,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("ComfyUI-native detection failed: %s", e)
    return None


def detect_model_type(unet_path: str) -> str | None:
    try:
        from safetensors import safe_open
        with safe_open(unet_path, framework="pt", device="cpu") as f:
            # ALBABIT-FIX: no key-count limit — listing keys only reads the
            # safetensors header (cheap), and some architectures (e.g. Mochi's
            # t5_yproj.weight) sort late alphabetically among hundreds of keys.
            keys = list(f.keys())
            for test_fn, arch in _ARCH_HEURISTICS:
                if test_fn(keys, f):
                    logger.info(
                        "Auto-detected architecture: %s from %s",
                        arch, unet_path.rsplit("/", 1)[-1],
                    )
                    return arch
            # 3.5: nothing here matched. ComfyUI's detector knows every family
            # the running ComfyUI can load, so ask it before giving up.
            return _detect_via_comfy(unet_path, f, keys)
    except ImportError:
        logger.debug("safetensors not available — skipping auto-detect")
    except Exception as e:
        logger.debug("Auto-detect failed: %s", e)
    return None


def latent_format(arch: str) -> str:
    return _FORMAT_MAP.get(arch, f"{arch}_{LATENT_CHANNELS.get(arch, 4)}ch")


def assemble_clip_paths(arch: str, unet_path: str | None = None, **slots) -> list[str]:
    slot_map = slots
    order = CLIP_SLOT_ORDER.get(arch, list(slot_map.keys()))
    paths = []
    for slot in order:
        val = slot_map.get(slot)
        if not val or val in ("None", ""):
            continue
        # ALBABIT-FIX: "Baked (from UNET)" for text_projection — LTX 2.3's
        # text_embedding_projection weights ship inside the main UNET
        # checkpoint, mirroring the native "LTXV Audio Text Encoder Loader"
        # node (which loads the UNET checkpoint a second time as a CLIP source).
        if val == "Baked (from UNET)":
            if unet_path:
                paths.append(unet_path)
            else:
                logger.warning("CLIP slot '%s' set to 'Baked (from UNET)' but no UNET path available.", slot)
            continue
        p = folder_paths.get_full_path("text_encoders", val)
        if p:
            paths.append(p)
        else:
            logger.warning("CLIP slot '%s' file not found: %s", slot, val)
    return paths


def get_clip_type_enum(model_type: str):
    mapping = {
        "flux": comfy.sd.CLIPType.FLUX,
        "sd3": comfy.sd.CLIPType.SD3,
        "sd3.5": comfy.sd.CLIPType.SD3,
        "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION,
        "sd1.5": comfy.sd.CLIPType.STABLE_DIFFUSION,
    }

    # ALBABIT-FIX: Cosmos / CogVideoX / Mochi resolve via CLIPType.{COSMOS,COGVIDEOX,MOCHI}.
    # Chroma -> CLIPType.CHROMA, Flux.2 -> CLIPType.FLUX2 (Flux.2 Klein via
    # _CLIP_TYPE_VARIANTS override above, since "FLUX2-KLEIN" isn't a real enum).
    for name in ("hunyuan_video", "wan", "wan_ti2v", "ltxv", "ltxav", "pixart", "aura_flow", "lumina2", "z_image",  # ALBABIT-FIX: "ltx" → "ltxv"
                  "cosmos", "cogvideox", "mochi", "chroma", "flux2", "flux2-klein",
                  "qwen_image", "krea2", "omnigen2", "longcat_image", "hunyuan_image",
                  "hunyuan_video_15", "hidream", "kandinsky5", "kandinsky5_image"):
        enum_name = name.upper().replace(".", "_")
        auto_variants = [enum_name, name.upper(), name.title().replace("_", "")]
        extra = _CLIP_TYPE_VARIANTS.get(name, [])
        all_variants = extra + [v for v in auto_variants if v not in extra]

        for variant in all_variants:
            if hasattr(comfy.sd.CLIPType, variant):
                mapping[name] = getattr(comfy.sd.CLIPType, variant)
                break
        else:
            mapping.setdefault(name, comfy.sd.CLIPType.STABLE_DIFFUSION)

    clip_type = mapping.get(model_type)
    if clip_type is None:
        enum_name = model_type.upper().replace(".", "_")
        if hasattr(comfy.sd.CLIPType, enum_name):
            clip_type = getattr(comfy.sd.CLIPType, enum_name)
        else:
            logger.warning("No CLIPType mapping for '%s', falling back to STABLE_DIFFUSION", model_type)
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

    return clip_type


def estimate_vram_usage(
    model_type: str,
    weight_dtype: str,
    clip_dtype: str = "fp16",
    has_loras: bool = False,
    has_controlnet: bool = False,
) -> float:
    base = _BASE_VRAM.get(model_type, 8.0)
    unet_mult = _DTYPE_MULT.get(weight_dtype, 1.0)
    clip_base = _BASE_CLIP_VRAM.get(model_type, 2.0)
    clip_mult = _CLIP_DTYPE_MULT.get(clip_dtype, 1.0)

    vram = (base * unet_mult) + (clip_base * clip_mult)
    if has_loras:
        vram += 0.5
    if has_controlnet:
        vram += 2.0
    return round(vram, 1)
