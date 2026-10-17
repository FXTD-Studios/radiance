"""Optional prompt upsampling for FLUX 3 Image Klein and Qwen-Image 2.1.

Both families ship a prompt upsampler, a Qwen3.5 9B LLM loaded with
CLIPLoader (``qwen3.5_9b_flux3_prompt_upsampler_*`` for FLUX 3,
``qwen3.5_9b_qwen_image_2.1_pe_*`` for Qwen-Image 2.1). It rewrites a short
request into the detailed caption the image model was trained on. This runs it
the way ComfyUI's official templates do: ComfyUI's own TextGenerate node, the
model authors' system prompt, and the templates' sampling settings.

The system prompts in ``upsampler_prompts/`` are copied unchanged from the
ComfyUI workflow templates (comfyui-workflow-templates, MIT licence):
image_flux3_klein_t2i / _image_edit and image_qwen_image_2_1_t2i / _image_edit.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

logger = logging.getLogger("radiance")

_PROMPT_DIR = Path(__file__).resolve().parent / "upsampler_prompts"

#: Per family: the system prompt files and the templates' TextGenerate settings
#: (t2i and edit differ only in presence_penalty).
UPSAMPLERS = {
    "flux3": {
        "label": "FLUX 3 Image Klein",
        "file_hint": "qwen3.5_9b_flux3_prompt_upsampler",
        "system": {"t2i": "flux3_t2i.txt", "edit": "flux3_edit.txt"},
        "max_length": 3584, "temperature": 1.0, "top_k": 1, "top_p": 0.95, "min_p": 0.0,
        "repetition_penalty": 1.0, "presence_penalty": {"t2i": 1.5, "edit": 0.0},
        "thinking": False, "mtp": "off",
    },
    "qwen_image21": {
        "label": "Qwen-Image 2.1",
        "file_hint": "qwen3.5_9b_qwen_image_2.1_pe",
        "system": {"t2i": "qwen_image21_t2i.txt", "edit": "qwen_image21_edit.txt"},
        "max_length": 4096, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
        "repetition_penalty": 1.0, "presence_penalty": {"t2i": 1.5, "edit": 0.0},
        "thinking": True, "mtp": "auto",
    },
}


def system_prompt(arch: str, mode: str) -> str:
    """The model authors' upsampler instructions for ``arch`` ('t2i' or 'edit')."""
    return (_PROMPT_DIR / UPSAMPLERS[arch]["system"][mode]).read_text(encoding="utf-8")


def _text_generate():
    try:
        from comfy_extras import nodes_textgen
    except ImportError as exc:  # older ComfyUI
        raise ValueError("Prompt: the prompt upsampler needs ComfyUI's Generate Text node "
                         "(comfy_extras/nodes_textgen.py). Update ComfyUI.") from exc
    return nodes_textgen.TextGenerate


def _batch(references):
    """The reference images as one IMAGE batch, as BatchImagesNode makes it."""
    images = [img[:, :, :, :3] for img in references.values()]
    if not images:
        return None
    try:
        from comfy_extras import nodes_post_processing
        return nodes_post_processing.BatchImagesNode.execute(
            images={f"image{i + 1}": img for i, img in enumerate(images)}).args[0]
    except (ImportError, AttributeError):
        import torch
        if all(img.shape[1:] == images[0].shape[1:] for img in images):
            return torch.cat(images, dim=0)
        logger.warning("[Encoder] Upsampler: this ComfyUI cannot batch reference images of "
                       "different sizes; the upsampler sees image_1 only.")
        return images[0]


def upsample(upsampler_clip, arch: str, prompt: str, references=None, seed: int = 0) -> str:
    """Rewrite ``prompt`` with the family's upsampler. Raises ValueError when it cannot run."""
    if arch not in UPSAMPLERS:
        raise ValueError(
            f"Prompt: the prompt upsampler is for FLUX 3 Image Klein and Qwen-Image 2.1, and "
            f"the text encoder resolves to '{arch}'. Turn upsample_prompt off.")
    if upsampler_clip is None:
        spec = UPSAMPLERS[arch]
        raise ValueError(
            f"Prompt: upsample_prompt is on and upsampler_clip is not connected. Load the "
            f"{spec['label']} upsampler ({spec['file_hint']}_*.safetensors) with a Load CLIP "
            f"node and connect it, or turn upsample_prompt off.")
    spec = UPSAMPLERS[arch]
    mode = "edit" if references else "t2i"
    node = _text_generate()
    kwargs = dict(
        clip=upsampler_clip, prompt=prompt, max_length=spec["max_length"],
        sampling_mode={
            "sampling_mode": "on", "temperature": spec["temperature"], "top_k": spec["top_k"],
            "top_p": spec["top_p"], "min_p": spec["min_p"],
            "repetition_penalty": spec["repetition_penalty"], "seed": int(seed),
            "presence_penalty": spec["presence_penalty"][mode],
        },
        image=_batch(references) if references else None,
        thinking=spec["thinking"], use_default_template=True,
        mtp=spec["mtp"], system_prompt=system_prompt(arch, mode),
    )
    accepted = inspect.signature(node.execute).parameters
    if "system_prompt" not in accepted:
        raise ValueError("Prompt: this ComfyUI's Generate Text node takes no system prompt, "
                         "which the upsampler needs. Update ComfyUI.")
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    text = node.execute(**kwargs).args[0]
    text = (text or "").strip()
    if not text:
        raise ValueError("Prompt: the upsampler returned no text. Check that upsampler_clip "
                         f"is the {spec['label']} upsampler, not the text encoder.")
    return text
