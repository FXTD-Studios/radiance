import math
import torch
import threading
import logging
from typing import Tuple

from radiance.core.tensor.chunking import chunks, frames_per_chunk

# Module logger
logger = logging.getLogger("radiance.depth")

# ═══════════════════════════════════════════════════════════════════════════════
#                           MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Model sizes and their HuggingFace identifiers
DEPTH_MODELS = {
    "Small (25M - Fast)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base (98M - Balanced)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large (335M - Best)": "depth-anything/Depth-Anything-V2-Large-hf",
}

#: Commit of each model repository that is loaded (3.5.0: pinned; it was
#: "main", so a change upstream would change every depth map). Hugging Face
#: checks every file against its sha256 as it downloads.
DEPTH_REVISIONS = {
    "depth-anything/Depth-Anything-V2-Small-hf": "5426e4f0f36572d16453bbda7a8389317b1bef99",
    "depth-anything/Depth-Anything-V2-Base-hf": "b1958afc87fb45a9e3746cb387596094de553ed8",
    "depth-anything/Depth-Anything-V2-Large-hf": "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
}

# Thread-safe cache for loaded models, keyed by (model_id, device_str).
# Separate entries per device prevent the in-place .to(device) race
# where two callers on different devices mutate the same nn.Module.
# Bounded LRU, not a plain dict. All three Depth Anything V2 tiers resident is
# ~1.83 GB pinned for the process lifetime, invisible to ComfyUI's model
# manager -- the sampler OOMs later with no obvious culprit. GPUModelCache
# moves an evicted module back to CPU before dropping it.
from radiance.model.cache import GPUModelCache  # noqa: E402

_model_cache = GPUModelCache(max_size=1)   # key: (model_id, device_str) -> model
_processor_cache: dict = {}  # key: model_id -> processor (small, CPU-only)
_cache_lock = threading.RLock()


def get_device(use_gpu: bool = True) -> torch.device:
    """Get the appropriate compute device (CUDA > MPS > CPU)."""
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def download_and_load_model(model_size: str, device: torch.device):
    """
    Download model from HuggingFace if not cached, then return on *device*.

    Cache stores models on **CPU** so that concurrent calls requesting
    different devices (CUDA vs MPS vs CPU) don't mutate each other's
    parameter storage.  The caller receives the model already moved to
    the requested device.
    """
    model_id = DEPTH_MODELS.get(model_size)
    if not model_id:
        logger.warning(
            f"Unknown model_size '{model_size}', falling back to Base"
        )
        model_id = DEPTH_MODELS["Base (98M - Balanced)"]

    device_str = str(device)
    cache_key  = (model_id, device_str)

    with _cache_lock:
        if cache_key not in _model_cache:
            # Download and cache on the requested device.
            # Each (model_id, device) pair gets its own parameter copy
            # so concurrent callers on different devices never race.
            # Downloads on first use through the shared consent gate
            # (RADIANCE_ALLOW_DOWNLOADS=0 or an offline flag loads only what is
            # cached). Base and Large are CC-BY-NC-4.0.
            from radiance.core.consent import downloads_allowed, refusal_message
            local_only = not downloads_allowed()
            try:
                from transformers import (
                    AutoImageProcessor,
                    AutoModelForDepthEstimation,
                )
            except ImportError as e:
                raise ImportError(
                    "transformers library required for Depth Anything V2.\n"
                    "Install with: pip install transformers"
                ) from e
            if not local_only:
                logger.info(f"Loading depth model: {model_size} ({model_id}); downloads on first use")
            try:
                if model_id not in _processor_cache:
                    _processor_cache[model_id] = AutoImageProcessor.from_pretrained(
                        model_id, revision=DEPTH_REVISIONS.get(model_id, "main"), local_files_only=local_only
                    )
                model = AutoModelForDepthEstimation.from_pretrained(
                    model_id, revision=DEPTH_REVISIONS.get(model_id, "main"), local_files_only=local_only
                )
                model.eval()
                logger.info(f"Depth model loaded: {model_id}")
            except Exception as e:
                if local_only:
                    raise FileNotFoundError(refusal_message(
                        f"Depth Anything V2 ({model_id})",
                        url=f"https://huggingface.co/{model_id}",
                        dest="the Hugging Face cache (HF_HOME)",
                    )) from e
                raise RuntimeError(
                    f"Failed to download depth model '{model_id}': {e}"
                ) from e

            # Store already on the target device. 3.5.0: half precision on
            # CUDA (half the VRAM, about twice the speed). Measured against
            # fp32 on the Small and Base models: mean difference 0.03-0.07 %
            # of the depth range, no NaNs. CPU and MPS stay fp32.
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            _model_cache.put(cache_key, model.to(device=device, dtype=dtype))
            logger.info(f"Depth model cached on {device_str}: {model_id}")

        model     = _model_cache.get(cache_key)
        processor = _processor_cache[model_id]

    # model is already on the correct device — no further .to() needed
    return model, processor


def _dpt_resize_size(h: int, w: int, out_h: int, out_w: int, multiple: int) -> Tuple[int, int]:
    """DPTImageProcessor's keep-aspect resize: scale as little as possible,
    then round each side to a multiple of `multiple`."""
    scale_h = out_h / h
    scale_w = out_w / w
    if abs(1 - scale_w) < abs(1 - scale_h):
        scale_h = scale_w
    else:
        scale_w = scale_h

    def fit(val: float) -> int:
        x = round(val / multiple) * multiple
        if x < multiple:   # never below one patch
            x = max(multiple, int(math.ceil(val / multiple)) * multiple)
        return int(x)

    return fit(scale_h * h), fit(scale_w * w)


def _resize_uint8(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """The processor's resize: torchvision bicubic with antialiasing on uint8
    (matches DPTImageProcessor to float precision). torch's own interpolate is
    the fallback when torchvision is missing (within about 3 levels)."""
    try:
        import torchvision.transforms.v2.functional as TF
        from torchvision.transforms import InterpolationMode
        return TF.resize(x, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    except ImportError:
        y = torch.nn.functional.interpolate(x.float(), size=(h, w), mode="bicubic",
                                            align_corners=False, antialias=True)
        return y.clamp(0, 255).round().to(torch.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#                           DEPTH MAP GENERATOR NODE
# ═══════════════════════════════════════════════════════════════════════════════


class RadianceDepthMapGenerator:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    """
    Depth Anything V2 — Monocular depth estimation.

    Video-safe: processes each frame independently, with batch-wide
    normalization for temporal consistency across frames.
    """

    MODEL_SIZES = list(DEPTH_MODELS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Display-encoded image or frame batch. Frames with values above 1.05 are Reinhard tone-mapped to 0..1 first; alpha is ignored."}),
                "model_size": (
                    cls.MODEL_SIZES,
                    {
                        "default": "Large (335M - Best)",
                        "tooltip": (
                            "Depth Anything V2 model size. "
                            "Small = fast previews, Large = best quality."
                        ),
                    },
                ),
            },
            "optional": {
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Normalize depth to 0-1 range. "
                            "For video, frames are standardized for temporal consistency."
                        ),
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert depth (white=far, black=near).",
                    },
                ),
                "blur_edges": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.5,
                        "display": "slider",
                        "tooltip": "Gaussian blur to smooth depth discontinuities.",
                    },
                ),
                "use_gpu": ("BOOLEAN", {"default": True,
                    "tooltip": "Run depth estimation on GPU. Requires a CUDA-capable device. Falls back to CPU if unavailable."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "generate_depth"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"

    DESCRIPTION = (
        "Depth Anything V2 monocular depth estimation. "
        "Video-safe — standardizes each frame with spatial-temporal alignment "
        "preventing flickering. Outputs 3-channel grayscale depth map. "
        "Connect to Depth of Field node for realistic defocus blur."
    )

    @torch.no_grad()
    def generate_depth(
        self,
        image: torch.Tensor,
        model_size: str,
        normalize: bool = True,
        invert: bool = False,
        blur_edges: float = 0.0,
        use_gpu: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Generate depth map from input image(s). Video-safe per-frame."""

        device = get_device(use_gpu)

        # Load model (auto-downloads on first call)
        model, processor = download_and_load_model(model_size, device)
        model_dtype = next(model.parameters()).dtype

        batch_size = image.shape[0]
        orig_h, orig_w = image.shape[1], image.shape[2]
        is_video = batch_size > 1
        if is_video:
            logger.info(f"Processing {batch_size} frames for depth estimation...")

        # 3.5.0: frames go through the model a few at a time, preprocessed on
        # the device, and each full-resolution depth frame goes to the CPU as
        # soon as it exists. This used to run the Hugging Face processor on
        # the CPU one frame at a time (a uint8 round trip through PIL per
        # frame) and keep every full-resolution depth frame on the GPU, then a
        # 3-channel copy of all of them: 240 frames of 4K needed about 32 GB of
        # VRAM. Preprocessing is the processor's own: uint8 quantisation,
        # keep-aspect resize to a multiple of 14 near 518 (bicubic,
        # antialiased), rescale and ImageNet normalisation.
        size = processor.size if isinstance(processor.size, dict) else {"height": 518, "width": 518}
        in_h, in_w = _dpt_resize_size(orig_h, orig_w, size.get("height", 518), size.get("width", 518),
                                      getattr(processor, "ensure_multiple_of", 14) or 14)
        mean = torch.tensor(getattr(processor, "image_mean", (0.485, 0.456, 0.406)), device=device).view(1, 3, 1, 1)
        std = torch.tensor(getattr(processor, "image_std", (0.229, 0.224, 0.225)), device=device).view(1, 3, 1, 1)

        depth_cpu = torch.empty((batch_size, orig_h, orig_w), dtype=torch.float32)
        # Per-frame statistics for the video standardisation below.
        means = torch.empty(batch_size, dtype=torch.float64)
        stds = torch.empty(batch_size, dtype=torch.float64)
        per = max(1, min(8, frames_per_chunk(orig_h, orig_w, 4, 6.0, device)))
        done = 0
        for a, b in chunks(batch_size, per):
            x = image[a:b, ..., :3].to(device=device, dtype=torch.float32)
            # HDR tonemap: Reinhard on absolute values, per frame, only for a
            # frame that goes above 1.05 (as before).
            hot = x.flatten(1).amax(dim=1) > 1.05
            if bool(hot.any()):
                xa = x.abs()
                x = torch.where(hot.view(-1, 1, 1, 1), torch.sign(x) * (xa / (1.0 + xa)), x)
            # The uint8 the processor received, then its own resize.
            x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 3, 1, 2)
            if (in_h, in_w) != (orig_h, orig_w):
                x = _resize_uint8(x, in_h, in_w)
            x = (x.float() / 255.0 - mean) / std
            depth = model(pixel_values=x.to(model_dtype)).predicted_depth.float()   # (n, mH, mW)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=(orig_h, orig_w), mode="bicubic", align_corners=False,
            ).squeeze(1)
            flat = depth.flatten(1)
            means[a:b] = flat.mean(dim=1).double().cpu()
            stds[a:b] = flat.std(dim=1).double().cpu()
            depth_cpu[a:b] = depth.cpu()
            del x, depth, flat
            if is_video and (b // 10) > (done // 10):
                logger.info(f"  Depth frame {b}/{batch_size}")
            done = b

        depth_batch = depth_cpu

        # ── Normalize ──
        # Depth Anything V2 outputs relative affine-invariant depth maps.
        # Natively doing global min/max causes severe flickering for videos,
        # because each frame has completely arbitrary scale and shift.
        if normalize:
            if is_video:
                # 1. Standardize each frame to Mean=0, Std=1.
                # This mathematically removes the arbitrary per-frame scale & shift.
                depth_batch.sub_(means.float().view(-1, 1, 1)).div_((stds.float() + 1e-8).view(-1, 1, 1))
            # 2. Normalize to [0,1].
            #     - For videos, frames are now aligned in scale/shift, so global Min/Max
            #       keeps relative distance consistent across the entire clip.
            #     - For single images, this normalizes the single frame correctly.
            d_min = depth_batch.min()
            d_max = depth_batch.max()
            depth_batch.sub_(d_min).div_(d_max - d_min + 1e-8)

        # ── Invert ──
        if invert:
            depth_batch = 1.0 - depth_batch

        # ── Edge blur ── (per frame, so a chunk at a time on the device)
        if blur_edges > 0:
            per_blur = max(1, frames_per_chunk(orig_h, orig_w, 1, 4.0, device))
            for a, b in chunks(batch_size, per_blur):
                depth_batch[a:b] = self._blur_depth_batch(depth_batch[a:b].to(device), blur_edges).cpu()

        # ── Output: 3-channel grayscale IMAGE (B, H, W, 3), built on the CPU ──
        result = depth_batch.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

        if is_video:
            logger.info(f"Depth estimation complete: {batch_size} frames")

        return (result,)

    @staticmethod
    def _blur_depth_batch(
        depth: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """
        Gaussian blur on a (B, H, W) depth tensor.
        Uses reflect-padding to avoid dark edge halos (v1.0 used zero-padding).
        """
        if sigma <= 0:
            return depth

        kernel_size = max(3, int(sigma * 6) | 1)  # Odd, minimum 3
        device = depth.device

        x = torch.arange(kernel_size, device=device, dtype=torch.float32)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # (B, H, W) → (B, 1, H, W) for conv2d
        d = depth.unsqueeze(1)
        pad = kernel_size // 2

        # Horizontal pass — reflect-pad to avoid edge darkening
        kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
        d = torch.nn.functional.pad(d, (pad, pad, 0, 0), mode="reflect")
        d = torch.nn.functional.conv2d(d, kernel_h, padding=0)

        # Vertical pass
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1)
        d = torch.nn.functional.pad(d, (0, 0, pad, pad), mode="reflect")
        d = torch.nn.functional.conv2d(d, kernel_v, padding=0)

        return d.squeeze(1)  # → (B, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
#                              NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "RadianceDepthMapGenerator": RadianceDepthMapGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceDepthMapGenerator": "◎ Radiance Depth Map",
}
