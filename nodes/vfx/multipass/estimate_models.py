"""Model loading for Multipass Estimate: MoGe-2 geometry and Marigold IID.

Everything here is imported lazily by the node, so the package still loads on
an install without diffusers, and the CI lane can test the pure parts (file
lists, paths, consent) without torch models.

Weights
-------
MoGe-2 ViT-L with normals (Microsoft, MIT), the ComfyUI repackage:
    Comfy-Org/MoGe @ 1484985258ba7ef45c314144fcf0e90924ffbf44
    geometry_estimation/moge_2_vitl_normal_fp16.safetensors (662 MB)
    -> ComfyUI/models/geometry_estimation/

Marigold IID v1.1 (ETH Zurich PRS, OpenRAIL++-M: commercial use allowed,
with the licence's use restrictions):
    prs-eth/marigold-iid-appearance-v1-1 @ e7280a0a0fc5a0df0b36050882b3d8b77da22fd9
        albedo + material (roughness, metallicity); also supplies the shared
        VAE, text encoder, tokenizer and scheduler (~2.5 GB)
    prs-eth/marigold-iid-lighting-v1-1 @ 08c3930bb641abf786ba44ce92547507ebefbc16
        albedo + shading + residual; only its UNet is needed (~1.7 GB)
    -> ComfyUI/models/radiance/marigold/<name>/

Downloads are pinned to those commits. Hugging Face verifies each LFS file
against its sha256 as it downloads. They happen only when the node's
``download_missing_models`` widget is on and nothing in the environment says
no (``RADIANCE_ALLOW_DOWNLOADS=0``, ``HF_HUB_OFFLINE=1``,
``TRANSFORMERS_OFFLINE=1``).
"""
from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("radiance.vfx.multipass.estimate_models")

# ── Pinned sources ──────────────────────────────────────────────────────────

MOGE_REPO = "Comfy-Org/MoGe"
MOGE_REVISION = "1484985258ba7ef45c314144fcf0e90924ffbf44"
MOGE_FILENAME = "moge_2_vitl_normal_fp16.safetensors"
MOGE_REPO_PATH = f"geometry_estimation/{MOGE_FILENAME}"
MOGE_SIZE = 661_859_924
MOGE_SHA256 = "cb1a692d03235671e959e81360d7b4d9f44aefadb1f852d6ca6aa17799d5e31f"
MOGE_PAGE_URL = f"https://huggingface.co/{MOGE_REPO}/blob/main/{MOGE_REPO_PATH}"
MOGE_FOLDER_KEY = "geometry_estimation"

MARIGOLD_APPEARANCE = "iid-appearance-v1-1"
MARIGOLD_LIGHTING = "iid-lighting-v1-1"

MARIGOLD_SOURCES: Dict[str, Dict[str, object]] = {
    MARIGOLD_APPEARANCE: {
        "repo": "prs-eth/marigold-iid-appearance-v1-1",
        "revision": "e7280a0a0fc5a0df0b36050882b3d8b77da22fd9",
        "size_mb": 2500,
        "files": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    MARIGOLD_LIGHTING: {
        "repo": "prs-eth/marigold-iid-lighting-v1-1",
        "revision": "08c3930bb641abf786ba44ce92547507ebefbc16",
        "size_mb": 1700,
        "files": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
}
MARIGOLD_LICENSE_URL = "https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1/blob/main/LICENSE"


class EstimateModelError(RuntimeError):
    """A model the node needs is missing or cannot run here. The message says why and what to do."""


# ── Consent ─────────────────────────────────────────────────────────────────

def downloads_permitted(widget_allows: bool) -> bool:
    """The widget is the per-graph yes; the environment can always say no.

    ``RADIANCE_ALLOW_DOWNLOADS=0`` and the Hugging Face offline flags win over
    the widget. ``RADIANCE_ALLOW_DOWNLOADS=1`` does not override a widget set
    to off: someone who switched it off on this node meant it.
    """
    from radiance.core.consent import downloads_allowed
    return bool(widget_allows) and downloads_allowed(default=True)


# ── Paths ───────────────────────────────────────────────────────────────────

def _geometry_dirs() -> List[Path]:
    dirs: List[Path] = []
    try:
        import folder_paths  # type: ignore
        get = getattr(folder_paths, "get_folder_paths", None)
        if callable(get):
            dirs.extend(Path(p) for p in (get(MOGE_FOLDER_KEY) or ()))
        base = getattr(folder_paths, "models_dir", None)
        if isinstance(base, str) and base:
            dirs.append(Path(base) / MOGE_FOLDER_KEY)
    except Exception:  # noqa: BLE001 - no ComfyUI
        pass
    from radiance.model.paths import radiance_model_dirs
    for d in radiance_model_dirs():
        dirs.append(d.parent / MOGE_FOLDER_KEY)
    seen, out = set(), []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _marigold_dirs(name: str) -> List[Path]:
    from radiance.model.paths import radiance_model_dirs
    return [d / "marigold" / name for d in radiance_model_dirs()]


def _first_writable(candidates: List[Path]) -> Optional[Path]:
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".radiance_write_probe"
            probe.write_bytes(b"")
            probe.unlink()
            return d
        except OSError:
            continue
    return None


def find_moge() -> Optional[Path]:
    for d in _geometry_dirs():
        p = d / MOGE_FILENAME
        if p.is_file() and p.stat().st_size == MOGE_SIZE:
            return p
    return None


def marigold_complete(path: Path, name: str) -> bool:
    files = MARIGOLD_SOURCES[name]["files"]
    return all((path / f).is_file() for f in files)  # type: ignore[union-attr]


def find_marigold(name: str) -> Optional[Path]:
    for d in _marigold_dirs(name):
        if marigold_complete(d, name):
            return d
    return None


# ── Downloads ───────────────────────────────────────────────────────────────

_dl_lock = threading.Lock()


def _refuse(what: str, size_mb: int, dest: str, url: str) -> EstimateModelError:
    return EstimateModelError(
        f"[Multipass Estimate] {what} (~{size_mb} MB) is not installed and downloads are off "
        f"(the node's download_missing_models widget, RADIANCE_ALLOW_DOWNLOADS=0 or HF_HUB_OFFLINE=1).\n"
        f"Install it from {url}\ninto {dest}"
    )


def ensure_moge(allow_download: bool) -> Path:
    found = find_moge()
    if found:
        return found
    target = _first_writable(_geometry_dirs())
    dest = str(target or "ComfyUI/models/geometry_estimation/")
    if not downloads_permitted(allow_download):
        raise _refuse(f"MoGe-2 ({MOGE_FILENAME})", 662, dest, MOGE_PAGE_URL)
    if target is None:
        raise EstimateModelError(f"[Multipass Estimate] No writable models/geometry_estimation folder for {MOGE_FILENAME}.")
    with _dl_lock:
        found = find_moge()
        if found:
            return found
        from huggingface_hub import hf_hub_download
        logger.info("[Radiance] Downloading MoGe-2 (~662 MB, MIT licence) to %s", target)
        staging = target / ".radiance_download"
        local = Path(hf_hub_download(
            repo_id=MOGE_REPO, filename=MOGE_REPO_PATH, revision=MOGE_REVISION,
            local_dir=str(staging),
        ))
        if local.stat().st_size != MOGE_SIZE:
            raise EstimateModelError(f"[Multipass Estimate] {MOGE_FILENAME} downloaded with the wrong size.")
        final = target / MOGE_FILENAME
        os.replace(local, final)
        # hf_hub_download leaves its lock and metadata files in the staging
        # folder; nothing reads them once the file is in place.
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        logger.info("[Radiance] Installed %s", final)
        return final


def ensure_marigold(name: str, allow_download: bool) -> Path:
    found = find_marigold(name)
    if found:
        return found
    src = MARIGOLD_SOURCES[name]
    url = f"https://huggingface.co/{src['repo']}"
    candidates = _marigold_dirs(name)
    target = _first_writable(candidates)
    dest = str(target or f"ComfyUI/models/radiance/marigold/{name}/")
    if not downloads_permitted(allow_download):
        raise _refuse(f"Marigold {name}", int(src["size_mb"]), dest, url)  # type: ignore[arg-type]
    if target is None:
        raise EstimateModelError(f"[Multipass Estimate] No writable models/radiance folder for Marigold {name}.")
    with _dl_lock:
        if marigold_complete(target, name):
            return target
        from huggingface_hub import snapshot_download
        logger.info(
            "[Radiance] Downloading Marigold %s (~%s MB) to %s\n"
            "           Weights licensed OpenRAIL++-M (commercial use allowed, with use restrictions): %s",
            name, src["size_mb"], target, MARIGOLD_LICENSE_URL,
        )
        snapshot_download(
            repo_id=str(src["repo"]), revision=str(src["revision"]),
            allow_patterns=list(src["files"]), local_dir=str(target),  # type: ignore[arg-type]
        )
        if not marigold_complete(target, name):
            raise EstimateModelError(f"[Multipass Estimate] Marigold {name} download is incomplete in {target}.")
        return target


# ── MoGe loader ─────────────────────────────────────────────────────────────

_moge_cache: Dict[str, object] = {}


def load_moge(allow_download: bool):
    """ComfyUI's native MoGe wrapper (comfy.ldm.moge), cached per process."""
    try:
        from comfy.ldm.moge.model import MoGeModel  # type: ignore
        import comfy.utils  # type: ignore
    except ImportError as exc:
        raise EstimateModelError(
            "[Multipass Estimate] This ComfyUI has no native MoGe support (comfy.ldm.moge). "
            "Update ComfyUI to a release that includes the MoGe nodes."
        ) from exc
    path = ensure_moge(allow_download)
    key = str(path)
    model = _moge_cache.get(key)
    if model is None:
        _moge_cache.clear()
        sd = comfy.utils.load_torch_file(str(path), safe_load=True)
        model = MoGeModel(sd)
        del sd
        _moge_cache[key] = model
    return prepare_moge(model)


def prepare_moge(model):
    """Half precision fails on CPU inside MoGe's antialiased resize; run fp32 there."""
    import torch
    dev = getattr(model, "load_device", None)
    if getattr(dev, "type", None) == "cpu" and getattr(model, "dtype", None) != torch.float32:
        model.model.float()
        model.dtype = torch.float32
    return model


# ── Marigold loader ─────────────────────────────────────────────────────────

_EMBEDDING_FILE = "radiance_empty_prompt_embedding.safetensors"


def _empty_embedding(appearance_dir: str, tok, text_model_cls, kw):
    """Marigold's only use of CLIP: the embedding of the empty prompt.

    Computed once, then cached beside the weights so later runs never load the
    680 MB text encoder again.
    """
    import torch
    from safetensors.torch import load_file, save_file
    cached = os.path.join(appearance_dir, _EMBEDDING_FILE)
    if os.path.isfile(cached):
        try:
            return load_file(cached)["embedding"]
        except Exception:  # noqa: BLE001 - recompute a damaged cache
            pass
    te = text_model_cls.from_pretrained(appearance_dir, subfolder="text_encoder", **kw)
    with torch.inference_mode():
        ids = tok("", padding="do_not_pad", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids
        emb = te(ids)[0].detach().float().clone()
    del te
    gc.collect()
    try:
        save_file({"embedding": emb.contiguous()}, cached)
    except OSError:
        pass
    return emb


class MarigoldIID:
    """Both Marigold IID models behind one VAE: the UNet is swapped per pass.

    Kept on the CPU between runs; moved to the compute device only while
    predicting. The text encoder is used once to compute the empty-prompt
    embedding and then dropped (Marigold is conditioned on the image only).
    """

    def __init__(self, appearance_dir: Path, lighting_dir: Path, dtype):
        try:
            from diffusers import (  # type: ignore
                AutoencoderKL, DDIMScheduler, MarigoldIntrinsicsPipeline, UNet2DConditionModel,
            )
            from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore
        except ImportError as exc:
            raise EstimateModelError(
                "[Multipass Estimate] Marigold needs diffusers>=0.33 and accelerate. "
                "Reinstall Radiance's requirements (pip install -r requirements.txt)."
            ) from exc
        import torch

        self.dtype = dtype
        kw = dict(variant="fp16", torch_dtype=dtype)
        a = str(appearance_dir)
        tok = CLIPTokenizer.from_pretrained(a, subfolder="tokenizer")
        self.empty_embedding = _empty_embedding(a, tok, CLIPTextModel, kw).to(dtype)
        self._shared = dict(
            vae=AutoencoderKL.from_pretrained(a, subfolder="vae", **kw),
            text_encoder=None,
            tokenizer=tok,
            scheduler=DDIMScheduler.from_pretrained(a, subfolder="scheduler"),
        )
        self.vae = self._shared["vae"]
        self._roots = {"appearance": a, "lighting": str(lighting_dir)}
        self._kw = kw
        self._pipe_cls = MarigoldIntrinsicsPipeline
        self._unet_cls = UNet2DConditionModel
        # fp32 UNets are 3.4 GB each; on the CPU hold one at a time.
        self._keep_both = dtype != torch.float32
        self.pipes: Dict[str, object] = {}
        self._device = torch.device("cpu")
        self._targets = {}
        for which, root in self._roots.items():
            import json as _json
            with open(os.path.join(root, "model_index.json"), "r", encoding="utf-8") as fh:
                self._targets[which] = tuple(_json.load(fh)["target_properties"]["target_names"])

    def _pipe(self, which: str):
        pipe = self.pipes.get(which)
        if pipe is not None:
            return pipe
        if not self._keep_both:
            for other in list(self.pipes):
                del self.pipes[other]
            gc.collect()
        root = self._roots[which]
        unet = self._unet_cls.from_pretrained(root, subfolder="unet", **self._kw).to(self._device)
        pipe = self._pipe_cls.from_pretrained(root, unet=unet, **self._shared)
        pipe.empty_text_embedding = self.empty_embedding
        pipe.set_progress_bar_config(disable=True)
        self.pipes[which] = pipe
        return pipe

    def target_names(self, which: str) -> Tuple[str, ...]:
        return self._targets[which]

    def to(self, device):
        import torch
        device = torch.device(device)
        if device == self._device:
            return self
        self.vae.to(device)
        for p in self.pipes.values():
            p.unet.to(device)
        self.empty_embedding = self.empty_embedding.to(device)
        for p in self.pipes.values():
            p.empty_text_embedding = self.empty_embedding
        self._device = device
        return self

    def predict(self, which: str, image_bchw, *, steps: int, resolution: int, ensemble: int, seed: int):
        """One frame (1,3,H,W) in sRGB [0,1] -> (T,3,H,W) float32 in [0,1] on the compute device."""
        import torch
        pipe = self._pipe(which)
        gen = torch.Generator(device=self._device).manual_seed(int(seed))
        with torch.inference_mode():
            out = pipe(
                image_bchw.to(self._device, self.dtype),
                num_inference_steps=int(steps),
                ensemble_size=int(ensemble),
                processing_resolution=int(resolution),
                match_input_resolution=True,
                generator=gen,
                output_type="pt",
            )
        return out.prediction.float()


_marigold_cache: Dict[str, MarigoldIID] = {}


def load_marigold(allow_download: bool, device) -> MarigoldIID:
    import torch
    a = ensure_marigold(MARIGOLD_APPEARANCE, allow_download)
    l = ensure_marigold(MARIGOLD_LIGHTING, allow_download)
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.float16
    key = f"{a}|{l}|{dtype}"
    model = _marigold_cache.get(key)
    if model is None:
        _marigold_cache.clear()
        gc.collect()
        model = MarigoldIID(a, l, dtype)
        _marigold_cache[key] = model
    return model


def release_moge_cache() -> None:
    """Drop the cached MoGe model (used before Marigold on memory-bound CPU runs)."""
    _moge_cache.clear()
    gc.collect()


def release_marigold(model: MarigoldIID) -> None:
    """Park the weights on the CPU so ComfyUI gets its VRAM back."""
    try:
        model.to("cpu")
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
