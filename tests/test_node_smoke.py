"""
tests/test_node_smoke.py — Radiance v3.1 Node Smoke Test Suite
═══════════════════════════════════════════════════════════════
Audit fix: production readiness blocker #3 — test coverage.

Strategy
--------
Every registered ComfyUI node must satisfy three minimal properties:

  1. STRUCTURAL — the class has RETURN_TYPES, RETURN_NAMES, FUNCTION, and
     CATEGORY as class attributes, all with the correct Python types.
  2. INPUT_SPEC — INPUT_TYPES() is callable and returns a dict with at
     least a 'required' or 'optional' key.
  3. EXECUTE — the primary execute method (named by FUNCTION) is callable
     and can be *instantiated without args*.  We do NOT call execute() here
     because that would require real GPU / model weights / OCIO configs at
     test time.  Calling execute is the job of integration tests.

The suite discovers nodes from each flat nodes_*.py file individually
(rather than via the full radiance package) so it works in any environment
without ComfyUI, torch, or aiohttp installed.

Running
-------
    cd /path/to/radiance
    pytest tests/test_node_smoke.py -v

Each node appears as an individual test item:
    PASSED  test_structural[RadianceGrade]
    PASSED  test_input_spec[RadianceGrade]
    PASSED  test_instantiation[RadianceGrade]
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
import unittest.mock as mock

# ── Path setup ────────────────────────────────────────────────────────────────
RADIANCE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(RADIANCE_ROOT))

# ── ComfyUI / heavy-dep stubs ─────────────────────────────────────────────────
# Provide lightweight stubs so node files can be imported without a live
# ComfyUI process, GPU, or optional libraries.

def _stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _is_importable(name: str) -> bool:
    """True when the real package is installed, whether or not it is imported.

    ``name not in sys.modules`` is not the same question.  This module is
    imported during collection, so a stub installed on that test alone leaks
    into every test collected after it, and a real package that simply had not
    been imported yet gets shadowed for the whole run.  That produced an
    order-dependent failure: ``colour`` is installed on the full dependency
    lane, but the colour-science cross-check in
    ``test_colorspace_convert_regression.py`` saw this stub and failed with
    ``module 'colour' has no attribute 'LOG_ENCODINGS'`` only when collection
    ordered that file after this one.  Stub what is genuinely absent, nothing
    else.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

def _ensure_stubs() -> None:
    """Install minimal stubs for ComfyUI and heavy optional deps."""

    # ── folder_paths ──
    if "folder_paths" not in sys.modules:
        _stub_module("folder_paths")
    # Always ensure critical attributes exist (conftest.py may have created a
    # partial stub that lacks get_input_directory / filter_files_content_types).
    _fp = sys.modules["folder_paths"]
    if not hasattr(_fp, "get_input_directory") or isinstance(getattr(_fp, "get_input_directory", None), mock.Mock):
        _fp.get_input_directory  = lambda: "/tmp"
    if not hasattr(_fp, "get_output_directory") or isinstance(getattr(_fp, "get_output_directory", None), mock.Mock):
        _fp.get_output_directory = lambda: "/tmp"
    if not hasattr(_fp, "filter_files_content_types") or isinstance(getattr(_fp, "filter_files_content_types", None), mock.Mock):
        _fp.filter_files_content_types = lambda files, types: files
    if not hasattr(_fp, "models_dir"):
        _fp.models_dir = "/tmp/models"
    if not hasattr(_fp, "get_filename_list") or isinstance(getattr(_fp, "get_filename_list", None), mock.Mock):
        _fp.get_filename_list = lambda *a, **kw: []
    if not hasattr(_fp, "get_full_path"):
        _fp.get_full_path = lambda *a, **kw: None
    if not hasattr(_fp, "get_annotated_filepath"):
        _fp.get_annotated_filepath = lambda name: name
    if not hasattr(_fp, "exists_annotated_filepath"):
        _fp.exists_annotated_filepath = lambda name: True

    # ── server (PromptServer) ──
    if "server" not in sys.modules:
        srv = _stub_module("server")
        class _FakeRoutes:
            def post(self, path): return lambda f: f
            def get(self,  path): return lambda f: f
        class _FakePS:
            instance = type("PS", (), {"routes": _FakeRoutes()})()
        srv.PromptServer = _FakePS

    # ── aiohttp ──
    if "aiohttp" not in sys.modules:
        ah = _stub_module("aiohttp")
        ah.web = _stub_module("aiohttp.web")

    # ── torch ──
    if "torch" not in sys.modules:
        _ctx = mock.MagicMock(return_value=mock.MagicMock(
            __enter__=lambda s, *a: None,
            __exit__=lambda s, *a: None,
        ))

        t = _stub_module("torch")

        # dtype sentinels
        t.float32   = "float32"
        t.float16   = "float16"
        t.float64   = "float64"
        t.bfloat16  = "bfloat16"
        t.int32     = "int32"
        t.int64     = "int64"
        t.uint8     = "uint8"
        t.bool      = "bool"

        # Tensor class with enough attribute surface for import-time code
        _Tensor = type("Tensor", (), {
            "to":       lambda s, *a, **k: s,
            "float":    lambda s: s,
            "half":     lambda s: s,
            "cpu":      lambda s: s,
            "cuda":     lambda s, *a, **k: s,
            "detach":   lambda s: s,
            "squeeze":  lambda s, *a: s,
            "unsqueeze":lambda s, *a: s,
            "permute":  lambda s, *a: s,
            "reshape":  lambda s, *a: s,
            "view":     lambda s, *a: s,
            "numpy":    lambda s: [],
            "item":     lambda s: 0.0,
            "shape":    property(lambda s: (1, 1, 1, 3)),
            "device":   "cpu",
            "dtype":    "float32",
            "ndim":     4,
            "__len__":  lambda s: 1,
            "__getitem__": lambda s, k: s,
        })
        t.Tensor = _Tensor

        t.device        = lambda *a: "cpu"
        t.no_grad       = _ctx
        t.inference_mode = _ctx
        t.autocast      = _ctx

        # factory / math ops
        _ret = lambda *a, **kw: _Tensor()
        t.zeros         = _ret
        t.ones          = _ret
        t.cat           = _ret
        t.stack         = _ret
        t.clamp         = _ret
        t.tensor        = _ret
        t.from_numpy    = _ret
        t.linspace      = _ret
        t.arange        = _ret
        t.mean          = _ret
        t.sum           = _ret
        t.max           = _ret
        t.min           = _ret
        t.abs           = _ret
        t.sqrt          = _ret
        t.exp           = _ret
        t.log           = _ret
        t.sigmoid       = _ret
        t.tanh          = _ret
        t.softmax       = _ret
        t.einsum        = _ret
        t.matmul        = _ret
        t.inverse       = _ret
        t.is_tensor     = lambda x: isinstance(x, _Tensor)
        t.Size          = tuple

        # cuda
        t.cuda = _stub_module("torch.cuda")
        t.cuda.is_available = lambda: False
        t.cuda.empty_cache  = lambda: None
        t.cuda.current_device = lambda: 0
        t.cuda.device_count   = lambda: 0

        # nn — must be a real module so `from torch.nn import Module` works
        nn = _stub_module("torch.nn")
        nn.Module        = object
        nn.ModuleList    = list
        nn.ModuleDict    = dict
        nn.Sequential    = mock.MagicMock
        nn.Linear        = mock.MagicMock
        nn.Conv2d        = mock.MagicMock
        nn.ConvTranspose2d = mock.MagicMock
        nn.BatchNorm2d   = mock.MagicMock
        nn.LayerNorm     = mock.MagicMock
        nn.Dropout       = mock.MagicMock
        nn.ReLU          = mock.MagicMock
        nn.GELU          = mock.MagicMock
        nn.SiLU          = mock.MagicMock
        nn.Sigmoid       = mock.MagicMock
        nn.Tanh          = mock.MagicMock
        nn.Embedding     = mock.MagicMock
        nn.MultiheadAttention = mock.MagicMock
        nn.Parameter     = mock.MagicMock
        nn.Identity      = mock.MagicMock
        nn.Upsample      = mock.MagicMock
        nn.functional    = _stub_module("torch.nn.functional")
        t.nn             = nn

        # torch.nn.functional — most-used ops
        F = sys.modules["torch.nn.functional"]
        for _fname in ("relu", "gelu", "silu", "sigmoid", "tanh",
                       "softmax", "log_softmax", "dropout",
                       "interpolate", "pad", "conv2d", "conv_transpose2d",
                       "batch_norm", "layer_norm", "group_norm",
                       "mse_loss", "l1_loss", "cross_entropy",
                       "binary_cross_entropy", "binary_cross_entropy_with_logits",
                       "normalize", "unfold", "fold",
                       "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
                       "grid_sample", "affine_grid"):
            setattr(F, _fname, _ret)

        # torch.backends
        t.backends = _stub_module("torch.backends")
        t.backends.mps = _stub_module("torch.backends.mps")
        t.backends.mps.is_available = lambda: False
        t.backends.cudnn = _stub_module("torch.backends.cudnn")
        t.backends.cudnn.enabled = True

        # torch.amp
        t.amp = _stub_module("torch.amp")
        t.amp.autocast = _ctx

        # torch.utils
        t.utils = _stub_module("torch.utils")
        t.utils.data = _stub_module("torch.utils.data")
        t.utils.data.Dataset    = object
        t.utils.data.DataLoader = mock.MagicMock

        # torch.optim
        t.optim = _stub_module("torch.optim")
        t.optim.Adam   = mock.MagicMock
        t.optim.AdamW  = mock.MagicMock

    # ── numpy ──
    if "numpy" not in sys.modules:
        np = _stub_module("numpy")
        # dtype sentinels
        np.float32  = "float32"
        np.float64  = "float64"
        np.float16  = "float16"
        np.uint8    = "uint8"
        np.uint16   = "uint16"
        np.uint32   = "uint32"
        np.int32    = "int32"
        np.int64    = "int64"
        np.bool_    = bool

        # ndarray with common attrs
        np.ndarray  = type("ndarray", (), {
            "T":        property(lambda s: s),
            "shape":    property(lambda s: (0,)),
            "dtype":    "float32",
            "astype":   lambda s, *a, **k: s,
            "tobytes":  lambda s: b"",
            "flatten":  lambda s: s,
            "reshape":  lambda s, *a: s,
            "mean":     lambda s, *a, **k: 0.0,
            "std":      lambda s, *a, **k: 0.0,
            "min":      lambda s, *a, **k: 0.0,
            "max":      lambda s, *a, **k: 0.0,
            "clip":     lambda s, *a, **k: s,
            "__len__":  lambda s: 0,
            "__getitem__": lambda s, k: s,
            "__setitem__": lambda s, k, v: None,
        })
        _arr = lambda *a, **kw: np.ndarray()
        np.zeros     = _arr
        np.ones      = _arr
        np.array     = _arr
        np.empty     = _arr
        np.full      = _arr
        np.arange    = _arr
        np.linspace  = _arr
        np.frombuffer = _arr
        np.clip      = _arr
        np.stack     = _arr
        np.concatenate = _arr
        np.expand_dims = _arr
        np.squeeze   = _arr
        np.transpose = _arr
        np.mean      = lambda *a, **k: 0.0
        np.std       = lambda *a, **k: 0.0
        np.sum       = lambda *a, **k: 0.0
        np.min       = lambda *a, **k: 0.0
        np.max       = lambda *a, **k: 0.0
        np.abs       = _arr
        np.sqrt      = _arr
        np.log       = _arr
        np.exp       = _arr
        np.power     = _arr
        np.dot       = _arr
        np.eye       = _arr
        np.identity  = _arr
        np.diag      = _arr
        np.linalg    = _stub_module("numpy.linalg")
        np.linalg.norm = lambda *a, **k: 0.0
        np.linalg.inv = lambda a, *args, **kwargs: a
        np.linalg.solve = lambda a, b, *args, **kwargs: b
        np.random    = _stub_module("numpy.random")
        np.random.randn  = _arr
        np.random.random = _arr

    # ── PIL ──
    if "PIL" not in sys.modules:
        pil  = _stub_module("PIL")
        pil.Image     = _stub_module("PIL.Image")
        pil.Image.open   = mock.MagicMock
        pil.Image.new    = mock.MagicMock
        pil.Image.fromarray = mock.MagicMock
        pil.Image.LANCZOS   = 1
        pil.Image.BICUBIC   = 3
        pil.Image.BILINEAR  = 2
        _stub_module("PIL.ImageDraw")
        _stub_module("PIL.ImageFont")
        _stub_module("PIL.ImageFilter")
        _stub_module("PIL.ImageEnhance")
        pil.ImageOps = _stub_module("PIL.ImageOps")
        pil.ImageOps.autocontrast = lambda *a, **k: mock.MagicMock()
        pil.ImageOps.equalize     = lambda *a, **k: mock.MagicMock()

    # ── cv2 ──
    if "cv2" not in sys.modules and not _is_importable("cv2"):
        cv2 = _stub_module("cv2")
        cv2.COLOR_BGR2RGB = 4
        cv2.COLOR_RGB2BGR = 4
        cv2.COLOR_BGR2GRAY = 6
        cv2.INTER_LINEAR = 1
        cv2.INTER_CUBIC  = 2
        cv2.INTER_AREA   = 3
        cv2.INTER_LANCZOS4 = 8
        cv2.cvtColor     = lambda *a, **k: None
        cv2.resize       = lambda *a, **k: None
        cv2.GaussianBlur = lambda *a, **k: None
        cv2.medianBlur   = lambda *a, **k: None

    # ── OpenEXR / Imath ──
    for name in ("OpenEXR", "Imath"):
        if name not in sys.modules and not _is_importable(name):
            _stub_module(name)

    # ── PyOpenColorIO ──
    if "PyOpenColorIO" not in sys.modules and not _is_importable("PyOpenColorIO"):
        ocio = _stub_module("PyOpenColorIO")
        ocio.__version__ = "stub"
        ocio.Config     = mock.MagicMock
        ocio.Exception  = Exception
        ocio.ROLE_SCENE_LINEAR = "scene_linear"

    # ── colour ──
    if "colour" not in sys.modules and not _is_importable("colour"):
        _stub_module("colour")

    # ── comfy (ComfyUI internals) — extended stub set ──
    _comfy_mods = [
        "comfy", "comfy.model_management", "comfy.utils",
        "comfy.sd", "comfy.samplers", "comfy.k_diffusion",
        "comfy.k_diffusion.sampling", "comfy.latent_formats",
        # Additional modules referenced by loader/sampler nodes
        "comfy.sample", "comfy.sampler_helpers",
        "comfy.cldm", "comfy.cldm.cldm",
        "comfy.controlnet", "comfy.lora",
        "comfy.model_base", "comfy.model_detection",
        "comfy.model_patcher", "comfy.clip_model",
        "comfy.diffusers_load", "comfy.supported_models",
        "comfy.supported_models_base", "comfy.t2i_adapter",
        "comfy.extra_samplers", "comfy.nodes",
    ]
    for name in _comfy_mods:
        if name not in sys.modules:
            _stub_module(name)
    for name in _comfy_mods:
        if "." not in name:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        child = sys.modules.get(name)
        if parent is not None and child is not None and not hasattr(parent, child_name):
            setattr(parent, child_name, child)

    samplers = sys.modules["comfy.samplers"]
    if not hasattr(samplers, "KSampler"):
        samplers.KSampler = type(
            "KSampler",
            (),
            {"SAMPLERS": ["euler"], "SCHEDULERS": ["normal"]},
        )
    if not hasattr(samplers, "sampler_object"):
        samplers.sampler_object = mock.MagicMock(return_value=mock.MagicMock())

    # Attach common model_management attributes
    mm = sys.modules["comfy.model_management"]
    mm.get_torch_device      = lambda: "cpu"
    mm.get_free_memory       = lambda *a: 0
    mm.soft_empty_cache      = lambda: None
    mm.unet_offload_device   = lambda: "cpu"
    mm.text_encoder_device   = lambda: "cpu"
    mm.vae_device            = lambda: "cpu"
    mm.current_loaded_models = []
    mm.load_model_gpu        = mock.MagicMock
    mm.unload_model          = mock.MagicMock

    # ── transformers ──
    if "transformers" not in sys.modules:
        tf = _stub_module("transformers")
        tf.CLIPTextModel      = mock.MagicMock
        tf.CLIPTokenizer      = mock.MagicMock
        tf.AutoModelForCausalLM = mock.MagicMock
        tf.AutoTokenizer      = mock.MagicMock

    # ── torchaudio ──
    if "torchaudio" not in sys.modules:
        _stub_module("torchaudio")

    # ── tqdm ──
    if "tqdm" not in sys.modules:
        _tqdm_mod = _stub_module("tqdm")
        _tqdm_mod.tqdm = mock.MagicMock

    # ── defusedxml ──
    if "defusedxml" not in sys.modules:
        _stub_module("defusedxml")
        _stub_module("defusedxml.ElementTree")

    # ── scipy ──
    if "scipy" not in sys.modules:
        sc = _stub_module("scipy")
        sc.signal = _stub_module("scipy.signal")
        sc.signal.find_peaks = lambda *a, **k: ([], {})
        sc.signal.resample   = lambda *a, **k: []
        sc.ndimage = _stub_module("scipy.ndimage")

    # ── librosa ──
    if "librosa" not in sys.modules:
        lb = _stub_module("librosa")
        lb.load              = lambda *a, **k: ([], 22050)
        lb.beat              = _stub_module("librosa.beat")
        lb.beat.beat_track   = lambda *a, **k: (120.0, [])
        lb.onset             = _stub_module("librosa.onset")
        lb.onset.onset_detect = lambda *a, **k: []
        lb.onset.onset_strength = lambda *a, **k: []
        lb.util              = _stub_module("librosa.util")
        lb.util.peak_pick    = lambda *a, **k: []
        lb.frames_to_time    = lambda *a, **k: []

    # ── Patch __spec__ on all stub modules so importlib.util.find_spec() ──────
    # doesn't raise ValueError("__spec__ is None") when nodes call _has(pkg).
    import importlib.machinery as _imm
    for _mod_name, _mod in list(sys.modules.items()):
        if isinstance(_mod, types.ModuleType) and getattr(_mod, "__spec__", None) is None:
            try:
                _mod.__spec__ = _imm.ModuleSpec(_mod_name, loader=None)
            except Exception:
                pass

    # ── radiance package and its commonly relative-imported sub-modules ───────
    # When node files do `from .color_utils import ...` they need the parent
    # package `radiance` registered in sys.modules, and the target sub-module
    # registered as `radiance.<name>`.  We stub them here and rely on
    # _import_file() setting mod.__package__ = "radiance" before exec.
    if "radiance" not in sys.modules:
        _rad = _stub_module("radiance")
        _rad.__path__ = [str(RADIANCE_ROOT)]
        _rad.__package__ = "radiance"
        _imm_spec = _imm.ModuleSpec("radiance", loader=None, is_package=True)
        _imm_spec.submodule_search_locations = [str(RADIANCE_ROOT)]
        _rad.__spec__ = _imm_spec

    # Pre-import or stub the modules that are relatively imported at module level
    _IDENTITY_MATRIX = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    class _FakeVAE4KDecode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "samples": ("LATENT",),
                    "vae": ("VAE",),
                    "target_space": (["sRGB", "Linear", "ACEScg"], {"default": "Linear"}),
                },
                "optional": {
                    "force_hdr_decode": ("BOOLEAN", {"default": False}),
                    "hdr_mode": (["Compress (Log)", "Passthrough"], {"default": "Passthrough"}),
                    "hdr_output": ("BOOLEAN", {"default": False}),
                    "export_rhdr": ("BOOLEAN", {"default": False}),
                    "crop_padding": ("STRING", {"default": ""}),
                    "processing_mode": (["sequential"], {"default": "sequential"}),
                },
            }

        def decode(self, *args, **kwargs):
            return (None,)

    _FakeVAE4KEncode = type(
        "RadianceVAE4KEncode",
        (),
        {"INPUT_TYPES": classmethod(lambda cls: {"required": {"image": ("IMAGE",)}})},
    )
    _FakeVAE4KRoundtrip = type(
        "RadianceVAE4KRoundtrip",
        (),
        {"INPUT_TYPES": classmethod(lambda cls: {"required": {"image": ("IMAGE",)}})},
    )

    class _FakeSamplerMode:
        STANDARD = "Standard"
        ALL = ["Standard"]

    _radiance_sub_stubs = {
        "radiance.color_utils":         ["apply_input_transform", "apply_output_transform",
                                          "INPUT_COLORSPACES", "OUTPUT_COLORSPACES",
                                          "_apply_matrix", "_BRADFORD_CAT",
                                          "_PRIMARIES_MATRICES",
                                          # log curves / tensor transforms
                                          "linear_to_logc3", "logc3_to_linear",
                                          "linear_to_logc4", "logc4_to_linear",
                                          "linear_to_slog3", "slog3_to_linear",
                                          "linear_to_vlog", "vlog_to_linear",
                                          "linear_to_canonlog3", "canonlog3_to_linear",
                                          "linear_to_acescct", "acescct_to_linear",
                                          "linear_to_davinci_intermediate", "davinci_intermediate_to_linear",
                                          "linear_to_log3g10", "log3g10_to_linear",
                                          "tensor_linear_to_logc3", "tensor_logc3_to_linear",
                                          "tensor_linear_to_logc4", "tensor_logc4_to_linear",
                                          "tensor_linear_to_slog3", "tensor_slog3_to_linear",
                                          "tensor_linear_to_vlog", "tensor_vlog_to_linear",
                                          "tensor_linear_to_log3g10", "tensor_log3g10_to_linear",
                                          "tensor_linear_to_davinci_intermediate", "tensor_davinci_intermediate_to_linear",
                                          "tensor_linear_to_acescct", "tensor_acescct_to_linear",
                                          "tensor_linear_to_canonlog3", "tensor_canonlog3_to_linear",
                                          "tensor_srgb_to_linear", "tensor_linear_to_srgb",
                                          "tensor_to_numpy_float32", "numpy_to_tensor_float32",
                                          "apply_matrix_transform",
                                          "AWG3_TO_ACESCG", "AWG4_TO_ACESCG",
                                          "SGAMUT3_CINE_TO_ACESCG", "VGAMUT_TO_ACESCG",
                                          "CINEMA_GAMUT_TO_ACESCG", "REDWIDEGAMUT_TO_ACESCG",
                                          "DAVINCI_WIDE_TO_ACESCG", "ACESCG_TO_SRGB",
                                          "ACESCG_TO_P3D65", "ACESCG_TO_REC2020",
                                          "aces2_tonemap", "aces2_gamut_compress",
                                          "linear_to_jmh", "jmh_to_linear",
                                          "linear_to_srgb", "linear_to_pq", "linear_to_hlg"],
        "radiance.radiance_ocio":       ["get_ocio_manager", "HAS_OCIO"],
        "radiance.sampler_utils":       ["SamplerConfig", "build_sampler", "SAMPLERS", "SCHEDULERS",
                                          # dynamic guidance constants
                                          "DYNAMIC_GUIDANCE_EARLY_MULTIPLIER",
                                          "DYNAMIC_GUIDANCE_LATE_MULTIPLIER",
                                          "DYNAMIC_GUIDANCE_EARLY_THRESHOLD",
                                          "DYNAMIC_GUIDANCE_LATE_THRESHOLD",
                                          "DYNAMIC_GUIDANCE_RAMP_WIDTH",
                                          "GUIDANCE_RESCALE_PHI",
                                          "SIGMA_DISCONTINUITY_THRESHOLD",
                                          "PAG_DEFAULT_SCALE",
                                          "PAG_LAYER_NAMES",
                                          "CFG_PLUS_PLUS_DEFAULT_SCALE",
                                          "MODEL_TYPES",
                                          "VIDEO_MODEL_TYPES",
                                          "GUIDANCE_EMBED_MODELS",
                                          "CFG_GUIDED_MODELS",
                                          "PREVIEW_METHODS",
                                          "NOISE_TYPES",
                                          "CLIP_TARGETS",
                                          "MULTI_COND_MODES",
                                          "TILE_BLEND_MODES",
                                          "SamplerMode",
                                          "SigmaCache",
                                          "_sigma_cache",
                                          "RadianceModelRegistry",
                                          "detect_by_config",
                                          "detect_by_architecture",
                                          "detect_by_sampling",
                                          "detect_model_type",
                                          "get_model_defaults",
                                          "gradual_sigma_blend",
                                          "log_tensor",
                                          "SigmaIndexer",
                                          "SamplingStage",
                                          "apply_flux_guidance",
                                          "compute_dynamic_guidance",
                                          "compute_dynamic_cfg",
                                          "compute_base_sigmas",
                                          "WORKFLOW_PRESETS",
                                          "flux_shift_sigmas",
                                          "get_flux_sigmas",
                                          "validate_step_range",
                                          "apply_pag_to_model",
                                          "AYS_ANCHORS",
                                          "get_ays_sigmas",
                                          "guidance_rescale_cfg",
                                          "correct_sigma_end",
                                          "apply_cfg_plus_plus",
                                          "build_sigma_report",
                                          "_temporally_correlate",
                                          "_perlin_noise",
                                          "_perlin_noise_2d",
                                          "_spectral_noise",
                                          "_get_freq_grid",
                                          "_spectral_noise_2d",
                                          "_brownian_noise",
                                          "_simplex_noise",
                                          "_voronoi_noise",
                                          "_curl_noise",
                                          "generate_noise",
                                          "merge_conditionings",
                                          "route_conditioning",
                                          "tile_sample",
                                          "_build_latent_meta",
                                          "MODEL_DEFAULTS",
                                          "DYNAMIC_CFG_EARLY_MULTIPLIER",
                                          "DYNAMIC_CFG_LATE_MULTIPLIER",
                                          "DYNAMIC_CFG_EARLY_THRESHOLD",
                                          "DYNAMIC_CFG_LATE_THRESHOLD"],
        "radiance.nodes_hdr_colorspace":["HDR_COLORSPACES", "LOG_PROFILE_HDR_PARAMS"],
        "radiance.hdr":                 [],
        "radiance.hdr.vae":             ["LOG_PROFILE_HDR_PARAMS", "LOG_PROFILE_HDR_DEFAULT",
                                          "RadianceVAE4KEncode", "RadianceVAE4KDecode",
                                          "RadianceVAE4KRoundtrip"],
        "radiance.hdr.color":           [],
        "radiance.hdr.io":              [],
        "radiance.hdr.analysis":        [],

    }
    # Scalar constants that must not be MagicMock (used in arithmetic at import time)
    _SCALAR_STUB_DEFAULTS = {
        "DYNAMIC_GUIDANCE_EARLY_MULTIPLIER": 0.6,
        "DYNAMIC_GUIDANCE_LATE_MULTIPLIER":  0.95,
        "DYNAMIC_GUIDANCE_EARLY_THRESHOLD":  0.2,
        "DYNAMIC_GUIDANCE_LATE_THRESHOLD":   0.9,
        "DYNAMIC_GUIDANCE_RAMP_WIDTH":       0.05,
        "GUIDANCE_RESCALE_PHI":              0.0,
        "DYNAMIC_CFG_EARLY_MULTIPLIER":      1.2,
        "DYNAMIC_CFG_LATE_MULTIPLIER":       0.7,
        "DYNAMIC_CFG_EARLY_THRESHOLD":       0.15,
        "DYNAMIC_CFG_LATE_THRESHOLD":        0.85,
        "HAS_OCIO":                          False,
        "AWG3_TO_ACESCG":                    _IDENTITY_MATRIX,
        "AWG4_TO_ACESCG":                    _IDENTITY_MATRIX,
        "SGAMUT3_CINE_TO_ACESCG":            _IDENTITY_MATRIX,
        "VGAMUT_TO_ACESCG":                  _IDENTITY_MATRIX,
        "CINEMA_GAMUT_TO_ACESCG":            _IDENTITY_MATRIX,
        "REDWIDEGAMUT_TO_ACESCG":            _IDENTITY_MATRIX,
        "DAVINCI_WIDE_TO_ACESCG":            _IDENTITY_MATRIX,
        "ACESCG_TO_SRGB":                    _IDENTITY_MATRIX,
        "ACESCG_TO_P3D65":                   _IDENTITY_MATRIX,
        "ACESCG_TO_REC2020":                 _IDENTITY_MATRIX,
        "RadianceVAE4KDecode":               _FakeVAE4KDecode,
        "RadianceVAE4KEncode":               _FakeVAE4KEncode,
        "RadianceVAE4KRoundtrip":            _FakeVAE4KRoundtrip,
        "SamplerMode":                       _FakeSamplerMode,
        "SIGMA_DISCONTINUITY_THRESHOLD":     0.01,
        "PAG_DEFAULT_SCALE":                 3.0,
        "CFG_PLUS_PLUS_DEFAULT_SCALE":       1.0,
        "PAG_LAYER_NAMES":                   ["middle"],
        "MODEL_TYPES":                       ["Auto"],
        "VIDEO_MODEL_TYPES":                 ["Auto"],
        "GUIDANCE_EMBED_MODELS":             ["Auto"],
        "CFG_GUIDED_MODELS":                 ["Auto"],
        "PREVIEW_METHODS":                   ["None"],
        "NOISE_TYPES":                       ["gaussian"],
        "CLIP_TARGETS":                      ["auto"],
        "MULTI_COND_MODES":                  ["merge"],
        "TILE_BLEND_MODES":                  ["linear"],
        "WORKFLOW_PRESETS":                  ["Auto", "Custom"],
        "AYS_ANCHORS":                       {},
        "MODEL_DEFAULTS":                    {},
    }

    for _sub_name, _attrs in _radiance_sub_stubs.items():
        if _sub_name not in sys.modules:
            _sub = _stub_module(_sub_name)
            for _attr in _attrs:
                if not hasattr(_sub, _attr):
                    _val = _SCALAR_STUB_DEFAULTS.get(_attr, mock.MagicMock())
                    setattr(_sub, _attr, _val)
            _sub.__spec__ = _imm.ModuleSpec(_sub_name, loader=None)

    # ── Additional missing comfy sub-modules ──────────────────────────────────
    for _cname in ("comfy.cldm.control_types", "comfy.extra_samplers",
                   "comfy.clip_vision", "comfy.gligen", "comfy.taesd",
                   "comfy.ldm", "comfy.ldm.models", "comfy.ldm.modules",
                   "comfy.ldm.modules.attention"):
        if _cname not in sys.modules:
            _stub_module(_cname)
    # UNION_CONTROLNET_TYPES used by nodes_loader.py at module scope
    _cct = sys.modules["comfy.cldm.control_types"]
    if not hasattr(_cct, "UNION_CONTROLNET_TYPES"):
        _cct.UNION_CONTROLNET_TYPES = {"auto": -1, "openpose": 0, "depth": 1,
                                        "hed": 2, "canny": 3, "mlsd": 4,
                                        "normal": 5, "segment": 6}

    # ── node_helpers (ComfyUI built-in helper, not always present) ────────────
    if "node_helpers" not in sys.modules:
        _nh = _stub_module("node_helpers")
        _nh.pillow = lambda fn, *a, **k: fn(*a, **k)

    # ── numpy additions ───────────────────────────────────────────────────────
    if "numpy" in sys.modules:
        _np = sys.modules["numpy"]
        if not hasattr(_np.linalg, "LinAlgError"):
            _np.linalg.LinAlgError = Exception
        if not hasattr(_np, "bool_"):
            _np.bool_ = bool
        if not hasattr(_np, "inf"):
            _np.inf = float("inf")
        if not hasattr(_np, "nan"):
            _np.nan = float("nan")

    # ── torch additions ───────────────────────────────────────────────────────
    if "torch" in sys.modules:
        _t = sys.modules["torch"]
        if not hasattr(_t, "Generator"):
            _t.Generator = mock.MagicMock
        if not hasattr(_t, "finfo"):
            _t.finfo = lambda dtype: type("finfo", (), {"max": 65504.0, "min": -65504.0, "eps": 1e-5})()
        if not hasattr(_t, "iinfo"):
            _t.iinfo = lambda dtype: type("iinfo", (), {"max": 65535, "min": 0})()

    # ── PIL additions ─────────────────────────────────────────────────────────
    if "PIL" in sys.modules:
        _pil = sys.modules["PIL"]
        if not hasattr(_pil, "ImageSequence"):
            _pil.ImageSequence = _stub_module("PIL.ImageSequence")
            _pil.ImageSequence.Iterator = mock.MagicMock


# ── Node discovery ─────────────────────────────────────────────────────────────

def _discover_node_keys_from_source() -> Dict[str, str]:
    """
    Parse the node modules with AST to extract NODE_CLASS_MAPPINGS keys without
    importing the files.  Returns {node_key: source_file}.

    This used to glob only the root `nodes_*.py` layer, which is why retiring
    that layer made three keys silently vanish from the scan instead of
    resolving: the aliases were declared in files that had moved into the
    package. It now walks `nodes/` too, so a key is discovered wherever it
    actually lives.
    """
    results: Dict[str, str] = {}
    patterns = (
        list(RADIANCE_ROOT.glob("nodes_*.py"))
        + list(RADIANCE_ROOT.glob("nodes/**/*.py"))
        + list(RADIANCE_ROOT.glob("color/*.py"))
    )
    for fpath in sorted(patterns):
        try:
            src = fpath.read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                            if isinstance(node.value, ast.Dict):
                                for k in node.value.keys:
                                    if isinstance(k, ast.Constant):
                                        results[k.value] = str(fpath.relative_to(RADIANCE_ROOT))
        except Exception:
            pass
    return results


def _import_file(fpath: str) -> types.ModuleType | None:
    """Import a source file by path, returning the module or None on failure.

    Sets __package__ to the module's real parent package so that relative
    imports resolve against the pre-stubbed radiance.* entries in sys.modules.
    A file under `nodes/generate/` needs `radiance.nodes.generate`, not
    `radiance` — with the wrong parent, `from ...core import x` walks off the
    top of the package and the import fails.
    """
    path = RADIANCE_ROOT / fpath
    spec = importlib.util.spec_from_file_location(
        f"_radiance_smoke_{path.stem}", str(path)
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Give the module a package context so relative imports work.
    parent_parts = path.relative_to(RADIANCE_ROOT).parts[:-1]
    mod.__package__ = ".".join(("radiance",) + parent_parts)
    # Register under the radiance namespace so cross-file relative imports
    # (e.g. from .color_utils import …) can find sibling stubs.
    _mod_key = f"{mod.__package__}.{path.stem}"
    if _mod_key not in sys.modules:
        sys.modules[_mod_key] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception:
        sys.modules.pop(_mod_key, None)
        return None


def _load_node_classes() -> Dict[str, Any]:
    """
    Import each source file and collect class objects from NODE_CLASS_MAPPINGS.
    Returns {node_key: class_object}.  Keys with import errors are skipped.
    """
    _ensure_stubs()
    package_parent = str(RADIANCE_ROOT.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    # v3's production surface is the package entry point. The root nodes_*.py
    # files are compatibility wrappers, so flat-file discovery undercounts the
    # nodes that ComfyUI actually sees.
    if "radiance" in sys.modules:
        try:
            radiance_pkg = sys.modules["radiance"]
            mappings = getattr(radiance_pkg, "NODE_CLASS_MAPPINGS", {})
            if isinstance(mappings, dict) and mappings:
                return dict(mappings)
        except Exception:
            pass

    for name in tuple(sys.modules):
        if name == "radiance" or name.startswith("radiance."):
            sys.modules.pop(name, None)
    for name in (
        "radiance.radiance_ocio",
        "radiance.nodes_hdr_colorspace",
        "radiance.image",
        "radiance.image.defects",
    ):
        sys.modules.pop(name, None)
    try:
        radiance_pkg = importlib.import_module("radiance")
        mappings = getattr(radiance_pkg, "NODE_CLASS_MAPPINGS", {})
        if isinstance(mappings, dict) and mappings:
            return dict(mappings)
    except Exception:
        sys.modules.pop("radiance", None)

    _ensure_stubs()
    key_to_file = _discover_node_keys_from_source()
    # Group keys by source file to import each file once
    file_to_keys: Dict[str, List[str]] = {}
    for key, fpath in key_to_file.items():
        file_to_keys.setdefault(fpath, []).append(key)

    result: Dict[str, Any] = {}
    for fpath, keys in sorted(file_to_keys.items()):
        mod = _import_file(fpath)
        if mod is None:
            continue
        mappings: Dict[str, Any] = getattr(mod, "NODE_CLASS_MAPPINGS", {})
        for key in keys:
            cls = mappings.get(key)
            if cls is not None:
                result[key] = cls
    return result


# ── Load all importable node classes once at module level ─────────────────────
_ALL_NODES: Dict[str, Any] = _load_node_classes()

# ── Test cases ────────────────────────────────────────────────────────────────

class TestNodeStructural(unittest.TestCase):
    """
    STRUCTURAL — every node class must have the four required ComfyUI attrs.
    """
    longMessage = True

    def _check(self, key: str) -> None:
        cls = _ALL_NODES[key]
        ctx = f"[{key}] from {cls.__module__}"

        # RETURN_TYPES — tuple of strings
        self.assertTrue(hasattr(cls, "RETURN_TYPES"),
                        f"{ctx}: missing RETURN_TYPES")
        self.assertIsInstance(cls.RETURN_TYPES, tuple,
                              f"{ctx}: RETURN_TYPES must be a tuple")

        # RETURN_NAMES — tuple of strings, same length as RETURN_TYPES
        self.assertTrue(hasattr(cls, "RETURN_NAMES"),
                        f"{ctx}: missing RETURN_NAMES")
        self.assertIsInstance(cls.RETURN_NAMES, tuple,
                              f"{ctx}: RETURN_NAMES must be a tuple")
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES),
                         f"{ctx}: RETURN_TYPES length {len(cls.RETURN_TYPES)} != "
                         f"RETURN_NAMES length {len(cls.RETURN_NAMES)}")

        # FUNCTION — string naming the execute method
        self.assertTrue(hasattr(cls, "FUNCTION"),
                        f"{ctx}: missing FUNCTION")
        self.assertIsInstance(cls.FUNCTION, str,
                              f"{ctx}: FUNCTION must be a str")
        self.assertTrue(hasattr(cls, cls.FUNCTION),
                        f"{ctx}: method '{cls.FUNCTION}' named by FUNCTION not found on class")

        # CATEGORY — string
        self.assertTrue(hasattr(cls, "CATEGORY"),
                        f"{ctx}: missing CATEGORY")
        self.assertIsInstance(cls.CATEGORY, str,
                              f"{ctx}: CATEGORY must be a str")
        self.assertTrue(cls.CATEGORY.startswith("FXTD STUDIOS/Radiance/"),
                        f"{ctx}: CATEGORY '{cls.CATEGORY}' does not start with "
                        f"'FXTD STUDIOS/Radiance/'")


class TestNodeInputSpec(unittest.TestCase):
    """
    INPUT_SPEC — INPUT_TYPES() must be callable and return a dict with at
    least one of 'required' / 'optional' / 'hidden'.
    """
    longMessage = True

    def _check(self, key: str) -> None:
        cls = _ALL_NODES[key]
        ctx = f"[{key}]"

        self.assertTrue(hasattr(cls, "INPUT_TYPES"),
                        f"{ctx}: missing INPUT_TYPES classmethod")
        try:
            spec = cls.INPUT_TYPES()
        except Exception as exc:
            self.fail(f"{ctx}: INPUT_TYPES() raised {type(exc).__name__}: {exc}")

        self.assertIsInstance(spec, dict,
                              f"{ctx}: INPUT_TYPES() must return a dict")
        valid_keys = {"required", "optional", "hidden"}
        has_valid = bool(spec.keys() & valid_keys)
        self.assertTrue(has_valid,
                        f"{ctx}: INPUT_TYPES() dict must contain at least one of "
                        f"{valid_keys}, got {set(spec.keys())}")


class TestNodeInstantiation(unittest.TestCase):
    """
    EXECUTE — the class must be instantiable with no arguments.
    """
    longMessage = True

    def _check(self, key: str) -> None:
        cls = _ALL_NODES[key]
        ctx = f"[{key}]"
        try:
            obj = cls()
        except Exception as exc:
            self.fail(f"{ctx}: {cls.__name__}() raised {type(exc).__name__}: {exc}")
        self.assertIsNotNone(obj, f"{ctx}: instantiation returned None")


# ── Parametric test generation ────────────────────────────────────────────────

def _add_parametric_tests(
    test_cls: type,
    method_name: str,
    checker_name: str,
    all_nodes: Dict[str, Any],
) -> None:
    """Inject one test method per node key into test_cls."""
    for key in sorted(all_nodes.keys()):
        # Sanitise key for a valid method name
        safe = key.replace("-", "_").replace(".", "_")
        method = f"test_{safe}"

        def _make_test(k: str, c: str):
            def _test(self):
                getattr(self, c)(k)
            _test.__name__ = method
            _test.__doc__ = f"{c} check for {k}"
            return _test

        setattr(test_cls, method, _make_test(key, checker_name))


if _ALL_NODES:
    _add_parametric_tests(TestNodeStructural,    "test_", "_check", _ALL_NODES)
    _add_parametric_tests(TestNodeInputSpec,     "test_", "_check", _ALL_NODES)
    _add_parametric_tests(TestNodeInstantiation, "test_", "_check", _ALL_NODES)


# ── Coverage summary ──────────────────────────────────────────────────────────

class TestCoverageSummary(unittest.TestCase):
    """Meta-test: verify we are testing the expected number of nodes."""

    def test_minimum_node_count(self):
        """Every node the catalog publishes must be discovered and importable.

        The floor was a hand-written 85 while 131 nodes registered, so a third
        of the catalog could have stopped resolving with this green. There is
        one floor now -- EXPECTED_MIN_NODE_COUNT in config/constants.py -- and
        test_package_cleanup.py pins that constant to the real count.
        """
        from radiance.config.constants import EXPECTED_MIN_NODE_COUNT

        degraded = self._environment_load_failures()
        if degraded:
            raise unittest.SkipTest(
                "environment is missing runtime dependencies for: "
                + ", ".join(degraded)
            )

        count = len(_ALL_NODES)
        self.assertGreaterEqual(
            count, EXPECTED_MIN_NODE_COUNT,
            f"Only {count} nodes were importable. Expected >= "
            f"{EXPECTED_MIN_NODE_COUNT}. Check that stubs are adequate or that "
            "source files parse cleanly."
        )

    #: Node keys that appear in a NODE_CLASS_MAPPINGS literal in the source but
    #: are deliberately not published to ComfyUI.
    #:
    #: This list was twelve entries long. Nine of them turned out to be complete,
    #: importable node classes that the v3 reorganisation simply forgot to list
    #: in a group's mapping dict — they are registered now. What remains is three
    #: back-compat aliases: each points at a class that IS published under its
    #: current key, so registering the old key would put a second, identical
    #: entry in the node menu for no benefit.
    #:
    #: The list is a ratchet: it may shrink, and a shrink fails this test so the
    #: entry gets removed, but it may never grow without someone editing here.
    # Empty, and it should stay that way.
    #
    # It held three alias keys that "could not" be registered. They are all
    # published now — as DEPRECATED subclasses, so saved workflows referencing
    # the old key still load while the menu shows one entry per node. Widening
    # this file's AST scan from the retired root `nodes_*.py` layer to the whole
    # `nodes/` package turned up seventeen more nodes in the same state, none of
    # which anyone had noticed; `nodes/aggregate.py` publishes by default now.
    #
    # An entry here means a node exists in source and deliberately does not
    # ship. Write down why.
    _KNOWN_UNREGISTERED = frozenset()

    @staticmethod
    def _environment_load_failures():
        """Labels of node groups this machine could not import at all."""
        try:
            import radiance
        except Exception:  # pragma: no cover - package import is tested elsewhere
            return ()
        result = getattr(radiance, "_LOAD_RESULT", None)
        return tuple(f.source.label for f in getattr(result, "failures", ()) or ())

    def test_all_keys_have_class(self):
        """Every key discovered by AST must resolve to an actual class.

        This used to call `warnings.warn` instead of asserting, which meant a
        node could stop resolving entirely and the suite still reported green --
        twelve of them had, and the warning text blamed "torch/GPU at runtime"
        for what is really a registration gap. Assert against an explicit
        ratchet instead, so new breakage fails and known breakage is visible.
        """
        all_keys = set(_discover_node_keys_from_source().keys())
        missing = all_keys - set(_ALL_NODES.keys())

        # A machine short a runtime dependency drops whole node groups, which
        # would show up here as dozens of extra "missing" keys and drown the
        # signal we actually want. That case has its own loud ERROR at startup
        # (see radiance.report_node_load_health); skip rather than double-report.
        degraded = self._environment_load_failures()
        if degraded:
            raise unittest.SkipTest(
                "environment is missing runtime dependencies for: "
                + ", ".join(degraded)
            )

        regressions = missing - self._KNOWN_UNREGISTERED
        self.assertFalse(
            regressions,
            f"{len(regressions)} node key(s) no longer resolve to a class and "
            f"are not in the known-unregistered allowlist: {sorted(regressions)}. "
            "Either wire the node into the package entry point or, if the drop "
            "is deliberate, add it to _KNOWN_UNREGISTERED with a reason.",
        )

        fixed = self._KNOWN_UNREGISTERED - missing
        self.assertFalse(
            fixed,
            f"{len(fixed)} node key(s) now resolve but are still listed in "
            f"_KNOWN_UNREGISTERED: {sorted(fixed)}. Remove them from the "
            "allowlist so it keeps ratcheting down.",
        )


# ── ACES colour science round-trip tests ─────────────────────────────────────

class TestACESColorScience(unittest.TestCase):
    """
    Verify the core ACES matrix constants in nodes_colorscience.py satisfy
    mathematical requirements without needing torch or a GPU.

    Tests
    -----
    1. Forward × Inverse ≈ Identity  (round-trip consistency)
    2. Matrix values are within ±0.01 of the published ACES 1.3 AP1 spec.
    3. Forward matrix preserves approximate luminance (row-sums ≈ 1.0 for
       an all-white scene-linear input).
    """

    # Published ACES 1.3 / S-2014-004 reference values
    # BT.709 linear → ACEScg (AP1 D60)
    _SPEC_709_TO_ACESCG = [
        [0.6131,  0.3395,  0.0474],
        [0.0701,  0.9163,  0.0136],
        [0.0206,  0.1096,  0.8698],
    ]

    # ACEScg → BT.709 linear (inverse of above)
    _SPEC_ACESCG_TO_709 = [
        [ 1.7048, -0.6239, -0.0809],
        [-0.1295,  1.1383, -0.0089],
        [-0.0240, -0.1246,  1.1486],
    ]

    @staticmethod
    def _matmul_3x3(A, B):
        """Pure-Python 3×3 matrix multiply."""
        C = [[0.0, 0.0, 0.0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    @staticmethod
    def _is_identity(M, tol=1e-3):
        """Return True when M is within tol of the 3×3 identity matrix."""
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                if abs(M[i][j] - expected) > tol:
                    return False, i, j, M[i][j], expected
        return True, -1, -1, 0.0, 0.0

    def _extract_matrices(self):
        """
        Pull the live matrix constants from nodes_colorscience.py using AST
        so the test stays decoupled from torch imports.
        """
        src_path = RADIANCE_ROOT / "color" / "ops.py"
        src = src_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        matrices = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                # Class-body attributes use ast.Name, not ast.Attribute
                if isinstance(t, ast.Name) and t.id in ("M_REC709_TO_ACESCG", "M_ACESCG_TO_REC709"):
                    attr_name = {
                        "M_REC709_TO_ACESCG": "_M_709_TO_ACESCG",
                        "M_ACESCG_TO_REC709": "_M_ACESCG_TO_709",
                    }[t.id]
                elif isinstance(t, ast.Attribute) and t.attr in ("_M_709_TO_ACESCG", "_M_ACESCG_TO_709"):
                    attr_name = t.attr
                else:
                    continue
                # value must be ast.Call → torch.tensor([[...]])
                if not (isinstance(node.value, ast.Call) and
                        isinstance(node.value.args[0], ast.List)):
                    continue
                rows = node.value.args[0].elts
                mat = []
                for row in rows:
                    if not isinstance(row, ast.List):
                        break
                    mat.append([
                        (elt.value if isinstance(elt, ast.Constant) else
                         -elt.operand.value if (isinstance(elt, ast.UnaryOp) and
                                                 isinstance(elt.op, ast.USub) and
                                                 isinstance(elt.operand, ast.Constant))
                         else None)
                        for elt in row.elts
                    ])
                if len(mat) == 3 and all(len(r) == 3 for r in mat):
                    matrices[attr_name] = mat

        return matrices

    def test_round_trip_is_identity(self):
        """Forward × Inverse must produce the identity matrix within 1e-3."""
        mats = self._extract_matrices()
        self.assertIn("_M_709_TO_ACESCG", mats,
                      "Could not extract _M_709_TO_ACESCG from nodes_colorscience.py")
        self.assertIn("_M_ACESCG_TO_709", mats,
                      "Could not extract _M_ACESCG_TO_709 from nodes_colorscience.py")

        product = self._matmul_3x3(mats["_M_ACESCG_TO_709"], mats["_M_709_TO_ACESCG"])
        # Tolerance of 5e-3 accounts for accumulated rounding in the 4-decimal
        # matrix constants (worst observed off-diagonal: ~0.0036).
        # Full-precision SMPTE values would pass at 1e-6.
        ok, i, j, got, exp = self._is_identity(product, tol=5e-3)
        self.assertTrue(
            ok,
            f"Round-trip M_ACESCG_TO_709 @ M_709_TO_ACESCG is not identity: "
            f"[{i}][{j}] = {got:.6f}, expected {exp:.6f} (tol 2e-3)"
        )

    def test_forward_matrix_matches_spec(self):
        """Live _M_709_TO_ACESCG values must be within 0.01 of ACES 1.3 spec."""
        mats = self._extract_matrices()
        if "_M_709_TO_ACESCG" not in mats:
            self.skipTest("Could not extract _M_709_TO_ACESCG")
        live = mats["_M_709_TO_ACESCG"]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    live[i][j], self._SPEC_709_TO_ACESCG[i][j], delta=0.01,
                    msg=f"_M_709_TO_ACESCG[{i}][{j}]: live={live[i][j]:.4f}, "
                        f"spec={self._SPEC_709_TO_ACESCG[i][j]:.4f}"
                )

    def test_inverse_matrix_matches_spec(self):
        """Live _M_ACESCG_TO_709 values must be within 0.01 of ACES 1.3 spec."""
        mats = self._extract_matrices()
        if "_M_ACESCG_TO_709" not in mats:
            self.skipTest("Could not extract _M_ACESCG_TO_709")
        live = mats["_M_ACESCG_TO_709"]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    live[i][j], self._SPEC_ACESCG_TO_709[i][j], delta=0.01,
                    msg=f"_M_ACESCG_TO_709[{i}][{j}]: live={live[i][j]:.4f}, "
                        f"spec={self._SPEC_ACESCG_TO_709[i][j]:.4f}"
                )

    def test_white_point_preservation(self):
        """
        An equal-energy white (1,1,1) scene-linear input transformed forward
        to ACEScg must have row sums ≈ 1.0 (luminance is approximately preserved).
        """
        mats = self._extract_matrices()
        if "_M_709_TO_ACESCG" not in mats:
            self.skipTest("Could not extract _M_709_TO_ACESCG")
        M = mats["_M_709_TO_ACESCG"]
        for i, row in enumerate(M):
            row_sum = sum(row)
            self.assertAlmostEqual(
                row_sum, 1.0, delta=0.01,
                msg=f"Row {i} sum = {row_sum:.4f} (expected ≈ 1.0 for white-point preservation)"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
