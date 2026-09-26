import torch
import numpy as np
import os
import threading
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

# Setup logger
logger = logging.getLogger("radiance.color.lut")


# ── What the LUT expects as input ────────────────────────────────────────────
#
# Most show and camera LUTs take a camera log signal (LogC3, S-Log3, ...) and
# the graph carries scene-linear light. 3.5.0: log_space used to raise the
# pixels to a power (10^x, 2^x, e^x), which matches no camera curve, so those
# LUTs could not be used correctly. It now encodes the linear image into the
# LUT's input space through color/encodings (OCIO when available, the analytic
# curve and gamut matrix otherwise), the same code Read and Write use.
#
# The old names stay accepted so saved graphs render as before; they are no
# longer offered in the menu.

_LEGACY_EXPONENTS = {"Log10": 10.0, "Log2": 2.0, "Natural Log (Ln)": float(np.e)}


def _lut_input_choices() -> List[str]:
    try:
        from .encodings import ENCODINGS
    except Exception:  # pragma: no cover - encodings always ships with the package
        return ["ARRI LogC3"]
    linear = {"Linear Rec.709 (sRGB)", "Linear Rec.2020", "Linear P3-D65", "ACEScg", "ACES2065-1"}
    names = [n for n in ENCODINGS if n not in linear]
    # the common show-LUT inputs first
    first = ["ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "Panasonic V-Log", "RED Log3G10",
             "Canon Log 3", "DaVinci Intermediate", "ACEScct"]
    return [n for n in first if n in names] + [n for n in names if n not in first]


def _working_spaces() -> List[str]:
    try:
        from .encodings import WORKING_SPACES
        return list(WORKING_SPACES)
    except Exception:  # pragma: no cover
        return ["Linear Rec.709 (sRGB)"]


def encode_for_lut(image: torch.Tensor, lut_input: str,
                   working_space: str = "Linear Rec.709 (sRGB)") -> torch.Tensor:
    """Scene-linear ``working_space`` RGB -> the signal the LUT was built for.

    Alpha (and any channel past the third) is passed through untouched."""
    if lut_input in _LEGACY_EXPONENTS:
        base = torch.as_tensor(_LEGACY_EXPONENTS[lut_input], dtype=image.dtype, device=image.device)
        return torch.cat([torch.pow(base, image[..., :3]), image[..., 3:]], dim=-1)
    from .encodings import encode
    rgb = image[..., :3].detach().float().cpu().numpy()
    out, _path = encode(rgb, lut_input, working_space)
    enc = torch.from_numpy(np.ascontiguousarray(out)).to(device=image.device, dtype=image.dtype)
    if image.shape[-1] > 3:
        enc = torch.cat([enc, image[..., 3:]], dim=-1)
    return enc


def _validate_lut_input(log_encoding) -> "bool | str":
    if log_encoding in _LEGACY_EXPONENTS or log_encoding in _lut_input_choices():
        return True
    return f"unknown LUT input encoding {log_encoding!r}"


@dataclass
class LUTData:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    """Container for parsed LUT information with metadata."""

    lut_tensor: torch.Tensor
    size: int
    domain_min: List[float]
    domain_max: List[float]
    filepath: str
    # FIX #4: store (mtime_ns, size) validator instead of MD5 hash.
    # Computing MD5 of a 3 MB .cube file on every cache-hit adds ~5-10 ms I/O.
    # os.stat() costs a single syscall (~1 µs) and detects both modification
    # time changes and in-place overwrites of the same size.
    file_validator: Tuple[int, int]  # (mtime_ns, file_size)
    # FIX #3: per-device tensor cache so we avoid a fresh host→device transfer
    # on every apply_lut() call even when the CPU tensor is already cached.
    _device_tensors: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def get_tensor_for_device(self, device: torch.device) -> torch.Tensor:
        """Return a device-resident tensor, caching it after the first transfer."""
        key = str(device)
        if key not in self._device_tensors:
            self._device_tensors[key] = self.lut_tensor.to(device)
        return self._device_tensors[key]


class LUTCache:
    """Thread-safe LUT cache with bounded size to prevent memory leaks."""

    _cache: Dict[str, LUTData] = {}
    _max_cache_size: int = 32
    _lock = threading.RLock()  # Reentrant lock for thread safety

    @classmethod
    def _get_validator(cls, filepath: str) -> Tuple[int, int]:
        """FIX #4: stat-based validator — near-zero cost vs full MD5 read."""
        try:
            st = os.stat(filepath)
            return (st.st_mtime_ns, st.st_size)
        except Exception as e:
            logger.debug(f"Could not stat LUT file {filepath}: {e}")
            return (0, 0)



    @classmethod
    def get(cls, filepath: str) -> Optional[LUTData]:
        with cls._lock:
            if filepath in cls._cache:
                cached = cls._cache[filepath]
                # FIX #4: validate with stat, not MD5
                current_validator = cls._get_validator(filepath)
                if current_validator == cached.file_validator:
                    return cached
                else:
                    del cls._cache[filepath]
            return None

    @classmethod
    def set(cls, filepath: str, lut_data: LUTData):
        with cls._lock:
            # Evict oldest if at capacity (FIFO eviction)
            while len(cls._cache) >= cls._max_cache_size:
                oldest_key = next(iter(cls._cache))
                del cls._cache[oldest_key]
                logger.debug(f"LUT cache evicted: {oldest_key}")
            cls._cache[filepath] = lut_data

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._cache.clear()
            logger.debug("LUT cache cleared")


class RadianceLUTApply:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    """
    Apply a 3D LUT (.cube) to an image with professional-grade color transformation.
    """

    LOG_ENCODINGS = _lut_input_choices()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to transform. Values should be in the encoding the LUT expects; anything outside the LUT's DOMAIN_MIN/MAX (0 to 1 by default) is clamped to its edge before lookup."}),
                "lut_file": (cls.get_lut_files(), {"tooltip": "3D .cube LUT from ComfyUI's models/luts folder."}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mix between the original image (0) and the full LUT result (1)."},
                ),
                "log_space": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Encode to LUT input",
                        "label_off": "Image as is",
                        "tooltip": "On: the image is scene-linear (in working_space) and is encoded into log_encoding, the input the LUT was built for, before the lookup. Use this for show and camera LUTs such as LogC3 to Rec.709. Off: the pixels go into the LUT unchanged, for an image already in the LUT's input space.",
                    },
                ),
            },
            "optional": {
                "log_encoding": (cls.LOG_ENCODINGS, {"default": "ARRI LogC3", "tooltip": "The encoding the LUT expects as input (camera log, ACEScct, sRGB, PQ...). Used when log_space is on."}),
                "clamp_output": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Clamp to 0-1. Disable for HDR."},
                ),
                "interpolation": (
                    ["Trilinear", "Tetrahedral"],
                    {
                        "default": "Trilinear",
                        "tooltip": "LUT interpolation between grid points. Tetrahedral is more accurate on hue and neutral axes but slightly slower.",
                    },
                ),
                "working_space": (_working_spaces(), {"default": "Linear Rec.709 (sRGB)", "tooltip": "The linear space the image is in, used when log_space is on to reach the LUT's input primaries."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, log_encoding="ARRI LogC3"):
        # Saved graphs may hold the pre-3.5.0 names (Log10, Log2, Ln), which
        # are no longer in the menu but still run as before.
        return _validate_lut_input(log_encoding)

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lut"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Apply a 3D LUT (.cube) with trilinear or tetrahedral interpolation. Encodes a scene-linear image into the LUT's input (LogC, S-Log3, ACEScct...) when asked, and keeps HDR (unclamped) output."

    @staticmethod
    def get_lut_files() -> List[str]:
        try:
            import folder_paths

            # Safe check avoiding KeyError if 'luts' isn't registered
            if "luts" not in folder_paths.folder_names_and_paths:
                # isinstance, not truthiness: a test that stubs folder_paths
                # with a bare MagicMock makes models_dir a truthy Mock, and
                # joining it created a literal "MagicMock/mock.models_dir/<id>/"
                # tree on disk. This runs from INPUT_TYPES(), so it fired on
                # every catalog load.
                models_dir = getattr(folder_paths, "models_dir", None)
                if not isinstance(models_dir, str) or not models_dir:
                    return ["No LUTs found"]

                luts_path = os.path.join(models_dir, "luts")
                if not os.path.isdir(luts_path):
                    # Do NOT create it here. ComfyUI calls INPUT_TYPES() on
                    # every node at startup and on every /object_info request;
                    # a widget enumerator must not have filesystem side
                    # effects. The directory is created on demand instead.
                    return ["No LUTs found"]
                folder_paths.add_model_folder_path("luts", luts_path)

            luts = folder_paths.get_filename_list("luts")
            return luts if luts else ["No LUTs found"]
        except Exception as e:
            logger.debug(f"Could not load LUT file list: {e}")
            return ["No LUTs found"]

    @staticmethod
    def parse_cube_file(filepath: str) -> LUTData:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LUT file not found: {filepath}")

        lut_data = []
        lut_size = None
        lut_1d_size = None      # FIX: track 1D size separately
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 1.0]

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("TITLE"):
                    continue

                if line.startswith("LUT_1D_SIZE"):
                    # FIX: parse 1D LUT size; data is collected in the same loop
                    lut_1d_size = int(line.split()[1])
                    continue

                if line.startswith("LUT_3D_SIZE"):
                    lut_size = int(line.split()[1])
                    continue
                if line.startswith("DOMAIN_MIN"):
                    domain_min = [float(x) for x in line.split()[1:4]]
                    continue
                if line.startswith("DOMAIN_MAX"):
                    domain_max = [float(x) for x in line.split()[1:4]]
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    try:
                        lut_data.append([float(x) for x in parts[:3]])
                    except ValueError:
                        continue

        if not lut_data:
            raise ValueError("No valid data found in LUT file.")

        # ── 1D LUT: promote to 3D identity-composition ───────────────────────
        # A 1D LUT applies the same 1D transfer function to each channel
        # independently. We promote it to a 3D LUT whose identity grid has the
        # per-channel curve baked in, so the existing trilinear/tetrahedral
        # interpolation path works without modification.
        # Promotion size: min(lut_1d_size, 64) — larger is more accurate but
        # wastes VRAM; 64³ is indistinguishable from a native 1D apply.
        if lut_1d_size is not None:
            lut_1d = np.array(lut_data, dtype=np.float32)  # (N, 3)
            N = lut_1d.shape[0]
            promote_size = min(lut_1d_size, 64)

            # Build index lookup: for each grid step, find nearest 1D entry
            grid = np.linspace(0.0, 1.0, promote_size, dtype=np.float32)
            indices = np.round(grid * (N - 1)).astype(np.int32).clip(0, N - 1)

            # Build 3D array: each axis controls one channel, others are identity
            lut_array = np.zeros((promote_size, promote_size, promote_size, 3),
                                 dtype=np.float32)
            for ri in range(promote_size):
                for gi in range(promote_size):
                    for bi in range(promote_size):
                        lut_array[ri, gi, bi, 0] = lut_1d[indices[ri], 0]
                        lut_array[ri, gi, bi, 1] = lut_1d[indices[gi], 1]
                        lut_array[ri, gi, bi, 2] = lut_1d[indices[bi], 2]

            lut_size = promote_size
            logger.info(
                f"[LUT] 1D LUT ({N} entries) promoted to {promote_size}³ 3D grid: "
                f"{filepath}"
            )
        else:
            # ── 3D LUT ───────────────────────────────────────────────────────
            if lut_size is None:
                lut_size = round(len(lut_data) ** (1 / 3))

            # FIX #1: .cube format stores entries R-fastest (innermost loop).
            # After reshape(S,S,S,3) in C order, arr[i,j,k] = entry for B=i, G=j, R=k.
            # Transposing axes (2,1,0) gives arr[R,G,B,c] — correct for the indexing.
            lut_array = np.array(lut_data, dtype=np.float32).reshape(
                lut_size, lut_size, lut_size, 3
            )
            lut_array = lut_array.transpose(2, 1, 0, 3)   # [B,G,R,c] → [R,G,B,c]

        lut_tensor = torch.from_numpy(lut_array.copy())   # copy needed after transpose/promote

        # FIX #4: use stat-based validator
        file_validator = LUTCache._get_validator(filepath)

        return LUTData(
            lut_tensor, lut_size, domain_min, domain_max, filepath, file_validator
        )

    @staticmethod
    def trilinear_interpolate(image, lut, lut_size, domain_min, domain_max):
        # Scale to domain
        domain_range = domain_max - domain_min
        coords = (image[..., :3] - domain_min) / domain_range
        coords = torch.clamp(coords, 0.0, 1.0) * (lut_size - 1)

        coords_floor = torch.floor(coords).long()
        coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
        coords_ceil = coords_floor + 1
        coords_frac = coords - coords_floor.float()

        x0, y0, z0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        x1, y1, z1 = coords_ceil[..., 0], coords_ceil[..., 1], coords_ceil[..., 2]

        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]

        xd, yd, zd = coords_frac[..., 0:1], coords_frac[..., 1:2], coords_frac[..., 2:3]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd

    @staticmethod
    def tetrahedral_interpolate(image, lut, lut_size, domain_min, domain_max):
        """Tetrahedral (Sakamoto) interpolation — more accurate than trilinear."""
        domain_range = domain_max - domain_min
        coords = (image[..., :3] - domain_min) / domain_range
        coords = torch.clamp(coords, 0.0, 1.0) * (lut_size - 1)

        coords_floor = torch.floor(coords).long()
        coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
        coords_ceil = coords_floor + 1
        coords_frac = coords - coords_floor.float()

        x0, y0, z0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        x1, y1, z1 = coords_ceil[..., 0], coords_ceil[..., 1], coords_ceil[..., 2]
        xd = coords_frac[..., 0]
        yd = coords_frac[..., 1]
        zd = coords_frac[..., 2]

        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]

        # Initialise with Case 1: R >= G >= B  (xd >= yd >= zd)
        result = (
            c000
            + (c100 - c000) * xd.unsqueeze(-1)
            + (c110 - c100) * yd.unsqueeze(-1)
            + (c111 - c110) * zd.unsqueeze(-1)
        )

        # Case 2: R >= B >= G  (xd >= zd >= yd)
        mask = (xd >= zd) & (zd >= yd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(
                m,
                c000
                + (c100 - c000) * xd.unsqueeze(-1)
                + (c101 - c100) * zd.unsqueeze(-1)
                + (c111 - c101) * yd.unsqueeze(-1),
                result,
            )

        # Case 3: B >= R >= G  (zd >= xd >= yd)
        mask = (zd >= xd) & (xd >= yd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(
                m,
                c000
                + (c001 - c000) * zd.unsqueeze(-1)
                + (c101 - c001) * xd.unsqueeze(-1)
                + (c111 - c101) * yd.unsqueeze(-1),
                result,
            )

        # Case 4: G >= R >= B  (yd >= xd >= zd)
        mask = (yd >= xd) & (xd >= zd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(
                m,
                c000
                + (c010 - c000) * yd.unsqueeze(-1)
                + (c110 - c010) * xd.unsqueeze(-1)
                + (c111 - c110) * zd.unsqueeze(-1),
                result,
            )

        # Case 5: G >= B >= R  (yd >= zd >= xd)
        mask = (yd >= zd) & (zd >= xd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(
                m,
                c000
                + (c010 - c000) * yd.unsqueeze(-1)
                + (c011 - c010) * zd.unsqueeze(-1)
                + (c111 - c011) * xd.unsqueeze(-1),
                result,
            )

        # Case 6: B >= G >= R  (zd >= yd >= xd)
        mask = (zd >= yd) & (yd >= xd)
        if mask.any():
            m = mask.unsqueeze(-1)
            result = torch.where(
                m,
                c000
                + (c001 - c000) * zd.unsqueeze(-1)
                + (c011 - c001) * yd.unsqueeze(-1)
                + (c111 - c011) * xd.unsqueeze(-1),
                result,
            )

        return result

    def apply_lut(
        self,
        image,
        lut_file,
        strength=1.0,
        log_space=False,
        log_encoding="ARRI LogC3",
        clamp_output=False,
        interpolation="Trilinear",
        working_space="Linear Rec.709 (sRGB)",
    ):
        if strength == 0.0:
            return (image,)

        if not lut_file:
            return (image,)

        lut_path = None
        if os.path.isabs(lut_file):
            lut_path = lut_file
            if not os.path.exists(lut_path):
                raise FileNotFoundError(f"LUT file not found: {lut_path}")
        else:
            try:
                import folder_paths
                lut_path = folder_paths.get_full_path("luts", lut_file)
            except Exception as e:
                logger.error(f"Failed to resolve LUT path for '{lut_file}': {e}")
                return (image,)

        if not lut_path or not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_file}")

        # Cache lookup
        lut_data = LUTCache.get(lut_path)
        if lut_data is None:
            try:
                lut_data = self.parse_cube_file(lut_path)
                LUTCache.set(lut_path, lut_data)
            except Exception as e:
                logger.warning(f"Error parsing LUT: {e}")
                return (image,)

        # Process
        device = image.device
        img_proc = image.clone()

        if log_space:
            img_proc = encode_for_lut(img_proc, log_encoding, working_space)

        # FIX #3: use cached device tensor — avoids repeated host→device transfer
        lut_tensor = lut_data.get_tensor_for_device(device)
        d_min = torch.tensor(lut_data.domain_min, device=device)
        d_max = torch.tensor(lut_data.domain_max, device=device)

        # Choose interpolation method
        if interpolation == "Tetrahedral":
            result = self.tetrahedral_interpolate(
                img_proc, lut_tensor, lut_data.size, d_min, d_max
            )
        else:
            result = self.trilinear_interpolate(
                img_proc, lut_tensor, lut_data.size, d_min, d_max
            )

        if image.shape[-1] > 3:
            result = torch.cat([result, img_proc[..., 3:]], dim=-1)

        if strength < 1.0:
            result = image * (1.0 - strength) + result * strength

        if clamp_output:
            result = torch.clamp(result, 0.0, 1.0)
        # No min=0 clamp when disabled — inverse LUTs and gamut maps produce valid negatives

        return (result,)


class RadianceLUTBlend:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    """
    Blend two LUTs together with various blend modes. Allows creative combination
    of color grades and seamless transitions between different looks.
    """

    BLEND_MODES = ["Linear", "Luminosity", "Saturation", "Hue"]

    @classmethod
    def INPUT_TYPES(cls):
        lut_files = RadianceLUTApply.get_lut_files()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to transform through both LUTs. Values outside each LUT's domain are clamped to its edge before lookup."}),
                "lut_a": (lut_files, {"tooltip": "First .cube LUT from models/luts. It is the base look: the non-Linear modes keep its luminance or chroma."}),
                "lut_b": (lut_files, {"tooltip": "Second .cube LUT from models/luts, blended towards by blend_factor."}),
                "blend_factor": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 = LUT A only, 1.0 = LUT B only",
                    },
                ),
                "blend_mode": (cls.BLEND_MODES, {"default": "Linear", "tooltip": "Linear: mix the two results. Luminosity: A's colour with mixed luminance. Saturation: mixed chroma at A's luminance. Hue: mixed chroma direction with A's chroma amount and luminance."}),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mix between the original image (0) and the blended LUT result (1)."},
                ),
                "clamp_output": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Clamp to 0-1. Disable for HDR."},
                ),
                # FIX #17: expose log_space and interpolation so blends work in log space
                "log_space": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Encode to LUT input",
                        "label_off": "Image as is",
                        "tooltip": "On: the image is scene-linear (in working_space) and is encoded into log_encoding, the input both LUTs were built for, before the lookups. Off: the pixels go into the LUTs unchanged.",
                    },
                ),
                "log_encoding": (RadianceLUTApply.LOG_ENCODINGS, {"default": "ARRI LogC3", "tooltip": "The encoding both LUTs expect as input (camera log, ACEScct, sRGB, PQ...). Used when log_space is on."}),
                "interpolation": (
                    ["Trilinear", "Tetrahedral"],
                    {"default": "Trilinear", "tooltip": "LUT interpolation used for both lookups. Tetrahedral is more accurate but slightly slower."},
                ),
                "working_space": (_working_spaces(), {"default": "Linear Rec.709 (sRGB)", "tooltip": "The linear space the image is in, used when log_space is on to reach the LUTs' input primaries."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, log_encoding="ARRI LogC3"):
        return _validate_lut_input(log_encoding)

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend_luts"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Blend two LUTs with various blend modes for creative color grading."

    def blend_luts(
        self,
        image,
        lut_a,
        lut_b,
        blend_factor,
        blend_mode,
        strength=1.0,
        clamp_output=False,
        log_space=False,
        log_encoding="ARRI LogC3",
        interpolation="Trilinear",
        working_space="Linear Rec.709 (sRGB)",
    ):
        if strength == 0.0:
            return (image,)

        # FIX #17: pass log_space and interpolation through to each apply_lut call
        lut_applier = RadianceLUTApply()
        result_a = lut_applier.apply_lut(
            image, lut_a, 1.0, log_space, log_encoding, False, interpolation, working_space
        )[0]
        result_b = lut_applier.apply_lut(
            image, lut_b, 1.0, log_space, log_encoding, False, interpolation, working_space
        )[0]

        if blend_mode == "Linear":
            blended = result_a * (1.0 - blend_factor) + result_b * blend_factor

        elif blend_mode == "Luminosity":
            # Take luminosity from blend, color from A
            luma_a = (
                0.2126 * result_a[..., 0]
                + 0.7152 * result_a[..., 1]
                + 0.0722 * result_a[..., 2]
            )
            luma_b = (
                0.2126 * result_b[..., 0]
                + 0.7152 * result_b[..., 1]
                + 0.0722 * result_b[..., 2]
            )
            luma_blend = luma_a * (1.0 - blend_factor) + luma_b * blend_factor

            luma_a_u = luma_a.unsqueeze(-1)
            luma_blend = luma_blend.unsqueeze(-1)
            blended = result_a * (luma_blend / (luma_a_u + 1e-10))

        elif blend_mode == "Saturation":
            # Take saturation from blend, luminosity from A
            luma_a = (
                0.2126 * result_a[..., 0]
                + 0.7152 * result_a[..., 1]
                + 0.0722 * result_a[..., 2]
            ).unsqueeze(-1)
            luma_b = (
                0.2126 * result_b[..., 0]
                + 0.7152 * result_b[..., 1]
                + 0.0722 * result_b[..., 2]
            ).unsqueeze(-1)

            sat_a = result_a - luma_a
            sat_b = result_b - luma_b
            sat_blend = sat_a * (1.0 - blend_factor) + sat_b * blend_factor

            blended = luma_a + sat_blend
            # FIX #15: sat_b is chroma relative to luma_b, not luma_a. When the
            # two LUTs shift luminance differently, adding sat_b to luma_a
            # introduces a tonal error ∝ (luma_a - luma_b)*blend_factor.
            # Restore the target luminance (luma_a) after the chroma blend.
            new_luma = (
                0.2126 * blended[..., 0]
                + 0.7152 * blended[..., 1]
                + 0.0722 * blended[..., 2]
            ).unsqueeze(-1)
            blended = blended * (luma_a / (new_luma + 1e-10))

        elif blend_mode == "Hue":
            # Take hue from B, saturation and luminosity from A
            luma_a = (
                0.2126 * result_a[..., 0]
                + 0.7152 * result_a[..., 1]
                + 0.0722 * result_a[..., 2]
            ).unsqueeze(-1)
            luma_b = (
                0.2126 * result_b[..., 0]
                + 0.7152 * result_b[..., 1]
                + 0.0722 * result_b[..., 2]
            ).unsqueeze(-1)

            # Normalise to get hue direction
            sat_a_mag = torch.sqrt(
                torch.sum((result_a - luma_a) ** 2, dim=-1, keepdim=True) + 1e-10
            )
            sat_b_dir = (result_b - luma_b) / (
                torch.sqrt(torch.sum((result_b - luma_b) ** 2, dim=-1, keepdim=True))
                + 1e-10
            )

            sat_a_dir = (result_a - luma_a) / (sat_a_mag + 1e-10)
            hue_blend = sat_a_dir * (1.0 - blend_factor) + sat_b_dir * blend_factor
            hue_blend = hue_blend / (
                torch.sqrt(torch.sum(hue_blend**2, dim=-1, keepdim=True)) + 1e-10
            )

            blended = luma_a + hue_blend * sat_a_mag

        else:
            blended = result_a * (1.0 - blend_factor) + result_b * blend_factor

        # Apply overall strength
        if strength < 1.0:
            blended = image * (1.0 - strength) + blended * strength

        if clamp_output:
            blended = torch.clamp(blended, 0.0, 1.0)

        return (blended,)


# =============================================================================
# NODE MAPPINGS
# FIX #2: NODE_CLASS_MAPPINGS was absent — both nodes were invisible to ComfyUI.
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceLUTApply": RadianceLUTApply,
    "RadianceLUTBlend": RadianceLUTBlend,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLUTApply": "◎ Radiance LUT Apply",
    "RadianceLUTBlend": "◎ Radiance LUT Blend",
    
}
