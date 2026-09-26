"""File colour encodings for Read and Write: transfer + primaries, with OCIO.

One table says what every encoding offered by RadianceRead / RadianceWrite
*is*: a transfer function, a set of primaries and a white point, and the name
of the same colorspace in OpenColorIO's built-in ACES studio config.

Conversion runs through OpenColorIO when it is installed (the studio config is
built into OCIO 2.2+, no file needed), which is what Nuke, Resolve, Flame and
every OCIO host resolve the same names to. Without OCIO the analytic path
below runs: the transfer functions in :mod:`radiance.color.transfer` plus
primaries matrices derived from the published chromaticities, adapted to
the ACES white with CAT02 or Bradford per gamut exactly as the studio config
does. ``tests/test_color_encodings.py`` holds the two paths to each
other, so a file decoded on a machine without OCIO matches one decoded with it.

Before 3.5 a camera-log read decoded the transfer only: an ARRI LogC4 plate
came out as linear AWG4 labelled "linear", i.e. with the camera's primaries
and visibly wrong saturation in any Rec.709 or ACEScg pipeline. Every
encoding now goes all the way to the working space.

HDR display encodings (PQ, HLG) are scene-referred here, BT.2408 style:
scene-linear 1.0 is reference (diffuse) white, placed at
``reference_white_nits`` (203 by default, the value Radiance's own HDR nodes
use). They are not OCIO display colorspaces, which would drag in a view
transform (the ACES output tonemap) that a file encoding must not apply.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from . import transfer as _tf

log = logging.getLogger("radiance.color.encodings")

try:  # optional at runtime, a declared dependency of a full install
    import PyOpenColorIO as _OCIO  # type: ignore
    HAS_OCIO = True
except Exception:  # noqa: BLE001
    _OCIO = None
    HAS_OCIO = False


# ── Primaries (CIE 1931 xy) ─────────────────────────────────────────────────

D65 = (0.3127, 0.3290)
ACES_WHITE = (0.32168, 0.33767)

#: name -> ((Rx, Ry), (Gx, Gy), (Bx, By), white)
PRIMARIES: Dict[str, Tuple] = {
    "Rec.709":        ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), D65),
    "Rec.2020":       ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), D65),
    "P3-D65":         ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),
    "AP1":            ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), ACES_WHITE),
    "AP0":            ((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770), ACES_WHITE),
    "ARRI Wide Gamut 3": ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020), D65),
    "ARRI Wide Gamut 4": ((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308), D65),
    "S-Gamut3":       ((0.730, 0.280), (0.140, 0.855), (0.100, -0.050), D65),
    "S-Gamut3.Cine":  ((0.766, 0.275), (0.225, 0.800), (0.089, -0.087), D65),
    "V-Gamut":        ((0.730, 0.280), (0.165, 0.840), (0.100, -0.030), D65),
    "Cinema Gamut":   ((0.740, 0.270), (0.170, 1.140), (0.080, -0.100), D65),
    "REDWideGamutRGB": ((0.780308, 0.304253), (0.121595, 1.493994), (0.095612, -0.084589), D65),
    "DaVinci Wide Gamut": ((0.8000, 0.3130), (0.1682, 0.9877), (0.0790, -0.1155), D65),
}

_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]], dtype=np.float64)


def _xy_to_XYZ(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def rgb_to_xyz(name: str) -> np.ndarray:
    (r, g, b, w) = PRIMARIES[name]
    m = np.stack([_xy_to_XYZ(*r), _xy_to_XYZ(*g), _xy_to_XYZ(*b)], axis=1)
    s = np.linalg.solve(m, _xy_to_XYZ(*w))
    return m * s


_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                   [-0.7036, 1.6975, 0.0061],
                   [0.0030, 0.0136, 0.9834]], dtype=np.float64)

#: Chromatic adaptation each gamut uses to reach the ACES white, as the ACES
#: studio config defines them: CAT02 for the camera gamuts whose vendor or
#: ACES IDT specifies it, Bradford for the ITU / DCI and remaining ones.
#: Measured against OCIO 2.5's studio-config-v4.0.0: every matrix matches to
#: 6e-8 with this table, and to only 4e-3 with Bradford everywhere.
_CAT_FOR_GAMUT = {
    "ARRI Wide Gamut 3": _CAT02, "ARRI Wide Gamut 4": _CAT02,
    "S-Gamut3": _CAT02, "S-Gamut3.Cine": _CAT02,
    "Cinema Gamut": _CAT02, "DaVinci Wide Gamut": _CAT02,
}


def _cat(src_white, dst_white, cone=_BRADFORD) -> np.ndarray:
    if np.allclose(src_white, dst_white):
        return np.eye(3)
    ws, wd = cone @ _xy_to_XYZ(*src_white), cone @ _xy_to_XYZ(*dst_white)
    return np.linalg.inv(cone) @ np.diag(wd / ws) @ cone


def _to_aces_xyz(name: str) -> np.ndarray:
    """RGB in ``name`` -> XYZ adapted to the ACES white (the config's hub)."""
    cone = _CAT_FOR_GAMUT.get(name, _BRADFORD)
    return _cat(PRIMARIES[name][3], ACES_WHITE, cone) @ rgb_to_xyz(name)


@lru_cache(maxsize=None)
def gamut_matrix(src: str, dst: str) -> np.ndarray:
    """3x3 matrix taking linear ``src`` RGB to linear ``dst`` RGB (column vectors).

    Routed through the ACES white with each gamut's own adaptation, which is
    how the ACES OCIO configs chain colorspaces; a direct D65 -> D65 path would
    differ from OCIO by the CAT02/Bradford round trip.
    """
    if src == dst:
        return np.eye(3, dtype=np.float64)
    return np.linalg.inv(_to_aces_xyz(dst)) @ _to_aces_xyz(src)


# ── Transfer functions ──────────────────────────────────────────────────────

PQ_M1, PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
PQ_C1, PQ_C2, PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
HLG_A, HLG_B = 0.17883277, 0.28466892
HLG_C = 0.5 - HLG_A * math.log(4 * HLG_A)
#: BT.2408: reference white sits at 75% HLG signal. Scene E for that signal.
HLG_REF_E = (math.exp((0.75 - HLG_C) / HLG_A) + HLG_B) / 12.0


def _pq_encode(x, ref_nits):
    L = np.clip(np.asarray(x, np.float64) * ref_nits / 10000.0, 0.0, 1.0)
    Lm = np.power(L, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * Lm) / (1.0 + PQ_C3 * Lm), PQ_M2).astype(np.float32)


def _pq_decode(v, ref_nits):
    V = np.power(np.clip(np.asarray(v, np.float64), 0.0, 1.0), 1.0 / PQ_M2)
    L = np.power(np.maximum(V - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * V), 1.0 / PQ_M1)
    return (L * 10000.0 / ref_nits).astype(np.float32)


#: BT.2100 HLG reference display (BT.2408 PQ <-> HLG conversion, OCIO's
#: "Rec.2100-HLG" 1000-nit display). System gamma 1.2 at 1000 nits.
HLG_DISPLAY_NITS = 1000.0
HLG_GAMMA = 1.2
_LUMA_2020 = np.array([0.2627, 0.6780, 0.0593], np.float64)


def _hlg_encode(x, ref_nits=None):
    """Display-linear Rec.2020 (1.0 = ref white) -> HLG, BT.2100 transcode.

    3.5.0: HLG now encodes the same picture PQ does. Linear light is placed
    at ``ref_nits`` (1.0 = 203 nits), the inverse HLG OOTF for the 1000-nit
    reference display turns it into scene light, then the OETF. Reference
    white lands on 75 %, 18 % grey on 43.6 %, 1000 nits on 100 %, matching
    OpenColorIO's DISPLAY - CIE-XYZ-D65_to_REC.2100-HLG-1000nit.
    """
    ref = float(ref_nits or DEFAULT_REFERENCE_WHITE_NITS)
    fd = np.maximum(np.asarray(x, np.float64), 0.0) * (ref / HLG_DISPLAY_NITS)
    yd = fd[..., :3] @ _LUMA_2020
    ys = np.power(np.maximum(yd, 1e-9), 1.0 / HLG_GAMMA)[..., None]
    E = np.clip(fd * np.power(ys, 1.0 - HLG_GAMMA), 0.0, 1.0)
    out = np.where(E <= 1 / 12, np.sqrt(3 * E), HLG_A * np.log(np.maximum(12 * E - HLG_B, 1e-12)) + HLG_C)
    return out.astype(np.float32)


def _hlg_decode(v, ref_nits=None):
    ref = float(ref_nits or DEFAULT_REFERENCE_WHITE_NITS)
    V = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    E = np.where(V <= 0.5, V * V / 3.0, (np.exp((V - HLG_C) / HLG_A) + HLG_B) / 12.0)
    ys = (E[..., :3] @ _LUMA_2020)[..., None]
    fd = E * np.power(np.maximum(ys, 1e-9), HLG_GAMMA - 1.0)
    return (fd * (HLG_DISPLAY_NITS / ref)).astype(np.float32)


def _gamma_encode(g):
    return lambda x, _r=None: (np.sign(x) * np.power(np.abs(x), 1.0 / g)).astype(np.float32)


def _gamma_decode(g):
    return lambda x, _r=None: (np.sign(x) * np.power(np.abs(x), g)).astype(np.float32)


def _wrap(fn):
    return lambda x, _r=None: np.asarray(fn(np.asarray(x, np.float32)), np.float32)


_IDENT = (lambda x, _r=None: np.asarray(x, np.float32))


def _logc3_encode(x, _r=None):
    x = np.asarray(x, np.float64)
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    return np.where(x > cut, c * np.log10(np.maximum(a * x + b, 1e-12)) + d, e * x + f).astype(np.float32)


def _logc3_decode(v, _r=None):
    v = np.asarray(v, np.float64)
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    return np.where(v > e * cut + f, (np.power(10.0, (v - d) / c) - b) / a, (v - f) / e).astype(np.float32)


def _slog3_encode(x, _r=None):
    # Sony's published S-Log3; the toe is linear for all x below the cut, so
    # negatives (out-of-gamut colours) are not clamped as color.transfer does.
    x = np.asarray(x, np.float64)
    hi = (420.0 + np.log10(np.maximum(x + 0.01, 1e-12) / 0.19) * 261.5) / 1023.0
    lo = (x * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
    return np.where(x >= 0.01125, hi, lo).astype(np.float32)


def _slog3_decode(v, _r=None):
    v = np.asarray(v, np.float64)
    hi = np.power(10.0, (v * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
    lo = (v * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)
    return np.where(v >= 171.2102946929 / 1023.0, hi, lo).astype(np.float32)


# ── Encodings ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Encoding:
    name: str
    gamut: str
    decode: Callable        # encoded -> linear (in `gamut`)
    encode: Callable        # linear (in `gamut`) -> encoded
    ocio: Optional[str]     # studio-config colorspace, None = analytic only
    hdr: bool = False       # takes reference_white_nits
    display_referred: bool = False


def _E(name, gamut, dec, enc, ocio=None, hdr=False, display=False):
    return Encoding(name, gamut, dec, enc, ocio, hdr, display)


ENCODINGS: Dict[str, Encoding] = {e.name: e for e in [
    _E("Linear Rec.709 (sRGB)", "Rec.709", _IDENT, _IDENT, "Linear Rec.709 (sRGB)"),
    _E("Linear Rec.2020", "Rec.2020", _IDENT, _IDENT, "Linear Rec.2020"),
    _E("Linear P3-D65", "P3-D65", _IDENT, _IDENT, "Linear P3-D65"),
    _E("ACEScg", "AP1", _IDENT, _IDENT, "ACEScg"),
    _E("ACES2065-1", "AP0", _IDENT, _IDENT, "ACES2065-1"),
    _E("ACEScct", "AP1", _wrap(_tf.acescct_to_linear), _wrap(_tf.linear_to_acescct), "ACEScct"),
    _E("sRGB", "Rec.709", _wrap(_tf.srgb_to_linear), _wrap(_tf.linear_to_srgb),
       "sRGB Encoded Rec.709 (sRGB)", display=True),
    _E("Rec.709 (BT.1886)", "Rec.709", _gamma_decode(2.4), _gamma_encode(2.4),
       "Gamma 2.4 Encoded Rec.709", display=True),
    # Analytic on purpose: ITU-R BT.709 specifies a 4.500 toe slope, OCIO's
    # "Camera Rec.709" uses the continuous 4.514 form (0.3% off in the toe)
    # and clamps negatives.
    _E("Rec.709 (camera OETF)", "Rec.709", _wrap(_tf.rec709_to_linear), _wrap(_tf.linear_to_rec709)),
    _E("Rec.2020 (BT.2020 OETF)", "Rec.2020", _wrap(_tf.rec2020_to_linear), _wrap(_tf.linear_to_rec2020)),
    _E("P3-D65 (Gamma 2.6)", "P3-D65", _gamma_decode(2.6), _gamma_encode(2.6), display=True),
    _E("PQ (ST.2084)", "Rec.2020", _pq_decode, _pq_encode, hdr=True, display=True),
    _E("HLG (BT.2100)", "Rec.2020", _hlg_decode, _hlg_encode, hdr=True, display=True),
    _E("ARRI LogC4", "ARRI Wide Gamut 4", _wrap(_tf.logc4_to_linear), _wrap(_tf.linear_to_logc4), "ARRI LogC4"),
    _E("ARRI LogC3", "ARRI Wide Gamut 3", _logc3_decode, _logc3_encode, "ARRI LogC3 (EI800)"),
    _E("Sony S-Log3", "S-Gamut3.Cine", _slog3_decode, _slog3_encode,
       "S-Log3 S-Gamut3.Cine"),
    _E("Sony S-Log3 S-Gamut3", "S-Gamut3", _slog3_decode, _slog3_encode,
       "S-Log3 S-Gamut3"),
    _E("Panasonic V-Log", "V-Gamut", _wrap(_tf.vlog_to_linear), _wrap(_tf.linear_to_vlog), "V-Log V-Gamut"),
    _E("Canon Log 3", "Cinema Gamut", _wrap(_tf.canonlog3_to_linear), _wrap(_tf.linear_to_canonlog3),
       "CanonLog3 CinemaGamut D55"),
    _E("RED Log3G10", "REDWideGamutRGB", _wrap(_tf.log3g10_to_linear), _wrap(_tf.linear_to_log3g10),
       "Log3G10 REDWideGamutRGB"),
    _E("DaVinci Intermediate", "DaVinci Wide Gamut", _wrap(_tf.davinci_intermediate_to_linear),
       _wrap(_tf.linear_to_davinci_intermediate), "DaVinci Intermediate WideGamut"),
]}

#: What an IMAGE tensor can be in, i.e. the working space of the graph.
WORKING_SPACES = ["Linear Rec.709 (sRGB)", "ACEScg", "Linear Rec.2020", "Linear P3-D65", "ACES2065-1"]

#: Names the pre-3.5 Write node used, mapped onto the table above. Kept as
#: aliases so saved graphs keep their meaning.
LEGACY_ALIASES = {
    "Rec.709": "Rec.709 (camera OETF)",
    "Rec.2020": "Rec.2020 (BT.2020 OETF)",
    "PQ (HDR10 / ST.2084)": "PQ (ST.2084)",
    "HLG (Hybrid Log-Gamma)": "HLG (BT.2100)",
}

DEFAULT_REFERENCE_WHITE_NITS = 203.0


def resolve(name: str) -> Encoding:
    name = LEGACY_ALIASES.get(name, name)
    if name not in ENCODINGS:
        raise ValueError(f"Unknown colour encoding {name!r}. Known: {', '.join(ENCODINGS)}")
    return ENCODINGS[name]


# ── OCIO ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def ocio_config(path: str = ""):
    """The config to use: explicit path, else $OCIO, else OCIO's built-in
    ACES studio config. None when OpenColorIO is not installed."""
    if not HAS_OCIO:
        return None
    import os
    p = (path or "").strip().strip('"').strip("'")
    if p:
        if p.startswith("ocio://"):
            return _OCIO.Config.CreateFromFile(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"OCIO config not found: {p}")
        return _OCIO.Config.CreateFromFile(p)
    env = os.environ.get("OCIO", "").strip()
    if env and (env.startswith("ocio://") or os.path.isfile(env)):
        return _OCIO.Config.CreateFromFile(env)
    return _OCIO.Config.CreateFromBuiltinConfig("studio-config-latest")


def ocio_config_name(path: str = "") -> str:
    cfg = ocio_config(path)
    return cfg.getName() if cfg is not None else "none"


@lru_cache(maxsize=128)
def _ocio_processor(src: str, dst: str, path: str):
    cfg = ocio_config(path)
    return cfg.getProcessor(src, dst).getDefaultCPUProcessor()


def ocio_apply(arr: np.ndarray, src: str, dst: str, config_path: str = "") -> np.ndarray:
    """Convert RGB (…, 3) float32 between two colorspaces of a config."""
    if not HAS_OCIO:
        raise RuntimeError("OpenColorIO is not installed (pip install opencolorio)")
    cfg = ocio_config(config_path)
    for n in (src, dst):
        if cfg.getColorSpace(n) is None:
            raise ValueError(f"OCIO colorspace {n!r} is not in config {cfg.getName()!r}")
    out = np.ascontiguousarray(arr[..., :3], dtype=np.float32).copy()
    flat = out.reshape(-1, 3)
    _ocio_processor(src, dst, config_path).applyRGB(flat)
    return flat.reshape(out.shape)


# ── Public conversions ──────────────────────────────────────────────────────

def _matrix_apply(arr: np.ndarray, m: np.ndarray) -> np.ndarray:
    if np.allclose(m, np.eye(3)):
        return arr
    return np.einsum("...c,dc->...d", arr, m.astype(np.float32)).astype(np.float32)


def _working_encoding(working: str) -> Encoding:
    if working not in WORKING_SPACES:
        raise ValueError(f"Unknown working space {working!r}. Known: {', '.join(WORKING_SPACES)}")
    return ENCODINGS[working]


def decode(arr: np.ndarray, encoding: str, working: str = "Linear Rec.709 (sRGB)",
           reference_white_nits: float = DEFAULT_REFERENCE_WHITE_NITS,
           use_ocio: bool = True) -> Tuple[np.ndarray, str]:
    """File values in ``encoding`` -> scene-linear ``working``. Returns (rgb, path)."""
    enc = resolve(encoding)
    wk = _working_encoding(working)
    rgb = np.asarray(arr[..., :3], np.float32)
    if use_ocio and HAS_OCIO and enc.ocio and wk.ocio:
        return ocio_apply(rgb, enc.ocio, wk.ocio), f"OCIO {enc.ocio} -> {wk.ocio}"
    lin = enc.decode(rgb, reference_white_nits) if enc.hdr else enc.decode(rgb)
    return _matrix_apply(lin, gamut_matrix(enc.gamut, wk.gamut)), \
        f"analytic {enc.name} -> {wk.name}"


def encode(arr: np.ndarray, encoding: str, working: str = "Linear Rec.709 (sRGB)",
           reference_white_nits: float = DEFAULT_REFERENCE_WHITE_NITS,
           use_ocio: bool = True) -> Tuple[np.ndarray, str]:
    """Scene-linear ``working`` -> file values in ``encoding``. Returns (rgb, path)."""
    enc = resolve(encoding)
    wk = _working_encoding(working)
    rgb = np.asarray(arr[..., :3], np.float32)
    if use_ocio and HAS_OCIO and enc.ocio and wk.ocio:
        return ocio_apply(rgb, wk.ocio, enc.ocio), f"OCIO {wk.ocio} -> {enc.ocio}"
    lin = _matrix_apply(rgb, gamut_matrix(wk.gamut, enc.gamut))
    out = enc.encode(lin, reference_white_nits) if enc.hdr else enc.encode(lin)
    return np.asarray(out, np.float32), f"analytic {wk.name} -> {enc.name}"


# ── File metadata ───────────────────────────────────────────────────────────

def chromaticities(gamut: str) -> Tuple[float, ...]:
    """(Rx, Ry, Gx, Gy, Bx, By, Wx, Wy) for the EXR `chromaticities` attribute."""
    r, g, b, w = PRIMARIES[gamut]
    return (*r, *g, *b, *w)


def gamut_from_chromaticities(values, tol: float = 2e-3) -> Optional[str]:
    """Name the gamut an EXR `chromaticities` attribute describes, if known."""
    try:
        v = [float(x) for x in values]
    except Exception:  # noqa: BLE001
        return None
    if len(v) != 8:
        return None
    for name in ("Rec.709", "AP0", "AP1", "Rec.2020", "P3-D65"):
        if all(abs(a - b) <= tol for a, b in zip(v, chromaticities(name))):
            return name
    return None


#: Gamut -> the linear encoding name carrying it (for Auto reads).
LINEAR_FOR_GAMUT = {"Rec.709": "Linear Rec.709 (sRGB)", "Rec.2020": "Linear Rec.2020",
                    "P3-D65": "Linear P3-D65", "AP1": "ACEScg", "AP0": "ACES2065-1"}


__all__ = [
    "ENCODINGS", "WORKING_SPACES", "LEGACY_ALIASES", "PRIMARIES", "HAS_OCIO",
    "DEFAULT_REFERENCE_WHITE_NITS", "Encoding", "resolve", "decode", "encode",
    "gamut_matrix", "ocio_config", "ocio_config_name", "ocio_apply",
    "chromaticities", "gamut_from_chromaticities", "LINEAR_FOR_GAMUT",
]
