"""Color space conversion matrices — all 3×3 matrix constants in one place.

Every matrix transforms from the named color space to ACEScg (AP1).
Derivations documented in the original color_utils.py at each constant.
"""
from __future__ import annotations

import numpy as np

# sRGB (Rec.709) → ACEScg (AP1)
SRGB_TO_ACESCG = np.array(
    [[0.613097, 0.339523, 0.047379],
     [0.070194, 0.916354, 0.013452],
     [0.020616, 0.109570, 0.869815]],
    dtype=np.float32,
)

# ACEScg (AP1) → sRGB (Rec.709)
ACESCG_TO_SRGB = np.array(
    [[1.7050509, -0.6217921, -0.0832588],
     [-0.1302564, 1.1408047, -0.0105483],
     [-0.0240033, -0.1289690, 1.1529723]],
    dtype=np.float32,
)

# ACES AP0 (2065-1) → AP1 (ACEScg)
ACES_AP0_TO_AP1 = np.array(
    [[1.4514393161, -0.2365107469, -0.2149285693],
     [-0.0765537734, 1.1762296998, -0.0996759264],
     [0.0083161484, -0.0060324498, 0.9977163014]],
    dtype=np.float32,
)

# ACEScg (AP1) → Rec.2020
ACESCG_TO_REC2020 = np.array(
    [[1.0258246, -0.0200540, -0.0057706],
     [-0.0023054, 1.0045847, -0.0022793],
     [-0.0050569, -0.0252857, 1.0303426]],
    dtype=np.float32,
)

# ACEScg (AP1) → P3-D65
ACESCG_TO_P3D65 = np.array(
    [[1.3792141, -0.3088546, -0.0703595],
     [-0.0693257, 1.0823507, -0.0130250],
     [-0.0021522, -0.0454616, 1.0476138]],
    dtype=np.float32,
)

# ARRI Wide Gamut 3 (LogC3) → ACEScg (AP1)
AWG3_TO_ACESCG = np.array(
    [[0.966739, 0.112970, -0.079709],
     [0.048587, 1.182211, -0.230797],
     [0.007259, -0.062251, 1.054992]],
    dtype=np.float32,
)

# ARRI Wide Gamut 4 (LogC4) → ACEScg (AP1)
AWG4_TO_ACESCG = np.array(
    [[1.090227, -0.030918, -0.059309],
     [-0.055780, 1.171290, -0.115510],
     [0.005438, -0.001619, 0.996182]],
    dtype=np.float32,
)

# Sony S-Gamut3.Cine → ACEScg (AP1)
SGAMUT3_CINE_TO_ACESCG = np.array(
    [[0.934762, 0.140984, -0.075745],
     [-0.049980, 1.258786, -0.208806],
     [-0.024671, -0.026186, 1.050856]],
    dtype=np.float32,
)

# Panasonic V-Gamut → ACEScg (AP1)
VGAMUT_TO_ACESCG = np.array(
    [[1.048663, 0.009553, -0.058216],
     [-0.029392, 1.145806, -0.116414],
     [-0.003317, -0.005608, 1.008925]],
    dtype=np.float32,
)

# Canon Cinema Gamut → ACEScg (AP1)
CINEMA_GAMUT_TO_ACESCG = np.array(
    [[1.109024, -0.001731, -0.107293],
     [-0.052497, 1.309053, -0.256556],
     [-0.003326, -0.217994, 1.221321]],
    dtype=np.float32,
)

# RED Wide Gamut RGB → ACEScg (AP1)
REDWIDEGAMUT_TO_ACESCG = np.array(
    [[1.150452, -0.068566, -0.081886],
     [-0.025698, 1.303876, -0.278178],
     [-0.066789, -0.318915, 1.385704]],
    dtype=np.float32,
)

# DaVinci Wide Gamut → ACEScg (AP1)
DAVINCI_WIDE_TO_ACESCG = np.array(
    [[1.100915, 0.004904, -0.105819],
     [-0.023181, 1.304576, -0.281395],
     [-0.085087, -0.127764, 1.212851]],
    dtype=np.float32,
)

# Hunt-Pointer-Estevez transform (RGB → LMS-like, used in JMh)
HPE_MATRIX = np.array(
    [[0.38971, 0.68898, -0.07868],
     [-0.22981, 1.18340, 0.04641],
     [0.00000, 0.00000, 1.00000]],
    dtype=np.float32,
)

HPE_MATRIX_INV = np.linalg.inv(HPE_MATRIX)


def apply_matrix_transform(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.dot(img, matrix.T)


def linear_srgb_to_acescg(img: np.ndarray) -> np.ndarray:
    """Convert linear sRGB (Rec.709) to ACEScg (AP1)."""
    return np.dot(img, SRGB_TO_ACESCG.T)


def acescg_to_linear_srgb(img: np.ndarray) -> np.ndarray:
    """Convert ACEScg (AP1) to linear sRGB (Rec.709)."""
    return np.dot(img, ACESCG_TO_SRGB.T)


# Rec.709 (D65) ↔ BT.2020 (D65).  Numpy twins of M_REC709_TO_BT2020 and
# M_BT2020_TO_REC709 in color/ops.py; the values are identical on purpose, and
# tests/test_color_matrices.py pins that they stay that way.  The writer needs
# the numpy form because it works on float32 arrays, not tensors.
SRGB_TO_REC2020 = np.array(
    [[0.6274040, 0.3292820, 0.0433136],
     [0.0690970, 0.9195400, 0.0113612],
     [0.0163916, 0.0880132, 0.8955950]],
    dtype=np.float32,
)

REC2020_TO_SRGB = np.array(
    [[1.6604910, -0.5876411, -0.0728499],
     [-0.1245505, 1.1328999, -0.0083494],
     [-0.0181508, -0.1005789, 1.1187297]],
    dtype=np.float32,
)


def linear_srgb_to_rec2020(img: np.ndarray) -> np.ndarray:
    """Convert linear Rec.709/sRGB primaries to linear BT.2020 primaries."""
    return np.dot(img, SRGB_TO_REC2020.T).astype(np.float32)


def rec2020_to_linear_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear BT.2020 primaries to linear Rec.709/sRGB primaries."""
    return np.dot(img, REC2020_TO_SRGB.T).astype(np.float32)
