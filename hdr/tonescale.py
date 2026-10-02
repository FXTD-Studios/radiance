"""ACES 2.0 tone scale reference points, shared by every tone scale in Radiance.

Radiance shipped two ACES 2.0 tone scales that disagreed with each other and
with the standard:

  * ``RadianceACES2Tonescale`` implements the real Daniele Evo curve but pinned
    18% grey to a fixed *fraction* of peak (10%), so grey scaled with the
    display: 10 nits at SDR, 100 at 1000, 400 at 4000. On a 4000-nit master
    that is nearly 24x the reference midtone.
  * ``RadianceACES2OutputTransform`` used a log-contrast + tanh approximation
    that held grey at ~18 nits on every peak — the right *shape*, since ACES
    2.0 does keep the midtone nearly still, but ~0.85 stop bright at SDR.

Neither needed a judgement call in the end: ACES 2.0 publishes the answer.
The Output Transform maps an ACES value of 0.18 to these display luminances,
which rise gently with peak rather than tracking it:

    peak (nits)   18% grey out (nits)
        100            10.000
        500            13.193
       1000            14.512
       2000            15.747
       4000            16.824

(The same table gives ACES 1.0 → 45.757 nits at a 100-nit peak, i.e. the
familiar ~48-nit diffuse white, which is how you can tell the cells are
luminances and not normalised values.)

Reference: ACES documentation, Output Transforms → Tone Mapping.
https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/
"""
from __future__ import annotations

import math
from typing import Tuple

# (peak luminance in nits, display luminance of an 0.18 ACES input, in nits)
ACES2_MIDGREY_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (100.0, 10.000),
    (500.0, 13.193),
    (1000.0, 14.512),
    (2000.0, 15.747),
    (4000.0, 16.824),
)

SCENE_MIDGREY = 0.18


def aces2_midgrey_nits(peak_nits: float) -> float:
    """Display luminance ACES 2.0 assigns to 18% scene grey at *peak_nits*.

    Exact at the five published anchors; log-log linear between them, and
    continued with the end slope outside the published range so a 250-nit or
    10000-nit target still gets a sensible, monotonic answer rather than a
    clamp.

    Log-log because the relationship is close to a power law — grey rises
    roughly as peak^0.14 — so interpolating there is smooth and stays
    monotonic, where linear interpolation would kink at every anchor.
    """
    peak = max(float(peak_nits), 1e-6)

    xs = [math.log(p) for p, _ in ACES2_MIDGREY_ANCHORS]
    ys = [math.log(g) for _, g in ACES2_MIDGREY_ANCHORS]
    x = math.log(peak)

    if x <= xs[0]:
        i, j = 0, 1
    elif x >= xs[-1]:
        i, j = len(xs) - 2, len(xs) - 1
    else:
        i = max(k for k in range(len(xs) - 1) if xs[k] <= x)
        j = i + 1

    t = (x - xs[i]) / (xs[j] - xs[i])
    return math.exp(ys[i] + t * (ys[j] - ys[i]))


def aces2_midgrey_fraction(peak_nits: float) -> float:
    """`aces2_midgrey_nits` expressed as a fraction of peak.

    This is the form the Daniele Evo parameterisation wants: it solves for the
    curve constant K from "18% scene grey lands at this fraction of peak".
    At 100 nits it is 0.10 — the value that was previously hard-coded for
    every peak, which is why SDR looked right and HDR did not.
    """
    peak = max(float(peak_nits), 1e-6)
    return aces2_midgrey_nits(peak) / peak


# ─────────────────────────────────────────────────────────────────────────────
# ACES 2.0 reference tone scale (the Output Transform's Tonescale_fwd)
# ─────────────────────────────────────────────────────────────────────────────
#
# Constants and formulas from the ACES 2.0 Output Transform (aces-core
# Lib.Academy.OutputTransform, tone scale parameters). Verified against the
# OCIO studio-config-v4.0.0_aces-v2.0 Output Transforms to 1e-3 nits for
# 0.18 / 1.0 / 10.0 at 500, 1000, 2000 and 4000 nits
# (tests/test_p0_color_fixes.py).

ACES2_TS_N_R = 100.0     # normalised white, nits
ACES2_TS_G = 1.15        # contrast exponent
ACES2_TS_C = 0.18        # scene mid grey
ACES2_TS_C_D = 10.013    # display mid grey at 100 nits
ACES2_TS_W_G = 0.14      # mid-grey rise per stop of peak
ACES2_TS_T_1 = 0.04      # shadow toe
ACES2_TS_R_HIT_MIN = 128.0
ACES2_TS_R_HIT_MAX = 896.0


class ACES2TonescaleParams:
    """Precomputed ACES 2.0 tone scale constants for one peak luminance."""

    def __init__(self, peak_nits: float, g: float = ACES2_TS_G, t_1: float = ACES2_TS_T_1):
        n = max(float(peak_nits), 1.0)
        n_r = ACES2_TS_N_R
        r_hit = ACES2_TS_R_HIT_MIN + (ACES2_TS_R_HIT_MAX - ACES2_TS_R_HIT_MIN) * (
            math.log(n / n_r) / math.log(10000.0 / 100.0))
        m_0 = n / n_r
        m_1 = 0.5 * (m_0 + math.sqrt(m_0 * (m_0 + 4.0 * t_1)))
        u = ((r_hit / m_1) / ((r_hit / m_1) + 1.0)) ** g
        m = m_1 / u
        w_i = math.log(n / 100.0) / math.log(2.0)
        c_t = ACES2_TS_C_D / n_r * (1.0 + w_i * ACES2_TS_W_G)
        g_ip = 0.5 * (c_t + math.sqrt(c_t * (c_t + 4.0 * t_1)))
        g_ipp2 = -(m_1 * (g_ip / m) ** (1.0 / g)) / ((g_ip / m) ** (1.0 / g) - 1.0)
        w_2 = ACES2_TS_C / g_ipp2
        s_2 = w_2 * m_1
        u_2 = ((r_hit / m_1) / ((r_hit / m_1) + w_2)) ** g
        self.peak_nits = n
        self.n = n / n_r          # peak in 100-nit units
        self.g = g
        self.t_1 = t_1
        self.s_2 = s_2
        self.m_2 = m_1 / u_2

    def forward(self, Y):
        """Scene luminance (1.0 = 100-nit diffuse white) -> display nits / 100.

        Accepts floats, numpy arrays and torch tensors. Negative input maps to
        0, as in the reference.
        """
        if isinstance(Y, (int, float)):
            Yp = max(float(Y), 0.0)
        else:
            Yp = Y.clip(0.0, None) if not hasattr(Y, "clamp") else Y.clamp(min=0.0)
        f = self.m_2 * (Yp / (Yp + self.s_2)) ** self.g
        return f * f / (f + self.t_1)


def aces2_tonescale_nits(Y: float, peak_nits: float) -> float:
    """Display luminance in nits for scene luminance ``Y`` at ``peak_nits``."""
    return float(ACES2TonescaleParams(peak_nits).forward(float(Y))) * ACES2_TS_N_R
