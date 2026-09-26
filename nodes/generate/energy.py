"""
◎ Radiance — Energy-Prioritized Sampling (EPS) producer

Provides:
  • RadianceEnergyMask — attach an energy mask to CONDITIONING.

RadianceSamplerPro has carried the *reader* for Energy-Prioritized Sampling
since v3.1: it scans the positive CONDITIONING for a ``radiance_energy_mask``
key and, when it finds one, registers a CFG patch that scales the
(cond − uncond) guidance vector up inside the masked region. Nothing ever
wrote that key, so the feature was unreachable from a graph (issue #40).
This node is the missing producer.

What EPS is for: in an HDR pipeline the specular highlights and practical
lights carry most of the perceptual "energy" of a frame, but CFG treats every
pixel identically. Painting a mask over those regions and pushing `priority`
up gives the sampler more guidance where detail matters and leaves the rest
of the frame at the base CFG, instead of raising CFG globally and burning the
midtones.

Wiring:

    CLIP Text Encode ──► conditioning ──┐
                                        ├─► ◎ Energy Mask ──► positive
    (any MASK source) ──► mask ─────────┘        │
                                                 └──► mask (pass-through,
                                                      for preview)

The mask is stored at whatever resolution it arrives in; the sampler resizes
it to the latent grid at sampling time, so it works for any output size and
for video models without the graph knowing the latent shape up front.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger("radiance.energy")

ENERGY_MASK_KEY = "radiance_energy_mask"
ENERGY_PRIORITY_KEY = "radiance_energy_priority"

#: Every energy layer applied so far, oldest first, as ``(mask, priority)``.
#: The two singular keys above are still written — they carry the *last*
#: layer — so anything that learned to read them during 3.3.0 beta keeps
#: working. The sampler reads this key.
ENERGY_LAYERS_KEY = "radiance_energy_layers"


def attach_energy_mask(
    conditioning: list,
    mask: torch.Tensor,
    priority: float,
) -> list:
    """Return a copy of *conditioning* with one more energy layer attached.

    Layers accumulate. Chaining two of these nodes used to be a silent data
    loss: the second call overwrote ``ENERGY_MASK_KEY`` on every entry, so the
    downstream node won and the upstream mask vanished without a log line.
    (The node's own docstring claimed the *upstream* one won, which was true
    only for the other chaining shape — two branches joined by Conditioning
    Combine, where the sampler stopped at the first entry carrying the key.
    Both shapes now stack.)

    Every entry gets the layer, not just the first: ComfyUI nodes downstream
    are free to reorder or drop entries. The same layer tuple object is shared
    by every entry, which is how the sampler tells "one node fanned across
    four entries" (one layer) from "two nodes combined" (two layers).

    Input dicts are copied, never mutated — a conditioning list is frequently
    fanned out to several branches of a graph and mutating it in place would
    leak EPS into all of them.
    """
    layer = (mask, float(priority))

    out = []
    for entry in conditioning:
        tensor, meta = entry[0], entry[1]
        new_meta = dict(meta)

        existing = new_meta.get(ENERGY_LAYERS_KEY)
        prior = tuple(existing) if isinstance(existing, (list, tuple)) else ()
        new_meta[ENERGY_LAYERS_KEY] = prior + (layer,)

        new_meta[ENERGY_MASK_KEY] = mask
        new_meta[ENERGY_PRIORITY_KEY] = float(priority)
        out.append((tensor, new_meta))
    return out


def normalize_energy_mask(
    mask: torch.Tensor,
    invert: bool = False,
    normalize: bool = False,
    gain: float = 1.0,
) -> torch.Tensor:
    """Condition a raw MASK into the [0, 1] weight field EPS expects.

    ComfyUI's MASK convention is (B, H, W) float in [0, 1], but masks reach
    this node from luminance extractions and depth passes too, which are not
    guaranteed to be normalised. Values outside [0, 1] would turn the CFG
    modifier negative (inverting guidance) or unbounded, so clamp last.
    """
    m = mask.detach().float()

    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim > 3:
        # e.g. a (B, 1, H, W) latent-shaped mask, or an IMAGE-shaped (B,H,W,C)
        # one that a user routed in through a converter.
        m = m.reshape(-1, m.shape[-2], m.shape[-1])

    if normalize:
        lo = m.amin()
        span = m.amax() - lo
        if float(span) > 1e-6:
            m = (m - lo) / span
        else:
            m = torch.zeros_like(m)

    if invert:
        m = 1.0 - m.clamp(0.0, 1.0)

    if gain != 1.0:
        m = m * gain

    return m.clamp(0.0, 1.0)


class RadianceEnergyMask:
    """
    ◎ Radiance Energy Mask

    Marks a region of the frame as high-energy so RadianceSamplerPro spends
    more guidance there. Feed the output into the sampler's `positive` input.

    The masked region is sampled with an effective guidance scale of
    ``cfg × (1 + priority)``; unmasked regions keep the sampler's `cfg`
    exactly. `priority` therefore behaves like a local CFG bonus:

      • 0.0        — no effect (the patch is skipped entirely)
      • 0.25–0.75  — useful range for highlight emphasis
      • > 1.5      — expect the usual over-guidance artefacts, locally

    Negative priority is allowed and *suppresses* guidance in the mask, which
    is the cheapest way to keep a background soft while a subject stays sharp.

    Chaining stacks. Two of these in series — or two branches joined by
    Conditioning Combine — add their contributions, so the effective guidance
    is ``cfg × (1 + Σ priority_i · mask_i)``. Layering a broad +0.2 over the
    whole subject and a tight +0.6 on the practicals does what it looks like
    it does. The combined modifier is floored at 0 so a stack of negative
    priorities suppresses guidance completely rather than inverting it.
    """

    CATEGORY = "FXTD STUDIOS/Radiance/◎ Generate"
    DESCRIPTION = "Attach an energy mask to conditioning for Energy-Prioritized Sampling."
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "MASK")
    RETURN_NAMES = ("conditioning", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Positive conditioning. Connect the output to RadianceSamplerPro's `positive`.",
                }),
                "mask": ("MASK", {
                    "tooltip": (
                        "High-energy region. White = boosted guidance, black = untouched. "
                        "Resized to the latent grid at sampling time, so any resolution works."
                    ),
                }),
                "priority": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Local guidance bonus inside the mask. Effective CFG there is "
                        "cfg × (1 + priority). 0 disables EPS; negative softens the region. "
                        "Chained Energy Mask nodes stack, so priorities add where masks overlap."
                    ),
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap masked and unmasked regions before applying priority.",
                }),
                "normalize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Rescale the mask so its darkest pixel is 0 and brightest is 1. "
                        "Use for luminance or depth passes that are not already in [0,1]."
                    ),
                }),
                "gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Multiplier applied to the mask before clamping to [0,1]. Hardens a soft mask.",
                }),
            },
        }

    def apply(
        self,
        conditioning: list,
        mask: torch.Tensor,
        priority: float = 0.5,
        invert: bool = False,
        normalize: bool = False,
        gain: float = 1.0,
    ) -> Tuple[list, torch.Tensor]:

        if not isinstance(conditioning, list) or not conditioning:
            raise ValueError(
                "RadianceEnergyMask: `conditioning` must be a non-empty CONDITIONING "
                "list. Connect a text encoder output."
            )
        if not isinstance(mask, torch.Tensor):
            raise ValueError(
                f"RadianceEnergyMask: `mask` must be a MASK tensor, got {type(mask).__name__}."
            )
        if mask.numel() == 0:
            raise ValueError("RadianceEnergyMask: `mask` is empty.")

        m = normalize_energy_mask(mask, invert=invert, normalize=normalize, gain=gain)

        coverage = float(m.mean())
        if priority == 0.0:
            logger.info("[Energy Mask] priority=0 — EPS attached but inert.")
        elif coverage < 1e-4:
            logger.warning(
                "[Energy Mask] the mask is effectively empty (mean=%.5f); EPS will "
                "have no visible effect. Check `invert` and the mask source.", coverage
            )
        elif coverage > 0.99:
            logger.warning(
                "[Energy Mask] the mask covers the whole frame (mean=%.3f); this is "
                "equivalent to raising cfg globally to cfg × %.2f.",
                coverage, 1.0 + float(priority),
            )
        else:
            logger.info(
                "[Energy Mask] attached: priority=%.2f, coverage=%.1f%%, mask=%s",
                float(priority), coverage * 100.0, tuple(m.shape),
            )

        out = attach_energy_mask(conditioning, m, priority)

        depth = len(out[0][1].get(ENERGY_LAYERS_KEY, ()))
        if depth > 1:
            logger.info(
                "[Energy Mask] stacking onto %d existing layer(s); the sampler will "
                "sum priority × mask across all %d.", depth - 1, depth,
            )

        return (out, m)


# ==============================================================================
# Node registration
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "RadianceEnergyMask": RadianceEnergyMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceEnergyMask": "◎ Radiance Energy Mask",
}
