"""Controls that reported success while doing nothing.

Five widgets shipped in a state where setting them changed no pixel and no log
line said so. Two are now real, one is real for two of its three options and
honest about the third, and two say plainly what they do and do not do.

A control that silently does nothing is worse than no control: it makes someone
believe a decision landed.
"""
import pathlib

import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


def _code(rel):
    """Source with comment lines stripped.

    These files describe the broken guard in prose, and a substring search
    matches the explanation as readily as the code — the same trap the Batch 0
    suite hit with a `strict=False` check that matched its own comment.
    """
    import re
    lines = []
    for line in _src(rel).split("\n"):
        stripped = re.sub(r"(?<!['\"])#.*$", "", line)
        lines.append(stripped)
    return "\n".join(lines)


# ── 1. PAG guarded on a key ComfyUI never sets ──────────────────────────────

def test_pag_reads_the_extra_option_that_actually_exists():
    """
    The patch bailed unless `extra_options["block_type"] == "middle"`. ComfyUI's
    attn1 patch sets `extra_options["block"] = ("middle", idx)` and
    `"block_index"` — there is no `block_type` key, so the `.get(..., "unknown")`
    default never matched and the patch returned q, k, v unmodified on every
    single call while still logging "PAG applied with scale ...".
    """
    src = _code("sampler_utils.py")
    body = src[src.index("def pag_attention_patch"):]
    body = body[:body.index("\n        model_pag.set_model_attn1_patch")]
    assert 'extra_options.get("block_type"' not in body, \
        "PAG is back to guarding on a key ComfyUI does not set"
    assert 'extra_options.get("block")' in body


def test_pag_scale_is_a_strength_not_a_gate():
    """`pag_scale` was only used as on/off and stashed where nothing read it,
    so 0.1 and 5.0 gave bit-identical results."""
    src = _src("sampler_utils.py")
    body = src[src.index("def pag_attention_patch"):]
    body = body[:body.index("\n        model_pag.set_model_attn1_patch")]
    assert "pag_scale" in body, "the scale still never reaches the perturbation"
    assert "k_out[start:end] = q[start:end]" not in body, \
        "the perturbation is back to an unweighted replacement"


def test_pag_log_line_does_not_overclaim():
    """It used to say "PAG applied"; this is a self-attention perturbation, not
    the Ahn et al. 2024 method, which needs a third forward pass."""
    src = _src("sampler_utils.py")
    assert "not the full Ahn et al. 2024 method" in src


# ── 2. Restart sampling did not implement Xu et al. ─────────────────────────

def test_restart_runs_the_segment_back_down_to_zero():
    """
    `s_vals[idx : idx+restart_count+2]` stopped a couple of steps in, leaving
    the latent partially noised: measured on a 20-step Karras schedule,
    restart_sigma 2.0 returned a latent still at sigma 0.7913.
    """
    src = _src("nodes/generate/sampler.py")
    body = src[src.index("def _apply_restarts"):]
    body = body[:body.index("\n        return result")]
    assert "sub_sigmas = s_vals[idx:]" in body
    assert "s_vals[idx:min(idx + restart_count + 2" not in body


def test_restart_noise_variance_matches_the_paper():
    """Alg. 2 adds noise of variance sigma_max^2 - sigma_min^2. The old code
    used `randn_like(result) * r_val` — a std of r_val, no subtraction."""
    src = _src("nodes/generate/sampler.py")
    body = src[src.index("def _apply_restarts"):]
    body = body[:body.index("\n        return result")]
    assert "sigma_max ** 2 - sigma_min ** 2" in body
    assert "torch.randn_like(result) * r_val" not in body


def test_restart_does_not_pass_a_noised_latent_as_the_noise_argument():
    """
    ComfyUI applies `noise_scaling(sigma0, noise, latent_image)` internally, so
    passing the noised latent as `noise=` amplified the signal by (1 + sigma0)
    and scaled the noise by sigma0 on top.
    """
    src = _src("nodes/generate/sampler.py")
    body = src[src.index("def _apply_restarts"):]
    body = body[:body.index("\n        return result")]
    assert "noise=torch.zeros_like(noisy)" in body
    assert "latent_image=noisy" in body


# ── 3. blend_mode appeared once — in its own signature ──────────────────────

@pytest.mark.real_torch
def test_blend_modes_are_actually_different():
    from radiance.nodes.upscale.upscale import _tile_weight_map

    dev = torch.device("cpu")
    gaussian = _tile_weight_map("gaussian_feather", 64, 64, 16, dev)
    linear = _tile_weight_map("linear", 64, 64, 16, dev)
    assert gaussian.shape == linear.shape == (1, 1, 64, 64)
    assert not torch.allclose(gaussian, linear), \
        "every blend_mode still produces the same weight map"


@pytest.mark.real_torch
def test_unimplemented_blend_mode_falls_back_and_says_so(caplog):
    import logging

    from radiance.nodes.upscale import upscale as up

    up._warned_blend_modes.clear()
    with caplog.at_level(logging.INFO):
        lap = up._tile_weight_map("laplacian_pyramid", 32, 32, 8, torch.device("cpu"))
    gauss = up._tile_weight_map("gaussian_feather", 32, 32, 8, torch.device("cpu"))
    assert torch.allclose(lap, gauss)
    assert any("not " in r.getMessage() and "implemented" in r.getMessage()
               for r in caplog.records), "the fallback is silent again"
    up._warned_blend_modes.clear()


@pytest.mark.real_torch
def test_tile_weights_stay_positive_and_bounded():
    from radiance.nodes.upscale.upscale import _tile_weight_map

    for mode in ("gaussian_feather", "linear", "laplacian_pyramid"):
        w = _tile_weight_map(mode, 48, 64, 12, torch.device("cpu"))
        assert float(w.min()) > 0.0, f"{mode} produces a zero weight"
        assert float(w.max()) <= 1.0 + 1e-6


def test_blend_mode_reaches_the_weight_map():
    src = _src("nodes/upscale/upscale.py")
    body = src[src.index("def tiled_upscale("):]
    body = body[:body.index("\n    # Normalise by accumulated weights")]
    assert "_tile_weight_map(blend_mode" in body, \
        "blend_mode is back to being ignored inside tiled_upscale"


# ── 4. chromatic_adaptation changed no pixel ───────────────────────────────

@pytest.mark.real_torch
def test_adaptation_widget_warns_that_it_is_baked_in(caplog):
    """
    `_get_gpu_adaptation_matrix` is defined and called from nowhere, so
    "Bradford" and "None" returned bit-identical tensors. It still does — the
    adaptation lives in the precomputed matrices — but it no longer does so
    silently.
    """
    import logging

    import radiance.hdr.color as hc

    hc._warned_adaptation_methods.clear()
    with caplog.at_level(logging.WARNING):
        hc._warn_adaptation_is_baked_in("Von Kries")
    assert any("no effect" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hc._warn_adaptation_is_baked_in("Von Kries")
    assert not caplog.records, "the warning repeats per call"
    hc._warned_adaptation_methods.clear()


def test_adaptation_tooltip_tells_the_truth():
    src = _src("hdr/color.py")
    block = src[src.index('"chromatic_adaptation": ('):]
    block = block[:block.index("),")]
    assert "does not currently alter" in block or "baked into" in block


# ── 5. The RELOAD button never rendered ─────────────────────────────────────

def test_reload_is_a_real_widget_not_a_hidden_input():
    """
    ComfyUI only populates a hidden key when its VALUE is one of the magic
    strings (PROMPT, UNIQUE_ID, EXTRA_PNGINFO, ...), so a widget spec in
    "hidden" is dropped: `reload` never reached read() or IS_CHANGED. The
    frontend builds widgets from required/optional only, so radiance_io.js
    could not find it and bailed before adding the button.
    """
    from radiance.nodes.io.write import RadianceRead

    spec = RadianceRead.INPUT_TYPES()
    assert "reload" in (spec.get("optional") or {}), \
        "reload is not a widget, so the RELOAD button cannot render"
    assert "reload" not in (spec.get("hidden") or {})


def test_reload_still_reaches_is_changed_and_read():
    import inspect

    from radiance.nodes.io.write import RadianceRead

    assert "reload" in inspect.signature(RadianceRead.IS_CHANGED).parameters
    assert "reload" in inspect.signature(RadianceRead.read).parameters
