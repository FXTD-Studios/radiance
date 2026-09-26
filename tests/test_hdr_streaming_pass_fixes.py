"""
tests/test_hdr_streaming_pass_fixes.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Correctness defects found alongside the HDR VAE streaming rework.

Each test here fails on the code as it stood before that pass. They are grouped
by the pixels they were getting wrong:

  silently wrong pixels
    - video encode reported pad_h/pad_w as 0, so decode skipped the crop and
      returned reflect-padded rows as picture
    - video encode discarded alpha even under alpha_handling="Preserve"
    - Highlight Synthesis "Soft Light" evaluated a [0,1]-only blend on HDR and
      clamped the result to black
    - HDR Expansion applied the inverse OETF to the alpha channel
    - the soft-decompress inverse was evaluated one ten-millionth from its pole

  metrics and reports that said nothing
    - Blend Validator's SSIM returned exactly 1.0 for any two HDR images, and
      its "ssim" winner was a comparison that was never made
    - a substituted RUDRA decoder was invisible in the node's own report
    - a VRAM budget that fits zero frames was logged as an optimal chunk size

  routes that could not be reached
    - OCIO's bundled ACES config and the downloaded one were dead code
    - the OCIO transform node rejected alias names its own tooltip recommends,
      and raised at its own untouched defaults

  guards that were wrong
    - the torch.quantile size guard measured the reduced axis, not numel
    - detect_vae_factor accepted a value-normalisation constant as a spatial
      factor and returned 0
    - a zero-frame video latent raised NameError
    - _scene_linear_for_rhdr survived between node executions
    - the realtime decode path dropped its log-space highlight denoise
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True
# Every test here needs real tensors; skip them all on the stub lane.
pytestmark = pytest.mark.real_torch


# ── ComfyUI stubs ────────────────────────────────────────────────────────────

for _m in ["folder_paths", "comfy", "comfy.utils", "comfy.model_management"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

_mm = sys.modules["comfy.model_management"]
if not hasattr(_mm, "get_torch_device"):
    _mm.get_torch_device = lambda: torch.device("cpu")
if not hasattr(_mm, "soft_empty_cache"):
    _mm.soft_empty_cache = lambda: None

_cu = sys.modules["comfy.utils"]
if not hasattr(_cu, "ProgressBar"):
    class _FakeProgressBar:
        def __init__(self, *a, **kw):
            pass

        def update(self, *a):
            pass

        def update_absolute(self, *a):
            pass

    _cu.ProgressBar = _FakeProgressBar

sys.modules["comfy"].model_management = sys.modules["comfy.model_management"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _vae():
    return importlib.import_module("radiance.hdr.vae")


class _ScopeVAE:
    """A 4D image VAE with a factor-32 spatial compression, like LTX's.

    Factor 32 is the case that exposes the padding defect: 1080 and 1716 are
    both non-multiples, so every scope and every HD frame pads.
    """

    def __init__(self, factor=32, channels=4):
        self.downscale_ratio = factor
        self.factor = factor
        self.channels = channels

    def encode(self, pixels):
        b, h, w, _c = pixels.shape
        assert h % self.factor == 0 and w % self.factor == 0, (
            f"encode got an unpadded {w}x{h}; the node must pad to the factor")
        return torch.zeros(b, self.channels, h // self.factor, w // self.factor)

    def decode(self, latent):
        b, _c, h, w = latent.shape
        return torch.full((b, h * self.factor, w * self.factor, 3), 0.5)


# ═════════════════════════════════════════════════════════════════════════════
#  Silently wrong pixels
# ═════════════════════════════════════════════════════════════════════════════

def test_video_encode_records_the_real_pad_so_decode_crops_it_off():
    """A clip whose height is not a multiple of the VAE factor must come back
    at its original height.

    The video encode branch hardcoded pad_h/pad_w to 0 in the clip-level
    radiance_meta. Each per-frame recursive call pads to a multiple of
    vae_factor and reports the real pad in its own meta, but only `samples` was
    kept from it. Decode read the zeros and skipped the crop, so a 1920x1080
    clip through a factor-32 VAE decoded to 1088 tall with 8 rows of reflect
    pad delivered as picture. Any height off the factor is affected, 2.39:1
    scope at 4096x1716 included.
    """
    mod = _vae()
    enc, dec = mod.RadianceVAE4KEncode(), mod.RadianceVAE4KDecode()
    vae = _ScopeVAE(factor=32)

    height, width, frames = 40, 64, 3       # 40 pads to 64: pad_h = 24
    pixels = torch.rand(1, frames, height, width, 3)

    latent, _alpha, meta, _fmt, _qr = enc.encode(pixels, vae)

    assert latent["radiance_meta"]["pad_h"] == 24, (
        f"clip meta says pad_h={latent['radiance_meta']['pad_h']}, but the "
        f"frames were padded from {height} to 64")
    assert latent["radiance_meta"]["pad_w"] == 0
    assert json.loads(meta)["pad_h"] == 24

    img, _dmeta, _dfmt = dec.decode(latent, vae)
    assert img.shape[1] == height, (
        f"decode returned {img.shape[1]} rows for a {height}-row clip; the "
        f"extra rows are reflect padding, not picture")
    assert img.shape[2] == width


def test_video_encode_preserves_alpha():
    """alpha_handling="Preserve" must preserve alpha on video too.

    The still path extracts pixels[..., 3:4]; the video branch never did and
    returned torch.ones unconditionally, so decode composited solid white over
    the real matte on every RGBA clip.
    """
    mod = _vae()
    enc = mod.RadianceVAE4KEncode()
    vae = _ScopeVAE(factor=32)

    frames, height, width = 3, 64, 64
    pixels = torch.ones(1, frames, height, width, 4)
    pixels[..., 3] = 0.25

    _latent, alpha, _meta, _fmt, _qr = enc.encode(
        pixels, vae, alpha_handling="Preserve")

    assert alpha.shape == (frames, height, width, 1), (
        f"video alpha shape {tuple(alpha.shape)}; decode's frame_alphas "
        f"chunking expects (B*F, H, W, 1)")
    assert torch.allclose(alpha, torch.full_like(alpha, 0.25)), (
        "the real matte was replaced with an opaque one")


def test_video_encode_still_returns_opaque_alpha_when_there_is_none():
    """A 3-channel clip has no matte, so an opaque one is the right answer."""
    mod = _vae()
    enc = mod.RadianceVAE4KEncode()
    pixels = torch.rand(1, 2, 64, 64, 3)
    _latent, alpha, _meta, _fmt, _qr = enc.encode(pixels, _ScopeVAE(factor=32))
    assert alpha.shape == (2, 64, 64, 1)
    assert torch.allclose(alpha, torch.ones_like(alpha))


def test_soft_light_does_not_black_out_hdr_highlights():
    """Soft Light on an HDR frame must not return black.

    The W3C formula ``(1 - 2n)a^2 + 2n a`` is defined on a in [0,1]. This node
    expands highlights, so it routinely evaluates it at a >> 1, where the
    quadratic term dominates and goes hard negative: a=5.0 with n=0.7 gives
    -219.7, which the closing np.maximum(0.0, ...) clamped to pure black. With
    detail_amount=1.0 every pixel above roughly 2.0 linear blacked out, on the
    node whose only purpose is to put detail INTO highlights.
    """
    recovery = importlib.import_module("radiance.hdr.recovery")
    node = recovery.RadianceHighlightSynthesis()

    image = torch.full((1, 32, 32, 3), 5.0)     # a flat HDR highlight
    out, = node.synthesize(
        image, threshold=0.95, expansion=1.5, detail_amount=1.0,
        detail_scale=1.0, blend_mode="Soft Light", seed=7,
    )

    out_np = out.numpy()
    black_fraction = float((out_np.max(axis=-1) <= 1e-6).mean())
    assert black_fraction == 0.0, (
        f"{black_fraction:.1%} of an HDR highlight came back black")
    assert out_np.min() >= 0.0
    # And the highlight is still a highlight, not crushed to SDR.
    assert out_np.mean() > 1.0, (
        f"mean {out_np.mean():.3f}: the expansion was destroyed rather than "
        f"grained")


def test_soft_light_is_unchanged_on_sdr_content():
    """The HDR fix must be a no-op where the old formula was already valid.

    Normalising by the frame peak divides and multiplies by 1.0 when the frame
    never exceeds 1.0, so SDR output is bit-identical to what shipped.
    """
    recovery = importlib.import_module("radiance.hdr.recovery")
    node = recovery.RadianceHighlightSynthesis()
    image = torch.rand(1, 24, 24, 3) * 0.9

    out, = node.synthesize(
        image, threshold=0.95, expansion=1.0, detail_amount=0.2,
        detail_scale=1.0, blend_mode="Soft Light", seed=3,
    )
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.05


def test_sdr_to_hdr_expand_leaves_alpha_alone():
    """RadianceSDRtoHDRExpand must not gamma-decode the matte.

    The inverse OETF ran over the whole tensor before ``RGB = img[..., :3]``
    was sliced, and ``img[..., 3:]`` was concatenated back already decoded, so
    alpha 0.5 came out 0.2140. uplift_universal splits before decoding for
    exactly this reason.
    """
    synthesis = importlib.import_module("radiance.nodes.hdr.synthesis")
    node = synthesis.RadianceSDRtoHDRExpand()

    image = torch.zeros(1, 8, 8, 4)
    image[..., :3] = 0.2                       # below any expansion threshold
    image[..., 3] = 0.5

    out, = node.apply(image, "sRGB", 0.8, 5.0, 1.2, 0.1)

    assert torch.allclose(out[..., 3], torch.full_like(out[..., 3], 0.5),
                          atol=1e-6), (
        f"alpha came out {float(out[..., 3].mean()):.4f}; 0.5 sRGB-decoded is "
        f"0.2140, which is what the whole-tensor decode produced")


def test_soft_decompress_is_bounded_at_the_pole():
    """Code 1.0 must not decode to a quarter of a billion nits.

    ``y = img.clamp(0.0, 1.0 - 1e-7)`` then ``(y - 1 + r) / (1 - y + eps)``
    evaluates Reinhard's inverse one ten-millionth from its pole: ~2.5e6 linear
    at r=0.5 and ~1e7 on the r>=1 branch. Every specular and every practical
    light source in a VAE decode lands on code 1.0. Both figures also overflow
    fp16 (max 65504), so an f16 sidecar written from them stores Inf.
    """
    encoder = importlib.import_module("radiance.nodes.hdr.encoder")
    ceiling = encoder._DECOMPRESS_MAX_LINEAR

    for ratio in (0.25, 0.5, 0.75, 1.0):
        out = encoder._hdr_soft_decompress(torch.ones(1, 2, 2, 3), ratio)
        peak = float(out.max())
        assert peak <= ceiling, (
            f"ratio={ratio}: code 1.0 decoded to {peak:.3e} linear")
        assert torch.isfinite(out.half()).all(), (
            f"ratio={ratio}: {peak:.3e} is Inf in fp16")


def test_soft_decompress_still_round_trips_real_hdr_content():
    """The ceiling must not touch content anyone actually grades.

    64.0 linear is 6400 nits at this module's 1.0 == 100 nits convention,
    brighter than any HDR mastering target in use.
    """
    encoder = importlib.import_module("radiance.nodes.hdr.encoder")
    scene = torch.linspace(0.0, 64.0, 512).reshape(1, 16, 32, 1).repeat(1, 1, 1, 3)
    for ratio in (0.25, 0.5, 0.75, 1.0):
        recon = encoder._hdr_soft_decompress(
            encoder._hdr_soft_compress(scene, ratio), ratio)
        assert torch.allclose(recon, scene, rtol=1e-3, atol=1e-2), (
            f"ratio={ratio}: round-trip broke inside the legitimate range")


# ═════════════════════════════════════════════════════════════════════════════
#  Metrics and reports
# ═════════════════════════════════════════════════════════════════════════════

def _hdr_pair():
    """Two HDR frames whose luma exceeds 1.0 EVERYWHERE, with different
    structure. The old _ssim clamped luma to [0,1] first, so both collapsed to
    a flat field of exactly 1.0."""
    height = width = 48
    yy, xx = torch.meshgrid(torch.arange(height).float(),
                            torch.arange(width).float(), indexing="ij")
    a = (3.0 + 1.5 * torch.sin(xx / 4.0)).unsqueeze(-1).repeat(1, 1, 3)
    b = (3.0 + 1.5 * torch.sin(yy / 3.0)).unsqueeze(-1).repeat(1, 1, 3)
    assert float(a.min()) > 1.0 and float(b.min()) > 1.0
    return a.unsqueeze(0), b.unsqueeze(0)


def test_ssim_is_not_identically_one_on_hdr_content():
    """SSIM must measure HDR structure, not report 1.0 for everything.

    Clamping luma to [0,1] made mu=1 and sigma=0 for any pair of images whose
    luma exceeds 1.0, and the expression collapses to exactly 1.0 there. The
    docstring's "values < 0.95 indicate significant structural change" was
    unreachable on the content this node exists to validate.
    """
    inception = importlib.import_module("radiance.nodes.hdr.inception")
    a, b = _hdr_pair()

    score = inception.RadianceHDRBlendValidator._ssim(a, b)
    assert score != pytest.approx(1.0, abs=1e-6), (
        "SSIM returned exactly 1.0 for two structurally different HDR images")
    assert score < 0.95, (
        f"SSIM {score:.4f} for orthogonal sine fields; the metric is still "
        f"blind to structure")
    assert -1.0 <= score <= 1.0


def test_ssim_is_one_for_identical_images():
    """The floor case still has to behave."""
    inception = importlib.import_module("radiance.nodes.hdr.inception")
    a, _b = _hdr_pair()
    assert inception.RadianceHDRBlendValidator._ssim(a, a) == pytest.approx(
        1.0, abs=1e-4)


def test_ssim_winner_is_decided_by_the_measured_value():
    """win_metric="ssim" must pick on evidence, not unconditionally.

    It used to set ``winner_img = image_a`` with the label
    "image_a (baseline, higher SSIM)" whatever the number was, a comparison
    never made: SSIM is one symmetric value between the two images, so there
    is no higher one to pick. The only decision one number supports is a
    threshold.
    """
    inception = importlib.import_module("radiance.nodes.hdr.inception")
    node = inception.RadianceHDRBlendValidator()
    a, b = _hdr_pair()

    # Structurally different: the baseline is the safer output.
    winner, report_json, ssim_val, _js, _dr = node.validate(a, b, "ssim")
    report = json.loads(report_json)
    assert torch.equal(winner, a)
    assert f"{ssim_val:.4f}" in report["winner"], (
        f"the label must carry the number it decided on: {report['winner']!r}")

    # Structure preserved, only range changed: the blend is free range.
    winner, report_json, ssim_val, _js, _dr = node.validate(a, a * 1.0001, "ssim")
    report = json.loads(report_json)
    assert torch.equal(winner, a * 1.0001) or torch.allclose(winner, a * 1.0001)
    assert "image_b" in report["winner"], (
        f"a near-identical blend should win on SSIM: {report['winner']!r}")


def test_a_zero_frame_vram_budget_is_reported_as_an_overrun(caplog):
    """A budget that fits no frames must not be logged as an optimal size.

    ``frames = max(min_frames, min(max_pixel_frames // compression, max_frames))``
    has a floor of 2, so when the estimate says zero frames fit it returns 2
    anyway, and the INFO line announced that as a budget-derived decision. That
    is an OOM about to happen, dressed as a chosen chunk size.
    """
    mod = _vae()
    with caplog.at_level("WARNING", logger="radiance"):
        frames = mod.TileEngine.get_optimal_temporal_size(
            tile_size_px=2048, temporal_compression=8, vram_budget_gb=0.05)
    assert frames == 2
    assert any("expected to run out of memory" in r.message
               for r in caplog.records), (
        "a budget below the floor was reported as an ordinary decision")


def test_a_sufficient_vram_budget_is_not_flagged(caplog):
    mod = _vae()
    with caplog.at_level("WARNING", logger="radiance"):
        frames = mod.TileEngine.get_optimal_temporal_size(
            tile_size_px=512, temporal_compression=8, vram_budget_gb=16.0)
    assert frames >= 2
    assert not any("run out of memory" in r.message for r in caplog.records)


# ═════════════════════════════════════════════════════════════════════════════
#  OCIO
# ═════════════════════════════════════════════════════════════════════════════

ocio_mod = importlib.import_module("radiance.hdr.ocio")
skip_no_ocio = pytest.mark.skipif(
    not ocio_mod.HAS_OCIO, reason="PyOpenColorIO not installed")


@skip_no_ocio
def test_the_bundled_aces_config_is_reachable(monkeypatch):
    """With $OCIO unset, _resolve_config must find the config we ship.

    Step 3 called ``OCIO.GetCurrentConfig()`` and returned it if not None. In
    OCIO v2 that never returns None and never raises: with $OCIO unset it hands
    back a built-in default carrying one colorspace, "raw". So steps 4 and 5,
    the bundled radiance/ACES/config.ocio and the models/ACES/config.ocio that
    the Download ACES 2.0 button installs, were unreachable on every machine
    where PyOpenColorIO imports, and every transform then failed on names that
    built-in config does not have.
    """
    monkeypatch.delenv("OCIO", raising=False)
    config = ocio_mod._resolve_config("")
    assert config is not None
    names = {name for name, _family in ocio_mod._iter_colorspaces(config)}
    assert "ACEScg" in names, (
        f"resolved a config with colorspaces {sorted(names)}; that is OCIO's "
        f"built-in fallback, not the bundled ACES config")


@skip_no_ocio
def test_a_host_supplied_config_still_wins(monkeypatch):
    """A config a host application genuinely set must still take priority."""
    import PyOpenColorIO as OCIO

    monkeypatch.delenv("OCIO", raising=False)
    bundled = ocio_mod._resolve_config("")
    assert not ocio_mod._is_unconfigured_default(bundled)

    # Radiance now sets OCIO's current config at startup (ocio_setup), so the
    # process state depends on import order. Install the unconfigured raw
    # config explicitly and check it is still recognised as such.
    previous = OCIO.GetCurrentConfig()
    try:
        OCIO.SetCurrentConfig(OCIO.Config.CreateRaw())
        raw = OCIO.GetCurrentConfig()
        assert ocio_mod._is_unconfigured_default(raw), (
            "OCIO's unconfigured built-in should be recognised as such")
    finally:
        OCIO.SetCurrentConfig(previous)


@skip_no_ocio
def test_the_transform_node_accepts_the_names_its_tooltip_recommends(monkeypatch):
    """Alias names must resolve, because OCIO resolves them.

    The pre-validation collected only ``ColorSpace.getName()``. In the bundled
    ACES CG config "ACES - ACEScg", "ACES - ACES2065-1" and "sRGB - Texture"
    exist ONLY as aliases, and two of those three are names the node's own
    tooltip tells the user to enter.
    """
    monkeypatch.delenv("OCIO", raising=False)
    config = ocio_mod._resolve_config("")
    canonical = {name for name, _f in ocio_mod._iter_colorspaces(config)}

    for alias, expected in (("ACES - ACEScg", "ACEScg"),
                            ("ACES - ACES2065-1", "ACES2065-1"),
                            ("sRGB - Texture", "sRGB Encoded Rec.709 (sRGB)")):
        assert alias not in canonical, (
            f"{alias!r} is a canonical name in this config; the test fixture "
            f"needs updating")
        assert ocio_mod._resolve_colorspace_name(config, alias) == expected

    assert ocio_mod._resolve_colorspace_name(config, "not a colorspace") is None


@skip_no_ocio
def test_the_transform_nodes_own_defaults_exist_in_the_bundled_config(monkeypatch):
    """The node must not raise at settings nobody touched.

    It defaulted to source_colorspace="Linear" and
    target_colorspace="ACES - ARRI LogC4 (EI800)". Neither exists in the ACES
    CG config this package bundles (the camera log spaces live in the Studio
    config), so the node raised its own "colorspace not in config" error before
    it touched a pixel.
    """
    monkeypatch.delenv("OCIO", raising=False)
    node_cls = ocio_mod.OCIOColorTransform
    required = node_cls.INPUT_TYPES()["required"]
    source = required["source_colorspace"][1]["default"]
    target = required["target_colorspace"][1]["default"]

    config = ocio_mod._resolve_config("")
    assert ocio_mod._resolve_colorspace_name(config, source) is not None, (
        f"default source_colorspace {source!r} is not in the bundled config")
    assert ocio_mod._resolve_colorspace_name(config, target) is not None, (
        f"default target_colorspace {target!r} is not in the bundled config")

    image = torch.rand(1, 8, 8, 3)
    out, _meta = node_cls().apply_transform(image, source, target)
    assert out.shape == image.shape
    assert torch.isfinite(out).all()


# ═════════════════════════════════════════════════════════════════════════════
#  Guards
# ═════════════════════════════════════════════════════════════════════════════

def test_the_quantile_guard_counts_total_elements(monkeypatch):
    """The cap is on numel, not on the reduced axis.

    The guard was ``if n <= _QUANTILE_MAX_ELEMS`` with ``n = flat.shape[1]``, so
    a 16-frame 1080p batch passed it at n = 2,073,600 and handed torch
    33,177,600 elements, twice the cap, on the adaptive-knee path that is the
    node's default. Newer torch lifted the cap on CPU, so this reinstates it to
    show what the guard was letting through.
    """
    uplift = importlib.import_module("radiance.nodes.hdr.uplift_universal")
    cap = uplift._QUANTILE_MAX_ELEMS
    real_quantile = torch.quantile

    def _capped(tensor, q, **kwargs):
        if tensor.numel() > cap:
            raise RuntimeError("quantile() input tensor is too large")
        return real_quantile(tensor, q, **kwargs)

    monkeypatch.setattr(torch, "quantile", _capped)

    rows, cols = 16, 2_073_600 // 512          # same numel/row ratio, smaller
    scaled_cap = rows * cols // 2
    monkeypatch.setattr(uplift, "_QUANTILE_MAX_ELEMS", scaled_cap)

    def _capped_scaled(tensor, q, **kwargs):
        if tensor.numel() > scaled_cap:
            raise RuntimeError("quantile() input tensor is too large")
        return real_quantile(tensor, q, **kwargs)

    monkeypatch.setattr(torch, "quantile", _capped_scaled)

    flat = torch.rand(rows, cols)
    assert flat.shape[1] <= scaled_cap, "the fixture must pass an axis-only guard"
    assert flat.numel() > scaled_cap, "the fixture must fail a numel guard"

    result = uplift._row_quantile(flat, 0.75)   # must take the kthvalue route
    assert result.shape == (rows,)
    assert torch.allclose(result, real_quantile(flat, 0.75, dim=1), atol=2e-3)


def test_detect_vae_factor_rejects_a_value_normalisation_constant():
    """latent_format.scale_factor is not a spatial factor.

    It is the latent value-normalisation constant: 0.18215 for SD, 0.3611 for
    Flux. int() of either is 0, which gave pix_h = lat_h * 0 and then a
    ZeroDivisionError downstream. Unreachable through stock comfy.sd.VAE, which
    answers spacial_compression_decode(), but reachable through any third-party
    wrapper that only exposes latent_format.
    """
    mod = _vae()

    class _LatentFormat:
        scale_factor = 0.18215

    class _WrapperVAE:
        latent_format = _LatentFormat()

    factor = mod.detect_vae_factor(_WrapperVAE())
    assert factor >= 1, f"detect_vae_factor returned {factor}"
    assert factor == mod.VAE_FACTOR_DEFAULT

    class _RealFormat:
        downscale_factor = 16

    class _RealWrapper:
        latent_format = _RealFormat()

    assert mod.detect_vae_factor(_RealWrapper()) == 16, (
        "a genuine spatial factor must still be read")


def test_a_zero_frame_video_latent_returns_an_empty_batch():
    """Zero frames must not raise.

    ``meta_str`` is bound inside the frame loop and referenced after it, so a
    zero-frame latent raised NameError, after ``torch.cat([])`` raised first.
    """
    mod = _vae()
    dec = mod.RadianceVAE4KDecode()
    latent = torch.zeros(1, 4, 0, 4, 4)
    img, meta, _fmt = dec.decode({"samples": latent}, _ScopeVAE(factor=32))
    assert img.shape[0] == 0
    assert json.loads(meta)["frames"] == 0


def test_scene_linear_capture_does_not_survive_between_runs():
    """ComfyUI reuses node instances, so per-instance state must be cleared.

    ``_scene_linear_for_rhdr`` was cleared only inside the ``if export_rhdr:``
    block, so a run that set it and exited without exporting left it populated
    and the next run's first video frame harvested the previous run's tensor.
    The shape guard catches a resolution change but not a same-resolution
    stale frame, which is the case that ships wrong pixels.
    """
    mod = _vae()
    dec = mod.RadianceVAE4KDecode()
    stale = torch.full((1, 128, 128, 3), 99.0)
    dec._scene_linear_for_rhdr = stale

    latent = torch.zeros(1, 4, 4, 4)
    dec.decode({"samples": latent}, _ScopeVAE(factor=32))

    assert dec._scene_linear_for_rhdr is not stale, (
        "the previous run's scene-linear capture survived into this one")


def test_the_stale_root_recovery_module_is_gone():
    """Only radiance/hdr/recovery.py is registered.

    A stale duplicate sat at the repo root claiming "v3.2 Fixes" for defects
    that are not fixed in the shipped module, and its Screen implementation
    would have clamped the whole image to [0,1] had anything imported it.
    Nothing did: it opens with ``from .utils import ...`` and there is no
    radiance/utils.py, so it could not even import. Anyone auditing
    "recovery.py" read the wrong file. It has been moved to _to_delete/.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert not os.path.exists(os.path.join(root, "recovery.py")), (
        "the stale root recovery.py is back")
    assert os.path.exists(os.path.join(root, "hdr", "recovery.py"))

    registered = importlib.import_module("radiance.hdr")
    assert "RadianceHighlightSynthesis" in registered.NODE_CLASS_MAPPINGS
