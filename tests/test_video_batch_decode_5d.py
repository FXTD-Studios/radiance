"""
tests/test_video_batch_decode_5d.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RadianceVideoBatchDecode threw away ComfyUI's unlimited-duration video decode.

It flattened [B,C,T,H,W] to [B*T,C,H,W] before calling vae.decode(). comfy.sd
computes ``dims = samples_in.ndim - 2``, so a 4-D input is an image decode: the
video path and its decode_tiled_3d() memory fallback (which shrinks the temporal
tile until one fits) were unreachable, and a causal-3D-conv VAE either raised or
decoded every latent frame in isolation, dropping the temporal upsample. For
LTX (temporal compression 8) that is T frames returned where (T-1)*8+1 are owed.

Both mitigation widgets were inert with it:
  * tile_decode probed vae.enable_tiling / vae.set_tiling, neither of which
    exists on any ComfyUI VAE, and printed "Tile decode: enabled" anyway.
  * tile_overlap was accepted and never referenced.

The same "report a number nobody produced" shape ran through the report line
("Decoded {T} frames" while B*T were returned) and through the three bare
`except Exception` handlers that turned an OOM into black pixels labelled as a
successful render.
"""

from __future__ import annotations

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")

RADIANCE_TORCH_GATED = True
# Every test here needs real tensors; skip them all on the stub lane.
pytestmark = pytest.mark.real_torch


def _t2v():
    mod = "radiance.nodes.video.t2v"
    if mod in sys.modules:
        return sys.modules[mod]
    return importlib.import_module(mod)


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakeVideoVAE:
    """Stand-in for comfy.sd.VAE with only what the decode path touches.

    Deliberately not a MagicMock: the node branches on real attribute absence
    (``callable(getattr(vae, "decode_tiled", None))``), and MagicMock hands out
    a truthy callable for every name asked of it.

    decode() rejects a 4-D latent the way a causal 3-D-conv video VAE does, and
    returns (B, T_pixel, H, W, C) for a 5-D one, matching comfy.sd.VAE.decode()'s
    closing ``movedim(1, -1)``.
    """

    def __init__(self, temporal_compression=8, spatial_compression=32):
        self._tc = temporal_compression
        self._sc = spatial_compression
        self.decode_calls = []
        self.decode_tiled_calls = []

    def temporal_compression_decode(self):
        return self._tc

    def spacial_compression_decode(self):
        return self._sc

    def _pixel_frames(self, lat_t):
        return max(0, lat_t * self._tc - (self._tc - 1))

    def _out(self, samples):
        b, _c, t, h, w = samples.shape
        return torch.zeros(b, self._pixel_frames(t), h * self._sc, w * self._sc, 3)

    def decode(self, samples, vae_options=None):
        self.decode_calls.append(tuple(samples.shape))
        if samples.ndim != 5:
            raise RuntimeError(
                "video VAE expects a 5-D latent; got %dD" % samples.ndim)
        return self._out(samples)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None,
                     tile_t=None, overlap_t=None):
        self.decode_tiled_calls.append(
            dict(tile_x=tile_x, tile_y=tile_y, overlap=overlap,
                 tile_t=tile_t, overlap_t=overlap_t))
        return self._out(samples)


class _NoTiledVAE(_FakeVideoVAE):
    """A VAE-like object with no tiled entry point at all, which is what the
    test suite (and any third-party VAE wrapper) actually presents."""

    decode_tiled = None

    def __getattribute__(self, name):
        if name == "decode_tiled":
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class _FakeImageVAE:
    """Non-temporal VAE: temporal_compression_decode() returns None, as
    comfy.sd.VAE does for every image VAE (its upscale_ratio is a plain int,
    so the tuple index in that helper raises and it returns None)."""

    def __init__(self, spatial_compression=8):
        self._sc = spatial_compression
        self.decode_calls = []
        self.decode_tiled_calls = []

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return self._sc

    def decode(self, samples, vae_options=None):
        self.decode_calls.append(tuple(samples.shape))
        b, _c, h, w = samples.shape
        return torch.zeros(b, h * self._sc, w * self._sc, 3)

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None,
                     tile_t=None, overlap_t=None):
        self.decode_tiled_calls.append(
            dict(tile_x=tile_x, tile_y=tile_y, overlap=overlap,
                 tile_t=tile_t, overlap_t=overlap_t))
        b, _c, h, w = samples.shape
        return torch.zeros(b, h * self._sc, w * self._sc, 3)


class _OOMVAE:
    """Every decode raises the way a real out-of-memory decode does."""

    def temporal_compression_decode(self):
        return 8

    def spacial_compression_decode(self):
        return 32

    def decode(self, samples, vae_options=None):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    def decode_tiled(self, samples, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")


def _decoder():
    return _t2v().RadianceVideoBatchDecode()


# ── 1. The 5-D latent reaches the VAE as 5-D ─────────────────────────────────

def test_five_d_latent_is_not_flattened_before_decode():
    """The pre-fix node called vae.decode() with [B*T, C, H, W]; this VAE
    raises on a 4-D latent exactly as a causal-3D-conv video VAE does, so the
    old code could not get through this test at all."""
    vae = _FakeVideoVAE(temporal_compression=8)
    latent = torch.zeros(1, 128, 6, 8, 8)

    frames, count, report = _decoder().decode(vae, {"samples": latent})

    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0] == (1, 128, 6, 8, 8), \
        "5-D latent must go to vae.decode() unflattened"
    assert count == frames.shape[0]


def test_frame_count_follows_the_temporal_upsample():
    """LTX-style temporal compression 8: 6 latent frames decode to
    (6-1)*8+1 = 41 pixel frames. The flattened path returned B*T = 6."""
    vae = _FakeVideoVAE(temporal_compression=8)
    latent = torch.zeros(1, 128, 6, 8, 8)

    frames, count, report = _decoder().decode(vae, {"samples": latent})

    assert count == 41
    assert frames.shape[0] == 41
    assert frames.ndim == 4, "IMAGE output must be [N,H,W,C]"


def test_report_frame_count_matches_the_returned_batch():
    """The old report said "Decoded {T} frames from 5D latent" while the
    function returned B*T, so the number never described the output. The new
    line must quote the count the caller can actually measure."""
    vae = _FakeVideoVAE(temporal_compression=4)
    latent = torch.zeros(2, 16, 5, 8, 8)

    frames, count, report = _decoder().decode(vae, {"samples": latent})

    expected = 2 * ((5 - 1) * 4 + 1)
    assert count == expected == frames.shape[0]
    assert f"Decoded {count} pixel frames" in report
    assert "CHECK:" not in report


def test_mismatch_between_expected_and_returned_frames_is_reported():
    """A VAE whose real output disagrees with its declared temporal compression
    (the MiniMax H3 taehv genuinely does) must be called out, not quietly
    accepted, and frame_count must stay the number actually returned."""
    vae = _FakeVideoVAE(temporal_compression=8)
    vae.temporal_compression_decode = lambda: 4   # lies about its own factor
    latent = torch.zeros(1, 128, 6, 8, 8)

    _frames, count, report = _decoder().decode(vae, {"samples": latent})

    assert count == 41         # what the VAE actually returned
    assert "CHECK:" in report  # against the 21 its declared factor implies


# ── 2. tile_decode routes to the real tiled entry point ──────────────────────

def test_tile_decode_calls_decode_tiled_with_temporal_args():
    """tile_decode probed enable_tiling/set_tiling, which exist on no ComfyUI
    VAE, so the loop was a no-op and nothing was ever tiled."""
    vae = _FakeVideoVAE(temporal_compression=8, spatial_compression=32)
    latent = torch.zeros(1, 128, 30, 8, 8)

    _frames, _count, report = _decoder().decode(
        vae, {"samples": latent}, tile_decode=True, tile_overlap=64)

    assert len(vae.decode_tiled_calls) == 1
    assert len(vae.decode_calls) == 0
    call = vae.decode_tiled_calls[0]
    # Latent units, not pixel units: comfy's VAEDecodeTiled divides every
    # pixel-space size by the compression before calling.
    assert call["tile_x"] == 512 // 32
    assert call["tile_y"] == 512 // 32
    assert call["tile_t"] == max(2, 64 // 8)
    assert call["overlap_t"] == max(1, min(call["tile_t"] // 2, 8 // 8))
    assert "decode_tiled" in report


def test_tile_overlap_widget_reaches_the_decode():
    """tile_overlap was accepted and never referenced in the function body."""
    vae = _FakeVideoVAE(spatial_compression=8)
    latent = torch.zeros(1, 16, 4, 8, 8)

    for overlap_px in (0, 32, 64):
        vae.decode_tiled_calls.clear()
        _decoder().decode(vae, {"samples": latent},
                          tile_decode=True, tile_overlap=overlap_px)
        assert vae.decode_tiled_calls[0]["overlap"] == overlap_px // 8, \
            "tile_overlap must reach vae.decode_tiled()'s overlap argument"


def test_report_quotes_the_clamped_overlap_not_the_requested_one():
    """ComfyUI clamps overlap to tile_size//4 when it is more than a quarter of
    the tile, so any tile_overlap above 128px is silently lowered. The report
    has to name what the decode used, or it repeats the defect being fixed."""
    vae = _FakeVideoVAE(spatial_compression=8)
    latent = torch.zeros(1, 16, 4, 8, 8)

    _frames, _count, report = _decoder().decode(
        vae, {"samples": latent}, tile_decode=True, tile_overlap=256)

    used = vae.decode_tiled_calls[0]["overlap"]
    assert used == (512 // 4) // 8          # clamped, not 256 // 8
    assert f"overlap={used}" in report


def test_expected_pixel_frames_matches_the_causal_vae_formula():
    """(T-1)*k + 1, which is comfy.sd.VAE's upscale_ratio[0] for every video
    VAE: lambda a: max(0, a*k - (k-1)). The dit.py temporal_compression values
    are the k in that expression, not a plain frames-per-latent-frame ratio."""
    fn = _t2v()._expected_pixel_frames

    assert fn(6, 8) == 41        # LTX-Video, temporal compression 8
    assert fn(7, 4) == 25        # HunyuanVideo / Wan2.x, temporal compression 4
    assert fn(5, 6) == 25        # Mochi-1, temporal compression 6
    assert fn(1, 8) == 1         # a single latent frame is a single pixel frame
    assert fn(12, None) == 12    # image VAE: no temporal upsample
    assert fn(12, 1) == 12


def test_tile_decode_omits_temporal_args_for_an_image_vae():
    """comfy's decode_tiled dispatcher only reaches its 3-D branch when the
    latent is 5-D; tile_t on a non-temporal VAE is meaningless."""
    vae = _FakeImageVAE()
    latent = torch.zeros(1, 4, 64, 64)

    _decoder().decode(vae, {"samples": latent}, tile_decode=True, tile_overlap=64)

    call = vae.decode_tiled_calls[0]
    assert call["tile_t"] is None
    assert call["overlap_t"] is None


def test_tile_decode_without_decode_tiled_says_so():
    """Degrade visibly: a VAE stub with no tiled entry point must still decode,
    and the report must state that the requested tiling did not happen. The old
    code printed "Tile decode: enabled" for a VAE that had no tiling at all."""
    vae = _NoTiledVAE()
    assert not hasattr(vae, "decode_tiled")
    latent = torch.zeros(1, 128, 4, 8, 8)

    frames, count, report = _decoder().decode(
        vae, {"samples": latent}, tile_decode=True)

    assert "Tile decode: UNAVAILABLE" in report
    assert "enabled" not in report
    assert count == frames.shape[0] == (4 - 1) * 8 + 1


# ── 3. 4-D image latents keep their old behaviour ────────────────────────────

def test_four_d_latent_is_passed_through_untouched():
    vae = _FakeImageVAE(spatial_compression=8)
    latent = torch.zeros(3, 4, 16, 16)

    frames, count, report = _decoder().decode(vae, {"samples": latent})

    assert vae.decode_calls == [(3, 4, 16, 16)]
    assert count == 3
    assert frames.shape == (3, 128, 128, 3)
    assert "pixel frames from" not in report   # temporal line is 5-D only


# ── 4. Failures are visible, never black pixels reported as success ──────────

def test_oom_during_decode_propagates():
    """torch.cuda.OutOfMemoryError subclasses RuntimeError subclasses
    Exception, so `except Exception: return torch.zeros(1,64,64,3)` caught
    exactly the case that matters and handed back black frames the report then
    counted as output."""
    latent = torch.zeros(1, 128, 40, 8, 8)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _decoder().decode(_OOMVAE(), {"samples": latent})


def test_oom_during_tiled_decode_propagates():
    latent = torch.zeros(1, 128, 40, 8, 8)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _decoder().decode(_OOMVAE(), {"samples": latent}, tile_decode=True)


def test_vae_decode_helper_raises_instead_of_returning_black():
    t2v = _t2v()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        t2v._vae_decode(_OOMVAE(), {"samples": torch.zeros(1, 128, 4, 8, 8)})


def test_comfy_sample_helper_raises_instead_of_returning_the_noise():
    """_comfy_sample returned the raw input noise on any failure; the caller
    wrapped it in a LATENT and reported a successful sample, so the graph
    carried on and decoded static."""
    t2v = _t2v()
    noise = torch.randn(1, 16, 4, 8, 8)

    class _Exploding:
        def get_model_object(self, name):
            raise RuntimeError("sampler blew up (simulated)")

    raised = None
    returned = object()
    try:
        returned = t2v._comfy_sample(
            _Exploding(), noise, 10, 7.0, "euler", "normal",
            [], [], {"samples": noise})
    except Exception as exc:          # noqa: BLE001 - the point of the test
        raised = exc

    assert raised is not None, "a failed sample must not return quietly"
    assert returned is not noise, "the raw input noise must never be the result"


# ── 5. Preview decode failures are reported, not dressed up ──────────────────

@pytest.mark.parametrize("node_name", ["RadianceT2VPipeline", "RadianceI2VPipeline"])
def test_preview_decode_failure_is_named_in_the_report(node_name):
    """Both pipelines swallowed a preview decode failure and returned a 64x64
    black batch, which the report then counted as "Preview frames: 200"."""
    t2v = _t2v()
    node = getattr(t2v, node_name)()

    frames, error = node._decode_preview(_OOMVAE(), {"samples": torch.zeros(1, 128, 4, 8, 8)}, 200)

    assert error is not None, "a failed preview decode must be reported"
    assert "OutOfMemoryError" in error
    assert frames.shape[0] == 200   # still a placeholder, now a labelled one


@pytest.mark.parametrize("node_name", ["RadianceT2VPipeline", "RadianceI2VPipeline"])
def test_preview_decode_success_returns_no_error(node_name):
    t2v = _t2v()
    node = getattr(t2v, node_name)()
    vae = _FakeVideoVAE(temporal_compression=8, spatial_compression=8)

    frames, error = node._decode_preview(vae, {"samples": torch.zeros(1, 128, 4, 8, 8)}, 25)

    assert error is None
    assert frames.shape[0] == (4 - 1) * 8 + 1


@pytest.mark.parametrize("node_name", ["RadianceT2VPipeline", "RadianceI2VPipeline"])
def test_preview_reshape_does_not_permute_a_channels_last_result(node_name):
    """comfy.sd.VAE.decode() ends with movedim(1, -1), so a 5-D result is
    (B, T, H, W, C). The old code read it as (B, C, T, H, W) and permuted,
    which shuffled the spatial axes into the batch."""
    t2v = _t2v()
    node = getattr(t2v, node_name)()
    vae = _FakeVideoVAE(temporal_compression=4, spatial_compression=8)
    # (1, 16, 3, 5, 7) -> (1, 9, 40, 56, 3)
    frames, error = node._decode_preview(vae, {"samples": torch.zeros(1, 16, 3, 5, 7)}, 9)

    assert error is None
    assert frames.shape == (9, 40, 56, 3)


# ── 6. The retired `tiling` control ──────────────────────────────────────────

def test_video_sampler_no_longer_offers_the_dead_tiling_widget():
    """It drove model.model.set_tiling(True). No ComfyUI release defines
    set_tiling or enable_tiling on a model or a VAE, so the branch never ran
    while the report printed "Tiling: enabled"."""
    t2v = _t2v()
    spec = t2v.RadianceVideoSampler.INPUT_TYPES()
    assert "tiling" not in spec.get("optional", {})
    assert "tiling" not in spec.get("required", {})


def test_video_sampler_says_so_when_a_saved_workflow_still_sends_tiling():
    t2v = _t2v()
    src = (t2v.RadianceVideoSampler.sample.__code__.co_consts)
    joined = " ".join(c for c in src if isinstance(c, str))
    assert "NOT APPLIED" in joined
    assert "set_tiling" not in joined


def test_no_module_anywhere_still_probes_the_nonexistent_tiling_methods():
    import pathlib
    import re
    src = pathlib.Path(_t2v().__file__).with_suffix(".py").read_text(encoding="utf-8")
    code = "\n".join(re.sub(r"(?<!['\"])#.*$", "", line) for line in src.split("\n"))
    assert "set_tiling(" not in code, "set_tiling exists on no ComfyUI object"
    assert "enable_tiling(" not in code
