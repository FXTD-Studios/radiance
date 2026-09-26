"""
Three long-standing defects from the README backlog, and one that was already
fixed but never pinned.

Each of these shipped in a state where the code ran, produced output, and told
you nothing was wrong.
"""
import os
import sys

import numpy as np
import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
#  Scene-cut threshold was relative to the batch maximum
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_frame(shift, h=32, w=32, seed=0):
    """A frame whose content moves smoothly — a pan, not a cut."""
    rng = np.random.default_rng(seed)
    base = rng.random((h, w + 64, 3)).astype(np.float32)
    return np.ascontiguousarray(base[:, shift:shift + w, :])


class TestSceneCutThresholdIsAbsolute:

    def test_a_cut_free_pan_reports_no_cuts(self):
        """The defect: normalising by the batch max forced the largest ripple
        in *any* clip to 1.0, so footage with no cut still reported one."""
        from radiance.nodes.ai.scene_cut import detect_cuts

        frames = np.stack([_gradient_frame(i) for i in range(24)])
        cuts, scores = detect_cuts(frames, threshold=0.35, min_shot_frames=4)

        assert scores.max() < 0.35, \
            f"test premise broken: a pan scored {scores.max():.3f}, above the threshold"
        assert cuts == [0], f"a cut-free pan reported cuts at {cuts}"

    def test_a_real_cut_is_still_found(self):
        from radiance.nodes.ai.scene_cut import detect_cuts

        rng = np.random.default_rng(1)
        shot_a = np.tile(rng.random((1, 32, 32, 3)).astype(np.float32), (10, 1, 1, 1)) * 0.15
        shot_b = np.ones((10, 32, 32, 3), dtype=np.float32) * 0.95
        frames = np.concatenate([shot_a, shot_b])

        cuts, _ = detect_cuts(frames, threshold=0.35, min_shot_frames=4)
        assert 10 in cuts, f"missed the cut at frame 10: {cuts}"

    def test_the_threshold_means_the_same_thing_on_every_clip(self):
        """The point of an absolute threshold: identical content, different
        batch composition, same verdict."""
        from radiance.nodes.ai.scene_cut import detect_cuts

        rng = np.random.default_rng(2)
        a = np.tile(rng.random((1, 32, 32, 3)).astype(np.float32), (6, 1, 1, 1))
        b = np.ones((6, 32, 32, 3), dtype=np.float32)
        pair = np.concatenate([a, b])

        cuts_short, _ = detect_cuts(pair, threshold=0.35, min_shot_frames=2)
        # Same cut, but now padded with quiet frames that change the batch max.
        padded = np.concatenate([a, b, b, b])
        cuts_long, _ = detect_cuts(padded, threshold=0.35, min_shot_frames=2)

        assert cuts_short == cuts_long, \
            f"the same cut was judged differently by batch: {cuts_short} vs {cuts_long}"

    def test_the_plot_and_the_cut_list_agree(self):
        """`_make_plot` always compared raw scores to the threshold while
        detection compared normalised ones, so the picture could contradict
        the result on the same input."""
        from radiance.nodes.ai.scene_cut import detect_cuts

        frames = np.stack([_gradient_frame(i) for i in range(16)])
        threshold = 0.35
        cuts, scores = detect_cuts(frames, threshold=threshold, min_shot_frames=1)

        plot_says_cut = [i + 1 for i, s in enumerate(scores) if s >= threshold]
        assert plot_says_cut == [c for c in cuts if c > 0]


# ─────────────────────────────────────────────────────────────────────────────
#  Tier-3 upscale ignored its scale input
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestTier3HonoursScale:

    def test_a_fixed_4x_backend_is_conformed_to_2x(self):
        """The defect: the SD x4 backend always returns 4x, and tiled_upscale
        then cropped to 2x — keeping the top-left quarter of the render."""
        from radiance.nodes.upscale.upscale import _conform_to_scale

        src = _t.zeros(1, 16, 16, 3)
        # A 4x render with a distinctive right-hand half.
        out4 = _t.zeros(1, 64, 64, 3)
        out4[:, :, 32:, :] = 1.0

        got = _conform_to_scale(src, out4, 2)

        assert tuple(got.shape) == (1, 32, 32, 3)
        # A crop would have dropped the bright half entirely.
        assert got[:, :, 16:, :].mean() > 0.9, "the right-hand half was cropped away"
        assert got[:, :, :16, :].mean() < 0.1

    def test_a_matching_scale_is_passed_through_untouched(self):
        from radiance.nodes.upscale.upscale import _conform_to_scale

        src = _t.zeros(1, 8, 8, 3)
        out = _t.rand(1, 32, 32, 3)
        assert _conform_to_scale(src, out, 4) is out

    def test_upsampling_a_short_backend_also_works(self):
        from radiance.nodes.upscale.upscale import _conform_to_scale

        src = _t.zeros(1, 8, 8, 3)
        out2 = _t.rand(1, 16, 16, 3)
        got = _conform_to_scale(src, out2, 4)
        assert tuple(got.shape) == (1, 32, 32, 3)

    def test_the_whole_frame_survives_not_a_corner(self):
        """Correlation against the input was measured at 0.0024 before the fix."""
        from radiance.nodes.upscale.upscale import _conform_to_scale

        src = _t.rand(1, 16, 16, 3)
        x = src.permute(0, 3, 1, 2)
        out4 = _t.nn.functional.interpolate(x, scale_factor=4, mode="nearest").permute(0, 2, 3, 1)

        got = _conform_to_scale(src, out4, 2)
        ref = _t.nn.functional.interpolate(x, scale_factor=2, mode="area").permute(0, 2, 3, 1)

        a = (got - got.mean()).flatten()
        b = (ref - ref.mean()).flatten()
        corr = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        assert corr > 0.95, f"conformed output barely resembles the source: r={corr:.4f}"

    def test_the_tier3_path_actually_calls_the_conform(self, monkeypatch):
        """Wiring, not just the helper: build the real tier-3 callable with a
        stand-in 4x backend and check what comes out is 2x of the input."""
        from radiance.nodes.upscale import upscale as up

        def _fake_backend(tile, device, **kw):
            x = tile.permute(0, 3, 1, 2)
            y = _t.nn.functional.interpolate(x, scale_factor=4, mode="nearest")
            return y.permute(0, 2, 3, 1)

        monkeypatch.setattr(up, "_diffusion_upscale_infer", _fake_backend)

        fn, label = up._build_upscale_fn("tier3", 2, _t.device("cpu"))
        out = fn(_t.rand(1, 16, 16, 3))

        assert tuple(out.shape) == (1, 32, 32, 3), \
            f"tier-3 at scale=2 returned {tuple(out.shape)}; the 4x render was not conformed"
        assert "diffusion" in label.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Tiled VAE blending — already correct, never pinned
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TORCH, reason="needs real torch")
class TestTiledVAEBlend:
    """Tiles used to be butt-joined, leaving a visible seam grid.

    The cosine ramp that fixed it had no test, so the audit notes still listed
    the bug as open months afterwards. These lock the property in.
    """

    @staticmethod
    def _slice(overlap_left, overlap_right, tile=64):
        from radiance.hdr.vae import TileEngine
        w = TileEngine.make_cosine_blend_weight_2d(
            tile, tile, 0, 0, overlap_left, overlap_right)
        a = w.numpy() if _t.is_tensor(w) else np.asarray(w)
        return a.reshape(tile, tile)[tile // 2]

    def test_an_interior_tile_is_feathered_not_flat(self):
        row = self._slice(16, 16)
        assert row[0] < 0.05, "tile edge is not ramped — tiles would butt-join"
        assert row[32] == pytest.approx(1.0, abs=1e-6), "tile centre is attenuated"

    def test_border_edges_are_not_ramped(self):
        """Feathering the outer frame edge would darken the image border."""
        row = self._slice(0, 16)
        assert row[0] == pytest.approx(1.0, abs=1e-6)

    def test_overlapping_tiles_reconstruct_a_flat_field(self):
        """The seam test: weights must sum to 1 everywhere after normalising."""
        tile, overlap, width = 64, 16, 200
        step = tile - overlap
        xs = list(range(0, width - tile + 1, step))
        if xs[-1] + tile < width:
            xs.append(width - tile)

        acc = np.zeros(width)
        wsum = np.zeros(width)
        for x in xs:
            row = self._slice(0 if x == 0 else overlap,
                              0 if x + tile >= width else overlap)
            acc[x:x + tile] += row * 1.0       # a constant-1 image
            wsum[x:x + tile] += row

        out = acc / np.maximum(wsum, 1e-8)
        assert np.abs(out - 1.0).max() < 1e-6, \
            f"seam of {np.abs(out - 1.0).max():.2e} across tile boundaries"
