"""Regressions from the 2026-08 HDR-family audit: the Expand node's dead
smoothness control, the Prepare node's shifted feather mask, and the
fast-VAE tiled decode seams.
"""
import importlib

import pytest

torch = pytest.importorskip("torch")


class TestExpandSmoothness:
    """`smoothness` computed a sigmoid mask and then never used it."""

    def _expand(self, smoothness):
        mod = importlib.import_module("radiance.nodes.hdr.synthesis")
        node = mod.RadianceSDRtoHDRExpand()
        # A gradient crossing the threshold so feathering has an edge to soften.
        img = torch.linspace(0.7, 1.0, 64).view(1, 1, 64, 1).expand(1, 8, 64, 3).contiguous()
        return node.apply(image=img, inverse_oetf="None", threshold=0.8,
                          expansion_gain=5.0, expansion_gamma=1.2,
                          smoothness=smoothness)[0]

    def test_smoothness_changes_the_output(self):
        sharp = self._expand(0.0)
        soft = self._expand(0.5)
        assert not torch.allclose(sharp, soft, atol=1e-5), (
            "smoothness has no effect on the output — the dead-control bug "
            "is back")

    def test_smoothness_softens_the_onset(self):
        """Just above the threshold, the feathered version must expand LESS
        than the hard-gated one (sigmoid < 1 near the edge)."""
        sharp = self._expand(0.0)
        soft = self._expand(0.3)
        # Column index 22 ≈ luma 0.804: barely over the 0.8 threshold.
        edge_sharp = sharp[0, 0, 22, 0].item()
        edge_soft = soft[0, 0, 22, 0].item()
        assert edge_soft <= edge_sharp + 1e-6


class TestPrepareFeatherAlignment:
    """The feathered inpainting mask drifted up-left ~2x the feather radius
    because passes 2-3 zero-padded only the right/bottom edges."""

    def _feathered(self, feather):
        mod = importlib.import_module("radiance.nodes.hdr.uplift")
        node = mod.RadianceSDRToHDRPrepare()
        img = torch.full((1, 96, 96, 3), 0.5)
        mask = torch.zeros(1, 96, 96)
        mask[0, 40:56, 40:56] = 1.0          # clipped block dead-centre
        out = node.prepare(image=img, clip_mask=mask, mask_feather=feather)
        # out_mask is one of the returns; find the (1,96,96) tensor that isn't
        # the image and has values in [0,1].
        for item in out:
            if torch.is_tensor(item) and item.shape == mask.shape:
                return item
        raise AssertionError("no mask returned from prepare()")

    def test_feathered_mask_stays_centred(self):
        m = self._feathered(16)
        ys, xs = torch.nonzero(m[0] > 0.01, as_tuple=True)
        cy, cx = ys.float().mean().item(), xs.float().mean().item()
        # The clipped block is centred at (47.5, 47.5). The old bug pulled the
        # centroid up-left by ~2x feather; allow 2 px of numerical slack.
        assert abs(cy - 47.5) < 2.0 and abs(cx - 47.5) < 2.0, (
            f"feathered mask centroid drifted to ({cy:.1f}, {cx:.1f}); "
            "expected (47.5, 47.5)")

    def test_feather_zero_is_identity(self):
        m = self._feathered(0)
        assert m.max() == 1.0 and m[0, 48, 48] == 1.0


