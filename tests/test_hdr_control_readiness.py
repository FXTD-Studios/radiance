"""HDR controls must affect every frame and the selected channel statistic."""
import pytest
import torch

from radiance.hdr.processing import HDRExposureBlend
from radiance.nodes.hdr.uplift import _make_clip_mask
from radiance.nodes.hdr.synthesis import RadianceHDRSynthesisEngine


@pytest.mark.parametrize("depth", [0, 2])
def test_synthesis_chroma_preserves_luminance_and_controls_highlight_saturation(depth):
    image = torch.tensor([0.95, 0.75, 0.45, 0.37]).expand(2, 16, 16, 4).clone()
    original = image.clone()
    node = RadianceHDRSynthesisEngine()
    neutral, mask = node.synthesize(image, 4.0, depth, 0.0)
    preserved, _ = node.synthesize(image, 4.0, depth, 1.0)
    midpoint, _ = node.synthesize(image, 4.0, depth, 0.5)
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    torch.testing.assert_close(neutral[..., :3] @ weights, preserved[..., :3] @ weights)
    torch.testing.assert_close(midpoint, (neutral + preserved) / 2)
    rgb = image[..., :3]
    torch.testing.assert_close(preserved[..., :3] / preserved[..., :1], rgb / rgb[..., :1])
    assert torch.all(neutral[..., 2] / neutral[..., 0] > rgb[..., 2] / rgb[..., 0])
    assert torch.all(preserved[..., 0] > image[..., 0])
    for result in (neutral, preserved, midpoint):
        torch.testing.assert_close(result[..., 3], image[..., 3])
    assert mask.shape == (2, 16, 16, 3)
    torch.testing.assert_close(image, original)


@pytest.mark.parametrize("chroma", [0.0, 0.5, 1.0])
def test_synthesis_unit_energy_is_identity(chroma):
    image = torch.rand(2, 16, 16, 4)
    result, _ = RadianceHDRSynthesisEngine().synthesize(image, 1.0, 2, chroma)
    torch.testing.assert_close(result, image)


@pytest.mark.parametrize("soft", [0.0, 0.1])
def test_clip_channel_mode_is_used_with_and_without_feathering(soft):
    image = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]])
    any_mask = _make_clip_mask(image, 0.97, "any", soft)
    all_mask = _make_clip_mask(image, 0.97, "all", soft)
    luma_mask = _make_clip_mask(image, 0.97, "luma", soft)
    assert any_mask.tolist() == [[[1.0, 1.0]]]
    assert all_mask.tolist() == [[[0.0, 1.0]]]
    assert luma_mask.tolist() == [[[0.0, 1.0]]]


@pytest.mark.parametrize("mode,value", [("any", 0.95), ("all", 0.91), ("luma", 0.925656)])
def test_clip_feather_uses_selected_signal(mode, value):
    image = torch.tensor([[[[0.95, 0.92, 0.91]]]])
    mask = _make_clip_mask(image, 0.97, mode, 0.1)
    assert mask.item() == pytest.approx((value - 0.87) / 0.1, abs=1e-5)


@pytest.mark.parametrize("method", HDRExposureBlend.BLEND_METHODS)
def test_blend_keeps_every_frame_and_matches_individual_runs(method):
    node = HDRExposureBlend()
    low = torch.rand(3, 16, 16, 3, generator=torch.Generator().manual_seed(4)) * 0.3
    high = low * 3
    result, masks, report = node.blend_exposures(low, high, method)
    assert result.shape == low.shape and masks.shape == low.shape
    for i in range(3):
        expected, mask, _ = node.blend_exposures(low[i:i+1], high[i:i+1], method)
        torch.testing.assert_close(result[i:i+1], expected)
        torch.testing.assert_close(masks[i:i+1], mask)
    assert "Frame 3:" in report


def test_exposure_weighted_discards_clipped_bracket_before_ev_compensation():
    low = torch.full((1, 8, 8, 3), 0.4)
    high = torch.ones_like(low)
    result, mask, _ = HDRExposureBlend().blend_exposures(low, high, "Exposure Weighted")
    torch.testing.assert_close(result, torch.full_like(low, 1.6))
    assert torch.all(mask == 1)


def test_middle_bracket_can_supply_all_detail():
    low = torch.zeros(1, 8, 8, 3)
    high = torch.ones_like(low)
    mid = torch.full_like(low, 0.5)
    result, _, _ = HDRExposureBlend().blend_exposures(low, high, "Exposure Weighted", mid_exposure=mid)
    torch.testing.assert_close(result, mid)


@pytest.mark.parametrize("method", HDRExposureBlend.BLEND_METHODS)
def test_black_brackets_are_finite(method):
    image = torch.zeros(2, 16, 16, 3)
    result, masks, report = HDRExposureBlend().blend_exposures(image, image, method)
    assert torch.isfinite(result).all() and torch.isfinite(masks).all()
    assert not result.any() and "0.0 stops" in report


def test_single_exposure_broadcast_and_mismatched_batches():
    node = HDRExposureBlend()
    low = torch.full((1, 8, 8, 3), 0.3)
    high = torch.full((3, 8, 8, 3), 0.8)
    result, _, _ = node.blend_exposures(low, high, "Exposure Weighted")
    assert result.shape[0] == 3
    with pytest.raises(ValueError, match="batch"):
        node.blend_exposures(low.expand(2, -1, -1, -1), high)



def test_monitor_gamma_switch_works_for_exposure_gamma_and_preserves_alpha():
    from radiance.nodes.hdr.delivery import RadianceHDRMonitor
    image = torch.tensor([[[[0.25, 0.5, 1.0, 0.3]]]])
    node = RadianceHDRMonitor()
    off, = node.monitor(image, operator="Exposure + Gamma", gamma=2.0, gamma_correct_sdr=False)
    on, = node.monitor(image, operator="Exposure + Gamma", gamma=2.0, gamma_correct_sdr=True)
    torch.testing.assert_close(off, image)
    torch.testing.assert_close(on[..., :3], image[..., :3].sqrt())
    torch.testing.assert_close(on[..., 3:], image[..., 3:])


def test_unbatched_brackets_remain_supported():
    image = torch.full((8, 8, 3), 0.5)
    result, mask, _ = HDRExposureBlend().blend_exposures(image, image, "Exposure Weighted")
    assert result.shape == (1, 8, 8, 3) and mask.shape == result.shape
