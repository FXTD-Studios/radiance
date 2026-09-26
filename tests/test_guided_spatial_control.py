"""Guided denoising uses spatial falloff without clipping HDR or image edges."""
import pytest
import torch
import torch.nn.functional as F

from radiance.nodes.generate.denoise import RadianceDenoise


@pytest.mark.parametrize("sigma", [0.5, 2.0, 75.0])
def test_flat_guide_matches_two_normalized_gaussian_means(sigma):
    # With a constant guide the local linear slope is zero, leaving two
    # spatial means. Build an independent full 2D kernel as the reference.
    image = torch.zeros(2, 3, 9, 11, dtype=torch.float64)
    image[:, :, 0, 0] = 4.0
    image[:, :, 4, 5] = -0.7
    guide = torch.ones(2, 1, 9, 11, dtype=torch.float64)
    axis = torch.arange(-3, 4, dtype=image.dtype)
    weights = torch.exp(-(axis[:, None].square() + axis[None, :].square()) / (2 * sigma**2))
    kernel = (weights / weights.sum()).expand(3, 1, 7, 7)
    norm = F.conv2d(torch.ones_like(image), kernel, padding=3, groups=3)
    expected = F.conv2d(image, kernel, padding=3, groups=3) / norm
    expected = F.conv2d(expected, kernel, padding=3, groups=3) / norm
    actual = RadianceDenoise._guided_filter_t(guide, image, 3, 0.02, sigma)
    torch.testing.assert_close(actual, expected)


def test_sigma_space_changes_the_guided_band_result():
    image = torch.zeros(1, 3, 17, 17)
    image[..., 8, 8] = 4.0
    guide = torch.ones(1, 1, 17, 17)
    narrow = RadianceDenoise._denoise_band_t(image, 1.0, "Guided", 4, 0.2, 0.5, guide)
    wide = RadianceDenoise._denoise_band_t(image, 1.0, "Guided", 4, 0.2, 4.0, guide)
    assert narrow[0, 0, 8, 8] > wide[0, 0, 8, 8]
    assert narrow[0, 0, 8, 12] < wide[0, 0, 8, 12]
    assert torch.isfinite(wide).all()


def test_guided_filter_preserves_constant_hdr_channels_and_small_images():
    image = torch.tensor([-0.25, 0.5, 4.0]).view(1, 3, 1, 1).expand(2, 3, 2, 3).clone()
    before = image.clone()
    result = RadianceDenoise._guided_filter_t(image, image, 9, 0.02, 2.0)
    torch.testing.assert_close(result, image)
    torch.testing.assert_close(image, before)
