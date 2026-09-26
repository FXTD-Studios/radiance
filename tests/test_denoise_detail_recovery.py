"""Detail recovery must not undo low-frequency denoising."""
import torch

from radiance.nodes.generate.denoise import RadianceDenoise


def test_detail_recovery_does_not_restore_a_constant_removed_offset(monkeypatch):
    # Remove all spatial bands to isolate the final recovery stage.
    monkeypatch.setattr(RadianceDenoise, "_denoise_band_t", lambda self, band, *args: torch.zeros_like(band))
    image = torch.tensor([0.2, 0.4, 2.0, 0.37]).view(1, 1, 1, 4).expand(2, 16, 16, 4).clone()
    result, = RadianceDenoise().denoise(image, 6, 0.2, 4.0, detail_recovery=1.0)
    torch.testing.assert_close(result[..., :3], torch.zeros_like(result[..., :3]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[..., 3], image[..., 3])


def test_detail_recovery_restores_texture_without_restoring_the_dc_level(monkeypatch):
    monkeypatch.setattr(RadianceDenoise, "_denoise_band_t", lambda self, band, *args: torch.zeros_like(band))
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checker = ((xx + yy) % 2).float() * 0.2 + 0.5
    image = checker.view(1, 32, 32, 1).expand(1, 32, 32, 3).clone()
    original = image.clone()
    node = RadianceDenoise()
    off, = node.denoise(image, 6, 0.2, 4.0, detail_recovery=0.0)
    half, = node.denoise(image, 6, 0.2, 4.0, detail_recovery=0.5)
    full, = node.denoise(image, 6, 0.2, 4.0, detail_recovery=1.0)
    assert full.std() > off.std() + 0.05
    assert abs(full.mean()) < 1e-5
    torch.testing.assert_close(half, (off + full) / 2)
    torch.testing.assert_close(image, original)
