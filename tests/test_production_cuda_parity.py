"""Real CUDA checks for the spatial controls changed during production audit."""
import pytest
import torch

from radiance.nodes.generate.denoise import RadianceDenoise
from radiance.nodes.vfx import masking, inpaint

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("sigma", [0.5, 4.0, 75.0])
def test_guided_spatial_cpu_cuda_agree(sigma):
    generator = torch.Generator().manual_seed(19)
    image = torch.rand(2, 3, 64, 96, generator=generator) * 4 - 0.2
    guide = image.mean(1, keepdim=True)
    cpu = RadianceDenoise._guided_filter_t(guide, image, 9, 0.02, sigma)
    gpu = RadianceDenoise._guided_filter_t(guide.cuda(), image.cuda(), 9, 0.02, sigma)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-4, atol=2e-5)


def test_trimap_cpu_cuda_agree_across_chunks(monkeypatch):
    image = torch.rand(3, 64, 96, 3, generator=torch.Generator().manual_seed(23)) * 3
    mask = torch.zeros(1, 64, 96)
    mask[:, 8:55, 20:78] = 1
    monkeypatch.setattr(masking, "frames_per_chunk", lambda *args: 1)
    monkeypatch.setattr(masking, "compute_device", lambda: torch.device("cpu"))
    expected = masking.RadianceLinearMatting().apply(image, mask, "GuidedFilter", 5, 0.001)
    monkeypatch.setattr(masking, "compute_device", lambda: torch.device("cuda"))
    actual = masking.RadianceLinearMatting().apply(image, mask, "GuidedFilter", 5, 0.001)
    for gpu, cpu in zip(actual, expected):
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-4, atol=2e-5)


def test_standard_stitch_cpu_cuda_agree(monkeypatch):
    image = torch.rand(3, 64, 96, 4, generator=torch.Generator().manual_seed(31)) * 4
    crop = torch.full((1, 32, 40, 4), 0.5)
    mask = torch.ones(1, 32, 40)
    bounds = dict(ymin=16, ymax=48, xmin=20, xmax=60)
    monkeypatch.setattr(inpaint, "frames_per_chunk", lambda *args: 1)
    monkeypatch.setattr(inpaint, "compute_device", lambda: torch.device("cpu"))
    expected = inpaint.RadianceHDRStitch().apply(image, crop, mask, bounds, "Standard", 6)
    monkeypatch.setattr(inpaint, "compute_device", lambda: torch.device("cuda"))
    actual = inpaint.RadianceHDRStitch().apply(image, crop, mask, bounds, "Standard", 6)
    for gpu, cpu in zip(actual, expected):
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-4, atol=2e-5)


def test_denoise_detail_recovery_cpu_cuda_agree():
    image = torch.rand(2, 32, 48, 4, generator=torch.Generator().manual_seed(41)) * 3
    node = RadianceDenoise()
    options = dict(d=6, sigmaColor=0.2, sigmaSpace=4.0, filter_type="Guided", detail_recovery=0.8)
    cpu, = node.denoise(image, **options)
    gpu, = node.denoise(image.cuda(), **options)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(gpu[..., 3].cpu(), image[..., 3])
