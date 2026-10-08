"""Denoise `motion_compensation`: a block search, coarse to fine, since 4.0.

Until 4.0 it picked, per pixel, the best of the nine 1-pixel offsets, so it
only followed motion of about a pixel per frame. It now estimates motion on a
pyramid with a block cost and warps the neighbour with edge repeat.
"""
import torch
import torch.nn.functional as F
import pytest

from radiance.nodes.generate.denoise import RadianceDenoise

MC = RadianceDenoise._motion_compensate_t


def _texture(size=160, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(1, 3, size, size, generator=g)
    t = F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, stride=1)   # mild blur
    return t[0].permute(1, 2, 0).contiguous()                                  # (H, W, 3)


def _pair(dy, dx, size=96, margin=24):
    tex = _texture(size + 2 * margin)
    frame = tex[margin:margin + size, margin:margin + size]
    neighbor = tex[margin - dy:margin - dy + size, margin - dx:margin - dx + size]
    return frame, neighbor


@pytest.mark.parametrize("dy, dx", [(1, 1), (3, 5), (-6, 4), (0, -9)])
def test_a_shifted_neighbour_is_aligned(dy, dx):
    frame, neighbor = _pair(dy, dx)
    aligned = MC(frame, neighbor)
    m = 12
    err = (aligned[m:-m, m:-m] - frame[m:-m, m:-m]).abs().mean()
    assert err < 1e-3, f"shift ({dy}, {dx}) left error {err:.4f}"


def test_zero_motion_is_an_identity():
    frame = _texture(64)
    assert torch.allclose(MC(frame, frame.clone()), frame, atol=1e-5)


def test_alpha_moves_with_the_colour():
    frame, neighbor = _pair(3, 5)
    alpha = torch.linspace(0, 1, neighbor.shape[1]).expand(neighbor.shape[0], -1).unsqueeze(-1)
    aligned = MC(torch.cat([frame, frame[..., :1]], -1), torch.cat([neighbor, alpha], -1))
    assert aligned.shape[-1] == 4
    m = 12
    expected = torch.linspace(0, 1, neighbor.shape[1])[m + 5:-m + 5]
    assert torch.allclose(aligned[m:-m, m:-m, 3][0], expected, atol=1e-4)


@pytest.mark.parametrize("shape", [(1, 1, 3), (5, 7, 3), (2, 40, 1)])
def test_tiny_frames_do_not_fail(shape):
    a = torch.rand(shape)
    assert MC(a, torch.rand(shape)).shape == shape


def test_it_is_far_better_than_no_compensation_on_real_motion():
    frame, neighbor = _pair(4, 7)
    m = 12
    before = (neighbor - frame)[m:-m, m:-m].abs().mean()
    after = (MC(frame, neighbor) - frame)[m:-m, m:-m].abs().mean()
    assert after < 0.02 * before
