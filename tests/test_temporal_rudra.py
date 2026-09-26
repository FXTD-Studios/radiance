"""Phase 3 temporal RUDRA architecture and recovery contract."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from radiance.temporal_rudra import (  # noqa: E402
    TemporalRUDRAResidual,
    load_temporal_rudra_weights,
    recover_temporal_residual,
)


def _sequence(frames: int = 5, height: int = 8, width: int = 8):
    values = torch.linspace(0.0, 1.0, frames).view(1, frames, 1, 1, 1)
    return values.expand(1, frames, height, width, 3).contiguous()


def test_temporal_model_returns_residual_and_two_confidence_maps():
    model = TemporalRUDRAResidual(width=8, blocks=1, max_flow=2.0)
    frames = _sequence()
    masks = torch.zeros(1, 5, 8, 8)
    residual, highlight_confidence, shadow_confidence = model(frames, masks, masks)
    assert residual.shape == (1, 5, 8, 8, 3)
    assert highlight_confidence.shape == (1, 5, 8, 8)
    assert shadow_confidence.shape == (1, 5, 8, 8)
    assert torch.isfinite(residual).all()
    assert 0.0 <= float(highlight_confidence.detach().min()) <= float(highlight_confidence.detach().max()) <= 1.0
    assert 0.0 <= float(shadow_confidence.detach().min()) <= float(shadow_confidence.detach().max()) <= 1.0


def test_temporal_model_rejects_isolated_frames():
    model = TemporalRUDRAResidual(width=8, blocks=1)
    with pytest.raises(ValueError, match="at least three"):
        model(torch.zeros(1, 1, 8, 8, 3),
              torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))


class _FixedTemporalModel:
    def __init__(self, residual: float, highlight_confidence: float,
                 shadow_confidence: float):
        self.residual = residual
        self.highlight_confidence = highlight_confidence
        self.shadow_confidence = shadow_confidence
        self.calls = 0

    def __call__(self, frames, highlight_mask, shadow_mask):
        self.calls += 1
        b, t, h, w, _ = frames.shape
        residual = torch.full((b, t, h, w, 3), self.residual,
                              device=frames.device, dtype=frames.dtype)
        h_conf = torch.full((b, t, h, w), self.highlight_confidence,
                            device=frames.device, dtype=frames.dtype)
        s_conf = torch.full((b, t, h, w), self.shadow_confidence,
                            device=frames.device, dtype=frames.dtype)
        return residual, h_conf, s_conf


def test_low_confidence_uses_deterministic_fallback_and_preserves_clean_pixels():
    source = torch.full((5, 4, 4, 3), 0.25)
    deterministic = torch.full_like(source, 2.0)
    highlights = torch.zeros(5, 4, 4)
    highlights[:, 1, 1] = 1.0
    shadows = torch.zeros_like(highlights)
    model = _FixedTemporalModel(0.5, 0.0, 0.0)
    out, h_conf, s_conf = recover_temporal_residual(
        source, deterministic, highlights, shadows, model, 10.0, 5, True,
    )
    assert torch.equal(out[:, 0, 0], source[:, 0, 0])
    assert torch.allclose(out[:, 1, 1], deterministic[:, 1, 1])
    assert not h_conf.any()
    assert not s_conf.any()
    assert model.calls == 5


def test_high_confidence_applies_residual_only_inside_matching_mask():
    source = torch.full((5, 4, 4, 3), 0.25)
    deterministic = torch.full_like(source, 2.0)
    highlights = torch.zeros(5, 4, 4)
    highlights[:, 1, 1] = 1.0
    shadows = torch.zeros_like(highlights)
    model = _FixedTemporalModel(0.1, 1.0, 0.75)
    out, h_conf, s_conf = recover_temporal_residual(
        source, deterministic, highlights, shadows, model, 10.0, 5, True,
    )
    assert torch.equal(out[:, 0, 0], source[:, 0, 0])
    assert torch.allclose(out[:, 1, 1], torch.full((5, 3), 1.25))
    assert torch.equal(h_conf, highlights)
    assert not s_conf.any()


def test_temporal_checkpoint_round_trip_is_strict(tmp_path):
    model = TemporalRUDRAResidual(width=8, blocks=2, max_flow=3.0)
    path = tmp_path / "temporal_rudra.pth"
    torch.save({
        "model": model.state_dict(),
        "temporal_rudra": {"width": 8, "blocks": 2, "max_flow": 3.0},
    }, path)
    loaded = load_temporal_rudra_weights(str(path), "cpu")
    assert isinstance(loaded, TemporalRUDRAResidual)
    assert loaded.width == 8
    assert loaded.blocks == 2
    assert loaded.max_flow == 3.0


def test_missing_temporal_checkpoint_returns_none(tmp_path):
    assert load_temporal_rudra_weights(str(tmp_path / "missing.pth"), "cpu") is None


def test_training_loss_is_finite_and_backpropagates():
    from radiance.scripts.training.train_temporal_rudra import TemporalRecoveryLoss

    model = TemporalRUDRAResidual(width=8, blocks=1, max_flow=2.0)
    source = _sequence()
    highlights = torch.zeros(1, 5, 8, 8)
    shadows = torch.zeros_like(highlights)
    highlights[:, :, 2:6, 2:6] = 1.0
    target = source + 0.5 * highlights[..., None]
    residual, h_conf, s_conf = model(source, highlights, shadows)
    losses = TemporalRecoveryLoss(10.0)(
        source, target, residual, highlights, shadows, h_conf, s_conf,
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
