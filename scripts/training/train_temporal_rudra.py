"""Train the Phase 3 motion-aligned temporal RUDRA residual model.

Input files are ``.npz`` sequences containing:

``sdr``
    Display-referred sRGB frames, ``[T,H,W,3]`` in ``[0,1]``.
``hdr``
    Registered scene-linear Rec.709 HDR targets with the same shape, where
    linear ``1.0`` represents 100 nits.

Use genuine consecutive HDR/RAW frames and generate varied SDR degradations
before training. Repeated still frames do not provide a temporal signal.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radiance.temporal_rudra import TemporalRUDRAResidual

logger = logging.getLogger("radiance.train_temporal_rudra")


def _srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055).clamp(min=1e-6) ** 2.4,
    )


def _luma(image: torch.Tensor) -> torch.Tensor:
    return (0.2126 * image[..., 0] +
            0.7152 * image[..., 1] +
            0.0722 * image[..., 2])


def _recovery_masks(
    sdr: torch.Tensor,
    linear: torch.Tensor,
    highlight_threshold: float = 0.98,
    shadow_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    evidence = torch.maximum(_luma(sdr), sdr.max(dim=-1).values)
    x = ((evidence - highlight_threshold) /
         max(1.0 - highlight_threshold, 1e-6)).clamp(0.0, 1.0)
    highlights = x * x * (3.0 - 2.0 * x)
    shadows = ((shadow_threshold - _luma(linear)) /
               shadow_threshold).clamp(0.0, 1.0)
    return highlights, shadows


class TemporalHDRSequenceDataset(Dataset):
    def __init__(self, pair_dir: str, window: int = 5, image_size: int = 256):
        self.files = sorted(Path(pair_dir).rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No temporal HDR .npz pairs found in {pair_dir}")
        self.window = int(window)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        with np.load(self.files[index]) as data:
            sdr = torch.from_numpy(data["sdr"].astype(np.float32))
            hdr = torch.from_numpy(data["hdr"].astype(np.float32))
        if sdr.shape != hdr.shape or sdr.ndim != 4 or sdr.shape[-1] != 3:
            raise ValueError(f"Invalid temporal pair shape in {self.files[index]}")
        if sdr.shape[0] < self.window:
            raise ValueError(
                f"{self.files[index]} has {sdr.shape[0]} frames; need {self.window}"
            )
        start = random.randint(0, sdr.shape[0] - self.window)
        sdr = sdr[start:start + self.window].clamp(0.0, 1.0)
        hdr = hdr[start:start + self.window].clamp(min=0.0)
        sdr = F.interpolate(
            sdr.permute(0, 3, 1, 2), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
        hdr = F.interpolate(
            hdr.permute(0, 3, 1, 2), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
        if random.random() < 0.5:
            sdr = torch.flip(sdr, dims=(2,))
            hdr = torch.flip(hdr, dims=(2,))
        linear = _srgb_to_linear(sdr)
        highlights, shadows = _recovery_masks(sdr, linear)
        return linear, hdr, highlights, shadows


class TemporalRecoveryLoss(nn.Module):
    """HDR, detail, temporal, identity, and confidence calibration losses."""

    def __init__(self, peak_scale: float):
        super().__init__()
        self.peak_scale = float(peak_scale)

    @staticmethod
    def _gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image[..., :, 1:, :] - image[..., :, :-1, :], \
               image[..., 1:, :, :] - image[..., :-1, :, :]

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        residual: torch.Tensor,
        highlight_mask: torch.Tensor,
        shadow_mask: torch.Tensor,
        highlight_confidence: torch.Tensor,
        shadow_confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        region = torch.maximum(highlight_mask, shadow_mask)
        proposal = (source + residual * self.peak_scale).clamp(min=0.0)
        prediction = source + region[..., None] * (proposal - source)
        log_prediction = torch.log1p(prediction)
        log_target = torch.log1p(target)
        error = (log_prediction - log_target).abs().mean(dim=-1)

        reconstruction = error.mean()
        highlight = (error * highlight_mask).sum() / highlight_mask.sum().clamp(min=1.0)
        shadow = (error * shadow_mask).sum() / shadow_mask.sum().clamp(min=1.0)
        pred_dx, pred_dy = self._gradients(log_prediction)
        target_dx, target_dy = self._gradients(log_target)
        detail = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
        temporal = F.l1_loss(
            log_prediction[:, 1:] - log_prediction[:, :-1],
            log_target[:, 1:] - log_target[:, :-1],
        )
        identity = ((proposal - source).abs().mean(dim=-1) * (1.0 - region)).mean()

        confidence_target = torch.exp(-4.0 * error.detach()).clamp(0.0, 1.0)
        confidence = (
            F.binary_cross_entropy(
                highlight_confidence, confidence_target, reduction="none",
            ) *
            highlight_mask
        ).sum() / highlight_mask.sum().clamp(min=1.0)
        confidence += (
            F.binary_cross_entropy(
                shadow_confidence, confidence_target, reduction="none",
            ) *
            shadow_mask
        ).sum() / shadow_mask.sum().clamp(min=1.0)

        total = (reconstruction + 2.0 * highlight + 1.5 * shadow +
                 0.2 * detail + 0.5 * temporal + 2.0 * identity +
                 0.1 * confidence)
        return {
            "loss": total,
            "reconstruction": reconstruction.detach(),
            "highlight": highlight.detach(),
            "shadow": shadow.detach(),
            "detail": detail.detach(),
            "temporal": temporal.detach(),
            "identity": identity.detach(),
            "confidence": confidence.detach(),
        }


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    dataset = TemporalHDRSequenceDataset(args.pair_dir, args.window, args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = TemporalRUDRAResidual(
        width=args.width, blocks=args.blocks, max_flow=args.max_flow,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.05,
    )
    loss_fn = TemporalRecoveryLoss(args.peak_nits / 100.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)

    model.train()
    for step in range(1, args.steps + 1):
        try:
            source, target, highlights, shadows = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            source, target, highlights, shadows = next(iterator)
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        highlights = highlights.to(device, non_blocking=True)
        shadows = shadows.to(device, non_blocking=True)
        residual, h_conf, s_conf = model(source, highlights, shadows)
        losses = loss_fn(
            source, target, residual, highlights, shadows, h_conf, s_conf,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            logger.info(
                "step=%d loss=%.5f temporal=%.5f confidence=%.5f",
                step, losses["loss"].item(), losses["temporal"].item(),
                losses["confidence"].item(),
            )
        if step % args.save_every == 0 or step == args.steps:
            path = output_dir / f"temporal_rudra_residual_step{step:06d}.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "temporal_rudra": {
                    "version": 1,
                    "width": args.width,
                    "blocks": args.blocks,
                    "max_flow": args.max_flow,
                    "window": args.window,
                    "peak_nits": args.peak_nits,
                    "prediction": "scene_linear_residual",
                },
            }, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, choices=(5, 7, 9), default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-flow", type=float, default=24.0)
    parser.add_argument("--peak-nits", type=float, default=1000.0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = train(build_parser().parse_args())
    logger.info("Saved final checkpoint: %s", checkpoint)
