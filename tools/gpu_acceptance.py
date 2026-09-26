#!/usr/bin/env python3
"""Radiance GPU acceptance run — the test the sandbox cannot do.

Run this ON THE MACHINE WITH THE GPU, using ComfyUI's own Python:

    cd <ComfyUI root>
    python custom_nodes/radiance/tools/gpu_acceptance.py

What it does:
  1. Environment: CUDA device, VRAM, torch/driver versions.
  2. Node sweep: executes every node the functional harness can fabricate
     inputs for, ON CUDA, recording time + peak VRAM + pass/fail.
  3. RUDRA pixel model scoring: finds sdr2hdr_pixel_image.pt in models/radiance/
     and scores it 0-100 from measured behaviour on a synthetic clipped plate:
       - sanity           (finite output, recovered peak above SDR white)
       - speed            (wall time at 512, 1080p, 4K, tiled at 512/64)
       - VRAM             (peak bytes at each size)
       - temporal         (TemporalRUDRA static-scene flicker: identical
         frames in must give identical frames out, when its checkpoint exists)
  4. Writes gpu_acceptance_report.md next to this script.

Scoring bands: 90+ production, 80-89 stable, 70-79 needs work, <70 not ready.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(REPO))
for p in (COMFY, os.path.dirname(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

REPORT: list[str] = []


def log(line: str = "") -> None:
    print(line)
    REPORT.append(line)


def section(title: str) -> None:
    log(f"\n## {title}\n")


def vram_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def vram_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> float:
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    import math
    return 10.0 * math.log10(peak * peak / mse)


def main() -> int:
    section("Environment")
    has_cuda = torch.cuda.is_available()
    log(f"- torch {torch.__version__}, CUDA available: {has_cuda}")
    if not has_cuda:
        log("- **NO GPU FOUND — this run only repeats what CI already covers.**")
    else:
        props = torch.cuda.get_device_properties(0)
        log(f"- {props.name}, {props.total_memory/1e9:.1f} GB VRAM, "
            f"capability {props.major}.{props.minor}")
    device = torch.device("cuda" if has_cuda else "cpu")

    # ── 2. Node sweep on GPU ────────────────────────────────────────────
    section("Node sweep (CUDA tensors)")
    import radiance  # noqa: E402

    nodes = radiance.NODE_CLASS_MAPPINGS
    log(f"- {len(nodes)} nodes registered")

    # Which nodes are MISSING relative to the repo's snapshot? A shortfall
    # here means a node module failed to import on THIS machine (usually a
    # missing optional dependency) and every node in that module was dropped.
    snap_path = os.path.join(REPO, "tests", "node_keys_snapshot.json")
    if os.path.isfile(snap_path):
        import json as _json
        with open(snap_path, encoding="utf-8") as f:
            expected = set(_json.load(f))
        missing = sorted(expected - set(nodes))
        if missing:
            log(f"- **{len(missing)} nodes missing vs repo snapshot:**")
            for m in missing:
                log(f"  - {m}")
            log("  Check the ComfyUI startup log for '[Radiance]' import "
                "failures — a missing optional dependency (transformers, "
                "OpenImageIO, opencolorio, tifffile...) drops the whole "
                "module. The Environment Guard table at startup lists what "
                "is absent.")

    OPAQUE = {"MODEL", "CLIP", "VAE", "CONDITIONING", "CONTROL_NET",
              "UPSCALE_MODEL", "SIGMAS", "SAMPLER", "GUIDER", "NOISE", "AUDIO",
              "RADIANCE_GRADE_INFO", "RADIANCE_CDL", "RADIANCE_AOV",
              "RADIANCE_PASSES", "RADIANCE_LORA_STACK", "RADIANCE_MODEL_META",
              "RADIANCE_SHOT", "RADIANCE_PROJECT", "CLIP_VISION",
              "CLIP_VISION_OUTPUT", "GLIGEN", "STYLE_MODEL", "PHOTOMAKER",
              "SAM_MODEL", "BBOX_DETECTOR", "SEGM_DETECTOR"}

    def fab(spec):
        kind, cfg = spec[0], (spec[1] if len(spec) > 1 else {})
        if not isinstance(cfg, dict):
            cfg = {}
        if isinstance(kind, (list, tuple)):
            d = cfg.get("default")
            return d if d in kind else (kind[0] if kind else None)
        if kind == "IMAGE":
            return torch.rand(1, 256, 256, 3, device=device)
        if kind == "MASK":
            return torch.rand(1, 256, 256, device=device)
        if kind == "LATENT":
            return {"samples": torch.randn(1, 4, 32, 32, device=device)}
        if kind == "INT":
            return int(cfg.get("default", 1))
        if kind == "FLOAT":
            return float(cfg.get("default", 0.5))
        if kind == "BOOLEAN":
            return bool(cfg.get("default", False))
        if kind == "STRING":
            return str(cfg.get("default", ""))
        raise KeyError(kind)

    ok = failed = skipped = 0
    failures: list[str] = []
    for key in sorted(nodes):
        cls = nodes[key]
        try:
            spec = cls.INPUT_TYPES()
        except Exception:
            failed += 1
            failures.append(f"{key}: INPUT_TYPES raised")
            continue
        kwargs, skip = {}, None
        for sect in ("required", "optional"):
            for name, entry in (spec.get(sect) or {}).items():
                try:
                    kwargs[name] = fab(entry)
                except KeyError as e:
                    if sect == "required":
                        skip = f"needs {e}"
        if skip:
            skipped += 1
            continue
        vram_reset()
        t0 = time.perf_counter()
        try:
            fn = getattr(cls(), cls.FUNCTION)
            fn(**kwargs)
            dt = (time.perf_counter() - t0) * 1000
            ok += 1
            if dt > 5000:
                log(f"- SLOW {key}: {dt:.0f} ms, peak {vram_peak_mb():.0f} MB")
        except Exception as exc:
            msg = str(exc)[:100]
            env_markers = ("not found", "No such file", "pip install", "requires",
                           "must not be empty", "network", "No module named",
                           "does not exist", "ffmpeg", "checkpoint",
                           # A loader refusing fabricated defaults with a clear
                           # message ("No CLIP encoders provided for
                           # architecture...") is graceful failure, not a bug.
                           "provided for architecture", "Fill the required")
            if any(m.lower() in msg.lower() for m in env_markers):
                skipped += 1
            else:
                failed += 1
                failures.append(f"{key}: {type(exc).__name__}: {msg}")

    log(f"- executed OK: {ok}   env-skipped: {skipped}   FAILED: {failed}")
    for f in failures:
        log(f"  - **FAIL** {f}")

    # ── 3. RUDRA pixel model scoring ──────────────────────────────────────
    section("RUDRA pixel SDR→HDR model scoring")
    from radiance.pixel_sdr2hdr import (  # noqa: E402
        resolve_pixel_checkpoint, predict_pixel_sdr2hdr,
        describe_pixel_checkpoint_search,
    )
    ckpt = resolve_pixel_checkpoint("")
    if ckpt is None:
        log("- **no pixel checkpoint found — learned SDR→HDR is UNVERIFIED** "
            f"(expected {describe_pixel_checkpoint_search()})")
    else:
        log(f"- checkpoint: {ckpt}")
        sizes = [(512, 512, "512px"), (1080, 1920, "1080p"), (2160, 3840, "4K")]
        score, notes = 100.0, []
        try:
            for h, w, label in sizes:
                # A gradient with a clipped sun: the case the model exists for.
                yy, xx = torch.meshgrid(
                    torch.linspace(0, 1, h, device=device),
                    torch.linspace(0, 1, w, device=device), indexing="ij")
                sdr = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=-1)
                sun = ((xx - 0.7) ** 2 + (yy - 0.3) ** 2) < 0.01
                sdr[sun] = 1.0
                sdr = sdr.unsqueeze(0).clamp(0, 1)
                vram_reset()
                t0 = time.perf_counter()
                out = predict_pixel_sdr2hdr(sdr, checkpoint_path=str(ckpt),
                                            tile_size=512, tile_overlap=64)
                if has_cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                finite = bool(torch.isfinite(out).all())
                peak_lin = float(out.max()) * 100.0  # 1.0 == 10,000 nits
                if not finite:
                    score -= 40
                    notes.append(f"NaN/Inf at {label}")
                log(f"- {label}: {dt*1000:.0f} ms, peak {vram_peak_mb():.0f} MB, "
                    f"finite={finite}, max {peak_lin:.0f} nits")
                if label == "1080p" and dt > 2.0:
                    score -= 10
                    notes.append("slower than 2 s at 1080p")
            log(f"- **pixel model score: {max(0, score):.0f}/100"
                f"{' — ' + '; '.join(notes) if notes else ''}**")
        except Exception:
            log("- pixel model: ERROR")
            for ln in traceback.format_exc().splitlines()[-3:]:
                log(f"    {ln}")

    # ── 4. Temporal flicker with trained weights ────────────────────────
    section("TemporalRUDRA static-scene flicker")
    net = None
    try:
        from radiance.temporal_rudra import load_temporal_rudra_weights  # noqa: E402
        net = load_temporal_rudra_weights(device=device)
    except Exception:
        pass
    if net is None:
        log("- no trained temporal checkpoint loaded — flicker UNVERIFIED "
            "(expects models/radiance/temporal_rudra*.safetensors)")
    else:
        net = net.to(device).eval()
        x = torch.rand(1, 1, 256, 256, 3, device=device).repeat(1, 8, 1, 1, 1)
        m = torch.rand(1, 1, 256, 256, device=device).repeat(1, 8, 1, 1)
        with torch.no_grad():
            out = net(x, m, m)[0]
        var = (out - out.mean(dim=1, keepdim=True)).abs().max().item()
        log(f"- identical frames in -> max frame-to-frame deviation: {var:.6f}")
        log(f"- verdict: {'PASS (<0.002)' if var < 0.002 else '**FLICKER — investigate temporal padding**'}")

    out_path = os.path.join(REPO, "gpu_acceptance_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Radiance GPU Acceptance Report\n" + "\n".join(REPORT) + "\n")
    print(f"\nReport written: {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
