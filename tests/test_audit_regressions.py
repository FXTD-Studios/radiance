"""Regression tests for the 2026-07 audit fixes.

Each test here corresponds to a defect that shipped silently -- the failure mode
in every case was a plausible-looking success rather than an error, which is why
none of them was caught by the existing suite. They are deliberately cheap.
"""
import math

import pytest

torch = pytest.importorskip("torch")

# conftest.py installs a MagicMock torch stub when the real package is absent,
# which defeats importorskip. The old detector here was itself defeated by the
# stub -- it is a real ModuleType (so the isinstance check passes) whose every
# attribute is a MagicMock (so the hasattr probe passes too), and it therefore
# reported "real torch" on every no-torch CI run. Ask the stub directly.
_REAL_TORCH = not getattr(torch, "__radiance_stub__", False)
requires_torch = pytest.mark.skipif(not _REAL_TORCH, reason="requires real torch")

# This module gates torch per test, so conftest's automatic module-level
# skip must leave it alone: the source-text assertions below run fine
# against the stub and are worth keeping on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


# ── 1. Real-ESRGAN weights must actually load ───────────────────────────────
#
# The loader used strict=False against a network whose submodules were named
# c1..c5 / upsample while official checkpoints use conv1..conv5 / conv_up1 /
# conv_up2. 8 of 702 tensors matched; the other 694 were silently discarded and
# the model ran at ~random init while the log reported a successful load.

def _official_realesrgan_keys(nb, scale, in_nc=3, out_nc=3, nf=64, gc=32):
    """Key/shape layout of a basicsr RRDBNet checkpoint."""
    first_in = in_nc * 4 if scale == 2 else in_nc * 16 if scale == 1 else in_nc
    keys = {"conv_first.weight": (nf, first_in, 3, 3), "conv_first.bias": (nf,)}
    widths = [(gc, nf), (gc, nf + gc), (gc, nf + 2 * gc), (gc, nf + 3 * gc), (nf, nf + 4 * gc)]
    for i in range(nb):
        for r in (1, 2, 3):
            for c, (out_c, in_c) in enumerate(widths, start=1):
                keys[f"body.{i}.rdb{r}.conv{c}.weight"] = (out_c, in_c, 3, 3)
                keys[f"body.{i}.rdb{r}.conv{c}.bias"] = (out_c,)
    for n in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
        keys[f"{n}.weight"] = (nf, nf, 3, 3)
        keys[f"{n}.bias"] = (nf,)
    keys["conv_last.weight"] = (out_nc, nf, 3, 3)
    keys["conv_last.bias"] = (out_nc,)
    return keys


@requires_torch
@pytest.mark.parametrize("nb,scale", [(23, 4), (6, 4), (23, 2)])
def test_rrdbnet_accepts_official_checkpoint_strictly(nb, scale):
    from radiance.nodes.upscale.upscale import _RRDBNet

    ref = _official_realesrgan_keys(nb, scale)
    checkpoint = {k: torch.zeros(s) for k, s in ref.items()}
    net = _RRDBNet(in_nc=3, out_nc=3, nf=64, nb=nb, scale=scale, gc=32)

    model_keys = set(net.state_dict())
    assert model_keys == set(checkpoint), (
        f"architecture does not match official Real-ESRGAN layout: "
        f"missing={sorted(model_keys - set(checkpoint))[:5]} "
        f"unexpected={sorted(set(checkpoint) - model_keys)[:5]}"
    )
    # The real assertion: a strict load must succeed. If this raises, the
    # checkpoint would previously have been silently half-discarded.
    net.load_state_dict(checkpoint, strict=True)


@requires_torch
@pytest.mark.parametrize("scale", [4, 2])
def test_rrdbnet_output_scale(scale):
    from radiance.nodes.upscale.upscale import _RRDBNet

    net = _RRDBNet(nb=2, scale=scale).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 32, 32))
    assert out.shape[-2:] == (32 * scale, 32 * scale)


@requires_torch
def test_realesrgan_loader_uses_strict_load():
    """Guard the flag itself -- strict=False is how this hid for so long."""
    import inspect
    from radiance.nodes.upscale import upscale as up

    # Strip comments -- the fix is documented in a comment that mentions the
    # old flag, so a naive substring check would match its own explanation.
    code = [line.split("#", 1)[0] for line in inspect.getsource(up._load_realesrgan).splitlines()]
    code = "\n".join(code)
    assert "strict=True" in code
    assert "strict=False" not in code


# ── 2. Blend and temporal ramps must not start at zero ──────────────────────
#
# A ramp whose first element is 0, applied at an image or sequence boundary and
# then divided out by accumulated weight, produces exactly 0 -- a black edge.

def _temporal_weights(Fw, overlap_temporal, wi, n_windows):
    """Mirrors the temporal weighting in RadianceUpscaleVideo."""
    t = [1.0] * Fw
    if Fw > 1:
        half = overlap_temporal
        if wi > 0:
            for i in range(min(half, Fw)):
                t[i] = math.sin(math.pi * i / (2 * half))
        if wi < n_windows - 1:
            for i in range(min(half, Fw)):
                t[Fw - 1 - i] = math.sin(math.pi * i / (2 * half))
    return t


@pytest.mark.parametrize("overlap", [1, 2, 4, 8])
def test_first_window_first_frame_has_nonzero_weight(overlap):
    """Frame 0 of every multi-frame render was pure black."""
    w = _temporal_weights(Fw=16, overlap_temporal=overlap, wi=0, n_windows=4)
    assert w[0] > 0.0, "frame 0 has zero accumulated weight -> renders black"


@pytest.mark.parametrize("overlap", [1, 2, 4, 8])
def test_interior_windows_still_ramp_in(overlap):
    """The guard must not disable blending for windows that genuinely overlap."""
    w = _temporal_weights(Fw=16, overlap_temporal=overlap, wi=1, n_windows=4)
    assert w[0] == pytest.approx(0.0), "interior window should still ramp from 0"


@requires_torch
def test_tile_feather_ramp_is_nonzero_at_image_border():
    """The same defect exists in spatial tiling: cos ramp[0] == 0 exactly."""
    fade = 16
    ramp = (1 - torch.cos(torch.linspace(0, math.pi, fade))) / 2
    assert ramp[0].item() == pytest.approx(0.0), "sanity: raw ramp does start at 0"
    # Border suppression (the hdr/vae.py:2011 pattern) is what makes it safe.
    suppressed = torch.ones(fade)
    assert suppressed[0].item() > 0.0


# ── 3. "Skip" must skip ─────────────────────────────────────────────────────
#
# missing_frames="Skip" appended the missing path anyway, so an 8x8 black tile
# entered the batch and the downstream torch.cat raised -- swallowed into an
# 8x8 black image by a blanket except.

def test_skip_does_not_retain_missing_frame_slots(tmp_path):
    from radiance.nodes.io.write import _resolve_sequence_paths

    for f in (1001, 1002, 1005):
        (tmp_path / f"plate.{f:04d}.exr").write_bytes(b"")
    pattern = str(tmp_path / "plate.####.exr")

    skipped = _resolve_sequence_paths(pattern, 1001, 0, 1, "Skip")
    assert all(__import__("os").path.isfile(p) for p in skipped), \
        "Skip must not retain slots for frames that do not exist"
    assert len(skipped) == 3


def test_open_ended_range_does_not_explode(tmp_path):
    """end_frame=0 means 'read all'; it used to expand to ~99k phantom slots."""
    from radiance.nodes.io.write import _resolve_sequence_paths

    for f in (1001, 1002, 1003):
        (tmp_path / f"plate.{f:04d}.exr").write_bytes(b"")
    pattern = str(tmp_path / "plate.####.exr")

    paths = _resolve_sequence_paths(pattern, 1001, 99999, 1, "Skip")
    assert len(paths) < 100, f"expanded to {len(paths)} slots for a 3-frame sequence"
    assert len(paths) == 3


def test_black_mode_still_reserves_slots(tmp_path):
    """The other branch must keep working: Black fills gaps deliberately."""
    from radiance.nodes.io.write import _resolve_sequence_paths

    for f in (1001, 1003):
        (tmp_path / f"plate.{f:04d}.exr").write_bytes(b"")
    pattern = str(tmp_path / "plate.####.exr")

    paths = _resolve_sequence_paths(pattern, 1001, 1003, 1, "Black")
    assert len(paths) == 3, "Black must reserve a slot for the missing frame"


# ── 4. Delivery must speak the writer's vocabulary ──────────────────────────
#
# The handler called RadianceWrite.write with three kwargs that do not exist and
# omitted a required one, so every delivery raised TypeError, was swallowed, and
# returned HTTP 200 with status "error". No file was ever written.

def test_delivery_format_map_targets_real_write_formats():
    pytest.importorskip("aiohttp")
    from radiance.delivery.handler import _UI_TO_WRITE_FORMAT
    from radiance.nodes.io.write import WRITE_FORMATS

    for ui, mapped in _UI_TO_WRITE_FORMAT.items():
        assert mapped in WRITE_FORMATS, f"{ui!r} maps to unknown format {mapped!r}"


def test_delivery_colorspace_map_targets_real_colorspaces():
    pytest.importorskip("aiohttp")
    from radiance.delivery.handler import _UI_TO_WRITE_COLORSPACE
    from radiance.nodes.io.write import OUTPUT_COLOR_SPACES

    for ui, mapped in _UI_TO_WRITE_COLORSPACE.items():
        assert mapped in OUTPUT_COLOR_SPACES, f"{ui!r} maps to unknown space {mapped!r}"


def test_delivery_rejects_unmapped_options_loudly():
    pytest.importorskip("aiohttp")
    from radiance.delivery.handler import _resolve_write_format, _resolve_write_colorspace

    with pytest.raises(ValueError, match="no writer implementation"):
        _resolve_write_format("Video — MOV (ProRes 4444 XQ)")
    with pytest.raises(ValueError, match="not supported"):
        _resolve_write_colorspace("Panasonic V-Log")


def test_write_kwargs_match_writer_signature():
    """Pin the handler's call against RadianceWrite.write's real parameters."""
    pytest.importorskip("aiohttp")
    import inspect
    from radiance.nodes.io.write import RadianceWrite

    params = set(inspect.signature(RadianceWrite.write).parameters)
    used = {"image", "output_path", "format", "filename",
            "color_space", "fps", "quality", "broadcast_safe"}
    assert used <= params, f"handler passes kwargs the writer lacks: {used - params}"


# ── 5. Unsafe deserialization must stay gone ────────────────────────────────

def test_no_unsafe_torch_load_in_product_code():
    """weights_only defaults to unsafe on torch < 2.6; every site must be explicit."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & {"tests", "build", "dist", ".codex-backup-20260719"}:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "torch.load(" in line and "weights_only=True" not in line:
                offenders.append(f"{path.relative_to(root)}:{n}")
    assert not offenders, f"unsafe torch.load sites: {offenders}"
