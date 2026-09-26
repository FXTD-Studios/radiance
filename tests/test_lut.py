"""
test_lut.py — LUT parsing and interpolation tests.

The static methods under test (_parse_cube, _apply_3d_lut, _apply_1d_lut)
are pure-NumPy functions extracted from radiance.color.lut.RadianceLUTApply.
They are inlined here so the test file can be collected without a live
ComfyUI or torch installation.

Covers:
  • _parse_cube: 3D and 1D .cube file parsing, axis order, comment stripping
  • _apply_3d_lut: tetrahedral interpolation
      - Identity LUT leaves image unchanged
      - Constant LUT maps everything to the constant
      - Primary colour corners map exactly
      - .cube axis order: R varies fastest, B slowest
  • _apply_1d_lut: linear interpolation per channel
  • Integration (torch-guarded): strength blend, missing-file error
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    # Check for real torch: the stub (MagicMock) has no __version__ string
    HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-numpy implementations (copied from radiance.color.lut.RadianceLUTApply)
#  — inlined so this test file has no torch / relative-import dependency.
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cube(path: str):
    size = None
    is_3d = True
    entries = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                is_3d = True
            elif line.upper().startswith("LUT_1D_SIZE"):
                size = int(line.split()[-1])
                is_3d = False
            elif line.upper().startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX")):
                continue
            else:
                try:
                    vals = list(map(float, line.split()))
                    if len(vals) == 3:
                        entries.append(vals)
                except ValueError:
                    continue

    if size is None:
        size = int(round(len(entries) ** (1 / 3))) if is_3d else len(entries)

    table = np.array(entries, dtype=np.float32)
    if is_3d:
        # .cube: R fastest, B slowest → after reshape [B,G,R,c]
        # Transpose to [R,G,B,c] so lookups use (R,G,B) order.
        table = table.reshape(size, size, size, 3).transpose(2, 1, 0, 3).copy()
    return table, size, is_3d


def _apply_3d_lut(img: np.ndarray, table: np.ndarray, size: int) -> np.ndarray:
    H, W, _ = img.shape
    s1 = size - 1

    r = np.clip(img[..., 0], 0.0, 1.0) * s1
    g = np.clip(img[..., 1], 0.0, 1.0) * s1
    b = np.clip(img[..., 2], 0.0, 1.0) * s1

    r0 = np.floor(r).astype(np.int32).clip(0, s1 - 1)
    g0 = np.floor(g).astype(np.int32).clip(0, s1 - 1)
    b0 = np.floor(b).astype(np.int32).clip(0, s1 - 1)
    r1 = (r0 + 1).clip(0, s1)
    g1 = (g0 + 1).clip(0, s1)
    b1 = (b0 + 1).clip(0, s1)

    dr = (r - r0)[..., np.newaxis]
    dg = (g - g0)[..., np.newaxis]
    db = (b - b0)[..., np.newaxis]

    c000 = table[r0, g0, b0]
    c100 = table[r1, g0, b0]
    c010 = table[r0, g1, b0]
    c110 = table[r1, g1, b0]
    c001 = table[r0, g0, b1]
    c101 = table[r1, g0, b1]
    c011 = table[r0, g1, b1]
    c111 = table[r1, g1, b1]

    mask_rg = dr >= dg
    mask_gb = dg >= db
    mask_rb = dr >= db

    out = np.where(
        mask_rg & mask_gb,
        c000 + dr * (c100 - c000) + dg * (c110 - c100) + db * (c111 - c110),
        np.where(
            mask_rg & mask_rb,
            c000 + dr * (c100 - c000) + db * (c101 - c100) + dg * (c111 - c101),
            np.where(
                mask_rb & ~mask_rg,
                c000 + db * (c001 - c000) + dr * (c101 - c001) + dg * (c111 - c101),
                np.where(
                    ~mask_rb & ~mask_gb,
                    c000 + db * (c001 - c000) + dg * (c011 - c001) + dr * (c111 - c011),
                    np.where(
                        ~mask_rg & mask_gb,
                        c000 + dg * (c010 - c000) + db * (c011 - c010) + dr * (c111 - c011),
                        c000 + dg * (c010 - c000) + dr * (c110 - c010) + db * (c111 - c110),
                    ),
                ),
            ),
        ),
    ).astype(np.float32)

    return out


def _apply_1d_lut(img: np.ndarray, table: np.ndarray, size: int) -> np.ndarray:
    out = np.empty_like(img)
    s1 = size - 1
    for c in range(3):
        x = np.clip(img[..., c], 0.0, 1.0) * s1
        x0 = np.floor(x).astype(np.int32).clip(0, s1 - 1)
        x1 = (x0 + 1).clip(0, s1)
        t = x - x0
        out[..., c] = table[x0, c] * (1.0 - t) + table[x1, c] * t
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers to build in-memory .cube files
# ─────────────────────────────────────────────────────────────────────────────

def _write_identity_3d_cube(path: str, size: int = 4):
    """Identity 3D LUT: R varies fastest, B slowest."""
    s1 = size - 1
    lines = [f"# Identity {size}x{size}x{size}", f"LUT_3D_SIZE {size}", ""]
    for b_idx in range(size):
        for g_idx in range(size):
            for r_idx in range(size):
                r, g, b = r_idx / s1, g_idx / s1, b_idx / s1
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_constant_3d_cube(path: str, size: int, rgb: tuple):
    lines = [f"LUT_3D_SIZE {size}", ""]
    for _ in range(size ** 3):
        lines.append(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_identity_1d_cube(path: str, size: int = 16):
    s1 = size - 1
    lines = [f"LUT_1D_SIZE {size}", ""]
    for i in range(size):
        v = i / s1
        lines.append(f"{v:.6f} {v:.6f} {v:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  _parse_cube
# ─────────────────────────────────────────────────────────────────────────────

class TestParseCube:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_3d_size_detected(self, tmp_path):
        p = str(tmp_path / "id.cube")
        _write_identity_3d_cube(p, size=4)
        table, size, is_3d = _parse_cube(p)
        assert is_3d is True
        assert size == 4
        assert table.shape == (4, 4, 4, 3)

    def test_1d_size_detected(self, tmp_path):
        p = str(tmp_path / "id1d.cube")
        _write_identity_1d_cube(p, size=16)
        table, size, is_3d = _parse_cube(p)
        assert is_3d is False
        assert size == 16
        assert table.shape == (16, 3)

    def test_3d_identity_values(self, tmp_path):
        """Entry [r_idx, g_idx, b_idx] == normalised (r, g, b)."""
        p = str(tmp_path / "id.cube")
        _write_identity_3d_cube(p, size=4)
        table, size, _ = _parse_cube(p)
        np.testing.assert_allclose(table[0, 0, 0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(table[3, 0, 0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(table[0, 3, 0], [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(table[0, 0, 3], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(table[3, 3, 3], [1.0, 1.0, 1.0], atol=1e-5)

    def test_comments_and_blank_lines_ignored(self, tmp_path):
        p = str(tmp_path / "commented.cube")
        with open(p, "w") as f:
            f.write("# comment\nTITLE \"Test\"\n\nLUT_3D_SIZE 2\n# skip\n"
                    "0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n"
                    "0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n")
        table, size, is_3d = _parse_cube(p)
        assert size == 2 and is_3d is True
        assert table.shape == (2, 2, 2, 3)

    def test_float32_dtype(self, tmp_path):
        p = str(tmp_path / "id.cube")
        _write_identity_3d_cube(p, size=2)
        table, _, _ = _parse_cube(p)
        assert table.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
#  _apply_3d_lut — Tetrahedral interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _make_identity_table(size: int) -> np.ndarray:
    s1 = size - 1
    table = np.zeros((size, size, size, 3), dtype=np.float32)
    for b_i in range(size):
        for g_i in range(size):
            for r_i in range(size):
                table[r_i, g_i, b_i] = [r_i / s1, g_i / s1, b_i / s1]
    return table


class TestApply3DLUT:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_identity_roundtrip(self):
        """Identity LUT must leave any image unchanged."""
        size = 8
        table = _make_identity_table(size)
        rng = np.random.default_rng(0)
        img = rng.random((16, 16, 3)).astype(np.float32)
        out = _apply_3d_lut(img, table, size)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_constant_lut(self):
        """LUT mapped to (0.5, 0.3, 0.1) everywhere → same constant out."""
        size = 4
        rgb = (0.5, 0.3, 0.1)
        table = np.tile(np.array(rgb, dtype=np.float32), (size, size, size, 1))
        img = np.random.default_rng(1).random((8, 8, 3)).astype(np.float32)
        out = _apply_3d_lut(img, table, size)
        np.testing.assert_allclose(out, np.full_like(out, rgb), atol=1e-5)

    def test_corner_black(self):
        size = 4
        img = np.zeros((1, 1, 3), dtype=np.float32)
        out = _apply_3d_lut(img, _make_identity_table(size), size)
        np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_corner_white(self):
        size = 4
        img = np.ones((1, 1, 3), dtype=np.float32)
        out = _apply_3d_lut(img, _make_identity_table(size), size)
        np.testing.assert_allclose(out[0, 0], [1.0, 1.0, 1.0], atol=1e-6)

    def test_primary_red_corner(self):
        size = 4
        img = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        out = _apply_3d_lut(img, _make_identity_table(size), size)
        np.testing.assert_allclose(out[0, 0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_primary_green_corner(self):
        size = 4
        img = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        out = _apply_3d_lut(img, _make_identity_table(size), size)
        np.testing.assert_allclose(out[0, 0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_primary_blue_corner(self):
        size = 4
        img = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
        out = _apply_3d_lut(img, _make_identity_table(size), size)
        np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 1.0], atol=1e-6)

    def test_cube_axis_order_r_fastest(self):
        """
        .cube stores R varying fastest.  Build a LUT where only R changes along
        [r_idx,0,0]: output R = r_idx/(size-1), output G=B=0.
        Input (0.5, 0, 0) should yield R≈0.5, G=0, B=0.
        """
        size = 4
        s1 = size - 1
        table = np.zeros((size, size, size, 3), dtype=np.float32)
        for r_i in range(size):
            table[r_i, 0, 0, 0] = r_i / s1
        img = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)
        out = _apply_3d_lut(img, table, size)
        assert out[0, 0, 0] == pytest.approx(0.5, abs=0.02)
        assert out[0, 0, 1] == pytest.approx(0.0, abs=1e-5)
        assert out[0, 0, 2] == pytest.approx(0.0, abs=1e-5)

    def test_output_shape_and_dtype(self):
        size = 4
        table = _make_identity_table(size)
        img = np.random.default_rng(2).random((32, 48, 3)).astype(np.float32)
        out = _apply_3d_lut(img, table, size)
        assert out.shape == (32, 48, 3)
        assert out.dtype == np.float32

    def test_larger_lut_size(self):
        """33×33×33 is the standard professional LUT size."""
        size = 33
        table = _make_identity_table(size)
        rng = np.random.default_rng(5)
        img = rng.random((8, 8, 3)).astype(np.float32)
        out = _apply_3d_lut(img, table, size)
        np.testing.assert_allclose(out, img, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
#  _apply_1d_lut — Linear interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _make_identity_1d_table(size: int) -> np.ndarray:
    s1 = size - 1
    t = np.zeros((size, 3), dtype=np.float32)
    for i in range(size):
        t[i] = [i / s1, i / s1, i / s1]
    return t


class TestApply1DLUT:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_identity_roundtrip(self):
        size = 32
        table = _make_identity_1d_table(size)
        img = np.random.default_rng(3).random((8, 8, 3)).astype(np.float32)
        out = _apply_1d_lut(img, table, size)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_invert_1d(self):
        """Inverted 1D LUT: 0→1, 1→0."""
        size = 16
        s1 = size - 1
        table = np.zeros((size, 3), dtype=np.float32)
        for i in range(size):
            v = 1.0 - i / s1
            table[i] = [v, v, v]
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = _apply_1d_lut(img, table, size)
        np.testing.assert_allclose(out[0, 0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(out[0, 0, 1], 0.5, atol=1e-4)
        np.testing.assert_allclose(out[0, 0, 2], 0.0, atol=1e-5)

    def test_channel_independence(self):
        """1D LUT with doubled R column leaves G/B unchanged."""
        size = 16
        table = _make_identity_1d_table(size)
        table[:, 0] = np.minimum(table[:, 0] * 2.0, 1.0)
        img = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
        out = _apply_1d_lut(img, table, size)
        assert out[0, 0, 0] == pytest.approx(0.5,  abs=1e-4)
        assert out[0, 0, 1] == pytest.approx(0.5,  abs=1e-4)
        assert out[0, 0, 2] == pytest.approx(0.75, abs=1e-4)

    def test_output_shape_and_dtype(self):
        size = 8
        table = _make_identity_1d_table(size)
        img = np.random.default_rng(4).random((10, 15, 3)).astype(np.float32)
        out = _apply_1d_lut(img, table, size)
        assert out.shape == (10, 15, 3)
        assert out.dtype == np.float32

    def test_boundary_at_zero(self):
        """Input 0.0 → exactly table[0]."""
        size = 8
        table = _make_identity_1d_table(size)
        img = np.zeros((1, 1, 3), dtype=np.float32)
        out = _apply_1d_lut(img, table, size)
        np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_boundary_at_one(self):
        """Input 1.0 → exactly table[-1]."""
        size = 8
        table = _make_identity_1d_table(size)
        img = np.ones((1, 1, 3), dtype=np.float32)
        out = _apply_1d_lut(img, table, size)
        np.testing.assert_allclose(out[0, 0], [1.0, 1.0, 1.0], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  Integration tests — require real torch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestApplyLUTIntegration:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    End-to-end tests that exercise radiance.color.lut.RadianceLUTApply.apply_lut,
    which wraps the numpy static methods with batch processing, strength
    blending, and alpha pass-through.
    """

    @pytest.fixture(scope="class")
    def node(self):
        # Import lazily — relative imports inside the package require the
        # full radiance package to be importable.  Under pytest the cwd is
        # the radiance root so this normally works when torch is available.
        try:
            from radiance.color.lut import RadianceLUTApply
        except ImportError:
            from radiance.color.lut import RadianceLUTApply
        return RadianceLUTApply()

    def test_strength_zero_returns_original(self, node, tmp_path):
        p = str(tmp_path / "const.cube")
        _write_constant_3d_cube(p, size=4, rgb=(0.5, 0.5, 0.5))
        import torch
        img = torch.rand(1, 8, 8, 3)
        result = node.apply_lut(img, p, strength=0.0)[0]
        np.testing.assert_allclose(result.cpu().numpy(), img.cpu().numpy(), atol=1e-5)

    def test_strength_one_fully_applies(self, node, tmp_path):
        p = str(tmp_path / "const.cube")
        rgb = (0.2, 0.5, 0.8)
        _write_constant_3d_cube(p, size=4, rgb=rgb)
        import torch
        img = torch.rand(1, 4, 4, 3)
        result = node.apply_lut(img, p, strength=1.0)[0]
        expected = np.full((1, 4, 4, 3), rgb, dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected, atol=1e-5)

    def test_missing_file_raises(self, node, tmp_path):
        import torch
        with pytest.raises(FileNotFoundError):
            node.apply_lut(torch.rand(1, 4, 4, 3), str(tmp_path / "nope.cube"))
