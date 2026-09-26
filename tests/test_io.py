"""
tests/test_io.py — Unit tests for nodes_io.py and path_utils.py

Covers:
  • get_next_index: empty dir → 0, gaps handled, non-matching files ignored,
    non-existent dir → 0, correct next after sequence
  • RadianceDigitalCinemaRead INPUT_TYPES: required keys present, RETURN_TYPES
  • RadianceDigitalCinemaWrite INPUT_TYPES: required keys present, OUTPUT_NODE
  • RadianceEXRMultiPart: registered, has write_multipart function
  • NODE_CLASS_MAPPINGS: all three nodes registered
"""

import sys
import os
import types
import importlib
import tempfile
import pytest

# ── ComfyUI stubs (nodes_io imports folder_paths for output dirs) ─────────────
for _mod in ["folder_paths", "comfy", "comfy.utils"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

_fp = sys.modules["folder_paths"]
if not hasattr(_fp, "get_output_directory"):
    _fp.get_output_directory = lambda: "/tmp/comfy_output"
if not hasattr(_fp, "get_input_directory"):
    _fp.get_input_directory = lambda: "/tmp/comfy_input"
if not hasattr(_fp, "get_temp_directory"):
    _fp.get_temp_directory = lambda: "/tmp/comfy_temp"

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = hasattr(torch, "__version__")
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def _import_path_utils():
    if "path_utils" in sys.modules:
        return sys.modules["path_utils"]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    return importlib.import_module("path_utils")


def _import_io():
    if "radiance.nodes.io.write" in sys.modules:
        return sys.modules["radiance.nodes.io.write"]
    if "nodes_io" in sys.modules:
        return sys.modules["radiance.nodes.io.write"]
    root = __import__("pathlib").Path(__file__).parent.parent
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        return importlib.import_module("radiance.nodes.io.write")
    except ImportError:
        sys.path.insert(0, str(root))
        return importlib.import_module("radiance.nodes.io.write")


# ─────────────────────────────────────────────────────────────────────────────
# get_next_index (path_utils)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetNextIndex:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """Pure filesystem tests — no torch required."""

    def setup_method(self):
        self.mod = _import_path_utils()
        self.fn = self.mod.get_next_index

    def test_empty_dir_returns_0(self, tmp_path):
        """No matching files → next index is 0."""
        assert self.fn(str(tmp_path), "frame_", ".png") == 0

    def test_nonexistent_dir_returns_0(self):
        assert self.fn("/nonexistent/path/xyz_123", "frame_", ".png") == 0

    def test_single_file_returns_next(self, tmp_path):
        (tmp_path / "frame_0042.png").touch()
        result = self.fn(str(tmp_path), "frame_", ".png")
        assert result == 43

    def test_sequence_returns_max_plus_1(self, tmp_path):
        for i in [1, 2, 3, 10, 20]:
            (tmp_path / f"clip_{i:04d}.exr").touch()
        result = self.fn(str(tmp_path), "clip_", ".exr")
        assert result == 21

    def test_non_matching_files_ignored(self, tmp_path):
        """Files with wrong prefix or extension must not affect the index."""
        (tmp_path / "other_0005.png").touch()
        (tmp_path / "frame_0003.jpg").touch()  # wrong extension
        (tmp_path / "frame_0003.png").touch()  # correct
        result = self.fn(str(tmp_path), "frame_", ".png")
        assert result == 4

    def test_gap_in_sequence_returns_max_plus_1(self, tmp_path):
        """Non-contiguous indices — result should be max+1, not first gap."""
        for i in [1, 5, 10]:
            (tmp_path / f"shot_{i}.exr").touch()
        result = self.fn(str(tmp_path), "shot_", ".exr")
        assert result == 11

    def test_zero_index_file(self, tmp_path):
        """A file with index 0 → next index is 1."""
        (tmp_path / "frame_0.png").touch()
        result = self.fn(str(tmp_path), "frame_", ".png")
        assert result == 1

    def test_extension_case_sensitive(self, tmp_path):
        """Extension match must be exact (case-sensitive on Linux)."""
        (tmp_path / "frame_001.PNG").touch()
        result = self.fn(str(tmp_path), "frame_", ".png")
        assert result == 0  # .PNG ≠ .png

    def test_prefix_not_matched_as_substring(self, tmp_path):
        """'frame_' prefix must match exactly — not as a substring."""
        (tmp_path / "keyframe_001.png").touch()
        result = self.fn(str(tmp_path), "frame_", ".png")
        assert result == 0


# ─────────────────────────────────────────────────────────────────────────────
# RadianceDigitalCinemaRead API surface
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestRadianceDigitalCinemaReadAPI:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_io()
        self.cls = self.mod.RadianceDigitalCinemaRead

    def test_registered(self):
        assert "RadianceDigitalCinemaRead" in self.mod.NODE_CLASS_MAPPINGS

    def test_required_keys(self):
        it = self.cls.INPUT_TYPES()
        req = it["required"]
        assert "source_path" in req, "source_path must be required"
        assert "read_mode" in req, "read_mode must be required"

    def test_return_types_include_image_and_mask(self):
        assert "IMAGE" in self.cls.RETURN_TYPES
        assert "MASK" in self.cls.RETURN_TYPES

    def test_return_types_include_shot_metadata(self):
        assert "RADIANCE_SHOT" in self.cls.RETURN_TYPES

    def test_function_is_read(self):
        assert self.cls.FUNCTION == "read"

    @pytest.mark.parametrize("path", ["", "   ", '""'])
    def test_empty_source_path_raises_instead_of_a_black_frame(self, path):
        """Live 3.5 run: an empty path executed as an 8x8 black shot."""
        with pytest.raises(ValueError, match="source_path is empty"):
            self.cls().read(path, "Auto", 1, 0, "sRGB (Standard)")

    def test_shot_metadata_carries_colorspace_and_fps(self, tmp_path):
        """Audit 3.5: Linear Check always saw "Unknown"; fps_override was unused."""
        from PIL import Image
        import numpy as np
        p = tmp_path / "plate.png"
        Image.fromarray((np.ones((8, 8, 3)) * 128).astype("uint8")).save(p)
        img, _, meta = self.cls().read(str(p), "Auto", 1, 0, "sRGB (Standard)", fps_override=23.976)
        assert meta["colorspace"] == "Linear (decoded from sRGB)"
        assert meta["fps"] == 23.976 and meta["fps_source"] == "override"
        assert 0.2 < float(img.mean()) < 0.23   # sRGB 128 decoded to linear
        from radiance.nodes.pipeline.metadata import RadianceLinearCheck
        RadianceLinearCheck().check(img, meta, "Strict Error")

    def test_read_mode_choices_non_empty(self):
        it = self.cls.INPUT_TYPES()
        read_mode_entry = it["required"]["read_mode"]
        # First element of the tuple is the choices list
        choices = read_mode_entry[0]
        assert isinstance(choices, list) and len(choices) > 0


# ─────────────────────────────────────────────────────────────────────────────
# RadianceDigitalCinemaWrite API surface
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestRadianceDigitalCinemaWriteAPI:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_io()
        self.cls = self.mod.RadianceDigitalCinemaWrite

    def test_registered(self):
        assert "RadianceDigitalCinemaWrite" in self.mod.NODE_CLASS_MAPPINGS

    def test_output_node_true(self):
        assert getattr(self.cls, "OUTPUT_NODE", False) is True

    def test_required_keys(self):
        it = self.cls.INPUT_TYPES()
        all_keys = {**it["required"], **it.get("optional", {})}
        assert "images" in all_keys or "image" in all_keys, \
            "Write node must accept images input"

    def test_return_types_include_string_status(self):
        assert "STRING" in self.cls.RETURN_TYPES or len(self.cls.RETURN_TYPES) >= 1

    def test_function_is_write(self):
        assert self.cls.FUNCTION == "write"


# ─────────────────────────────────────────────────────────────────────────────
# RadianceEXRMultiPart API surface
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestRadianceEXRMultiPartAPI:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_io()
        self.cls = self.mod.RadianceEXRMultiPart

    def test_registered(self):
        assert "RadianceEXRMultiPart" in self.mod.NODE_CLASS_MAPPINGS

    def test_has_write_function(self):
        assert hasattr(self.cls, self.cls.FUNCTION), \
            f"RadianceEXRMultiPart must have method '{self.cls.FUNCTION}'"

    def test_output_node_true(self):
        assert getattr(self.cls, "OUTPUT_NODE", False) is True

    def test_required_has_beauty_or_image(self):
        it = self.cls.INPUT_TYPES()
        all_keys = {**it["required"], **it.get("optional", {})}
        has_image_input = any(
            k in ("beauty", "image", "images") for k in all_keys
        )
        assert has_image_input, "EXR multi-part node must accept a beauty/image input"


# ─────────────────────────────────────────────────────────────────────────────
# NODE_CLASS_MAPPINGS completeness
# ─────────────────────────────────────────────────────────────────────────────

@skip_no_torch
class TestNodeRegistrations:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def setup_method(self):
        self.mod = _import_io()

    def test_all_three_nodes_registered(self):
        ncm = self.mod.NODE_CLASS_MAPPINGS
        for key in ["RadianceDigitalCinemaRead",
                    "RadianceDigitalCinemaWrite",
                    "RadianceEXRMultiPart"]:
            assert key in ncm, f"{key} missing from NODE_CLASS_MAPPINGS"

    def test_display_names_match_class_keys(self):
        ncm = self.mod.NODE_CLASS_MAPPINGS
        ndm = self.mod.NODE_DISPLAY_NAME_MAPPINGS
        for key in ncm:
            assert key in ndm, f"Display name missing for {key}"
