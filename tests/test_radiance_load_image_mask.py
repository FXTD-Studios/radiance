"""
tests/test_radiance_load_image_mask.py — Unit tests for RadianceLoadImageMask node.
"""

import sys
import os
import types
import importlib
import tempfile
import hashlib
import math
import pytest
from PIL import Image

# ── ComfyUI stubs ─────────────────────────────────────────────────────────────
for _mod in ["folder_paths", "comfy", "comfy.utils", "node_helpers"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Ensure stubs are set on the active folder_paths module
_fp = sys.modules["folder_paths"]
_fp.get_input_directory = lambda: "/tmp/comfy_input"
_fp.filter_files_content_types = lambda files, types: files
_fp.get_annotated_filepath = lambda name: name
_fp.exists_annotated_filepath = lambda name: True

_nh = sys.modules["node_helpers"]
if not hasattr(_nh, "pillow"):
    _nh.pillow = lambda fn, *args, **kwargs: fn(*args, **kwargs)

# ── Real torch check ──────────────────────────────────────────────────────────
try:
    import torch
    # conftest.py installs a MagicMock stub under the name "torch" when the real
    # package is absent, so a bare successful import proves nothing. The stub
    # tags itself; anything else is the genuine article.
    HAS_TORCH = not getattr(torch, "__radiance_stub__", False)
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def _import_mask():
    # Force reload of mask module to bind to the current sys.modules["folder_paths"]
    sys.modules.pop("radiance.nodes.io.mask", None)
    sys.modules.pop("nodes.io.mask", None)
    
    root = __import__("pathlib").Path(__file__).parent.parent
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        return importlib.import_module("radiance.nodes.io.mask")
    except ImportError:
        sys.path.insert(0, str(root))
        return importlib.import_module("nodes.io.mask")


@skip_no_torch
class TestRadianceLoadImageMask:
    @property
    def fp(self):
        return sys.modules["folder_paths"]

    def setup_method(self):
        self.mod = _import_mask()
        self.cls = self.mod.RadianceLoadImageMask
        self.node = self.cls()

    def test_registration(self):
        """Verify that the node is registered properly."""
        assert "RadianceLoadImageMask" in self.mod.NODE_CLASS_MAPPINGS
        assert self.mod.NODE_DISPLAY_NAME_MAPPINGS["RadianceLoadImageMask"] == "◎ Radiance Load Image"

    def test_api_surface(self):
        """Test signatures of input, outputs, function name, etc."""
        assert self.cls.RETURN_TYPES == ("IMAGE", "MASK")
        assert self.cls.RETURN_NAMES == ("image", "mask")
        assert self.cls.FUNCTION == "load_image"

    def test_validate_inputs(self):
        """Test input validator helper methods."""
        # When folder_paths says it exists
        self.fp.exists_annotated_filepath = lambda name: True
        assert self.cls.VALIDATE_INPUTS("dummy.png") is True

        # When it doesn't exist
        self.fp.exists_annotated_filepath = lambda name: False
        assert "Invalid image file" in self.cls.VALIDATE_INPUTS("missing.png")

    def test_load_image_fallback_no_companion(self):
        """Load image with NO companion _radmask.png, verifying correct shape generation."""
        self.fp.exists_annotated_filepath = lambda name: True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple primary image with transparent alpha channel (RGBA)
            img_path = os.path.join(tmpdir, "primary.png")
            img = Image.new("RGBA", (100, 80), color=(255, 0, 0, 128))
            img.save(img_path)

            # Override get_annotated_filepath to return our temp file
            self.fp.get_annotated_filepath = lambda name: img_path

            # Load it
            image_tensor, mask_tensor = self.node.load_image("primary.png")

            # Check shapes:
            # Image shape: [batch, height, width, channels] -> [1, 80, 100, 3]
            assert image_tensor.ndim == 4
            assert image_tensor.shape == (1, 80, 100, 3)
            # Mask shape: [batch, height, width] -> [1, 80, 100]
            assert mask_tensor.ndim == 3
            assert mask_tensor.shape == (1, 80, 100)

            # Verification of default alpha mask inversion (1.0 - alpha)
            # 128 / 255.0 = 0.50196
            # Inverted alpha should be ~0.498
            assert torch.allclose(mask_tensor[0, 0, 0], torch.tensor(1.0 - 128/255.0), atol=1e-3)

    def test_load_image_with_companion_override(self):
        """Load image WITH a companion _radmask.png and verify companion alpha overrides primary."""
        self.fp.exists_annotated_filepath = lambda name: True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Primary image
            img_path = os.path.join(tmpdir, "primary.png")
            img = Image.new("RGBA", (100, 80), color=(255, 0, 0, 128))
            img.save(img_path)

            # Companion image with different alpha
            comp_path = os.path.join(tmpdir, "primary_radmask.png")
            comp_img = Image.new("RGBA", (100, 80), color=(0, 0, 0, 200))
            comp_img.save(comp_path)

            # Override get_annotated_filepath
            self.fp.get_annotated_filepath = lambda name: img_path

            # Load it
            image_tensor, mask_tensor = self.node.load_image("primary.png")

            # Validate sizes and values
            assert image_tensor.shape == (1, 80, 100, 3)
            assert mask_tensor.shape == (1, 80, 100)

            # Companion mask extracts 'A' channel directly (200 / 255.0 = ~0.784)
            # without inversion in the companion code branch
            assert torch.allclose(mask_tensor[0, 0, 0], torch.tensor(200/255.0), atol=1e-3)

    def test_is_changed_hashing(self):
        """Verify IS_CHANGED accurately digests active components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "primary.png")
            img = Image.new("RGB", (10, 10), color="blue")
            img.save(img_path)

            self.fp.get_annotated_filepath = lambda name: img_path

            # Hash initial state
            h1 = self.cls.IS_CHANGED("primary.png")

            # Touch companion mask file and check if hash changes
            comp_mask_path = os.path.join(tmpdir, "primary_radmask.png")
            with open(comp_mask_path, "wb") as f:
                f.write(b"companion_raster_data")

            h2 = self.cls.IS_CHANGED("primary.png")
            assert h1 != h2

            # Touch companion metadata and check if hash changes again
            comp_meta_path = os.path.join(tmpdir, "primary_radmask_meta.json")
            with open(comp_meta_path, "wb") as f:
                f.write(b'{"polygons":[]}')

            h3 = self.cls.IS_CHANGED("primary.png")
            assert h2 != h3

    def test_is_changed_exception_safety(self):
        """Verify that any exception (missing files, etc.) returns nan safely without raising."""
        self.fp.get_annotated_filepath = lambda name: "/nonexistent/path/primary.png"
        res = self.cls.IS_CHANGED("primary.png")
        assert math.isnan(res)
