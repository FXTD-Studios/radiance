"""
tests/test_sampler_preview.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Regression tests for the preview_method widget on RadianceSamplerPro.

Pins four compounding defects that made the widget inert and broke
ComfyUI's own preview:

  1. The "TAESD" branch imported `TAESDDecoder` from comfy.taesd.taesd.
     That name exists in neither ComfyUI 0.32.0 nor 0.36.0 (the module
     exports TAESD), so TAESD always downgraded to Latent2RGB.
  2. Neither branch decoded anything — the imported symbol was unused and
     both values produced the same callback.
  3. The callback sent `(x0,)`, a 1-tuple of raw latents, where the
     protocol wants (format, PIL.Image, max_size).  server.py's
     send_image_with_metadata() indexes image_data[2], so every previewed
     step raised IndexError server side.
  4. disable_pbar was wired to a flag that was True whenever the widget
     was not "None", so a failed previewer also killed Comfy's own bar.
"""

from __future__ import annotations

import enum
import inspect
import sys
import types
import unittest
from unittest.mock import MagicMock

from radiance.nodes.generate.sampler import (
    RadianceSamplerPro,
    _make_phase_preview_callback,
    _resolve_latent_previewer,
)


# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI latent_preview / cli_args stand-ins
# ─────────────────────────────────────────────────────────────────────────────

class _LatentPreviewMethod(enum.Enum):
    """Verbatim copy of comfy.cli_args.LatentPreviewMethod (0.32.0 = 0.36.0)."""

    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        return None


class _FakePreviewer:
    """Stands in for latent_preview.LatentPreviewer."""

    def __init__(self):
        self.calls = []

    def decode_latent_to_preview_image(self, preview_format, x0):
        self.calls.append((preview_format, x0))
        return (preview_format, f"<image {x0}>", 512)


class _PreviewEnv:
    """Installs fake `latent_preview` + `comfy.cli_args` modules."""

    def __init__(self, previewer=None):
        self.previewer = previewer
        self.seen_methods = []
        self.get_previewer_args = []
        self.args = types.SimpleNamespace(
            preview_method=_LatentPreviewMethod.NoPreviews
        )
        self._saved = {}

    def _get_previewer(self, device, latent_format):
        self.seen_methods.append(self.args.preview_method)
        self.get_previewer_args.append((device, latent_format))
        return self.previewer

    def __enter__(self):
        lp = types.ModuleType("latent_preview")
        lp.get_previewer = self._get_previewer

        cli_args = types.ModuleType("comfy.cli_args")
        cli_args.args = self.args
        cli_args.LatentPreviewMethod = _LatentPreviewMethod

        for name, mod in (("latent_preview", lp), ("comfy.cli_args", cli_args)):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc):
        for name, mod in self._saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        return False


def _fake_model(latent_format="LF"):
    model = MagicMock()
    model.load_device = "cuda:0"
    model.model.latent_format = latent_format
    return model


class _PbarRecorder:
    def __init__(self):
        self.updates = []

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total, preview))


# ─────────────────────────────────────────────────────────────────────────────
#  Defects 1 + 2 — the widget must actually select a decoder
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveLatentPreviewer(unittest.TestCase):

    def test_taesd_is_not_downgraded_to_latent2rgb(self):
        previewer = _FakePreviewer()
        with _PreviewEnv(previewer) as env:
            result = _resolve_latent_previewer(_fake_model(), "TAESD")

        self.assertIs(result, previewer)
        self.assertEqual(env.seen_methods, [_LatentPreviewMethod.TAESD])

    def test_latent2rgb_selects_that_method(self):
        previewer = _FakePreviewer()
        with _PreviewEnv(previewer) as env:
            _resolve_latent_previewer(_fake_model(), "Latent2RGB")
        self.assertEqual(env.seen_methods, [_LatentPreviewMethod.Latent2RGB])

    def test_previewer_is_built_from_model_device_and_latent_format(self):
        with _PreviewEnv(_FakePreviewer()) as env:
            _resolve_latent_previewer(_fake_model("FLUX_LF"), "TAESD")
        self.assertEqual(env.get_previewer_args, [("cuda:0", "FLUX_LF")])

    def test_global_preview_method_is_restored(self):
        """Other nodes in the same prompt must not inherit the override."""
        with _PreviewEnv(_FakePreviewer()) as env:
            env.args.preview_method = _LatentPreviewMethod.Latent2RGB
            _resolve_latent_previewer(_fake_model(), "TAESD")
            self.assertEqual(env.args.preview_method, _LatentPreviewMethod.Latent2RGB)

    def test_none_method_builds_nothing(self):
        with _PreviewEnv(_FakePreviewer()) as env:
            self.assertIsNone(_resolve_latent_previewer(_fake_model(), "None"))
        self.assertEqual(env.seen_methods, [])

    def test_unknown_method_returns_none(self):
        with _PreviewEnv(_FakePreviewer()) as env:
            self.assertIsNone(_resolve_latent_previewer(_fake_model(), "Nonsense"))
        self.assertEqual(env.seen_methods, [])

    def test_missing_latent_format_returns_none(self):
        model = MagicMock()
        model.load_device = "cpu"
        model.model.latent_format = None
        with _PreviewEnv(_FakePreviewer()):
            self.assertIsNone(_resolve_latent_previewer(model, "TAESD"))

    def test_comfy_returning_no_previewer_is_not_fatal(self):
        with _PreviewEnv(None):
            self.assertIsNone(_resolve_latent_previewer(_fake_model(), "TAESD"))

    def test_nonexistent_taesd_symbol_is_no_longer_imported(self):
        """comfy.taesd.taesd has never exported TAESDDecoder, in 0.32 or 0.36."""
        import radiance.nodes.generate.sampler as sampler_mod

        source = inspect.getsource(sampler_mod)
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "import" in line and "taesd" in line
        ]
        self.assertEqual(offenders, [])


# ─────────────────────────────────────────────────────────────────────────────
#  Defect 3 — the preview payload shape
# ─────────────────────────────────────────────────────────────────────────────

class TestPreviewCallbackPayload(unittest.TestCase):

    def test_callback_sends_the_protocol_three_tuple(self):
        """Old callback sent `(x0,)`; server.py indexes image_data[2]."""
        pbar = _PbarRecorder()
        previewer = _FakePreviewer()
        callback = _make_phase_preview_callback(pbar, previewer, 0)

        callback(0, "latent0", "x", 10)

        (value, total, preview), = pbar.updates
        self.assertEqual(value, 1)
        self.assertEqual(total, 10)
        self.assertEqual(len(preview), 3)
        self.assertEqual(preview[0], "JPEG")
        self.assertEqual(preview[2], 512)
        # send_image_with_metadata() does exactly this and used to IndexError.
        self.assertIsNotNone(preview[2])

    def test_callback_actually_decodes_the_latent(self):
        pbar = _PbarRecorder()
        previewer = _FakePreviewer()
        callback = _make_phase_preview_callback(pbar, previewer, 0)

        callback(3, "latent_at_3", "x", 10)

        self.assertEqual(previewer.calls, [("JPEG", "latent_at_3")])

    def test_callback_offsets_by_phase_start(self):
        pbar = _PbarRecorder()
        callback = _make_phase_preview_callback(pbar, _FakePreviewer(), 12)

        callback(2, "latent", "x", 40)

        self.assertEqual(pbar.updates[0][0], 15)

    def test_callback_unwraps_nested_latents(self):
        class _Nested:
            is_nested = True
            tensors = ("first", "second")

        previewer = _FakePreviewer()
        callback = _make_phase_preview_callback(_PbarRecorder(), previewer, 0)

        callback(0, _Nested(), "x", 4)

        self.assertEqual(previewer.calls, [("JPEG", "first")])

    def test_decode_failure_still_reports_progress(self):
        class _Broken:
            def decode_latent_to_preview_image(self, fmt, x0):
                raise RuntimeError("decode blew up")

        pbar = _PbarRecorder()
        callback = _make_phase_preview_callback(pbar, _Broken(), 0)

        callback(0, "latent", "x", 5)

        self.assertEqual(pbar.updates, [(1, 5, None)])

    def test_no_previewer_still_reports_progress(self):
        pbar = _PbarRecorder()
        callback = _make_phase_preview_callback(pbar, None, 0)

        callback(4, "latent", "x", 5)

        self.assertEqual(pbar.updates, [(5, 5, None)])


# ─────────────────────────────────────────────────────────────────────────────
#  Defect 4 — disable_pbar must not fire without a working previewer
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressBarWiring(unittest.TestCase):
    """Source-level guards, matching the house pattern in test_sampler_regression."""

    def setUp(self):
        self.source = inspect.getsource(RadianceSamplerPro.sample)

    def test_use_custom_preview_requires_a_previewer(self):
        """It must be set inside the `previewer_ref is not None` branch only."""
        self.assertIn("previewer_ref = _resolve_latent_previewer(", self.source)
        before, _, after = self.source.partition("use_custom_preview = True")
        self.assertIn("if previewer_ref is not None:", before)
        # Nothing else may flip the flag on.
        self.assertNotIn("use_custom_preview = True", after)

    def test_callback_is_none_without_a_progress_bar(self):
        """A None callback lets comfy.sample.sample_custom run its own bar."""
        self.assertIn("if pbar_ref is None:", self.source)
        self.assertIn("_make_phase_preview_callback(", self.source)

    def test_raw_latent_tuple_is_no_longer_sent(self):
        self.assertNotIn("total_steps, (x0,)", self.source)


if __name__ == "__main__":
    unittest.main()
