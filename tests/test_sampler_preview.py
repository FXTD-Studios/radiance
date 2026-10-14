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
  4. disable_pbar was wired to the preview, so any preview (a failed one
     too) switched off Comfy's console bar and its step count.

Also covers "TAESD" fetching a missing decoder into models/vae_approx and
previewing LTX-2 latents through taeltx2_3, which ComfyUI does not wire.
"""

from __future__ import annotations

import enum
import inspect
import os
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import radiance.nodes.generate.sampler as sampler_mod
from radiance.core.model_fetch import ModelFetchError
from radiance.nodes.generate.sampler import (
    RadianceSamplerPro,
    _TAESD_DECODERS,
    _ensure_taesd_decoder,
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


class _LTXAV:
    """Stands in for comfy.latent_formats.LTXAV."""

    taesd_decoder_name = None


class _FakeVAE:
    """Stands in for comfy.sd.VAE built from a video TAE state dict."""

    def __init__(self, sd):
        self.sd = sd
        self.first_stage_model = types.SimpleNamespace(show_progress_bar=True)


class _FakeTAEHVPreviewer(_FakePreviewer):
    def __init__(self, taesd):
        super().__init__()
        self.taesd = taesd


class _PreviewEnv:
    """Installs fake `latent_preview`, `comfy.cli_args`, `comfy.latent_formats`
    and `comfy.sd` modules, and a decoder lookup that finds *decoder*."""

    def __init__(self, previewer=None, decoder="/vae_approx/taef1_decoder.pth"):
        self.previewer = previewer
        self._ensure = patch.object(sampler_mod, "_ensure_taesd_decoder", return_value=decoder)
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
        lp.TAEHVPreviewerImpl = _FakeTAEHVPreviewer

        cli_args = types.ModuleType("comfy.cli_args")
        cli_args.args = self.args
        cli_args.LatentPreviewMethod = _LatentPreviewMethod

        latent_formats = types.ModuleType("comfy.latent_formats")
        latent_formats.LTXAV = _LTXAV

        sd = types.ModuleType("comfy.sd")
        sd.VAE = _FakeVAE

        for name, mod in (("latent_preview", lp), ("comfy.cli_args", cli_args),
                          ("comfy.latent_formats", latent_formats), ("comfy.sd", sd)):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        self.ensure = self._ensure.start()
        return self

    def __exit__(self, *exc):
        self._ensure.stop()
        for name, mod in self._saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        return False


def _fake_model(latent_format=None):
    if latent_format is None:
        latent_format = types.SimpleNamespace(taesd_decoder_name="taef1_decoder")
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
        flux = types.SimpleNamespace(taesd_decoder_name="taef1_decoder")
        with _PreviewEnv(_FakePreviewer()) as env:
            _resolve_latent_previewer(_fake_model(flux), "TAESD")
        self.assertEqual(env.get_previewer_args, [("cuda:0", flux)])

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

    def test_taesd_fetches_the_decoder_before_comfy_loads_it(self):
        with _PreviewEnv(_FakePreviewer()) as env:
            _resolve_latent_previewer(_fake_model(), "TAESD")
        env.ensure.assert_called_once_with("taef1_decoder")
        self.assertEqual(env.seen_methods, [_LatentPreviewMethod.TAESD])

    def test_latent2rgb_fetches_nothing(self):
        with _PreviewEnv(_FakePreviewer()) as env:
            _resolve_latent_previewer(_fake_model(), "Latent2RGB")
        env.ensure.assert_not_called()

    def test_a_model_without_a_decoder_previews_in_latent2rgb(self):
        """Mochi, LTX-Video 0.9: get_previewer() warned on every run."""
        mochi = types.SimpleNamespace(taesd_decoder_name=None)
        with _PreviewEnv(_FakePreviewer()) as env:
            _resolve_latent_previewer(_fake_model(mochi), "TAESD")
        env.ensure.assert_not_called()
        self.assertEqual(env.seen_methods, [_LatentPreviewMethod.Latent2RGB])

    def test_a_decoder_that_could_not_be_fetched_previews_in_latent2rgb(self):
        with _PreviewEnv(_FakePreviewer(), decoder=None) as env:
            _resolve_latent_previewer(_fake_model(), "TAESD")
        self.assertEqual(env.seen_methods, [_LatentPreviewMethod.Latent2RGB])

    def test_the_console_names_the_decoder_in_use(self):
        with _PreviewEnv(_FakePreviewer()), self.assertLogs(sampler_mod.logger, "INFO") as logs:
            _resolve_latent_previewer(_fake_model(), "TAESD")
        self.assertIn("[Radiance] Preview: TAESD (taef1_decoder.pth)", logs.output[-1])

    def test_the_console_says_when_a_model_has_no_decoder(self):
        mochi = types.SimpleNamespace(taesd_decoder_name=None)
        with _PreviewEnv(_FakePreviewer()), self.assertLogs(sampler_mod.logger, "INFO") as logs:
            _resolve_latent_previewer(_fake_model(mochi), "TAESD")
        self.assertIn("[Radiance] Preview: Latent2RGB (no TAESD decoder for this model)", logs.output[-1])

    def test_taesd_is_the_default(self):
        required = RadianceSamplerPro.INPUT_TYPES()["required"]
        self.assertEqual(required["preview_method"][1]["default"], "TAESD")

    def test_ltx2_previews_through_taeltx2_3(self):
        """ComfyUI gives LTX-2 latents no decoder name, so get_previewer()
        could only fall back to Latent2RGB."""
        with _PreviewEnv(_FakePreviewer(), decoder="/m/taeltx2_3.safetensors") as env, \
                patch.object(sampler_mod.comfy.utils, "load_torch_file", return_value={"w": 1}, create=True) as load:
            result = _resolve_latent_previewer(_fake_model(_LTXAV()), "TAESD")
        env.ensure.assert_called_once_with("taeltx2_3")
        load.assert_called_once_with("/m/taeltx2_3.safetensors")
        self.assertIsInstance(result, _FakeTAEHVPreviewer)
        self.assertEqual(result.taesd.sd, {"w": 1})
        self.assertFalse(result.taesd.first_stage_model.show_progress_bar)
        self.assertEqual(env.seen_methods, [])

    def test_ltx2_without_its_decoder_falls_back_to_latent2rgb(self):
        with _PreviewEnv(_FakePreviewer(), decoder=None) as env:
            _resolve_latent_previewer(_fake_model(_LTXAV()), "TAESD")
        self.assertEqual(env.seen_methods, [_LatentPreviewMethod.Latent2RGB])

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
#  The decoder "TAESD" needs, fetched when it is missing
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureTaesdDecoder(unittest.TestCase):

    MODELS = os.path.abspath(os.path.join("ComfyUI", "models"))
    OWN = os.path.join(MODELS, "vae_approx")
    SHARED = os.path.abspath(os.path.join("StabilityMatrix", "Models", "ApproxVAE"))

    def _folder_paths(self, files, folders=(OWN, SHARED)):
        fp = types.SimpleNamespace(
            models_dir=self.MODELS,
            get_filename_list=MagicMock(return_value=files),
            get_full_path=MagicMock(side_effect=lambda kind, name: f"/vae_approx/{name}"),
            get_folder_paths=MagicMock(return_value=list(folders)),
        )
        return patch.object(sampler_mod, "folder_paths", fp)

    def setUp(self):
        sampler_mod._TAESD_FAILED.clear()

    def tearDown(self):
        sampler_mod._TAESD_FAILED.clear()

    def test_a_decoder_already_installed_is_used_as_is(self):
        """Whatever version the user put there: never replaced."""
        with self._folder_paths(["taef1_decoder.safetensors"]), \
                patch.object(sampler_mod, "fetch") as fetch:
            path = _ensure_taesd_decoder("taef1_decoder")
        self.assertEqual(path, "/vae_approx/taef1_decoder.safetensors")
        fetch.assert_not_called()

    def test_a_missing_decoder_goes_to_the_folder_the_user_added(self):
        """Stability Matrix lists its shared ApproxVAE after ComfyUI's own folder."""
        url, sha256, size = _TAESD_DECODERS["taeltx2_3"]
        with self._folder_paths([]), \
                patch.object(sampler_mod, "fetch", side_effect=lambda u, dest, **kw: dest) as fetch:
            path = _ensure_taesd_decoder("taeltx2_3")
        dest = os.path.join(self.SHARED, "taeltx2_3.safetensors")
        self.assertEqual(path, dest)
        fetch.assert_called_once_with(url, dest, sha256=sha256, size=size, label="Preview decoder taeltx2_3")

    def test_without_an_added_folder_it_goes_to_comfyuis_own(self):
        with self._folder_paths([], folders=(self.OWN,)), \
                patch.object(sampler_mod, "fetch", side_effect=lambda u, dest, **kw: dest):
            path = _ensure_taesd_decoder("taehv")
        self.assertEqual(path, os.path.join(self.OWN, "taehv.safetensors"))

    def test_a_failed_fetch_returns_none(self):
        with self._folder_paths([]), \
                patch.object(sampler_mod, "fetch", side_effect=ModelFetchError("offline")):
            self.assertIsNone(_ensure_taesd_decoder("taehv"))

    def test_a_failed_fetch_is_not_retried_until_restart(self):
        """Offline, every run warned again, or waited on the network."""
        with self._folder_paths([]), \
                patch.object(sampler_mod, "fetch", side_effect=ModelFetchError("offline")) as fetch:
            _ensure_taesd_decoder("taehv")
            self.assertIsNone(_ensure_taesd_decoder("taehv"))
        self.assertEqual(fetch.call_count, 1)
        with self._folder_paths(["taehv.safetensors"]):
            self.assertEqual(_ensure_taesd_decoder("taehv"), "/vae_approx/taehv.safetensors")

    def test_an_unknown_decoder_is_not_fetched(self):
        with self._folder_paths([]), patch.object(sampler_mod, "fetch") as fetch:
            self.assertIsNone(_ensure_taesd_decoder("taemochi"))
        fetch.assert_not_called()

    def test_every_pin_names_its_file_a_commit_and_a_digest(self):
        for name, (url, sha256, size) in _TAESD_DECODERS.items():
            self.assertTrue(url.rsplit("/", 1)[1].startswith(name), name)
            self.assertRegex(url, r"/[0-9a-f]{40}/", name)
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", sha256), name)
            self.assertGreater(size, 1_000_000, name)


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
        self.assertIsNone(total)     # the bar keeps its total over every stage
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

        self.assertEqual(pbar.updates, [(1, None, None)])

    def test_no_previewer_still_reports_progress(self):
        pbar = _PbarRecorder()
        callback = _make_phase_preview_callback(pbar, None, 0)

        callback(4, "latent", "x", 5)

        self.assertEqual(pbar.updates, [(5, None, None)])

    def test_a_later_stage_does_not_shrink_the_bar(self):
        """Stage two (steps 10-19 of 20) reported its own 10 as the total, so
        the bar sat full from the end of stage one."""
        pbar = _PbarRecorder()
        callback = _make_phase_preview_callback(pbar, None, 10)

        callback(2, "latent", "x", 10)

        self.assertEqual(pbar.updates, [(13, None, None)])


# ─────────────────────────────────────────────────────────────────────────────
#  Defect 4 — the preview must not switch off the console bar
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressBarWiring(unittest.TestCase):
    """Source-level guards, matching the house pattern in test_sampler_regression."""

    def setUp(self):
        self.source = inspect.getsource(RadianceSamplerPro.sample)

    def test_console_bar_follows_comfy_setting_only(self):
        """Like ComfyUI's KSampler: on unless Comfy's progress bar is off."""
        self.assertIn("previewer_ref = _resolve_latent_previewer(", self.source)
        self.assertIn("disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED", self.source)
        self.assertNotIn("use_custom_preview", self.source)

    def test_the_node_bar_exists_without_a_preview(self):
        """sample_custom only draws a console bar, so the node had no progress
        at all with preview_method None."""
        before, _, _ = self.source.partition('if preview_method != "None":')
        self.assertIn("pbar_ref = comfy.utils.ProgressBar(", before)
        self.assertNotIn("if pbar_ref is None:", self.source)
        self.assertIn("_make_phase_preview_callback(", self.source)

    def test_raw_latent_tuple_is_no_longer_sent(self):
        self.assertNotIn("total_steps, (x0,)", self.source)


if __name__ == "__main__":
    unittest.main()
