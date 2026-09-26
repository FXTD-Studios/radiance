"""
tests/test_lora_apply.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Regression tests for radiance.nodes.generate.lora — RadianceHDRLoRAApply.

Pins two confirmed P0 defects:

  1. LoRA apply corrupted the loader's cached model.
     ModelPatcher.clone() shares the underlying nn.Module and its
     nn.Parameter storage, so the old `module.weight.add_(delta)` wrote
     straight into the cached model: deltas accumulated on every queue,
     bypassing the node changed nothing, and an fp8_e4m3fn UNET raised
     NotImplementedError partway through and left the model half patched.

  2. Prefix extraction assumed every key was kohya `lora_unet_*` with
     `.lora_down.weight` / `.lora_up.weight`.  PEFT/diffusers exports
     yielded zero deltas and were logged at INFO as a success.

The FakeModelPatcher below mirrors ComfyUI 0.32.0 / 0.36.0 ModelPatcher
semantics exactly: clone() shares .model and copies the patch table,
add_patches() records into the clone's own table and drops keys that are
not in the model state dict.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from radiance.core.errors import RadianceError
from radiance.nodes.generate.lora import (
    RadianceHDRLoRAApply,
    _apply_lora_to_model,
    _build_weight_key_index,
    _iter_lora_pairs,
    _lookup_weight_key,
    _strip_lora_name_prefix,
)


# ─────────────────────────────────────────────────────────────────────────────
#  ComfyUI ModelPatcher stand-in
# ─────────────────────────────────────────────────────────────────────────────

class _Inner(nn.Module):
    """Minimal BaseModel stand-in: one diffusion_model with one Linear."""

    def __init__(self, out_features: int = 6, in_features: int = 4):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.double_blocks = nn.Module()
        self.diffusion_model.double_blocks.img_attn = nn.Linear(
            in_features, out_features, bias=False
        )
        with torch.no_grad():
            self.diffusion_model.double_blocks.img_attn.weight.fill_(0.25)


class FakeModelPatcher:
    """Reproduces the ModelPatcher contract the node depends on."""

    WEIGHT_KEY = "diffusion_model.double_blocks.img_attn.weight"

    def __init__(self, model, patches=None):
        self.model = model
        self.patches = {} if patches is None else patches
        self.load_device = torch.device("cpu")

    def clone(self):
        # Matches comfy/model_patcher.py: get_clone_model_override() returns
        # self.model itself, and n.patches starts empty then copies each list.
        n = FakeModelPatcher(self.model)
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        applied = []
        model_sd = self.model.state_dict()
        for k in patches:
            if k in model_sd:
                applied.append(k)
                current = self.patches.get(k, [])
                current.append((strength_patch, patches[k], strength_model, None, None))
                self.patches[k] = current
        return applied


class _Fp8Inner(nn.Module):
    """BaseModel stand-in whose weight is fp8_e4m3fn, as on an RTX 4080 run."""

    def __init__(self, out_features: int = 6, in_features: int = 4):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.double_blocks = nn.Module()
        linear = nn.Module()
        linear.weight = nn.Parameter(
            torch.full((out_features, in_features), 0.25).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.diffusion_model.double_blocks.img_attn = linear


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA fixtures
# ─────────────────────────────────────────────────────────────────────────────

RANK = 2
OUT, IN = 6, 4


def _down_up():
    down = torch.full((RANK, IN), 0.5)          # A: rank × in
    up = torch.full((OUT, RANK), 0.5)           # B: out × rank
    return down, up


def _kohya_lora():
    down, up = _down_up()
    return {
        "lora_unet_double_blocks_img_attn.lora_down.weight": down,
        "lora_unet_double_blocks_img_attn.lora_up.weight": up,
        "lora_unet_double_blocks_img_attn.alpha": torch.tensor(float(RANK)),
    }


def _peft_lora():
    down, up = _down_up()
    return {
        "diffusion_model.double_blocks.img_attn.lora_A.weight": down,
        "diffusion_model.double_blocks.img_attn.lora_B.weight": up,
    }


def _transformer_lora():
    down, up = _down_up()
    return {
        "transformer.double_blocks.img_attn.lora_A.weight": down,
        "transformer.double_blocks.img_attn.lora_B.weight": up,
    }


def _lora_dict(tensors):
    return {"tensors": tensors, "metadata": {}, "path": "/tmp/fake_hdr_lora.safetensors"}


# ─────────────────────────────────────────────────────────────────────────────
#  Defect 1 — shared-model corruption
# ─────────────────────────────────────────────────────────────────────────────

class TestSharedModelIsNotMutated(unittest.TestCase):

    def test_apply_leaves_cached_model_weights_untouched(self):
        """Old code wrote delta into the shared nn.Parameter; weights moved."""
        patcher = FakeModelPatcher(_Inner())
        before = patcher.model.state_dict()[FakeModelPatcher.WEIGHT_KEY].clone()

        out_model, _ = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=1.0
        )

        after = patcher.model.state_dict()[FakeModelPatcher.WEIGHT_KEY]
        self.assertTrue(torch.equal(before, after))
        # The clone shares the module, so it must see the same untouched weight.
        self.assertTrue(
            torch.equal(before, out_model.model.state_dict()[FakeModelPatcher.WEIGHT_KEY])
        )

    def test_patch_recorded_on_clone_only(self):
        """The source patcher's patch table must stay empty."""
        patcher = FakeModelPatcher(_Inner())
        out_model, _ = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=1.0
        )

        self.assertEqual(patcher.patches, {})
        self.assertIn(FakeModelPatcher.WEIGHT_KEY, out_model.patches)
        self.assertEqual(len(out_model.patches[FakeModelPatcher.WEIGHT_KEY]), 1)
        self.assertIsNot(out_model, patcher)

    def test_requeue_does_not_accumulate_deltas(self):
        """Old code added another delta to the cached model on every queue."""
        patcher = FakeModelPatcher(_Inner())
        before = patcher.model.state_dict()[FakeModelPatcher.WEIGHT_KEY].clone()

        first, _ = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=1.0
        )
        second, _ = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=1.0
        )

        after = patcher.model.state_dict()[FakeModelPatcher.WEIGHT_KEY]
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(len(first.patches[FakeModelPatcher.WEIGHT_KEY]), 1)
        self.assertEqual(len(second.patches[FakeModelPatcher.WEIGHT_KEY]), 1)

    def test_recorded_patch_is_the_expected_diff(self):
        """diff = (alpha / rank) * up @ down, with strength as strength_patch."""
        patcher = FakeModelPatcher(_Inner())
        out_model, _ = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=0.75
        )

        strength_patch, value, strength_model, offset, function = \
            out_model.patches[FakeModelPatcher.WEIGHT_KEY][0]
        self.assertAlmostEqual(strength_patch, 0.75, places=6)
        self.assertEqual(strength_model, 1.0)
        self.assertIsNone(offset)
        self.assertIsNone(function)

        patch_type, payload = value
        self.assertEqual(patch_type, "diff")
        down, up = _down_up()
        expected = (float(RANK) / RANK) * (up @ down)
        self.assertEqual(tuple(payload[0].shape), (OUT, IN))
        self.assertTrue(torch.allclose(payload[0], expected, atol=1e-6))

    def test_fp8_weight_does_not_raise(self):
        """Tensor.add_ has no float8_e4m3fn kernel; the old path raised here."""
        patcher = FakeModelPatcher(_Fp8Inner())
        before = (
            patcher.model.state_dict()[FakeModelPatcher.WEIGHT_KEY]
            .to(torch.float32)
            .clone()
        )

        out_model, _ = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=1.0
        )

        after = patcher.model.state_dict()[FakeModelPatcher.WEIGHT_KEY].to(torch.float32)
        self.assertTrue(torch.equal(before, after))
        self.assertIn(FakeModelPatcher.WEIGHT_KEY, out_model.patches)


# ─────────────────────────────────────────────────────────────────────────────
#  Defect 1b — key layouts and the silent zero-delta success
# ─────────────────────────────────────────────────────────────────────────────

class TestLoRAKeyLayouts(unittest.TestCase):

    def test_kohya_layout_still_applies(self):
        patcher = FakeModelPatcher(_Inner())
        applied = _apply_lora_to_model(patcher.clone(), _kohya_lora(), 1.0)
        self.assertEqual(applied, 1)

    def test_peft_diffusion_model_layout_applies(self):
        """lora_A / lora_B under diffusion_model.* used to yield 0 deltas."""
        patcher = FakeModelPatcher(_Inner())
        applied = _apply_lora_to_model(patcher.clone(), _peft_lora(), 1.0)
        self.assertEqual(applied, 1)

    def test_transformer_prefix_layout_applies(self):
        patcher = FakeModelPatcher(_Inner())
        applied = _apply_lora_to_model(patcher.clone(), _transformer_lora(), 1.0)
        self.assertEqual(applied, 1)

    def test_text_encoder_prefix_is_stripped_not_sliced(self):
        """lora_te_* was sliced by len('lora_unet_'), mangling the path."""
        self.assertEqual(
            _strip_lora_name_prefix("lora_te_text_model_encoder_layers_0_mlp_fc1"),
            "text_model_encoder_layers_0_mlp_fc1",
        )
        self.assertEqual(
            _strip_lora_name_prefix("lora_unet_double_blocks_0_img_attn"),
            "double_blocks_0_img_attn",
        )

    def test_iter_pairs_finds_both_layouts(self):
        names = {name for name, _, _, _ in _iter_lora_pairs(_kohya_lora())}
        self.assertEqual(names, {"lora_unet_double_blocks_img_attn"})
        names = {name for name, _, _, _ in _iter_lora_pairs(_peft_lora())}
        self.assertEqual(names, {"diffusion_model.double_blocks.img_attn"})

    def test_prefixed_name_wins_over_the_stripped_reading(self):
        """A model that really has a `transformer` submodule must still match."""
        index = _build_weight_key_index(
            ["diffusion_model.transformer.blocks.0.attn.weight"]
        )
        self.assertEqual(
            _lookup_weight_key(index, "transformer.blocks.0.attn"),
            "diffusion_model.transformer.blocks.0.attn.weight",
        )

    def test_unpaired_down_key_is_ignored(self):
        down, _ = _down_up()
        pairs = list(_iter_lora_pairs({"lora_unet_x.lora_down.weight": down}))
        self.assertEqual(pairs, [])

    def test_zero_applied_raises_instead_of_reporting_success(self):
        """A LoRA that matched nothing is a failed run, not a quiet no-op."""
        patcher = FakeModelPatcher(_Inner())
        down, up = _down_up()
        stray = {
            "lora_unet_no_such_module.lora_down.weight": down,
            "lora_unet_no_such_module.lora_up.weight": up,
        }
        with self.assertRaises(RadianceError) as ctx:
            RadianceHDRLoRAApply().apply(patcher, _lora_dict(stray), strength=1.0)
        self.assertIn("no LoRA delta matched", str(ctx.exception))
        self.assertEqual(patcher.patches, {})

    def test_strength_zero_still_short_circuits(self):
        patcher = FakeModelPatcher(_Inner())
        out_model, ratio = RadianceHDRLoRAApply().apply(
            patcher, _lora_dict(_kohya_lora()), strength=0.0
        )
        self.assertIs(out_model, patcher)
        self.assertEqual(patcher.patches, {})
        self.assertAlmostEqual(ratio, 0.5, places=6)


# ─────────────────────────────────────────────────────────────────────────────
#  Non-ModelPatcher fallback (plain nn.Module pipelines and tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlainModuleFallback(unittest.TestCase):

    def test_deepcopy_path_merges_and_leaves_source_alone(self):
        source = _Inner()
        before = source.diffusion_model.double_blocks.img_attn.weight.clone()

        out_model, _ = RadianceHDRLoRAApply().apply(
            source, _lora_dict(_kohya_lora()), strength=1.0
        )

        self.assertTrue(
            torch.equal(before, source.diffusion_model.double_blocks.img_attn.weight)
        )
        down, up = _down_up()
        expected = before + (up @ down)
        self.assertTrue(
            torch.allclose(
                out_model.diffusion_model.double_blocks.img_attn.weight,
                expected,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
