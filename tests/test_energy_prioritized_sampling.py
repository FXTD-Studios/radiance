"""
Energy-Prioritized Sampling: producer → conditioning → sampler reader.

Issue #40: RadianceSamplerPro has read `radiance_energy_mask` out of the
positive CONDITIONING since v3.1, but no node ever wrote that key, so the
feature could not be reached from a graph. The two tests that existed
(`TestEnergyPrioritizedSampling` in test_sampler_regression.py) hid this:
one re-implemented the parser loop inside the test body and asserted on its
own copy, the other did the modifier arithmetic inline. Both passed without
executing a single line of nodes_sampler.py.

These tests drive the real producer node and the real reader, and pin the
two crashes the reader was carrying: a 4-way shape unpack that dies on the
5-D latents every video model produces, and an expand() that dies on a mask
whose batch is neither 1 nor B.
"""
import os
import sys

import pytest

try:
    import torch as _t
    _HAS_TORCH = isinstance(getattr(_t, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    pytest.skip("torch not installed", allow_module_level=True)

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.generate.energy import (
    ENERGY_LAYERS_KEY,
    ENERGY_MASK_KEY,
    ENERGY_PRIORITY_KEY,
    RadianceEnergyMask,
)
from radiance.nodes.generate.sampler import _collect_energy_layers, _make_energy_cfg_patch


def _conditioning(batch=1):
    return [(torch.zeros(batch, 77, 2048), {"pooled_output": torch.zeros(batch, 1280)})]


def _half_mask(h=64, w=64):
    """Left half black, right half white."""
    m = torch.zeros(1, h, w)
    m[:, :, w // 2:] = 1.0
    return m


def _args(cond, uncond, cfg=7.0):
    return {
        "cond_denoised": cond,
        "uncond_denoised": uncond,
        "cond_scale": cfg,
        "denoised": uncond + cfg * (cond - uncond),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Producer
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMaskNode:

    def test_the_node_writes_the_keys_the_sampler_reads(self):
        """The whole point of #40: producer output must satisfy the reader."""
        out, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.75)

        layers = _collect_energy_layers(out)

        assert len(layers) == 1, "the sampler did not find the mask the node attached"
        assert layers[0][1] == pytest.approx(0.75)

    def test_every_entry_carries_the_keys(self):
        cond = _conditioning() + _conditioning()
        out, _ = RadianceEnergyMask().apply(cond, _half_mask(), priority=0.5)

        assert all(ENERGY_MASK_KEY in meta for _, meta in out)
        assert all(meta[ENERGY_PRIORITY_KEY] == 0.5 for _, meta in out)

    def test_the_input_conditioning_is_not_mutated(self):
        """A conditioning list is routinely fanned out to several branches."""
        cond = _conditioning()
        RadianceEnergyMask().apply(cond, _half_mask(), priority=0.5)

        assert ENERGY_MASK_KEY not in cond[0][1]

    def test_the_returned_mask_is_the_conditioned_one(self):
        out, preview = RadianceEnergyMask().apply(
            _conditioning(), _half_mask(), priority=0.5, invert=True
        )
        assert torch.equal(out[0][1][ENERGY_MASK_KEY], preview)

    def test_invert_swaps_the_region(self):
        _, m = RadianceEnergyMask().apply(
            _conditioning(), _half_mask(), priority=0.5, invert=True
        )
        assert m[0, 0, 0].item() == pytest.approx(1.0)
        assert m[0, 0, -1].item() == pytest.approx(0.0)

    def test_normalize_rescales_an_unnormalised_pass(self):
        """Luminance and depth passes are not guaranteed to sit in [0,1]."""
        raw = torch.full((1, 8, 8), 4.0)
        raw[:, :, 4:] = 12.0

        _, m = RadianceEnergyMask().apply(
            _conditioning(), raw, priority=0.5, normalize=True
        )
        assert m.min().item() == pytest.approx(0.0)
        assert m.max().item() == pytest.approx(1.0)

    def test_out_of_range_masks_are_clamped(self):
        """An unclamped mask would flip the CFG modifier negative."""
        raw = torch.tensor([[[-3.0, 5.0]]])
        _, m = RadianceEnergyMask().apply(_conditioning(), raw, priority=0.5)

        assert m.min().item() >= 0.0
        assert m.max().item() <= 1.0

    def test_gain_hardens_a_soft_mask(self):
        soft = torch.full((1, 4, 4), 0.25)
        _, m = RadianceEnergyMask().apply(_conditioning(), soft, priority=0.5, gain=4.0)
        assert m.mean().item() == pytest.approx(1.0)

    def test_a_non_tensor_mask_is_rejected_at_the_node(self):
        with pytest.raises(ValueError):
            RadianceEnergyMask().apply(_conditioning(), "not a mask", priority=0.5)

    def test_empty_conditioning_is_rejected(self):
        with pytest.raises(ValueError):
            RadianceEnergyMask().apply([], _half_mask(), priority=0.5)

    def test_node_contract(self):
        cls = RadianceEnergyMask
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
        assert hasattr(cls, cls.FUNCTION)
        assert "conditioning" in cls.INPUT_TYPES()["required"]
        assert "mask" in cls.INPUT_TYPES()["required"]

    def test_the_node_is_published_to_comfyui(self):
        """#40 was reader-only code. A producer nobody can place is the same bug."""
        from radiance.nodes.generate import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        assert NODE_CLASS_MAPPINGS.get("RadianceEnergyMask") is RadianceEnergyMask
        assert "RadianceEnergyMask" in NODE_DISPLAY_NAME_MAPPINGS


# ─────────────────────────────────────────────────────────────────────────────
#  Reader
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyMaskParser:

    def test_no_mask_returns_no_layers(self):
        assert _collect_energy_layers(_conditioning()) == []

    def test_non_list_conditioning_is_tolerated(self):
        assert _collect_energy_layers(None) == []

    def test_a_junk_mask_value_is_skipped_not_crashed(self):
        positive = [(torch.zeros(1, 77, 2048), {ENERGY_MASK_KEY: "nonsense"})]
        assert _collect_energy_layers(positive) == []

    def test_a_malformed_layer_entry_is_skipped_not_crashed(self):
        positive = [(torch.zeros(1, 77, 2048), {ENERGY_LAYERS_KEY: ["not a pair"]})]
        assert _collect_energy_layers(positive) == []

    def test_a_non_numeric_priority_falls_back(self):
        positive = [(torch.zeros(1, 77, 2048), {
            ENERGY_MASK_KEY: _half_mask(),
            ENERGY_PRIORITY_KEY: "loud",
        })]
        layers = _collect_energy_layers(positive)
        assert layers[0][1] == pytest.approx(1.0)

    def test_a_beta_conditioning_without_the_layers_key_still_reads(self):
        """3.3.0-beta wrote only the singular keys; those graphs keep working."""
        positive = [(torch.zeros(1, 77, 2048), {
            ENERGY_MASK_KEY: _half_mask(),
            ENERGY_PRIORITY_KEY: 0.4,
        })]
        layers = _collect_energy_layers(positive)
        assert len(layers) == 1
        assert layers[0][1] == pytest.approx(0.4)


# ─────────────────────────────────────────────────────────────────────────────
#  Stacking
#
#  Two Energy Mask nodes used to be a silent data loss, in two different ways
#  depending on how they were wired, and the node docstring described only one
#  of them — incorrectly. In series, the second `attach_energy_mask` overwrote
#  the singular key on every entry, so the DOWNSTREAM node won. Joined by
#  Conditioning Combine, the reader stopped at the first entry carrying the
#  key, so the UPSTREAM node won. Either way one mask vanished with no log
#  line. Both shapes now stack.
# ─────────────────────────────────────────────────────────────────────────────

class TestChainedEnergyMasksStack:

    def test_two_nodes_in_series_produce_two_layers(self):
        first, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.25)
        second, _ = RadianceEnergyMask().apply(first, _half_mask(), priority=0.5)

        layers = _collect_energy_layers(second)
        assert [p for _, p in layers] == [pytest.approx(0.25), pytest.approx(0.5)]

    def test_two_branches_combined_produce_two_layers(self):
        """The Conditioning Combine shape: one list, entries from both branches."""
        a, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.25)
        b, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=2.0)

        layers = _collect_energy_layers(a + b)
        assert [p for _, p in layers] == [pytest.approx(0.25), pytest.approx(2.0)]

    def test_one_node_fanned_across_many_entries_is_still_one_layer(self):
        """The counting trap: attach writes the layer onto every entry."""
        cond = _conditioning() + _conditioning() + _conditioning()
        out, _ = RadianceEnergyMask().apply(cond, _half_mask(), priority=0.5)

        assert len(_collect_energy_layers(out)) == 1

    def test_a_shallow_copy_of_the_meta_dict_does_not_double_count(self):
        """`conditioning_set_values` and friends copy the dict, not the layer."""
        out, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.5)
        copied = [(t, dict(meta)) for t, meta in out]

        assert len(_collect_energy_layers(out + copied)) == 1

    def test_the_priorities_add_where_the_masks_overlap(self):
        cond = torch.full((1, 4, 4, 4), 5.0)
        uncond = torch.ones(1, 4, 4, 4)
        full = torch.ones(1, 4, 4)

        patch = _make_energy_cfg_patch([(full, 0.25), (full, 0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        # 1 + 0.25 + 0.5 = 1.75  →  1 + 4 * 1.75 = 8.0
        assert out[0, 0, 0, 0].item() == pytest.approx(8.0)

    def test_disjoint_masks_each_apply_only_to_their_own_region(self):
        cond = torch.full((1, 4, 8, 8), 5.0)
        uncond = torch.ones(1, 4, 8, 8)

        left = torch.zeros(1, 8, 8)
        left[:, :, :4] = 1.0
        right = torch.zeros(1, 8, 8)
        right[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch([(left, 0.5), (right, 1.0)])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out[0, 0, 0, 0].item() == pytest.approx(7.0)   # 1 + 4*1.5
        assert out[0, 0, 0, 7].item() == pytest.approx(9.0)   # 1 + 4*2.0

    def test_a_stack_of_negatives_suppresses_but_never_inverts(self):
        """Two -1.0 layers would give a modifier of -1: guidance backwards."""
        cond = torch.full((1, 4, 4, 4), 5.0)
        uncond = torch.ones(1, 4, 4, 4)
        full = torch.ones(1, 4, 4)

        patch = _make_energy_cfg_patch([(full, -1.0), (full, -1.0)])
        out = patch(_args(cond, uncond, cfg=1.0))

        # Clamped to 0 → cond_eps == uncond, not uncond - (cond-uncond).
        assert out[0, 0, 0, 0].item() == pytest.approx(1.0)

    def test_layers_of_different_resolutions_are_each_resampled(self):
        cond = torch.full((1, 4, 16, 16), 5.0)
        uncond = torch.ones(1, 4, 16, 16)

        patch = _make_energy_cfg_patch([
            (torch.ones(1, 512, 512), 0.25),
            (torch.ones(1, 8, 8), 0.25),
        ])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out[0, 0, 8, 8].item() == pytest.approx(1.0 + 4.0 * 1.5, abs=1e-3)

    def test_an_inert_layer_does_not_cancel_an_active_one(self):
        """priority=0 layers are dropped at the call site, not summed in."""
        first, _ = RadianceEnergyMask().apply(_conditioning(), _half_mask(), priority=0.0)
        second, _ = RadianceEnergyMask().apply(first, _half_mask(), priority=0.6)

        active = [(m, p) for m, p in _collect_energy_layers(second) if p != 0.0]
        assert [p for _, p in active] == [pytest.approx(0.6)]


# ─────────────────────────────────────────────────────────────────────────────
#  CFG patch
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyCfgPatch:

    def test_guidance_is_boosted_inside_the_mask_only(self):
        cond = torch.full((1, 4, 8, 8), 5.0)
        uncond = torch.ones(1, 4, 8, 8)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch([(mask, 0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        # cfg=1 → denoised == cond_eps. Unmasked keeps cond (5.0); masked gets
        # uncond + (cond-uncond)*1.5 = 1 + 4*1.5 = 7.0
        assert out[0, 0, 0, 0].item() == pytest.approx(5.0)
        assert out[0, 0, 0, 7].item() == pytest.approx(7.0)

    def test_a_zero_mask_is_a_no_op(self):
        cond = torch.randn(1, 4, 8, 8)
        uncond = torch.randn(1, 4, 8, 8)
        args = _args(cond, uncond)

        patch = _make_energy_cfg_patch([(torch.zeros(1, 8, 8), 0.9)])
        assert torch.allclose(patch(args), args["denoised"], atol=1e-5)

    def test_negative_priority_suppresses_the_region(self):
        cond = torch.full((1, 4, 4, 4), 5.0)
        uncond = torch.ones(1, 4, 4, 4)

        patch = _make_energy_cfg_patch([(torch.ones(1, 4, 4), -0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        # 1 + 4*0.5 = 3.0, i.e. half the guidance
        assert out[0, 0, 0, 0].item() == pytest.approx(3.0)

    def test_video_latents_do_not_crash(self):
        """The old reader unpacked B, C, H, W and raised ValueError on 5-D."""
        cond = torch.full((1, 16, 5, 8, 8), 5.0)
        uncond = torch.ones(1, 16, 5, 8, 8)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch([(mask, 0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out.shape == cond.shape
        # The mask is spatial, so it applies identically on every frame.
        for t in range(5):
            assert out[0, 0, t, 0, 0].item() == pytest.approx(5.0)
            assert out[0, 0, t, 0, 7].item() == pytest.approx(7.0)

    def test_a_mask_at_image_resolution_is_resized_to_the_latent(self):
        cond = torch.full((1, 4, 64, 64), 5.0)
        uncond = torch.ones(1, 4, 64, 64)

        patch = _make_energy_cfg_patch([(_half_mask(512, 512), 0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out[0, 0, 0, 0].item() == pytest.approx(5.0, abs=1e-3)
        assert out[0, 0, 0, 63].item() == pytest.approx(7.0, abs=1e-3)

    def test_a_single_mask_broadcasts_over_a_latent_batch(self):
        cond = torch.full((4, 4, 8, 8), 5.0)
        uncond = torch.ones(4, 4, 8, 8)

        patch = _make_energy_cfg_patch([(torch.ones(1, 8, 8), 0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out.shape == cond.shape
        assert out[3, 0, 0, 0].item() == pytest.approx(7.0)

    def test_a_mismatched_mask_batch_does_not_crash(self):
        """expand() cannot broadcast 3 → 4; the old reader tried anyway."""
        cond = torch.full((4, 4, 8, 8), 5.0)
        uncond = torch.ones(4, 4, 8, 8)

        patch = _make_energy_cfg_patch([(torch.ones(3, 8, 8), 0.5)])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out.shape == cond.shape

    def test_ltxav_packed_latent_no_ops_with_one_warning(self, monkeypatch):
        """LTX-AV's packed latent (comfy.utils.pack_latents) is (B, 1, N) --
        3D, below the 4-D floor this patch assumes. Must not corrupt the
        sample, and must warn once, not on every step."""
        import radiance.nodes.generate.sampler as ns

        warnings = []
        monkeypatch.setattr(ns.logger, "warning", lambda *a, **k: warnings.append(a))

        cond = torch.full((1, 1, 64), 5.0)
        uncond = torch.ones(1, 1, 64)
        args = _args(cond, uncond, cfg=1.0)

        patch = _make_energy_cfg_patch([(torch.ones(1, 4, 4), 0.5)])
        out1 = patch(args)
        out2 = patch(args)

        assert torch.equal(out1, args["denoised"])
        assert torch.equal(out2, args["denoised"])
        assert len(warnings) == 1

    def test_composes_through_the_post_cfg_chain_not_manual_wrapping(self):
        """EPS is registered on set_model_sampler_post_cfg_function, a list
        ComfyUI chains automatically (comfy/samplers.py:600-603) -- it takes
        no existing_cfg_fn and must not depend on the incoming
        args["denoised"] (an earlier chain link's output) to produce its own
        result, so it composes correctly regardless of chain position."""
        cond = torch.full((1, 4, 4, 4), 5.0)
        uncond = torch.ones(1, 4, 4, 4)

        patch = _make_energy_cfg_patch([(torch.ones(1, 4, 4), 0.5)])
        args = _args(cond, uncond, cfg=1.0)
        args["denoised"] = torch.zeros_like(cond)  # as if an earlier link already ran
        out = patch(args)

        assert out[0, 0, 0, 0].item() == pytest.approx(7.0), \
            "EPS's own result must not be discarded in favor of the incoming denoised value"

    def test_mismatched_cond_uncond_shapes_pass_through(self):
        args = {
            "cond_denoised": torch.randn(1, 4, 8, 8),
            "uncond_denoised": torch.randn(1, 4, 4, 4),
            "cond_scale": 7.0,
            "denoised": torch.zeros(1, 4, 8, 8),
        }

        patch = _make_energy_cfg_patch([(torch.ones(1, 8, 8), 0.5)])
        assert torch.equal(patch(args), args["denoised"])

    def test_the_resized_mask_is_reused_across_steps(self):
        """The mask resize is per-step work that only depends on geometry."""
        cond = torch.full((1, 4, 8, 8), 5.0)
        uncond = torch.ones(1, 4, 8, 8)

        patch = _make_energy_cfg_patch([(_half_mask(256, 256), 0.5)])
        first = patch(_args(cond, uncond, cfg=1.0))

        with_interpolate_banned = torch.nn.functional.interpolate
        try:
            def _boom(*a, **k):
                raise AssertionError("mask was re-interpolated on a later step")
            torch.nn.functional.interpolate = _boom
            second = patch(_args(cond, uncond, cfg=1.0))
        finally:
            torch.nn.functional.interpolate = with_interpolate_banned

        assert torch.equal(first, second)


class TestEnergyCfgPatchPackedMultiModality:
    """LTX-AV's packed (video+audio) latent arrives as a flat (B, 1, N)
    tensor. These pin the unpack -> mask video -> re-pack path that replaced
    the old silent no-op (see project_radiance_sampler_deferred memory).

    comfy.utils.pack_latents/unpack_latents aren't in the test stub (only
    ProgressBar is), so tests that need them monkeypatch in a faithful
    reimplementation of the real algorithm (comfy/utils.py: reshape each
    tensor to (B, 1, -1), concatenate on the last dim; the shapes list is
    what makes the inverse slicing possible).
    """

    @staticmethod
    def _fake_pack(latents):
        shapes = [tuple(t.shape) for t in latents]
        flat = [t.reshape(t.shape[0], 1, -1) for t in latents]
        return torch.cat(flat, dim=-1), shapes

    @staticmethod
    def _fake_unpack(combined, shapes):
        if len(shapes) <= 1:
            return [combined]
        out, remaining = [], combined
        for shape in shapes:
            cut = 1
            for d in shape[1:]:
                cut *= d
            piece, remaining = remaining[..., :cut], remaining[..., cut:]
            out.append(piece.reshape([piece.shape[0]] + list(shape[1:])))
        return out

    def _patch_comfy_utils(self, monkeypatch):
        import radiance.nodes.generate.sampler as ns
        monkeypatch.setattr(ns.comfy.utils, "pack_latents", self._fake_pack, raising=False)
        monkeypatch.setattr(ns.comfy.utils, "unpack_latents", self._fake_unpack, raising=False)
        return ns

    def test_video_stream_gets_masked_audio_stream_is_untouched(self, monkeypatch):
        ns = self._patch_comfy_utils(monkeypatch)

        video_cond = torch.full((1, 4, 8, 8), 5.0)
        video_uncond = torch.ones(1, 4, 8, 8)
        audio_cond = torch.full((1, 2, 10), 3.0)
        audio_uncond = torch.full((1, 2, 10), 2.0)

        packed_cond, shapes = ns.comfy.utils.pack_latents([video_cond, audio_cond])
        packed_uncond, _ = ns.comfy.utils.pack_latents([video_uncond, audio_uncond])

        mask = torch.zeros(1, 8, 8)
        mask[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch([(mask, 0.5)], latent_shapes=shapes)
        out = patch(_args(packed_cond, packed_uncond, cfg=1.0))

        out_video, out_audio = ns.comfy.utils.unpack_latents(out, shapes)

        # Same math as test_guidance_is_boosted_inside_the_mask_only.
        assert out_video[0, 0, 0, 0].item() == pytest.approx(5.0)
        assert out_video[0, 0, 0, 7].item() == pytest.approx(7.0)
        # Audio: EPS never touches it, plain CFG only (cfg=1 -> equals cond).
        assert torch.allclose(out_audio, audio_cond)

    def test_single_entry_latent_shapes_is_treated_as_non_packed(self):
        """len(latent_shapes) <= 1 means "not actually multi-modal"; must
        not attempt to unpack a single-stream latent that already has a
        real shape."""
        cond = torch.full((1, 4, 8, 8), 5.0)
        uncond = torch.ones(1, 4, 8, 8)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, 4:] = 1.0

        patch = _make_energy_cfg_patch([(mask, 0.5)], latent_shapes=[tuple(cond.shape)])
        out = patch(_args(cond, uncond, cfg=1.0))

        assert out[0, 0, 0, 0].item() == pytest.approx(5.0)
        assert out[0, 0, 0, 7].item() == pytest.approx(7.0)

    def test_falls_back_to_warning_when_unpacking_raises(self, monkeypatch):
        """If comfy.utils.unpack_latents fails for any reason (e.g. the
        latent doesn't actually match latent_shapes), the sampler must not
        crash; it falls through to the existing unsupported-shape warning."""
        import radiance.nodes.generate.sampler as ns

        def _boom(*a, **k):
            raise RuntimeError("shape mismatch")
        monkeypatch.setattr(ns.comfy.utils, "unpack_latents", _boom, raising=False)
        warnings = []
        monkeypatch.setattr(ns.logger, "warning", lambda *a, **k: warnings.append(a))

        cond = torch.full((1, 1, 64), 5.0)
        uncond = torch.ones(1, 1, 64)
        args = _args(cond, uncond, cfg=1.0)

        patch = _make_energy_cfg_patch(
            [(torch.ones(1, 4, 4), 0.5)],
            latent_shapes=[(1, 4, 2, 4, 4), (1, 2, 10)],
        )
        out = patch(args)

        assert torch.equal(out, args["denoised"])
        assert len(warnings) == 1
