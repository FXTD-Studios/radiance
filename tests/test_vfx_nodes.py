import pytest
import torch
import json
import os
import shutil

from radiance.nodes.vfx.masking import (
    RadianceSAMModelLoader,
    RadianceSAMGenerator,
    RadianceMultiMaskVisualPicker,
    RadianceLinearMatting,
)
from radiance.nodes.vfx.plate import (
    RadianceHDRGrainMatcher,
    RadianceSubpixelStabilizer,
)
from radiance.nodes.vfx.inpaint import (
    RadianceHDRCrop,
    RadianceHDRStitch,
    RadianceTemporalStitchStabilizer,
)
from radiance.nodes.vfx.mask_propagate import RadianceVideoMaskPropagator

def test_sam_nodes_are_hidden_and_refuse_to_pretend():
    # 3.5.0 ships no SAM runtime. The two nodes stay registered so saved
    # graphs open, are hidden from the menu, and fail loudly on execution
    # instead of returning discs drawn around the click points.
    for cls in (RadianceSAMModelLoader, RadianceSAMGenerator):
        assert getattr(cls, "DEPRECATED", False) is True
    with pytest.raises(RuntimeError, match="does not ship a SAM runtime"):
        RadianceSAMModelLoader().load("sam2.1_hiera_large.pt", "cpu", False, "float32")
    with pytest.raises(RuntimeError, match="does not ship a SAM runtime"):
        RadianceSAMGenerator().generate(
            image=torch.ones((1, 8, 8, 3)), sam_model={}, points="[[4, 4]]", point_labels="[1]")


def test_linear_matting_lists_only_what_it_runs():
    methods = RadianceLinearMatting.INPUT_TYPES()["required"]["method"][0]
    assert methods == ["GuidedFilter"]


def test_masking_suite():
    image = torch.ones((2, 64, 64, 3), dtype=torch.float32)
    mask = torch.zeros((2, 64, 64), dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0

    # 3. Test Picker
    picker = RadianceMultiMaskVisualPicker()
    masks_batch = torch.ones((4, 2, 64, 64), dtype=torch.float32)
    picked_mask = picker.pick(masks_batch, 1)[0]
    assert picked_mask.shape == (4, 64, 64)
    
    # 4. Test Matting (Guided Filter)
    matting = RadianceLinearMatting()
    alpha, fore = matting.apply(
        image=image,
        mask=mask,
        method="GuidedFilter",
        trimap_dilation=4,
        eps=1e-4
    )
    assert alpha.shape == (2, 64, 64)
    assert fore.shape == (2, 64, 64, 3)


def test_plate_suite():
    # 1. Test Exposure-Relative Grain Matcher
    target = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 5.0  # high HDR exposure
    ref = torch.ones((1, 32, 32, 3), dtype=torch.float32) * 2.0
    
    # Add random noise to simulate grain
    ref = ref + torch.randn_like(ref) * 0.1
    
    matcher = RadianceHDRGrainMatcher()
    grained = matcher.apply(
        target=target,
        reference=ref,
        intensity=1.0,
        kernel_size=3,
        r_gain=1.0,
        g_gain=1.0,
        b_gain=1.0
    )[0]
    
    assert grained.shape == (1, 32, 32, 3)
    # Check that it didn't clip the high dynamic range output (it stays around 5)
    assert torch.max(grained) > 4.0
    
    # 2. Test Subpixel Stabilizer
    seq = torch.ones((3, 32, 32, 3), dtype=torch.float32)
    stabilizer = RadianceSubpixelStabilizer()
    stab_seq, disp = stabilizer.apply(seq, 0, 16)
    assert stab_seq.shape == (3, 32, 32, 3)
    # 3 channels, not 2. This assertion used to pin (3, 32, 32, 2), which is not
    # a valid ComfyUI IMAGE -- and RETURN_TYPES declares displacements_xy as
    # IMAGE, so wiring it into PreviewImage or any downstream node crashed on
    # the missing third channel. Blue is the unused vector component.
    assert disp.shape == (3, 32, 32, 3)
    assert torch.equal(disp[..., 2], torch.zeros_like(disp[..., 2]))


def test_inpainting_suite():
    image = torch.ones((2, 64, 64, 3), dtype=torch.float32)
    mask = torch.zeros((2, 64, 64), dtype=torch.float32)
    mask[:, 16:48, 16:48] = 1.0  # active region
    
    # 1. Test Crop (multiples of 16)
    cropper = RadianceHDRCrop()
    crop_img, crop_mask, stitch_data = cropper.apply(image, mask, 1.5, 16)
    
    assert crop_img.shape[0] == 2
    assert crop_img.shape[1] % 16 == 0
    assert crop_img.shape[2] % 16 == 0
    assert stitch_data["ymin"] >= 0
    
    # 2. Test Stitch
    stitcher = RadianceHDRStitch()
    stitched_img, blend_mask = stitcher.apply(
        original_image=image,
        cropped_image=crop_img,
        cropped_mask=crop_mask,
        stitcher_data=stitch_data,
        blend_mode="Linear_Laplacian",
        feather_radius=4
    )
    assert stitched_img.shape == (2, 64, 64, 3)
    assert blend_mask.shape == (2, 64, 64)
    
    # 3. Test Temporal Stabilizer
    temp_stabilizer = RadianceTemporalStitchStabilizer()
    smoothed_masks = temp_stabilizer.apply(mask, 1.5)[0]
    assert smoothed_masks.shape == (2, 64, 64)


@pytest.mark.parametrize("blend_mode", ["Linear_Laplacian", "Linear_Gaussian", "Standard"])
def test_stitch_single_frame_every_blend_mode(blend_mode):
    """A still is B == 1. The default Laplacian path squeezed the batch axis
    away and then did a 4-D permute, so it raised on every single frame; the
    suite only ever stitched a batch of two. Found by the live 3.5 run."""
    image = torch.full((1, 64, 64, 3), 0.25)
    mask = torch.zeros((1, 64, 64))
    mask[:, 16:48, 16:48] = 1.0
    crop_img, crop_mask, data = RadianceHDRCrop().apply(image, mask, 1.5, 16)
    crop_img = crop_img * 0 + 4.0          # scene-linear, above 1.0
    out, blend = RadianceHDRStitch().apply(image, crop_img, crop_mask, data, blend_mode, 4)
    assert out.shape == (1, 64, 64, 3)
    assert blend.shape == (1, 64, 64)
    assert float(out[0, 32, 32, 0]) > 3.0, "the crop was not composited back"
    assert float(out[0, 2, 2, 0]) == pytest.approx(0.25, abs=1e-3), "outside the mask changed"


def test_mask_propagator_warps_a_keyframe():
    propagator = RadianceVideoMaskPropagator()
    masks_seq = torch.zeros((3, 64, 64), dtype=torch.float32)
    masks_seq[0, 10:20, 10:20] = 1.0  # keyframe

    # Flow vectors shape [3, H, W, 3] representing +2 pixels shift
    flow = torch.zeros((3, 64, 64, 3), dtype=torch.float32)
    flow[..., 0] = 2.0  # shift right

    propagated = propagator.propagate(masks_seq, flow, "Forward")[0]

    assert propagated.shape == (3, 64, 64)
    # Frame 1 received the keyframe warped 2 pixels right
    assert propagated[1, 15, 17].item() > 0.5


def test_roto_node_is_removed():
    import radiance
    assert "RadianceVectorMaskDraw" not in radiance.NODE_CLASS_MAPPINGS


def test_mask_propagator_follows_the_motion():
    from radiance.nodes.vfx.motion import RadianceOpticalFlow
    H = W = 96
    torch.manual_seed(0)
    tex = torch.rand(24, 24)
    imgs = torch.zeros(4, H, W, 3)
    masks = torch.zeros(4, H, W)
    for i in range(4):
        x0 = 20 + 6 * i
        imgs[i, 30:54, x0:x0 + 24, :] = tex[..., None] * 0.8 + 0.2
        masks[i, 30:54, x0:x0 + 24] = 1
    vec = RadianceOpticalFlow().analyze(imgs, "Medium", 1.0, False)[0]

    def iou(a, b):
        return float((a * b).sum() / (a + b - a * b).sum())
    seed = masks.clone(); seed[1:] = 0
    fwd = RadianceVideoMaskPropagator().propagate(seed, vec, "Forward")[0]
    assert min(iou((fwd[i] > 0.5).float(), masks[i]) for i in range(4)) > 0.85
    seed = masks.clone(); seed[:3] = 0
    bwd = RadianceVideoMaskPropagator().propagate(seed, vec, "Backward")[0]
    assert min(iou((bwd[i] > 0.5).float(), masks[i]) for i in range(4)) > 0.85


def test_camera_sync_reads_shutter_alias_frames_and_refuses_abc(tmp_path):
    from radiance.nodes.vfx.camera import RadianceCameraSync
    node = RadianceCameraSync()
    cam, fl, fs, sh = node.sync('{"focal_length": 50, "shutter": 90}', 0)
    assert (fl, sh) == (50.0, 90) and cam["defaults_used"] == ["f_stop"]
    anim = '{"frames": [{"focal_length": 24}, {"focal_length": 85}]}'
    assert node.sync(anim, 1)[1] == 85.0 and node.sync(anim, 9)[1] == 85.0
    with pytest.raises(ValueError, match="Alembic"):
        node.sync("{}", 0, camera_file=str(tmp_path / "cam.abc"))
    with pytest.raises(FileNotFoundError):
        node.sync("{}", 0, camera_file=str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="not valid JSON"):
        node.sync("{focal", 0)


def test_motion_blur_conserves_energy_when_asked():
    from radiance.nodes.vfx.motion_blur import RadianceMotionBlur
    img = torch.full((1, 32, 32, 3), 0.1)
    img[:, 16, 16, :] = 50.0
    vec = torch.zeros(1, 32, 32, 3)
    vec[..., 0] = 8.0
    out = RadianceMotionBlur().apply(img, vec, 360.0, 9, True)[0]
    assert abs(float(out.sum()) - float(img.sum())) / float(img.sum()) < 0.02
    assert float(out[0, 5, 5, 0]) == pytest.approx(0.1, abs=1e-4)
