import torch
import numpy as np
import pytest
from radiance.hdr.processing import HDRExposureBlend

def test_blend_exposures_deghost_no_name_error():
    # Construct synthetic images
    # 2 exposure brackets of size 128x128x3
    low = torch.ones((1, 128, 128, 3), dtype=torch.float32) * 0.2
    high = torch.ones((1, 128, 128, 3), dtype=torch.float32) * 0.8
    
    blender = HDRExposureBlend()
    
    # Run with ghost_removal=True
    # This should not raise a NameError
    result, mask, info = blender.blend_exposures(
        low_exposure=low,
        high_exposure=high,
        blend_method="Mertens Fusion",
        ghost_removal=True
    )
    
    assert isinstance(result, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert "stops" in info

def test_blend_exposures_shadow_highlight_mask_not_discarded():
    # Construct synthetic images
    low = torch.ones((1, 128, 128, 3), dtype=torch.float32) * 0.2
    high = torch.ones((1, 128, 128, 3), dtype=torch.float32) * 0.8
    
    blender = HDRExposureBlend()
    
    # Run with blend_method="Shadow/Highlight Mask"
    result, mask, info = blender.blend_exposures(
        low_exposure=low,
        high_exposure=high,
        blend_method="Shadow/Highlight Mask",
        ghost_removal=False
    )
    
    assert isinstance(result, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    # The mask returned by _shadow_highlight_blend is a 3-channel visualization mask (shadow, midtone, highlight)
    # Stretched to match image dimensions.
    # Dimensions: (1, H, W, 3) or (H, W, 3)
    assert mask.ndim in (3, 4)
    assert mask.shape[-1] == 3
