import unittest
import torch
import numpy as np
import math
import sys
from unittest.mock import MagicMock
import pytest

# Only the tests marked @pytest.mark.real_torch below need genuine tensors; the
# rest run fine against conftest's stub. Opt out of the automatic module-level
# skip so they keep running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


# Mock ComfyUI specific modules before importing
sys.modules['aiohttp'] = MagicMock()
sys.modules['server'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.nested_tensor'] = MagicMock()

try:
    from radiance.viewer_utils import (
        _lut_logc3, _idt_logc3,
        _lut_vlog, _idt_vlog,
        _M_SRGB_TO_ACESCG, apply_lut, LUT_MODES
    )
    from radiance.sampler_utils import (
        gradual_sigma_blend,
        flux_shift_sigmas
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from viewer_utils import (
        _lut_logc3, _idt_logc3,
        _lut_vlog, _idt_vlog,
        _M_SRGB_TO_ACESCG, apply_lut, LUT_MODES
    )
    from sampler_utils import (
        gradual_sigma_blend,
        flux_shift_sigmas
    )


class TestViewerMathTransforms(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def test_logc3_invertibility(self):
        """Test that LogC3 encoding and IDT decoding are mathematically invertible."""
        test_values = np.array([0.0, 0.18, 1.0, 10.0, 50.0], dtype=np.float32)
        
        # Linear -> LogC3 -> Linear
        encoded = _lut_logc3(test_values)
        decoded = _idt_logc3(encoded)
        
        # Check precision up to 1e-4
        np.testing.assert_allclose(decoded, test_values, rtol=1e-3, atol=1e-4)

    def test_vlog_invertibility(self):
        """Test that V-Log encoding and IDT decoding are mathematically invertible."""
        test_values = np.array([0.0, 0.18, 1.0, 10.0], dtype=np.float32)
        
        encoded = _lut_vlog(test_values)
        decoded = _idt_vlog(encoded)
        
        np.testing.assert_allclose(decoded, test_values, rtol=1e-3, atol=1e-4)

    @pytest.mark.real_torch
    def test_apply_lut_passthrough(self):
        """Test that passthrough LUT mode does not alter the tensor."""
        dummy_tensor = torch.rand((2, 3, 256, 256))
        # Find index of Passthrough, default is 0 if not found
        try:
            pt_idx = LUT_MODES.index("Passthrough")
        except ValueError:
            pt_idx = 0
            
        result = apply_lut(dummy_tensor.clone(), pt_idx)
        self.assertTrue(torch.equal(dummy_tensor, result))

    def test_matrix_acescg(self):
        """Test the 32-bit float matrix for sRGB to ACEScg conversion."""
        # _M_SRGB_TO_ACESCG should be a 3x3 matrix
        self.assertEqual(_M_SRGB_TO_ACESCG.shape, (3, 3))
        
        # Verify the sum of the rows for D65 whitepoint preservation
        # White in sRGB should be White in ACEScg
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        aces_white = np.matmul(white, _M_SRGB_TO_ACESCG.T)
        
        # It should be close to 1.0 in all channels
        np.testing.assert_allclose(aces_white, white, rtol=1e-3, atol=1e-3)


class TestSamplerMathTransforms(unittest.TestCase):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    @pytest.mark.real_torch
    def test_flux_shift_sigmas(self):
        """Test flux sigma shifting logic."""
        sigmas = torch.tensor([10.0, 5.0, 1.0, 0.1, 0.0])
        shift = 3.0
        
        shifted = flux_shift_sigmas(sigmas, shift)
        
        # For shift=1.0, it should be identical
        self.assertTrue(torch.equal(flux_shift_sigmas(sigmas, 1.0), sigmas))
        
        # For shift=3.0, it should be different and mathematically correct
        denominator = 1.0 + (shift - 1.0) * sigmas
        expected = shift * sigmas / denominator
        torch.testing.assert_close(shifted, expected)

    # Order-dependent without real torch: another module in the suite swaps the
    # shared stub for a numpy-backed shim whose indexing yields plain floats.
    @pytest.mark.real_torch
    def test_gradual_sigma_blend(self):
        """Test smooth blending between two sigma schedules."""
        sigmas_a = torch.tensor([10.0, 5.0, 2.0, 1.0])
        sigmas_b = torch.tensor([8.0, 4.0, 1.5, 0.5])
        
        blend_steps = 2
        blended = gradual_sigma_blend(sigmas_a, sigmas_b, blend_steps)
        
        # The blend modifies sigmas_b in-place (clone returned).
        # The first element should be the last element of sigmas_a.
        self.assertAlmostEqual(blended[0].item(), sigmas_a[-1].item())
        
        # The last elements beyond blend_steps should remain unchanged from sigmas_b
        self.assertAlmostEqual(blended[-1].item(), sigmas_b[-1].item())


if __name__ == '__main__':
    unittest.main()
