import pytest
import numpy as np
import scipy as sp
from auto4dstem.Viz.viz import strain_tensor_for_real 
@pytest.fixture
def mock_affine_matrices():
    """Create mock affine matrices for testing."""
    return np.array([
        [[1.1, 0.1], [0.1, 0.9]],
        [[1.2, 0.2], [0.2, 0.8]],
        [[1.0, 0.0], [0.0, 1.0]]
    ])

def test_strain_tensor_for_real_output_shape(mock_affine_matrices):
    """Test that the function returns the correct output shapes."""
    im_size = (1, 3)  # Assuming 1x3 image size for this example
    exx_ae, eyy_ae, exy_ae = strain_tensor_for_real(mock_affine_matrices, im_size)
    
    assert exx_ae.shape == im_size, "Output exx_ae should have the correct shape."
    assert eyy_ae.shape == im_size, "Output eyy_ae should have the correct shape."
    assert exy_ae.shape == im_size, "Output exy_ae should have the correct shape."

def test_strain_tensor_for_real_identity():
    """Test the function with identity matrices."""
    M_init = np.array([np.eye(2), np.eye(2)])  # Identity matrices
    im_size = (1, 2)
    exx_ae, eyy_ae, exy_ae = strain_tensor_for_real(M_init, im_size)
    
    expected_exx = np.zeros(im_size)
    expected_eyy = np.zeros(im_size)
    expected_exy = np.zeros(im_size)
    
    np.testing.assert_array_almost_equal(exx_ae, expected_exx, decimal=6)
    np.testing.assert_array_almost_equal(eyy_ae, expected_eyy, decimal=6)
    np.testing.assert_array_almost_equal(exy_ae, expected_exy, decimal=6)

def test_strain_tensor_for_real_sample_index(mock_affine_matrices):
    """Test the function with a sample index provided."""
    im_size = (1, 3)
    sample_index = np.array([0, 3])  # Only two samples from the 4-pixel image
    
    exx_ae, eyy_ae, exy_ae = strain_tensor_for_real(mock_affine_matrices, im_size, sample_index=sample_index)
    
    # Check that only the sample indices are populated correctly
    expected_exx = [[-0.1, -0.2, 0. ]]
    expected_eyy = [[0.1, 0.2, 0. ]]
    expected_exy = [[0.1, 0.2, 0. ]]
    
    np.testing.assert_array_almost_equal(exx_ae, expected_exx, decimal=6)
    np.testing.assert_array_almost_equal(eyy_ae, expected_eyy, decimal=6)
    np.testing.assert_array_almost_equal(exy_ae, expected_exy, decimal=6)
