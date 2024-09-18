import pytest
import numpy as np
import scipy as sp
from auto4dstem.Viz.viz import strain_tensor 

@pytest.fixture
def mock_affine_matrices():
    """Create a mock affine matrix dataset for testing."""
    return np.random.randn(100, 100, 2, 2) * 0.1 + np.eye(2)  # Near identity matrices with small random noise

@pytest.fixture
def mock_reference_region():
    """Define a mock reference region."""
    return (30, 60, 10, 40)

def test_strain_tensor_output_shape(mock_affine_matrices):
    """Test that the function returns output arrays with the correct shapes."""
    im_size = (100, 100)
    exx_ae, eyy_ae, exy_ae = strain_tensor(mock_affine_matrices, im_size)
    
    assert exx_ae.shape == im_size, "Output exx_ae should have the correct shape."
    assert eyy_ae.shape == im_size, "Output eyy_ae should have the correct shape."
    assert exy_ae.shape == im_size, "Output exy_ae should have the correct shape."

def test_strain_tensor_identity_matrices():
    """Test the function with identity matrices."""
    M_init = np.array([np.eye(2) for _ in range(10000)]).reshape(100, 100, 2, 2)  # Identity matrices
    im_size = (100, 100)
    exx_ae, eyy_ae, exy_ae = strain_tensor(M_init, im_size)
    
    # Expectation: no deformation, hence zeros in exx, eyy, exy
    np.testing.assert_array_almost_equal(exx_ae, np.zeros(im_size), decimal=6)
    np.testing.assert_array_almost_equal(eyy_ae, np.zeros(im_size), decimal=6)
    np.testing.assert_array_almost_equal(exy_ae, np.zeros(im_size), decimal=6)

def test_strain_tensor_reference_region(mock_affine_matrices, mock_reference_region):
    """Test that the function correctly handles the reference region."""
    im_size = (100, 100)
    exx_ae, eyy_ae, exy_ae = strain_tensor(mock_affine_matrices, im_size, ref_region=mock_reference_region)
    
    # Compute the expected transformation for the reference region
    M_ref = np.median(mock_affine_matrices[mock_reference_region[0]:mock_reference_region[1],
                                           mock_reference_region[2]:mock_reference_region[3]], axis=(0, 1))
    
    for rx in range(mock_reference_region[0], mock_reference_region[1]):
        for ry in range(mock_reference_region[2], mock_reference_region[3]):
            T = mock_affine_matrices[rx, ry] @ np.linalg.inv(M_ref)
            u, p = sp.linalg.polar(T, side='left')
            transformation = np.array([
                [p[0, 0] - 1, p[0, 1]],
                [p[0, 1], p[1, 1] - 1],
            ])
            assert np.isclose(exx_ae[rx, ry], transformation[1, 1], atol=1e-6)
            assert np.isclose(eyy_ae[rx, ry], transformation[0, 0], atol=1e-6)
            assert np.isclose(exy_ae[rx, ry], transformation[0, 1], atol=1e-6)
