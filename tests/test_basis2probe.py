import pytest
import numpy as np
from auto4dstem.Viz.viz import basis2probe 

def test_basis2probe_identity():
    """Test that basis2probe works correctly for identity matrices."""
    rotation = np.array([[1, 0], [1, 0]])  # No rotation
    scale_shear = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])  # Identity scaling
    
    result = basis2probe(rotation, scale_shear)
    
    expected = np.array([
        [[1, 0], [0, 1]],  # Identity matrix
        [[1, 0], [0, 1]]   # Identity matrix
    ])
    
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

def test_basis2probe_scaling():
    """Test that basis2probe handles scaling correctly."""
    rotation = np.array([[1, 0], [1, 0]])  # No rotation
    scale_shear = np.array([[2, 0, 0, 3], [4, 0, 0, 5]])  # Different scaling factors
    
    result = basis2probe(rotation, scale_shear)
    
    expected = np.array([
        [[0.5, 0], [0, 1/3]],  # Inverse scaling for first matrix
        [[0.25, 0], [0, 0.2]]  # Inverse scaling for second matrix
    ])
    
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

def test_basis2probe_rotation():
    """Test that basis2probe handles rotation correctly."""
    rotation = np.array([[0, 1], [-1, 0]])  # 90 degree rotation
    scale_shear = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])  # Identity scaling
    
    result = basis2probe(rotation, scale_shear)
    
    expected = np.array([
        [[0, 1], [-1, 0]],  # 90 degree rotation matrix
        [[-1, 0], [0, -1]]   # -90 degree rotation matrix
    ])
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

def test_basis2probe_combined():
    """Test that basis2probe correctly combines rotation and scaling."""
    rotation = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [0, 1]])  # 45 degree rotation and no rotation
    scale_shear = np.array([[2, 0, 0, 2], [1, 0, 0, 1]])  # Uniform scaling for first, identity for second
    
    result = basis2probe(rotation, scale_shear)
    
    expected = np.array([
        [[np.sqrt(2)/4, np.sqrt(2)/4], [-np.sqrt(2)/4, np.sqrt(2)/4]],  # Combined matrix
        [[0, 1], [-1, 0]]  # Identity matrix
    ])
    print('2......................',result)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)
