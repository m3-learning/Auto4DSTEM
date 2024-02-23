import pytest
import os
import numpy as np
from auto4dstem.Data.DataProcess import STEM4D_DataSet

def test_initialization():
    dataset = STEM4D_DataSet(data_dir='data.npy')
    assert dataset.data_dir == 'data.npy'
    assert dataset.background_weight == 0.10
    assert dataset.x_size == 200
    assert dataset.y_size == 200

def test_load_npy_data():
    # Create a mock numpy file
    data = np.random.rand(10, 10, 28, 28)
    np.save('data.npy', data)
    
    dataset = STEM4D_DataSet(data_dir='data.npy',
                            crop = ((4,24),(4,24)),
                            )
    assert dataset.stem4d_data.shape == (100, 1, 20, 20)  # Expected shape after formatting

def test_rotation():
    rotation = np.array([[1, 0], [0, 1]])
    stem4d_data = np.random.rand(2, 10, 10)
    
    dataset = STEM4D_DataSet(data_dir='data.npy')
    dataset.rotate_data(stem4d_data, rotation)
    
    assert len(dataset.stem4d_rotation) == 2

def test_rotation_mismatch():
    rotation = np.array([[1, 0], [0, 1], [1, 1]])
    stem4d_data = np.random.rand(2, 10, 10)
    
    dataset = STEM4D_DataSet(data_dir='data.npy')
    
    with pytest.raises(ValueError, match='The rotation size and image size do not match each other'):
        dataset.rotate_data(stem4d_data, rotation)
    
    os.remove('data.npy')
