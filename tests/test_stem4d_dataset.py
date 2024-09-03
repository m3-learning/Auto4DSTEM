import pytest
import numpy as np
from unittest import mock
from dataclasses import dataclass, field
import h5py
from skimage import filters
from auto4dstem.Data.DataProcess import STEM4D_DataSet 

@pytest.fixture
def mock_stem4d_data():
    """Creates a mock 4D STEM dataset"""
    return np.random.rand(5, 5, 256, 256)  # Simulated 4D STEM data

@pytest.fixture
def mock_h5_file(tmp_path, mock_stem4d_data):
    """Creates a mock HDF5 file with simulated 4D STEM data"""
    file_path = tmp_path / "test_data.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("output4D", data=mock_stem4d_data)
    return file_path

@pytest.fixture
def dataset_instance(mock_h5_file):
    """Returns an instance of STEM4D_DataSet initialized with the mock file"""
    return STEM4D_DataSet(data_dir=str(mock_h5_file))

def test_dataset_initialization(dataset_instance):
    """Test if the dataset initializes and computes sizes correctly"""
    assert dataset_instance.x_size == 200
    assert dataset_instance.y_size == 200
    assert dataset_instance.stem4d_data.shape == (25, 1, 200, 200)

def test_load_data(mock_h5_file, mock_stem4d_data):
    """Test if data loads correctly from an HDF5 file"""
    dataset = STEM4D_DataSet(data_dir=str(mock_h5_file))
    assert dataset.stem4d_data.shape == (25, 1, 200, 200)
    np.testing.assert_almost_equal(dataset.stem4d_data.squeeze(), mock_stem4d_data[:,:, 28:228, 28:228].reshape(-1,200,200), decimal=5)

def test_format_data(dataset_instance, mock_stem4d_data):
    """Test the format_data method for correct cropping and transposing"""
    cropped_data = mock_stem4d_data[:,:, 28:228, 28:228]
    dataset_instance.format_data(cropped_data)
    assert dataset_instance.stem4d_data.shape == (25, 1, 200, 200)

def test_generate_background_noise(dataset_instance, mock_stem4d_data):
    """Test the generate_background_noise method"""
    mock_stem4d_data = mock_stem4d_data[:,:,28:228,28:228].reshape(-1,1,200,200)
    dataset_instance.generate_background_noise(mock_stem4d_data, 0.1, 1e5)
    assert dataset_instance.stem4d_data.shape == mock_stem4d_data.shape
    assert np.any(dataset_instance.stem4d_data != mock_stem4d_data)  # Check that noise was actually added
    
def test_rotate_data(dataset_instance, mock_stem4d_data):
    """Test the rotate_data method"""
    mock_stem4d_data = mock_stem4d_data[:,:,28:228,28:228].reshape(-1,1,200,200)
    rotation_angles = np.array([[np.cos(np.pi/6), np.sin(np.pi/6)] for _ in range(mock_stem4d_data.shape[0]*mock_stem4d_data.shape[1])])
    dataset_instance.rotate_data(mock_stem4d_data, rotation_angles)
    assert hasattr(dataset_instance, 'stem4d_rotation')  # Ensure rotated data is stored
    assert len(dataset_instance.stem4d_rotation) == mock_stem4d_data.shape[0]*mock_stem4d_data.shape[1]

def test_filter_sobel(dataset_instance, mock_stem4d_data):
    """Test the filter_sobel method"""
    mock_stem4d_data = mock_stem4d_data[:,:,28:228,28:228].reshape(-1,1,200,200)
    dataset_instance.filter_sobel(mock_stem4d_data)
    assert np.all(dataset_instance.stem4d_data <= 2)  # Intensity should be scaled up to 2

