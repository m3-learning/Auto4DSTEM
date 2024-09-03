import pytest
import numpy as np
import torch
import torch.nn.functional as F
from unittest import mock
import h5py
from auto4dstem.Data.DataProcess import data_translated  # Replace with the actual module name

@pytest.fixture
def mock_stem4d_data():
    """Creates a mock 4D STEM dataset"""
    return np.random.rand(10, 1, 128, 128)  # Simulated 4D STEM data

@pytest.fixture
def mock_h5_file(tmp_path, mock_stem4d_data):
    """Creates a mock HDF5 file with simulated 4D STEM data"""
    file_path = tmp_path / "test_data.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("output4D", data=mock_stem4d_data)
    return file_path

@pytest.fixture
def mock_translation():
    """Creates a mock translation matrix"""
    return np.random.uniform(-1, 1, size=(10, 2))  # Simulated translations

def test_load_data_from_h5(mock_h5_file, mock_translation, tmp_path):
    """Test loading data from an HDF5 file"""
    data_translated(
        data_path=str(mock_h5_file),
        translation=mock_translation,
        save_path=str(tmp_path / "output")
    )
    # Verify the output file exists
    output_file = tmp_path / "output_translated_version.npy"
    assert output_file.exists(), "The translated data file should be saved."

def test_transpose_and_crop(mock_h5_file, mock_translation, tmp_path):
    """Test transposing and cropping the dataset"""
    # Call the function with mock data
    data_translated(
        data_path=str(mock_h5_file),
        translation=mock_translation,
        crop=((2, 122), (2, 122)),
        transpose=(0, 1, 2, 3),
        save_path=str(tmp_path / "output")
    )
    # Load the translated data
    translated_data = np.load(tmp_path / "output_translated_version.npy")
    assert translated_data.shape == (10, 120, 120), "Data should be cropped to the correct size."

def test_apply_translation(mock_h5_file, mock_translation, tmp_path):
    """Test that translations are correctly applied"""
    data_translated(
        data_path=str(mock_h5_file),
        translation=mock_translation,
        crop=((2, 122), (2, 122)),
        save_path=str(tmp_path / "output")
    )
    translated_data = np.load(tmp_path / "output_translated_version.npy")
    assert translated_data.shape == (10, 120, 120), "The translated data should have the correct shape."
    # Additional checks can be done to verify specific translation effects

def test_save_translated_data(mock_h5_file, mock_translation, tmp_path):
    """Test that the translated data is saved correctly"""
    data_translated(
        data_path=str(mock_h5_file),
        translation=mock_translation,
        crop=((2, 122), (2, 122)),
        save_path=str(tmp_path / "output")
    )
    # Check if the file was saved
    output_file = tmp_path / "output_translated_version.npy"
    assert output_file.exists(), "The translated data file should be saved."
