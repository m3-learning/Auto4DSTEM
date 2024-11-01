import pytest
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import field
from unittest import mock
from auto4dstem.nn.Train_Function import TrainClass

# Assuming TrainClass is defined and imported correctly
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
def train_class_fixture(mock_stem4d_data, mock_h5_file,tmp_path):
    """Fixture to provide an instance of TrainClass with a temporary data directory."""
    rotation_angles = np.array([[np.cos(np.pi/6), np.sin(np.pi/6)] for _ in range(mock_stem4d_data.shape[0]*mock_stem4d_data.shape[1])])
    return TrainClass(data_dir=str(mock_h5_file),transpose= (0, 1, 2, 3),learned_rotation = rotation_angles, folder_path = str(tmp_path))

def test_train_class_initialization(train_class_fixture):
    """Test to ensure that TrainClass initializes with default parameters correctly."""
    assert isinstance(train_class_fixture, TrainClass)
    assert train_class_fixture.device == torch.device("cpu")
    assert train_class_fixture.seed == 42
    assert train_class_fixture.crop == ((28, 228), (28, 228))
    assert train_class_fixture.learning_rate == 3e-5

    # Check that the dataset has been reset (calling reset_dataset)
    assert hasattr(train_class_fixture, 'data_class')
    assert hasattr(train_class_fixture, 'data_set')
    assert hasattr(train_class_fixture, 'rotate_data')
    

def test_crop_one_image(train_class_fixture):
    """Test the crop_one_image method."""
    
    # Run the method
    train_class_fixture.crop_one_image(index=0, clim=[0, 1], cmap="viridis")

    # Check that the pick_1_image attribute is correctly assigned
    assert hasattr(train_class_fixture, 'pick_1_image')
    assert train_class_fixture.pick_1_image.shape == (256, 256)

def test_visual_noise(train_class_fixture,tmp_path):
    """Test the visual_noise method."""
    mock_stem_data = np.random.rand(256, 256)
    train_class_fixture.pick_1_image = mock_stem_data
    # Run the method
    noise_level = [0.1, 0.5]
    train_class_fixture.visual_noise(noise_level=noise_level, clim=[0, 1], file_name="test", cmap="viridis")
    # Check that the file was saved
    saved_file = tmp_path / f'test_generated_{noise_level}_noise.svg'
    assert saved_file.exists()

def test_lr_circular(train_class_fixture):
    """Test the lr_circular method."""
    lr = train_class_fixture.lr_circular(epoch=10, step_size_up=20, min_rate=3e-5, max_rate=2e-4)
    assert lr == 0.000115  # Expected value based on the input parameters

def test_reset_model(train_class_fixture):
    """Test the reset_model method to ensure model components are correctly initialized."""

    # Run the reset_model method
    encoder, decoder, join, optimizer = train_class_fixture.reset_model()

    assert encoder is not None
    assert decoder is not None
    assert join is not None
    assert optimizer is not None

def test_reset_loss_class(train_class_fixture):
    """Test the reset_model method to ensure model components are correctly initialized."""

    # Run the reset_model method
    loss_fn = train_class_fixture.reset_model()

    assert loss_fn is not None

def test_show_pickup_dots_with_six_dots(train_class_fixture):
    """Test the show_pickup_dots function with exactly 6 dots."""
    x_axis = [0, 0, 0, 1, 2, 3]
    y_axis = [0, 1, 2, 0, 1, 2]
    
    train_class_fixture.show_pickup_dots(x_axis, y_axis)
    expected_indices = []
    for i in range(6):
        expected_indices.append(y_axis[i] * 5 + x_axis[i])
    # Check that sample_series contains the correct indices
    np.testing.assert_array_equal(train_class_fixture.sample_series, expected_indices)

def test_show_pickup_dots_sets_mean_image(train_class_fixture):
    """Test that mean_real_space_domain is correctly set if not already initialized."""
    x_axis = [0, 0, 0, 1, 2, 3]
    y_axis = [0, 1, 2, 0, 1, 2]
    
    # Before running the function, mean_real_space_domain should be None
    assert train_class_fixture.mean_real_space_domain is None
    
    train_class_fixture.show_pickup_dots(x_axis, y_axis)
    
    # After running the function, mean_real_space_domain should be set
    assert train_class_fixture.mean_real_space_domain is not None

def test_show_transforming_sample(train_class_fixture,tmp_path):
    """Test that mean_real_space_domain is correctly set if not already initialized."""
    x_axis = [0, 0, 0, 1, 2, 3]
    y_axis = [0, 1, 2, 0, 1, 2]
    # Before running the function, mean_real_space_domain should be None
    assert train_class_fixture.mean_real_space_domain is None
    train_class_fixture.show_pickup_dots(x_axis, y_axis)
    assert train_class_fixture.join is None
    # create join model and load into class
    train_class_fixture.revise_affine = False
    _,_,join,_ = train_class_fixture.reset_model()
    train_class_fixture.join = join
    train_class_fixture.show_transforming_sample(file_name='test')
    saved_file = tmp_path / f'test_show_affine_process_of_pickup_samples.svg'
    # After running the function, figure should be saved correctly
    assert saved_file.exists()

def test_predict(train_class_fixture, tmp_path):

    # Call the predict function
    train_class_fixture.revise_affine = False
    _,_,join,_ = train_class_fixture.reset_model()
    train_class_fixture.join = join
    train_class_fixture.predict(save_strain=True, save_rotation=True, save_translation=True, save_classification=True, save_base=True, file_name="test")

    # Check that each np.save call was made with correct arguments
    saved_file = tmp_path / f'test_1_train_process_scale_shear.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_1_train_process_rotation.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_1_train_process_translation.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_1_train_process_classification.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_1_train_process_generated_base.npy'
    assert saved_file.exists()

def test_predict_with_sample_index(train_class_fixture, tmp_path):

    # Call the predict function
    train_class_fixture.revise_affine = False
    _,_,join,_ = train_class_fixture.reset_model()
    train_class_fixture.join = join
    sample_index = np.arange(10)
    train_class_fixture.predict(train_process='2',sample_index = sample_index, save_strain=True, 
                                save_rotation=True, save_translation=True, save_classification=True, save_base=True, file_name="test")

    # Check that each np.save call was made with correct arguments
    saved_file = tmp_path / f'test_2_train_process_scale_shear.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_2_train_process_rotation.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_2_train_process_translation.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_2_train_process_classification.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_2_train_process_generated_base.npy'
    assert saved_file.exists()

def test_save_predict(train_class_fixture, tmp_path):

    # Call the predict function
    train_class_fixture.revise_affine = False
    _,_,join,_ = train_class_fixture.reset_model()
    train_class_fixture.join = join
    train_class_fixture.predict()
    train_class_fixture.save_predict(save_strain=True, save_rotation=True, save_translation=True,file_name="test")

    # Check that each np.save call was made with correct arguments
    saved_file = tmp_path / f'test_1_train_process_scale_shear.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_1_train_process_rotation.npy'
    assert saved_file.exists()
    saved_file = tmp_path / f'test_1_train_process_translation.npy'
    assert saved_file.exists()

def count_files_with_suffix(directory, suffix):
    """Helper function to count files with a particular suffix in a directory."""
    path = Path(directory)
    return len(list(path.glob(f"*{suffix}")))

def test_train_process(train_class_fixture, tmp_path):

    # Call the predict function
    train_class_fixture.revise_affine = False
    # Add mask 
    train_class_fixture.fixed_mask = [torch.ones([train_class_fixture.data_set.shape[-1],
                                                  train_class_fixture.data_set.shape[-2]],
                                                 dtype = torch.bool)]
    # Add folder path
    train_class_fixture.folder_path = str(tmp_path)
    # Set epcoh
    train_class_fixture.epochs = 1
    # Train the model
    train_class_fixture.train_process()
    # Check that each np.save call was made with correct arguments
    file_count = count_files_with_suffix(train_class_fixture.folder_path, ".pkl")
    
    assert file_count >0 , f"No .pkl files were found in {train_class_fixture.folder_path}"

    # Clean up: Remove all .pkl files created during the test
    for pkl_file in Path(train_class_fixture.folder_path).glob("*.pkl"):
        os.remove(pkl_file)
