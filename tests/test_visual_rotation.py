import pytest
import numpy as np
import matplotlib.pyplot as plt
from auto4dstem.Viz.viz import visual_rotation  # Replace with the actual module name

# Mock add_colorbar function
def add_colorbar(im, ax):
    plt.colorbar(im, ax=ax)

@pytest.fixture
def mock_rotation_data():
    """Create mock rotation data for testing."""
    return np.array([
        [np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))],
        [np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))],
        [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))]
    ])

@pytest.fixture
def mock_classification_data():
    """Create mock classification data for testing."""
    classification = np.zeros((256, 256))
    sample_index = np.random.choice(np.arange(256*256), 500, replace=False)
    bkg_index = np.array([i for i in range(256*256) if i not in sample_index])
    classification.flat[sample_index] = 1
    return classification, sample_index, bkg_index

def test_visual_rotation_output(mock_rotation_data):
    """Test that the visual_rotation function returns the correct adjusted rotation values."""
    result = visual_rotation(mock_rotation_data, img_size=(1, 3), clim=[0, 60], save_figure = False)
    expected_result = np.array([10, 20, 30]).reshape(1, 3)  # Expected degree values
    print('rotation_out:',result)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=1)

def test_visual_rotation_plot(mock_rotation_data):
    """Test that the visual_rotation function correctly generates plots."""
    visual_rotation(mock_rotation_data, img_size=(1, 3), clim=[0, 60], save_figure = False)
    
    # Check if a figure was created
    assert len(plt.get_fignums()) > 0, "A figure should have been created."
    
    # Cleanup
    plt.close()

def test_visual_rotation_save_figure(mock_rotation_data, tmp_path):
    """Test that the figure is saved correctly."""
    save_path = tmp_path
    name_ = 'test_figure'
    visual_rotation(mock_rotation_data, img_size=(1, 3), clim=[0, 60], folder_name=str(save_path),title_name = name_)
    
    # Check if the file was saved
    saved_file = save_path / f'Rotation_map_on_{name_}.svg'
    assert saved_file.exists(), "The plot should be saved as an SVG file."
    
    # Cleanup
    plt.close()
