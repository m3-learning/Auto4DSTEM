import pytest
import numpy as np
import matplotlib.pyplot as plt
from auto4dstem.Viz.viz import compare_rotation  

# Mock add_colorbar function
def add_colorbar(im, ax):
    plt.colorbar(im, ax=ax)

@pytest.fixture
def mock_strain_map():
    """Create a mock strain map for testing."""
    strain_map = np.random.randn(4, 256, 256)
    return strain_map

@pytest.fixture
def mock_rotation_data():
    """Create mock rotation data for testing."""
    return np.array([[np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))] for _ in range(256*256)])

@pytest.fixture
def mock_classification_data():
    """Create mock classification data for testing."""
    classification = np.zeros((256, 256))
    sample_index = np.random.choice(np.arange(256*256), 500, replace=False)
    bkg_index = np.array([i for i in range(256*256) if i not in sample_index])
    classification.flat[sample_index] = 1
    return classification, sample_index, bkg_index

def test_compare_rotation_output(mock_strain_map, mock_rotation_data):
    """Test that compare_rotation returns the correct adjusted rotation values."""
    theta_correlation, theta_ae = compare_rotation(
        mock_strain_map, 
        mock_rotation_data, 
        img_size=(256, 256), 
        clim=[0, 60],
        save_figure=False
    )
    
    assert theta_correlation.shape == (256, 256), "Theta correlation should have the correct shape."
    assert theta_ae.shape == (256, 256), "Theta ae should have the correct shape."

def test_compare_rotation_plot(mock_strain_map, mock_rotation_data):
    """Test that compare_rotation correctly generates plots."""
    compare_rotation(
        mock_strain_map, 
        mock_rotation_data, 
        img_size=(256, 256), 
        clim=[0, 60],
        save_figure=False
    )
    
    # Check if a figure was created
    assert len(plt.get_fignums()) > 0, "A figure should have been created."
    
    # Cleanup
    plt.close()

def test_compare_rotation_save_figure(mock_strain_map, mock_rotation_data, tmp_path):
    """Test that the figure is saved correctly."""
    save_path = tmp_path
    name_ = 'test_figure'
    compare_rotation(
        mock_strain_map, 
        mock_rotation_data, 
        img_size=(256, 256), 
        clim=[0, 60],
        folder_name=str(save_path),
        title_name=name_,
    )

    # Check if the file was saved
    saved_file = save_path / f'Rotation_comparison_on_{name_}.svg'
    assert saved_file.exists(), "The plot should be saved as an SVG file."
    
    # Cleanup
    plt.close()
