import pytest
import numpy as np
import matplotlib.pyplot as plt
from auto4dstem.Viz.viz import hist_plotter  

@pytest.fixture
def mock_image_data():
    """Create mock image data for testing."""
    return np.random.randn(100, 100)

def test_hist_plotter_creation(mock_image_data):
    """Test that hist_plotter creates a histogram correctly."""
    fig, ax = plt.subplots()
    hist_plotter(ax, mock_image_data)
    
    # Ensure that a histogram was created
    assert len(ax.patches) > 0, "The histogram should have been created."

    # Check that the number of bins is correct
    counts, bin_edges = np.histogram(mock_image_data.reshape(-1), bins=200, range=[-0.03, 0.03])
    assert len(ax.patches) == 200, f"Expected 200 bins, but got {len(ax.patches)}."

    # Cleanup
    plt.close(fig)

def test_hist_plotter_with_custom_params(mock_image_data):
    """Test that hist_plotter handles custom color, alpha, and clim parameters correctly."""
    fig, ax = plt.subplots()
    hist_plotter(ax, mock_image_data, color="red", alpha=0.5, clim=[-0.1, 0.1])
    
    # Ensure that the histogram color is correct
    for patch in ax.patches:
        assert patch.get_facecolor()[:3] == (1.0, 0.0, 0.0), "The histogram color should be red."
    
    # Ensure that the alpha value is correct
    for patch in ax.patches:
        assert patch.get_alpha() == 0.5, "The histogram alpha should be 0.5."

    # Check that the histogram range matches the custom clim
    counts, bin_edges = np.histogram(mock_image_data.reshape(-1), bins=200, range=[-0.1, 0.1])
    assert len(ax.patches) == 200, f"Expected 200 bins, but got {len(ax.patches)}."

    # Cleanup
    plt.close(fig)
