import pytest
import matplotlib.pyplot as plt
from auto4dstem.Viz.viz import normalized_strain_matrices  

# Mock hist_plotter function
def hist_plotter(ax, data, color, clim):
    ax.hist(data, color=color, range=clim)

@pytest.fixture
def sample_strain_data():
    """Create sample strain data."""
    import numpy as np
    # Create tuples of sample data
    strain1 = (np.random.normal(0, 0.01, 1000), np.random.normal(0, 0.01, 1000),
               np.random.normal(0, 0.01, 1000), np.random.normal(0, 10, 1000))
    strain2 = (np.random.normal(0, 0.02, 1000), np.random.normal(0, 0.02, 1000),
               np.random.normal(0, 0.02, 1000), np.random.normal(0, 15, 1000))
    return [strain1, strain2]

def test_normalized_strain_matrices_not_enough_colors(sample_strain_data):
    """Test that the function returns an error if there aren't enough colors."""
    result = normalized_strain_matrices(sample_strain_data, color_list=['red'])
    assert result == "not enough color for show", "Function should return an error message when there aren't enough colors."

def test_normalized_strain_matrices_plot(sample_strain_data):
    """Test that the function correctly generates a plot."""
    result = normalized_strain_matrices(sample_strain_data, save_figure=False)
    
    # Check if a figure was created
    assert len(plt.get_fignums()) > 0, "A figure should have been created."
    
    # Cleanup
    plt.close()

def test_normalized_strain_matrices_save_figure(sample_strain_data, tmp_path):
    """Test that the plot is saved correctly."""
    file_path = tmp_path / "test_plot"
    normalized_strain_matrices(sample_strain_data, file_name=str(file_path))
    
    # Check if the file was saved
    saved_file = file_path.with_name(f"{file_path.name}_normalized_strain_histogram.svg")
    assert saved_file.exists(), "The plot should be saved as an SVG file."
    
    # Cleanup
    plt.close()
