import pytest
import numpy as np
import matplotlib.pyplot as plt
from auto4dstem.Viz.viz import visual_performance_plot

@pytest.fixture
def sample_data():
    """Create sample data for plotting."""
    x_list = np.linspace(0, 10, 50)
    auto = np.sin(x_list) / 10
    py4d = np.cos(x_list) / 10
    auto_yerr = np.random.rand(len(x_list)) / 100
    py4d_yerr = np.random.rand(len(x_list)) / 100
    return x_list, auto, py4d, auto_yerr, py4d_yerr

def test_visual_performance_plot_fill_between(sample_data, tmp_path):
    """Test the plot with fill_between=True."""
    x_list, auto, py4d, auto_yerr, py4d_yerr = sample_data
    
    # Call the function
    visual_performance_plot(
        x_list=x_list,
        auto=auto,
        py4d=py4d,
        auto_yerr=auto_yerr,
        py4d_yerr=py4d_yerr,
        fill_between=True,
        errorbar=False,
        save_figure=False,  # Don't save the figure to avoid file I/O in this test
    )
    
    # Check if the plot was created by counting axes
    assert len(plt.gcf().get_axes()) > 0, "The plot should have at least one axis"
    
    # Cleanup
    plt.close()

def test_visual_performance_plot_errorbar(sample_data, tmp_path):
    """Test the plot with errorbar=True."""
    x_list, auto, py4d, auto_yerr, py4d_yerr = sample_data
    
    # Call the function
    visual_performance_plot(
        x_list=x_list,
        auto=auto,
        py4d=py4d,
        auto_yerr=auto_yerr,
        py4d_yerr=py4d_yerr,
        fill_between=False,
        errorbar=True,
        save_figure=False,  # Don't save the figure to avoid file I/O in this test
    )
    
    # Check if the plot was created by counting axes
    assert len(plt.gcf().get_axes()) > 0, "The plot should have at least one axis"
    
    # Cleanup
    plt.close()

def test_visual_performance_plot_save_figure(sample_data, tmp_path):
    """Test if the plot is saved correctly."""
    x_list, auto, py4d, auto_yerr, py4d_yerr = sample_data
    save_path = tmp_path
    print('path',save_path)
    # Call the function
    visual_performance_plot(
        x_list=x_list,
        auto=auto,
        py4d=py4d,
        auto_yerr=auto_yerr,
        py4d_yerr=py4d_yerr,
        fill_between=True,
        errorbar=False,
        save_figure=True,
        folder_path=save_path,
        title=""
    )
    
    # Check if the file was saved
    saved_file = save_path.with_suffix(".svg")
    assert saved_file.exists(), "The plot should be saved as a .svg file"
    
    # Cleanup
    plt.close()
