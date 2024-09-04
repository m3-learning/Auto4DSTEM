import pytest
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Assuming add_colorbar is defined in your_module
from auto4dstem.Viz.viz import add_colorbar  # Replace with the actual module name

def test_add_colorbar():
    """Test if the colorbar is added correctly to a subplot"""
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    
    # Create a sample image data to plot
    data = np.random.rand(10, 10)
    norm = Normalize(vmin=0, vmax=1)
    
    # Display the image on the subplot
    im = ax.imshow(data, norm=norm)
    
    # Call the function to add a colorbar
    add_colorbar(im, ax)
    
    # Check if a colorbar was added
    assert len(fig.axes) == 2, "The figure should have two axes (main plot and colorbar)"
    
    # Clean up after the test
    plt.close(fig)

