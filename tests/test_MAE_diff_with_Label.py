import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest import mock
from auto4dstem.Viz.viz import MAE_diff_with_Label

def test_MAE_diff_with_Label(tmp_path):
    # Mock data for the test
    diff_list = [np.random.rand(10, 10) - 0.5 for _ in range(8)]
    diff_range = [-0.1, 0.1]
    rotation_range = [-10, 10]
    noise_intensity = 0.5

    # Set up a temporary folder for saving the figure
    folder_name = tmp_path
    save_path = folder_name / f'Performance_Comparison_{format(noise_intensity, ".2f")}Percent_BKG.svg'

    # Mock the add_colorbar function to avoid unnecessary complexities during the test
    with mock.patch('auto4dstem.Viz.viz.add_colorbar') as mock_add_colorbar:
        # Run the function
        MAE_diff_with_Label(
            diff_list=diff_list,
            diff_range=diff_range,
            rotation_range=rotation_range,
            noise_intensity=noise_intensity,
            folder_name=str(folder_name),
            save_figure=True
        )

        # Ensure that the figure was saved correctly
        assert save_path.exists(), f"Figure not saved to {save_path}"

        # Ensure that add_colorbar was called the correct number of times (once per subplot)
        assert mock_add_colorbar.call_count == 8, "add_colorbar not called the correct number of times"
