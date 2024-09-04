import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest import mock
from auto4dstem.Viz.viz import Strain_Compare

# Sample test function
def test_Strain_Compare(tmp_path):
    # Mock input data
    diff_list = [np.random.randn(256, 256) for _ in range(8)]
    ae_xx_diff_range = [-0.03, 0.03]
    ae_yy_diff_range = [-0.03, 0.03]
    ae_xy_diff_range = [-0.03, 0.03]
    cross_xx_diff_range = [-0.03, 0.03]
    cross_yy_diff_range = [-0.03, 0.03]
    cross_xy_diff_range = [-0.03, 0.03]
    rotation_range = [-40, 30]
    ref_rotation_range = [-40, 30]
    title_name = "test"
    folder_name = str(tmp_path)

    # Call the function
    Strain_Compare(
        diff_list=diff_list,
        ae_xx_diff_range=ae_xx_diff_range,
        ae_yy_diff_range=ae_yy_diff_range,
        ae_xy_diff_range=ae_xy_diff_range,
        cross_xx_diff_range=cross_xx_diff_range,
        cross_yy_diff_range=cross_yy_diff_range,
        cross_xy_diff_range=cross_xy_diff_range,
        rotation_range=rotation_range,
        ref_rotation_range=ref_rotation_range,
        title_name=title_name,
        folder_name=folder_name,
        cmap_strain="RdBu_r",
        cmap_rotation="viridis",
        data_index=None,
        save_figure=True
    )

    # Check if the file was saved
    saved_file = tmp_path / f"Strain_Map_{title_name}.svg"
    assert saved_file.exists()

    # Clean up the generated figure to avoid memory issues in tests
    plt.close('all')

