import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from auto4dstem.Viz.viz import visual_strain_magnitude

def test_visual_strain_magnitude_with_reference(tmp_path):
    # Mock input data
    s_xx = np.random.randn(256, 256)
    s_yy = np.random.randn(256, 256)
    ref_xx = np.random.randn(256, 256)
    ref_yy = np.random.randn(256, 256)
    title_name = "test"
    folder_name = str(tmp_path)

    # Call the function
    visual_strain_magnitude(
        s_xx=s_xx,
        s_yy=s_yy,
        title_name=title_name,
        folder_name=folder_name,
        cmap="RdBu_r",
        ref_xx=ref_xx,
        ref_yy=ref_yy,
        strain_range=[-3, 3],
        ref_range=[-3, 3],
        img_size=(256, 256),
        save_figure=True
    )

    # Check if the comparison figure was saved
    saved_file = tmp_path / f"{title_name}_Strain_Magnitude_Comparison.svg"
    assert saved_file.exists()

    # Clean up the generated figure to avoid memory issues in tests
    plt.close('all')


def test_visual_strain_magnitude_without_reference(tmp_path):
    # Mock input data
    s_xx = np.random.randn(256, 256)
    s_yy = np.random.randn(256, 256)
    title_name = "test_no_ref"
    folder_name = str(tmp_path)

    # Call the function without reference data
    visual_strain_magnitude(
        s_xx=s_xx,
        s_yy=s_yy,
        title_name=title_name,
        folder_name=folder_name,
        cmap="RdBu_r",
        strain_range=[-3, 3],
        img_size=(256, 256),
        only_real=True,
        save_figure=True
    )

    # Check if the performance figure was saved
    saved_file = tmp_path / f"{title_name}_Strain_Magnitude_Performance.svg"
    assert saved_file.exists()

    # Clean up the generated figure to avoid memory issues in tests
    plt.close('all')
