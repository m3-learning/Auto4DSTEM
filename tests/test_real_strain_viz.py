import pytest
import numpy as np
import matplotlib.pyplot as plt
from auto4dstem.Viz.viz import real_strain_viz 

@pytest.fixture
def mock_diff_list():
    """Create a mock diff_list for testing."""
    return [np.random.randn(100, 100) * 0.01 for _ in range(4)]

@pytest.fixture
def mock_data_index():
    """Create a mock data_index for testing."""
    return np.random.choice(10000, 5000, replace=False)

def test_real_strain_viz_saving(mock_diff_list, tmp_path):
    """Test that the function saves the figure correctly."""
    save_path = tmp_path / "figures"
    save_path.mkdir()
    file_name = str(save_path / "test_title")

    real_strain_viz(mock_diff_list, "test_title", folder_name=save_path, save_figure=True)

    # Check if the figure file was saved
    saved_file = save_path / f"test_title_Strain_Map_of_Experimental_4DSTEM.svg"
    assert saved_file.exists(), f"The file {saved_file} should have been saved."

    # Cleanup
    saved_file.unlink()

