import pytest
import numpy as np
import os
import h5py
from pathlib import Path
from auto4dstem.Viz.viz import visualize_real_4dstem

@pytest.fixture
def mock_data():
    # Creating mock rotation and scale_shear data
    rotation = np.random.rand(256*256, 2)
    scale_shear = np.random.rand(256*256, 4)

    # Create temporary file paths for mock data
    with h5py.File('mock_py4dstem_real.h5', 'w') as f:
        f.create_dataset('strain_map_root/strain_map/data', data=np.random.rand(4, 256, 256))

    return rotation, scale_shear, 'mock_py4dstem_real.h5'


def test_visualize_real_4dstem_init(mock_data, tmp_path):
    rotation, scale_shear, file_py4DSTEM = mock_data

    viz_real_4dstem = visualize_real_4dstem(rotation=rotation,
                                            scale_shear=scale_shear,
                                            file_py4DSTEM=file_py4DSTEM,
                                            folder_name=str(tmp_path),
                                            rotation_range = [0,60],
                                            ref_rotation_range = [0,60])

    # Check if the object is created properly
    assert isinstance(viz_real_4dstem, visualize_real_4dstem)

    # Check if the folder was created
    assert os.path.exists(tmp_path), "Output folder was not created."

    # Check if strain maps are loaded properly
    assert hasattr(viz_real_4dstem, 'strain_map'), "Strain map was not loaded."

    # Verify the shape of loaded data
    assert viz_real_4dstem.strain_map.shape == (4, 256, 256), "Strain map shape mismatch."


@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    def remove_files():
        if os.path.exists('mock_py4dstem_real.h5'):
            os.remove('mock_py4dstem_real.h5')
    request.addfinalizer(remove_files)
