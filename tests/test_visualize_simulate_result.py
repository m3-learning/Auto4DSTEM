import pytest
import numpy as np
import os
import h5py
from pathlib import Path
from auto4dstem.Viz.viz import visualize_simulate_result

@pytest.fixture
def mock_data():
    # Creating mock rotation and scale_shear data
    rotation = np.random.rand(256*256, 2)
    scale_shear = np.random.rand(256*256, 4)
    # Create temporary file paths for mock data
    with h5py.File('mock_py4dstem.h5', 'w') as f:
        f.create_dataset('strain_map_root/strain_map/data', data=np.random.rand(4, 256, 256))

    np.save('rotation_label_2.npy', np.random.rand(256, 256))
    np.save('Label_strain_xx.npy', np.random.rand(256, 256))
    np.save('Label_strain_yy.npy', np.random.rand(256, 256))
    np.save('Label_shear_xy.npy', np.random.rand(256, 256))
    
    return rotation, scale_shear, 'mock_py4dstem.h5'


def test_visualize_simulate_result_init(mock_data, tmp_path):
    rotation, scale_shear, file_py4DSTEM = mock_data

    viz_sim_res = visualize_simulate_result(rotation=rotation,
                                            scale_shear=scale_shear,
                                            file_py4DSTEM=file_py4DSTEM,
                                            folder_name=str(tmp_path))

    # Check if the object is created properly
    assert isinstance(viz_sim_res, visualize_simulate_result)

    # Check if the folder was created
    assert os.path.exists(tmp_path), "Output folder was not created."

    # Check if labels and strain maps are loaded properly
    assert hasattr(viz_sim_res, 'label_rotation'), "Label rotation was not loaded."
    assert hasattr(viz_sim_res, 'strain_map'), "Strain map was not loaded."

    # Verify the shape of loaded data
    assert viz_sim_res.label_rotation.shape == (256, 256), "Label rotation shape mismatch."
    assert viz_sim_res.strain_map.shape == (4, 256, 256), "Strain map shape mismatch."


@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    def remove_files():
        for filename in ['mock_py4dstem.h5', 'rotation_label_2.npy', 'Label_strain_xx.npy', 'Label_strain_yy.npy', 'Label_shear_xy.npy']:
            if os.path.exists(filename):
                os.remove(filename)
    request.addfinalizer(remove_files)
