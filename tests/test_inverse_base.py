import pytest
import torch
import h5py
import numpy as np
from unittest import mock
from auto4dstem.Viz.util import (
    inverse_base, center_mask_list_function
) 


# Assuming the inverse_base function and center_mask_list_function are imported from the appropriate module
# from your_module import inverse_base, center_mask_list_function

# Mock center_mask_list_function for testing purposes
#@mock.patch("your_module.center_mask_list_function")
def test_inverse_base():
    # Prepare mock data
    test_file = "test_data"
    input_mask_list = [torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)]
    coef = 2.5
    radius = 5
    
    # Create a dummy HDF5 file structure with a "base" dataset
    with h5py.File(test_file + ".h5", "w") as f:
        base_data = np.random.rand(1, 1, 10, 10)
        base_data = torch.tensor(base_data,dtype=torch.float)
        f.create_dataset("base", data=base_data)
    
    # Define expected output of center_mask_list_function
    # expected_center_mask_list = [torch.tensor([[1, 1], [0, 0]], dtype=torch.float)]
    # expected_rotate_center = torch.tensor([1, 1])
    expected_center_mask_list, expected_rotate_center = center_mask_list_function(
        base_data, input_mask_list, coef, radius=radius
    )
    
    # Run the inverse_base function
    result_center_mask_list, result_rotate_center = inverse_base(test_file, input_mask_list, coef, radius)
    
    assert torch.equal(result_center_mask_list[0], expected_center_mask_list[0]), "The center mask list output is not as expected."
    assert torch.equal(result_rotate_center, expected_rotate_center), "The rotate center output is not as expected."

    # Clean up test file
    import os
    os.remove(test_file + ".h5")

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_inverse_base.py
