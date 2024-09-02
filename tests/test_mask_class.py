import pytest
import numpy as np
import torch
from auto4dstem.Viz.util import (
    mask_class
)

# Assuming the mask_class and mask_function are imported from the appropriate module
# from your_module import mask_class, mask_function

# Mocking the mask_function to avoid dependencies and test the methods independently
def mock_mask_function(img, radius, center_coordinates):
    mask = np.zeros_like(img, dtype=bool)
    center_y, center_x = center_coordinates
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                mask[y, x] = True
    return mask

@pytest.fixture
def mask_obj():
    # Replace the mask_function with the mock version
    return mask_class(img_size=[200, 200])

@pytest.mark.parametrize("radius", [10, 20])
def test_mask_single(mask_obj, radius):
    # Test the mask_single method
    mask_tensor, mask_list = mask_obj.mask_single(radius)
    
    # Assert that the mask tensor has the correct shape
    assert mask_tensor.shape == tuple(mask_obj.img_size), "The mask tensor shape is incorrect."
    
    # Assert that the mask_list contains only one mask
    assert len(mask_list) == 1, "The mask list should contain only one mask."

    # Manually create the expected mask
    expected_mask = mock_mask_function(mask_obj.img, radius, mask_obj.center_coordinates)
    expected_mask_tensor = torch.tensor(expected_mask)

    # Assert that the mask tensor matches the expected mask
    assert torch.equal(mask_tensor, expected_mask_tensor), "The mask tensor does not match the expected output."

@pytest.mark.parametrize("radius_1, radius_2", [(10, 20), (30, 40)])
def test_mask_ring(mask_obj, radius_1, radius_2):
    # Test the mask_ring method
    mask_tensor, mask_list = mask_obj.mask_ring(radius_1, radius_2)
    
    # Assert that the mask tensor has the correct shape
    assert mask_tensor.shape == tuple(mask_obj.img_size), "The mask tensor shape is incorrect."
    
    # Assert that the mask_list contains only one mask
    assert len(mask_list) == 1, "The mask list should contain only one mask."

    # Manually create the expected ring mask
    inner_mask = mock_mask_function(mask_obj.img, radius_1, mask_obj.center_coordinates)
    outer_mask = mock_mask_function(mask_obj.img, radius_2, mask_obj.center_coordinates)
    expected_mask = ~inner_mask * outer_mask
    expected_mask_tensor = torch.tensor(expected_mask)

    # Assert that the mask tensor matches the expected ring mask
    assert torch.equal(mask_tensor, expected_mask_tensor), "The ring mask tensor does not match the expected output."

def test_mask_round(mask_obj):
    # Test the mask_round method
    radius = 10
    center_list = [(50, 50), (150, 150), (100, 100)]

    mask_tensor, mask_list = mask_obj.mask_round(radius, center_list)
    
    # Assert that the mask tensor has the correct shape
    assert mask_tensor.shape == tuple(mask_obj.img_size), "The mask tensor shape is incorrect."
    
    # Assert that the mask_list contains the correct number of masks
    assert len(mask_list) == len(center_list), "The mask list length does not match the number of center coordinates."

    # Manually create the expected round mask
    expected_mask = np.zeros(mask_obj.img_size, dtype=bool)
    for center in center_list:
        temp_mask = mock_mask_function(mask_obj.img, radius, center)
        expected_mask += temp_mask
    expected_mask_tensor = torch.tensor(expected_mask)

    # Assert that the mask tensor matches the expected round mask
    assert torch.equal(mask_tensor, expected_mask_tensor), "The round mask tensor does not match the expected output."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_mask_class.py
