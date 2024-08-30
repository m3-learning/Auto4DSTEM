import pytest
import torch
import torch.nn.functional as F
from auto4dstem.Viz.util import (
    upsample_mask,
    
)  

# Assuming the upsample_mask function is imported from the appropriate module
# from your_module import upsample_mask

def test_upsample_mask():
    # Prepare mock data
    input_size = 2
    up_size = 4
    
    mask_list = [
        torch.tensor([[1, 0], [0, 1]], dtype=torch.bool),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)
    ]
    
    # Expected upsampled masks
    expected_mask_list = [
        torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=torch.bool),
        torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0]
        ], dtype=torch.bool)
    ]
    
    # Run the upsample_mask function
    result_mask_list = upsample_mask(mask_list, input_size, up_size)
    
    # Assertions
    for result_mask, expected_mask in zip(result_mask_list, expected_mask_list):
        assert torch.equal(result_mask, expected_mask), "The upsampled mask does not match the expected output."

def test_upsample_mask_no_resize_needed():
    # Prepare mock data where no resize is needed
    input_size = 4
    up_size = 4
    
    mask_list = [
        torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=torch.bool)
    ]
    
    # Run the upsample_mask function
    result_mask_list = upsample_mask(mask_list, input_size, up_size)
    
    # Assertions
    assert result_mask_list == mask_list, "The mask list should not change when the up_size is equal to the input_size."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_upsample_mask.py
