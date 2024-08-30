import pytest
import torch
import torch.nn.functional as F
import math
from auto4dstem.Viz.util import (
    rotate_mask_list
)

# Assuming the rotate_mask_list function is imported from the appropriate module
# from your_module import rotate_mask_list

def test_rotate_mask_list_90_degrees():
    # Prepare mock data: a simple mask list with a single mask
    mask_list = [
        torch.tensor([
            [1, 0],
            [0, 0]
        ], dtype=torch.bool)
    ]
    
    theta_ = math.pi / 2  # 90 degrees

    # Expected output after 90 degrees rotation (clockwise)
    expected_rotated_masks = [
        torch.tensor([
            [0, 0],
            [1, 0]
        ], dtype=torch.bool)
    ]

    # Run the rotate_mask_list function
    rotated_masks, rotated_sum = rotate_mask_list(mask_list, torch.tensor(theta_))

    # Assertions
    assert len(rotated_masks) == len(expected_rotated_masks), "The length of the rotated mask list does not match the expected output."
    
    for rotated_mask, expected_mask in zip(rotated_masks, expected_rotated_masks):
        assert torch.equal(rotated_mask, expected_mask), "The rotated mask does not match the expected output."

    assert torch.equal(rotated_sum, expected_rotated_masks[0]), "The summed rotated mask does not match the expected output."

def test_rotate_mask_list_no_rotation():
    # Prepare mock data: a simple mask list with a single mask
    mask_list = [
        torch.tensor([
            [1, 0],
            [0, 0]
        ], dtype=torch.bool)
    ]
    
    theta_ = 0  # No rotation

    # Expected output should be the same as the input mask_list
    expected_rotated_masks = mask_list

    # Run the rotate_mask_list function
    rotated_masks, rotated_sum = rotate_mask_list(mask_list, torch.tensor(theta_))

    # Assertions
    assert len(rotated_masks) == len(expected_rotated_masks), "The length of the rotated mask list does not match the expected output."
    
    for rotated_mask, expected_mask in zip(rotated_masks, expected_rotated_masks):
        assert torch.equal(rotated_mask, expected_mask), "The rotated mask does not match the expected output."

    assert torch.equal(rotated_sum, expected_rotated_masks[0]), "The summed rotated mask does not match the expected output."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_rotate_mask_list.py
