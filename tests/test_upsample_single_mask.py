import pytest
import torch
import torch.nn.functional as F
from auto4dstem.Viz.util import (
    upsample_single_mask,
)  

# Assuming the upsample_single_mask function is imported from the appropriate module
# from your_module import upsample_single_mask

def test_upsample_single_mask():
    # Prepare mock data
    original_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    up_size = [4, 4]  # Desired output size

    # Expected upsampled mask
    expected_up_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=torch.bool)

    # Run the upsample_single_mask function
    result_up_mask = upsample_single_mask(original_mask, up_size)

    # Assertions
    assert torch.equal(result_up_mask, expected_up_mask), "The upsampled mask does not match the expected output."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_upsample_single_mask.py
