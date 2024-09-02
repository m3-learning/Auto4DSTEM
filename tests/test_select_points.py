import pytest
import numpy as np
import matplotlib.pyplot as plt
from auto4dstem.Viz.util import (
    select_points
)

# Assuming the select_points function is imported from the appropriate module
# from your_module import select_points

def test_select_points():
    # Prepare mock data
    data = data = np.array([np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]]),\
                            np.array([[1, 2], [3, 0]]), np.array([[0, 3], [2, 1]])])
    mask = np.array([[True, False], [False, True]])
    threshold = 3

    # Expected output: indices of data points where the loss_map > threshold
    expected_index = np.array([0, 1])  # Example based on the mock data

    # Run the select_points function
    result_index = select_points(data, mask, threshold)

    # Assertions
    assert np.array_equal(result_index, expected_index), "The indices of selected points do not match the expected output."


def test_select_points_no_mask():
    # Prepare mock data without applying a mask
    data = np.random.rand(16, 4, 4)
    mask = np.ones_like(data[0], dtype=bool)  # No masking; all values considered
    threshold = 1e10

    # Expected output: index of data points where the loss_map > threshold
    expected_index = np.array([])  # No point should exceed the threshold of 6

    # Run the select_points function
    result_index = select_points(data, mask, threshold)

    # Assertions
    assert np.array_equal(result_index, expected_index), "The indices of selected points do not match the expected output when no mask is applied."

def test_select_points_with_img_size():
    # Prepare mock data
    data = np.random.rand(16, 4, 4)  # 16 images of size 4x4
    mask = np.ones_like(data[0], dtype=bool)  # No masking; all values considered
    threshold = 0.5
    img_size = [4, 4]  # Image size is predefined

    # Expected behavior: indices of data points where the loss_map > threshold
    result_index = select_points(data, mask, threshold, img_size=img_size)

    # Since data is random, we'll just ensure the output is reasonable
    assert result_index.size <= 16, "The result index size should not exceed the number of images."
    assert result_index.ndim == 1, "The result index should be a 1D array."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_select_points.py
