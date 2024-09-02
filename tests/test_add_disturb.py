import pytest
import numpy as np
from auto4dstem.Viz.util import (
    add_disturb
)

# Assuming the add_disturb function is imported from the appropriate module
# from your_module import add_disturb

def test_add_disturb_no_rotation():
    # Prepare mock data with zero rotation (cos=1, sin=0)
    rotation = np.array([[1.0, 0.0], [1.0, 0.0]])
    dist = 0  # No additional disturbance

    # Expected output should be the same as input
    expected_rotation = rotation

    # Run the add_disturb function
    result_rotation = add_disturb(rotation, dist)

    # Assertions
    assert np.allclose(result_rotation, expected_rotation), "The rotation values do not match the expected output with no disturbance."

def test_add_disturb_with_20_degrees():
    # Prepare mock data with zero rotation (cos=1, sin=0)
    rotation = np.array([[1.0, 0.0], [1.0, 0.0]])
    dist = 20  # Add 20 degrees

    # Expected output after adding 20 degrees
    expected_angles = np.deg2rad(20)  # Convert 20 degrees to radians
    expected_rotation = np.array([[np.cos(expected_angles), np.sin(expected_angles)],
                                  [np.cos(expected_angles), np.sin(expected_angles)]])

    # Run the add_disturb function
    result_rotation = add_disturb(rotation, dist)

    # Assertions
    assert np.allclose(result_rotation, expected_rotation), "The rotation values do not match the expected output after adding 20 degrees."

def test_add_disturb_negative_rotation():
    # Prepare mock data with 90 degrees rotation (cos=0, sin=1)
    rotation = np.array([[0.0, 1.0], [0.0, 1.0]])
    dist = -20  # Subtract 20 degrees

    # Expected output after subtracting 20 degrees
    expected_angles = np.deg2rad(70)  # Convert 70 degrees to radians (90 - 20)
    expected_rotation = np.array([[np.cos(expected_angles), np.sin(expected_angles)],
                                  [np.cos(expected_angles), np.sin(expected_angles)]])

    # Run the add_disturb function
    result_rotation = add_disturb(rotation, dist)

    # Assertions
    assert np.allclose(result_rotation, expected_rotation), "The rotation values do not match the expected output after subtracting 20 degrees."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_add_disturb.py
