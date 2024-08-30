import pytest
import numpy as np
import cv2
from auto4dstem.Viz.util import (
    mask_function
)  
# Assuming the mask_function is imported from the appropriate module
# from your_module import mask_function

def test_mask_function():
    # Create a test image of size 200x200
    test_img = np.zeros((200, 200), dtype=np.float32)
    
    # Set parameters
    radius = 7
    center_coordinates = (100, 100)
    
    # Expected mask
    image_2 = cv2.circle(test_img, center_coordinates, radius, 100, -1)
    image_2 = np.array(image_2)
    expected_mask = image_2 == 100
    expected_mask = np.array(expected_mask)
    
    # Call the function
    result_mask = mask_function(test_img, radius, center_coordinates)
    
    # Verify that the mask returned matches the expected mask
    assert np.array_equal(result_mask, expected_mask), "The mask function did not produce the expected output."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_mask_function.py
