import pytest
import numpy as np
import os
from auto4dstem.Viz.util import (
    generate_classification
)
# Assuming the generate_classification function is imported from the appropriate module
# from your_module import generate_classification

def test_generate_classification_default():
    # Prepare mock data
    sample_index = np.array([0,1,8])
    img_size = [4, 4]  # Small image size for easy testing
    sample_position = 0
    save_file = False

    # Expected classification array
    expected_classification = np.array([
        [1, 0],  # sample index 0
        [1, 0],  # sample index 1
        [0, 1],  # bkg index 2
        [0, 1],  # bkg index 3
        [0, 1],  # bkg index 4
        [0, 1],  # bkg index 5
        [0, 1],  # bkg index 6
        [0, 1],  # bkg index 7
        [1, 0],  # sample index 8 (sample_index[2] % (img_size[0] * img_size[1]))
        [0, 1],  # bkg index 9
        [0, 1],  # bkg index 10
        [0, 1],  # bkg index 11
        [0, 1],  # bkg index 12
        [0, 1],  # bkg index 13
        [0, 1],  # bkg index 14
        [0, 1],  # bkg index 15
    ])

    # Run the generate_classification function
    result_classification = generate_classification(
        sample_index=sample_index, 
        sample_position=sample_position, 
        img_size=img_size, 
        save_file=save_file
    )

    # Assertions
    assert np.array_equal(result_classification, expected_classification), "The classification array does not match the expected output."

def test_generate_classification_save_file(tmpdir):
    # Prepare mock data
    sample_index = np.array([0, 1])
    img_size = [4, 4]  # Small image size for easy testing
    sample_position = 1
    file_path = os.path.join(tmpdir, "test_classification")

    # Run the generate_classification function with file saving
    generate_classification(
        sample_index=sample_index, 
        sample_position=sample_position, 
        img_size=img_size, 
        save_file=True, 
        file_path=file_path
    )

    # Check if the file was saved correctly
    saved_file = f"{file_path}_classification.npy"
    assert os.path.exists(saved_file), "The classification file was not saved correctly."

    # Load the saved file and verify its contents
    saved_classification = np.load(saved_file)
    expected_classification = np.zeros([img_size[0] * img_size[1], 2])
    expected_classification[sample_index, sample_position] = 1
    expected_classification[[i for i in range(img_size[0] * img_size[1]) if i not in sample_index], int(1 - sample_position)] = 1

    assert np.array_equal(saved_classification, expected_classification), "The saved classification array does not match the expected output."

def test_generate_classification_sample_position_1():
    # Prepare mock data
    sample_index = np.array([2, 3])
    img_size = [4, 4]  # Small image size for easy testing
    sample_position = 1
    save_file = False

    # Expected classification array
    expected_classification = np.zeros([img_size[0] * img_size[1], 2])
    expected_classification[sample_index, sample_position] = 1
    expected_classification[[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], int(1 - sample_position)] = 1

    # Run the generate_classification function
    result_classification = generate_classification(
        sample_index=sample_index, 
        sample_position=sample_position, 
        img_size=img_size, 
        save_file=save_file
    )

    # Assertions
    assert np.array_equal(result_classification, expected_classification), "The classification array does not match the expected output for sample_position=1."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_generate_classification.py
