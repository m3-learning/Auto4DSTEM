import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from unittest import mock
from auto4dstem.Viz.util import (
    find_nearby_dot_group
)


@pytest.fixture
def sample_image():
    # Create a sample image with random values
    img = np.array([[1,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,1]])
    return img

def test_set_cluster(sample_image):
    # Initialize the class with a sample image
    dot_group = find_nearby_dot_group(sample_image)
    dot_group.set_cluster()
    # Check if the clusters attribute is set correctly
    assert hasattr(dot_group, 'clusters'), "The clusters attribute should be set after running set_cluster."
    assert hasattr(dot_group, 'new_array'), "The new_array attribute should be set after running set_cluster."
    assert len(dot_group.new_array) > 0, "The new_array should contain coordinates of pixels exceeding the threshold."

    # Ensure that the clusters array has the correct length
    assert len(dot_group.clusters) == len(dot_group.new_array), "The clusters array length should match the length of new_array."

def test_center_cor_list(sample_image):
    # Initialize the class with a sample image
    dot_group = find_nearby_dot_group(sample_image)

    dot_group.set_cluster(threshold=0.5,eps=1)
    # Run the center_cor_list method
    cor_list = dot_group.center_cor_list()
    plt.close()

    # Verify that cor_list contains the correct number of center coordinates
    assert len(cor_list) == 3, "There should be three clusters, so cor_list should have two coordinates."

    # Check if the calculated centers are correct
    expected_centers = [[0, 0], [2, 2], [4, 4]]
    assert cor_list == expected_centers, f"The center coordinates do not match the expected values: {expected_centers}"

def test_no_clusters_found(sample_image):
    # Initialize the class with a sample image
    dot_group = find_nearby_dot_group(sample_image)

    # Run the center_cor_list method
    dot_group.set_cluster(threshold=2,eps=1)

    # Verify that cor_list is empty
    assert len(dot_group.new_array) == 0, "If no clusters are found, cor_list should be empty."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_find_nearby_dot_group.py

