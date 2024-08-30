import pytest
import torch
import numpy as np
from auto4dstem.Viz.util import (
    center_mask_list_function, 
    upsample_mask,
    center_of_mass,
    mask_function
)

# Assuming the center_mask_list_function, upsample_mask, center_of_mass, and mask_function 
# are imported from the appropriate module
# from your_module import center_mask_list_function, upsample_mask, center_of_mass, mask_function

def test_center_mask_list_function():
    # Prepare mock data
    image = torch.tensor(np.random.rand(1, 1, 10, 10), dtype=torch.float)
    mask_list = [torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.bool)]
    coef = 2.0
    radius = 3
    input_size = mask_list[0].shape[-1]
    up_size = image.shape[-1]
    # Mock the outputs of the upsample_mask, center_of_mass, and mask_function
    up_mask_list = upsample_mask(mask_list, input_size, up_size)
    # initial the new mask list 
    expected_center_mask_list = []
    mean_ = np.zeros([image.shape[-2], image.shape[-1]])
    # generate the mask list
    for j, mask in enumerate(up_mask_list):
        mask_ = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])

        new_image = image * mask_
        # compute coordinate with center of mass
        center_x, center_y = center_of_mass(new_image.squeeze(), mask_.squeeze(), coef)
        center_x = int(np.round(np.array(center_x)))
        center_y = int(np.round(np.array(center_y)))
        print(center_x, center_y)
        # create small mask region using center coordinate
        small_mask = mask_function(
            mean_, radius=radius, center_coordinates=(center_y, center_x)
        )
        # switch type into tensor
        small_mask = torch.tensor(small_mask, dtype=torch.bool)

        expected_center_mask_list.append(small_mask)
    # resize the mask list 
    expected_center_mask_list = upsample_mask(expected_center_mask_list, up_size, input_size)
    # create whole mask region in one image
    expected_rotate_mask_up = torch.clone(expected_center_mask_list[0])

    # for i in range(1, len(expected_center_mask_list)):
    #     expected_rotate_mask_up += expected_center_mask_list[i]

    # Run the center_mask_list_function
    center_mask_list, rotate_mask_up = center_mask_list_function(image, mask_list, coef, radius)

    # Assertions
    assert len(center_mask_list) == len(expected_center_mask_list), "The length of the center mask list does not match the input mask list."

    for center_mask, expected_mask in zip(center_mask_list,expected_center_mask_list):
        assert torch.equal(center_mask, expected_mask), "The center mask does not match the expected output."

    assert torch.equal(rotate_mask_up, expected_rotate_mask_up), "The rotated mask sum does not match the expected output."

