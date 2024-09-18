import pytest
import torch
import numpy as np
from unittest import mock
import torch.nn.functional as F
from auto4dstem.nn.CC_ST_AE import (
    revise_size_on_affine_gpu,
    crop_small_square
    )
from auto4dstem.Viz.util import (
    mask_function, 
    center_of_mass
)
# Assuming the revise_size_on_affine_gpu function and its dependencies are imported from the appropriate module
# from your_module import revise_size_on_affine_gpu, mask_function, center_of_mass, crop_small_square

@pytest.fixture
def mock_device():
    return torch.device('cpu')

@pytest.fixture
def mock_image():
    return torch.rand(1, 1, 128, 128)  # A single image tensor with random values

@pytest.fixture
def mock_image_2():
    return torch.rand(2, 1, 128, 128)  # A single image tensor with random values

@pytest.fixture
def mock_mask_list():
    return [torch.ones(1, 128, 128,dtype=torch.bool)]  # A simple mask covering the entire image

@pytest.fixture
def mock_theta():
    return torch.eye(2, 3).unsqueeze(0)  # An identity affine transformation

@pytest.fixture
def mock_theta_2():
    return torch.eye(2, 3).unsqueeze(0).repeat(2,1,1)  # An identity affine transformation


@pytest.fixture
def mock_adj_para():
    return torch.ones(1)  # Adjustment parameter set to 1

@pytest.fixture
def mock_adj_para_2():
    return torch.ones(2)  # Adjustment parameter set to 1

@mock.patch("auto4dstem.Viz.util.mask_function")
@mock.patch("auto4dstem.Viz.util.center_of_mass")
@mock.patch("auto4dstem.nn.CC_ST_AE.crop_small_square")
def test_revise_size_on_affine_gpu_no_adj(
    mock_crop_small_square,
    mock_center_of_mass,
    mock_mask_function,
    mock_device,
    mock_image,
    mock_mask_list,
    mock_theta,
):
    # Mock the outputs of the dependencies
    mock_center_of_mass.return_value = (64, 64)  # Center of the image
    mock_crop_small_square.return_value = ((52, 76), (52, 76))  # Small square around the center
    mock_mask_function.return_value = np.ones((24, 24), dtype=bool)  # A small mask

    result = revise_size_on_affine_gpu(
        image=mock_image,
        mask_list=mock_mask_list,
        batch_size=1,
        theta=mock_theta,
        device=mock_device,
        adj_para=None,
        radius=12,
        coef=1.5,
        pare_reverse=False,
        affine_mode="bicubic",
        dot_size=4,
    )

    assert result.shape == mock_image.shape, "The output image should have the same shape as the input image."

@mock.patch("auto4dstem.Viz.util.mask_function")
@mock.patch("auto4dstem.Viz.util.center_of_mass")
@mock.patch("auto4dstem.nn.CC_ST_AE.crop_small_square")
def test_revise_size_on_affine_gpu_with_adj(
    mock_crop_small_square,
    mock_center_of_mass,
    mock_mask_function,
    mock_device,
    mock_image,
    mock_mask_list,
    mock_theta,
    mock_adj_para,
):
    # Mock the outputs of the dependencies
    mock_center_of_mass.return_value = (64, 64)  # Center of the image
    mock_crop_small_square.return_value = ((52, 76), (52, 76))  # Small square around the center
    mock_mask_function.return_value = np.ones((24, 24), dtype=bool)  # A small mask

    result = revise_size_on_affine_gpu(
        image=mock_image,
        mask_list=mock_mask_list,
        batch_size=1,
        theta=mock_theta,
        device=mock_device,
        adj_para=mock_adj_para,
        radius=12,
        coef=1.5,
        pare_reverse=False,
        affine_mode="bicubic",
        dot_size=4,
    )

    assert result.shape == mock_image.shape, "The output image should have the same shape as the input image."
    # Additional checks can be added here to verify the intensity adjustments

@mock.patch("auto4dstem.Viz.util.mask_function")
@mock.patch("auto4dstem.Viz.util.center_of_mass")
@mock.patch("auto4dstem.nn.CC_ST_AE.crop_small_square")
def test_revise_size_on_affine_gpu_with_adj_batch_2(
    mock_crop_small_square,
    mock_center_of_mass,
    mock_mask_function,
    mock_device,
    mock_image_2,
    mock_mask_list,
    mock_theta_2,
    mock_adj_para_2,
):
    # Mock the outputs of the dependencies
    mock_center_of_mass.return_value = (64, 64)  # Center of the image
    mock_crop_small_square.return_value = ((52, 76), (52, 76))  # Small square around the center
    mock_mask_function.return_value = np.ones((24, 24), dtype=bool)  # A small mask

    result = revise_size_on_affine_gpu(
        image=mock_image_2,
        mask_list=mock_mask_list,
        batch_size=2,
        theta=mock_theta_2,
        device=mock_device,
        adj_para=mock_adj_para_2,
        radius=12,
        coef=1.5,
        pare_reverse=False,
        affine_mode="bicubic",
        dot_size=4,
    )

    assert result.shape == mock_image_2.shape, "The output image should have the same shape as the input image."
    # Additional checks can be added here to verify the intensity adjustments
    
@mock.patch("auto4dstem.Viz.util.mask_function")
@mock.patch("auto4dstem.Viz.util.center_of_mass")
@mock.patch("auto4dstem.nn.CC_ST_AE.crop_small_square")
def test_revise_size_on_affine_gpu_pare_reverse(
    mock_crop_small_square,
    mock_center_of_mass,
    mock_mask_function,
    mock_device,
    mock_image,
    mock_mask_list,
    mock_theta,
    mock_adj_para,
):
    # Mock the outputs of the dependencies
    mock_center_of_mass.return_value = (64, 64)  # Center of the image
    mock_crop_small_square.return_value = ((52, 76), (52, 76))  # Small square around the center
    mock_mask_function.return_value = np.ones((24, 24), dtype=bool)  # A small mask

    result = revise_size_on_affine_gpu(
        image=mock_image,
        mask_list=mock_mask_list,
        batch_size=1,
        theta=mock_theta,
        device=mock_device,
        adj_para=mock_adj_para,
        radius=12,
        coef=1.5,
        pare_reverse=True,
        affine_mode="bicubic",
        dot_size=4,
    )

    assert result.shape == mock_image.shape, "The output image should have the same shape as the input image."
    # Additional checks can be added here to verify the intensity adjustments with pare_reverse=True

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_revise_size_on_affine_gpu.py
