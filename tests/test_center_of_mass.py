import torch
import torch.nn.functional as F
import pytest
from auto4dstem.Viz.util import (
    center_of_mass,
)  


def test_center_of_mass():
    # Create a test image and mask
    img = torch.tensor([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    mask = torch.tensor([[0, 1, 1], [0, 1, 0], [0, 0, 0]])

    # Test with default coef (1.5)
    weighted_x, weighted_y = center_of_mass(img, mask, coef=1.5)

    # Compute expected values
    # mask selects elements 2, 3, 5
    mean_mass = torch.mean(img[mask])  # mean of [2.0, 3.0, 5.0] = 3.3333
    mass = F.relu(
        img[mask] - 1.5 * mean_mass
    )  # mass = relu([2-5, 3-5, 5-5]) = [0, 0, 0]
    sum_mass = torch.sum(mass)  # sum_mass = 0
    if sum_mass == 0:
        expected_x = torch.sum(torch.tensor([0, 0, 1])) / 3.0  # Average of [0, 0, 1]
        expected_y = torch.sum(torch.tensor([1, 2, 1])) / 3.0  # Average of [1, 2, 1]
    else:
        expected_x = torch.sum(torch.tensor([0, 0, 1]) * mass) / sum_mass
        expected_y = torch.sum(torch.tensor([1, 2, 1]) * mass) / sum_mass

    # Check the output
    assert torch.isclose(weighted_x, expected_x, atol=1e-6)
    assert torch.isclose(weighted_y, expected_y, atol=1e-6)

    # Additional test case with a different coef
    weighted_x, weighted_y = center_of_mass(img, mask, coef=1.0)
    mean_mass = torch.mean(img[mask])
    mass = F.relu(img[mask] - 1.0 * mean_mass)
    sum_mass = torch.sum(mass)
    if sum_mass == 0:
        expected_x = torch.sum(torch.tensor([0, 0, 1])) / 3.0
        expected_y = torch.sum(torch.tensor([1, 2, 1])) / 3.0
    else:
        expected_x = torch.sum(torch.tensor([0, 0, 1]) * mass) / sum_mass
        expected_y = torch.sum(torch.tensor([1, 2, 1]) * mass) / sum_mass

    assert torch.isclose(weighted_x, expected_x, atol=1e-6)
    assert torch.isclose(weighted_y, expected_y, atol=1e-6)


if __name__ == "__main__":
    pytest.main()
