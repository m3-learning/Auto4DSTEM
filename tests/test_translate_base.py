import torch
import torch.nn.functional as F
import pytest
from auto4dstem.viz.util import (
    translate_base,
    center_of_mass,
)


def test_translate_base():
    # Create a test image and mask
    img = torch.tensor([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    mask_ = torch.tensor([[0, 1, 1], [0, 1, 0], [0, 0, 0]])

    # Test translation by (0, 0), no translation
    add_x = 0.0
    add_y = 0.0
    expected_x, expected_y = center_of_mass(img, mask_, coef=0.5)
    weight_x, weight_y = translate_base(add_x, add_y, img, mask_, coef=0.5)

    assert torch.isclose(weight_x, expected_x, atol=1e-6)
    assert torch.isclose(weight_y, expected_y, atol=1e-6)

    # Test translation by (1, 0), moving right by 1 unit
    add_x = 1.0
    add_y = 0.0
    translated_img = torch.clone(img)
    translated_img[:, :-1] = img[:, 1:]  # Simulating the effect of translation
    weight_x, weight_y = translate_base(add_x, add_y, img, mask_, coef=0.5)

    assert torch.isclose(weight_x, torch.tensor([0],dtype=torch.float), atol=1e-6)
    assert torch.isclose(weight_y, torch.tensor([1.2698],dtype=torch.float), atol=1e-6)

    # Test translation by (0, 1), moving down by 1 unit
    add_x = 0.0
    add_y = 1.0
#    translated_img = torch.clone(img)
    test_img = torch.clone(img).unsqueeze(0).unsqueeze(1)
    add_trans = torch.tensor(
        [[1.0000, 0.0000, add_x], [0.0000, 1.0000, add_y]], dtype=torch.float
    ).unsqueeze(0)

    grid_ = F.affine_grid(add_trans, test_img.size())

    translated_img = F.grid_sample(test_img, grid_, mode="bicubic").squeeze()  # Simulating the effect of translation
    expected_x, expected_y = center_of_mass(translated_img, mask_, coef=0.5)
    weight_x, weight_y = translate_base(add_x, add_y, img, mask_, coef=0.5)

    assert torch.isclose(weight_x, expected_x, atol=1e-6)
    assert torch.isclose(weight_y, expected_y, atol=1e-6)

    # Test translation by (-1, -1), moving left and up by 1 unit each
    add_x = -1.0
    add_y = -1.0
#    translated_img = torch.clone(img)
    test_img = torch.clone(img).unsqueeze(0).unsqueeze(1)
    add_trans = torch.tensor(
        [[1.0000, 0.0000, add_x], [0.0000, 1.0000, add_y]], dtype=torch.float
    ).unsqueeze(0)

    grid_ = F.affine_grid(add_trans, test_img.size())

    translated_img = F.grid_sample(test_img, grid_, mode="bicubic").squeeze()  # Simulating the effect of translation  # Simulating the effect of translation
    expected_x, expected_y = center_of_mass(translated_img, mask_, coef=0.5)
    weight_x, weight_y = translate_base(add_x, add_y, img, mask_, coef=0.5)

    assert torch.isclose(weight_x, expected_x, atol=1e-6)
    assert torch.isclose(weight_y, expected_y, atol=1e-6)

# Additional edge cases can be added here


# if __name__ == "__main__":
#     pytest.main()
