from auto4dstem.masks.masks import mask_function
import numpy as np
from auto4dstem.nn.CC_ST_AE.utils import enforce_transformation_boundary, get_coordinate_range
import torch


def crop_single_diffraction_spot(
    center_coordinates: torch.Tensor, radius: int = 50, max_: int = 200, **kwargs
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Function to crop small square image for reverse affine operation

    Args:
        center_coordinates (torch.tensor): coordinates of diffraction spots after COM
        radius (int, optional): radius of small square for reverse affine operation. Defaults to 50
        max_ (int, optional): image size. Defaults to 200

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: (x_range, y_range) containing start and end coordinates
    """
    # Round and adjust coordinates to stay within bounds
    x = enforce_transformation_boundary(
        torch.round(center_coordinates[0]), radius, max_
    )
    y = enforce_transformation_boundary(
        torch.round(center_coordinates[1]), radius, max_
    )

    # Calculate coordinate ranges
    return get_coordinate_range(x, radius), get_coordinate_range(y, radius)


def apply_mask(
    image: torch.Tensor, mask: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Apply a binary mask to an image tensor.

    Args:
        image (torch.Tensor): Input image tensor to be masked
        mask (torch.Tensor): Binary mask tensor
        batch_size (int): Number of images in the batch
        device (torch.device): Device to place tensors on

    Returns:
        torch.Tensor: Masked image tensor where values outside the mask region are set to 0
    """
    if mask.shape[0] != batch_size:
        mask_ = (
            mask.squeeze()
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat(batch_size, 1, 1, 1)
            .to(device)
        )
    else:
        mask_ = mask.reshape(batch_size, 1, mask.shape[-2], mask.shape[-1]).to(device)

    # only keep values inside mask region
    masked_image = image * mask_.to(device)

    return masked_image, mask_


def create_square_mask(device: torch.device, radius: int, dot_size: int, **kwargs) -> torch.Tensor:
    """
    Creates a square mask with a small circle in the center.

    Args:
        device (torch.device): The device to perform computations on (e.g., CPU or GPU).
        radius (int): The radius of the square mask.
        dot_size (int): The size of the small circle in the center of the mask.

    Returns:
        torch.tensor: A boolean tensor representing the square mask with a small circle in the center.
    """
    initialize_square = np.zeros([radius * 2, radius * 2])

    # Crop small circle only to include diffraction spots
    dot_size = int(dot_size)
    small_square_mask = mask_function(
        initialize_square, radius=dot_size, center_coordinates=(radius, radius)
    )
    small_square_mask = torch.tensor(small_square_mask, dtype=torch.bool).to(device)
    return small_square_mask