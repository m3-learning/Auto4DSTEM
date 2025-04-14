from auto4dstem.masks.masks import Mask
import numpy as np
from auto4dstem.nn.CC_ST_AE.masks import create_square_mask
from auto4dstem.nn.CC_ST_AE.masks import apply_mask, crop_single_diffraction_spot
from auto4dstem.viz.util import center_of_mass, find_nearby_dot_group
import torch
import torch.nn.functional as F


def intensity_adjustment(
    device,
    intensity_adjustment_factor,
    divide_by_intensity_adjustment,
    small_square_mask,
    i,
    small_image,
):
    """
    Adjusts the intensity of a small image region based on given parameters and applies an affine transformation.

    Args:
        device (torch.device): The device to perform computations on (e.g., CPU or GPU).
        intensity_adjustment_factor (torch.tensor): Adjustment parameters for intensity scaling.
        divide_by_intensity_adjustment (bool): Flag to determine the direction of intensity adjustment.
        affine_mode (str): The mode for affine transformation (e.g., 'bilinear').
        small_square_mask (torch.tensor): Mask to specify the region of interest for intensity adjustment.
        img (torch.tensor): The original image tensor to be modified.
        i (int): Index of the current image in a batch.
        x_coordinate (tuple): Tuple containing the start and end x-coordinates for cropping.
        y_coordinate (tuple): Tuple containing the start and end y-coordinates for cropping.
        small_image (torch.tensor): The small image region to be adjusted.
        re_grid (torch.tensor): The grid for affine transformation.

    Returns:
        None: The function modifies the input image tensor in place.
    """
    small_image_copy = torch.clone(small_image.squeeze()).to(device)

    # Adjust the intensity of the small image region
    if divide_by_intensity_adjustment:
        small_image_copy[small_square_mask] /= intensity_adjustment_factor[i]
    else:
        small_image_copy[small_square_mask] *= intensity_adjustment_factor[i]

    small_image_copy = small_image_copy.unsqueeze(0).unsqueeze(1)

    return small_image_copy


def reverse_affine_transform_gpu(
    image,
    mask_positions,
    theta,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    intensity_adjustment_factor=None,
    radius=12,
    coef=1.5,
    divide_by_intensity_adjustment=False,
    affine_mode="bicubic",
    intensity_adjustment_radius=4,
    batch_size=None,
):
    """Reverse affine transform diffraction spots in an image.

    Args:
        image (torch.Tensor): Input image containing diffraction spots
        mask_positions (list): List of binary mask tensors indicating spot positions
        batch_size (int): Number of images in the batch
        theta (torch.Tensor): Affine transformation matrix containing scale and shear parameters
        device (torch.device): Device to place tensors on
        intensity_adjustment_factor (float, optional): Factor to adjust spot intensities. Defaults to None.
        radius (int, optional): Size of square region around each spot for transformation. Defaults to 12.
        coef (float, optional): Threshold coefficient for center of mass calculation. Defaults to 1.5.
        divide_by_intensity_adjustment (bool, optional): Whether to divide (True) or multiply (False) by intensity factor. Defaults to False.
        affine_mode (str, optional): Interpolation mode for affine grid sampling. Defaults to 'bicubic'.
        dot_size (int, optional): Radius of circular region for intensity adjustment. Defaults to 4.

    Returns:
        torch.Tensor: Image with reverse affine transformed diffraction spots
    """

    batch_size = image.shape[0] if batch_size is None else batch_size

    # Initializes the square image for reverse affine operation
    small_square_mask = create_square_mask(device, radius, intensity_adjustment_radius)

    img = torch.clone(image).to(device)

    # create identity matrix to make affine matrix into size [batch,3,3] for computing inverse matrix
    identity = (
        torch.tensor([0, 0, 1], dtype=torch.float)
        .reshape(1, 1, 3)
        .repeat(batch_size, 1, 1)
        .to(device)
    )

    # create 3x3 affine matrix
    new_theta = torch.cat((theta, identity), axis=1).to(device)

    # computing inverse matrix
    inverse_theta = torch.linalg.inv(new_theta)[:, 0:2].to(device)

    # replicate each mask into the same size of input
    for j, mask in enumerate(mask_positions):
        masked_image = apply_mask(image, mask, batch_size, device)

        for i in range(batch_size):
            # extract center coordinates of each diffraction spots
            center_x, center_y = center_of_mass(
                masked_image[i].squeeze(), mask[i].squeeze(), coef
            )
            center = torch.tensor([center_x, center_y]).to(device)

            # extract coordinates of corners of  small square image which has diffraction spots
            x_coordinate, y_coordinate = crop_single_diffraction_spot(
                center_coordinates=center.clone(), radius=radius, max_=img.shape[-1]
            )

            # crop the small image according to coordinates
            single_diffraction_spot_image = (
                img[i]
                .squeeze()[
                    x_coordinate[0] : x_coordinate[1], y_coordinate[0] : y_coordinate[1]
                ]
                .unsqueeze(0)
                .unsqueeze(1)
                .clone()
                .to(device)
            )

            # apply inverse affine transform on small images
            inverse_affine_matrix = F.affine_grid(
                inverse_theta[i].unsqueeze(0).to(device),
                single_diffraction_spot_image.size(),
            ).to(device)

            if intensity_adjustment_factor is not None:
                single_diffraction_spot_image = intensity_adjustment(
                    device,
                    intensity_adjustment_factor,
                    divide_by_intensity_adjustment,
                    small_square_mask,
                    img,
                    i,
                    single_diffraction_spot_image,
                )

            reverse_affine_transformation_single_diffraction_spot = F.grid_sample(
                single_diffraction_spot_image, inverse_affine_matrix, mode=affine_mode
            )
            img[
                i,
                :,
                x_coordinate[0] : x_coordinate[1],
                y_coordinate[0] : y_coordinate[1],
            ] = reverse_affine_transformation_single_diffraction_spot.squeeze()

    return img


def spatial_transformation(img, matrix, mask_0=None, reverse_affine=True, **kwargs):
    """function for spatial translation

    Args:
        img (torch.tensor): image with diffraction spots
        matrix (torch.tensor): affine transformation matrix
        mask_0 (torch.tensor, optional): mask of diffraction spots. Defaults to None.
        reverse_affine (bool, optional): switch multiplying or dividing adj_para . Defaults to True.

    Returns:
        torch.tensor: image after spatial translation
    """

    image_threshold = kwargs.get("image_threshold", 0.3)

    # Copy from the sample image
    temp_image = np.copy(img).squeeze()

    temp_image = torch.tensor(temp_image, dtype=torch.float).unsqueeze(0).unsqueeze(1)

    # apply affine transformation
    theta_1 = torch.tensor(matrix, dtype=torch.float)

    # Apply matrix to image
    grid = F.affine_grid(theta_1.unsqueeze(0), temp_image.size())
    temp_image = F.grid_sample(temp_image, grid).squeeze()

    # make the image binary
    temp_image[temp_image < image_threshold] = 0
    temp_image[temp_image >= image_threshold] = 1

    if mask_0 is not None:
        temp_image[mask_0] = 0

    # generate mask around the spot on image
    if reverse_affine:
        generate_mask = find_nearby_dot_group(temp_image)
        generate_mask.set_cluster()
        center_coord = generate_mask.center_cor_list()
        mask_class_ = Mask(img_size=img.shape)
        mask_tensor, mask_list = mask_class_.mask_round(
            radius=10, center_list=center_coord
        )
        temp_image = reverse_affine_transform_gpu(
            temp_image.unsqueeze(0).unsqueeze(1),
            mask_list,
            theta_1.unsqueeze(0),
            torch.device("cpu"),
            intensity_adjustment_factor=None,
            radius=15,
            coef=1.5,
            divide_by_intensity_adjustment=False,
            affine_mode="bicubic",
        ).squeeze()
        temp_image[temp_image < 0.3] = 0
        temp_image[temp_image >= 0.3] = 1
        return temp_image

    return temp_image


def apply_affine_transformation_to_image(x, scale_shear, rotation, translation, inverse_affine=False, **kwargs):
    device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    affine_mode = kwargs.get("affine_mode", "bilinear")


    scale_shear_grid = F.affine_grid(scale_shear.to(device), x.size()).to(device)
    rotation_grid = F.affine_grid(rotation.to(device), x.size()).to(device)
    translation_grid = F.affine_grid(translation.to(device), x.size()).to(device)
    
    if inverse_affine:
        order = [translation_grid, rotation_grid, scale_shear_grid]
    else:
        order = [scale_shear_grid, rotation_grid, translation_grid]
        
    for grid in order:
        x = F.grid_sample(x, grid, mode=affine_mode)
    
    return x, scale_shear_grid, rotation_grid, translation_grid


def generate_inverse_affine(scale_shear, rotation, translation, identity, **kwargs):
    device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scale_shear_affine = torch.cat((scale_shear, identity), axis=1).to(device)
    rotation_affine = torch.cat((rotation, identity), axis=1).to(device)
    translation_affine = torch.cat((translation, identity), axis=1).to(device)

    # generate inverse affine matrix
    inverse_scale_shear = torch.linalg.inv(scale_shear_affine)[:, 0:2].to(device)
    inverse_rotation = torch.linalg.inv(rotation_affine)[:, 0:2].to(device)
    inverse_translation = torch.linalg.inv(translation_affine)[:, 0:2].to(device)
    return inverse_scale_shear, inverse_rotation, inverse_translation