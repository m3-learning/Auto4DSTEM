import numpy as np

from auto4dstem.nn.CC_ST_AE.FPGA import conv_block_fpga, identity_block_fpga
from auto4dstem.nn.CC_ST_AE.ktop import ktop_layer

from ...masks.masks import Mask, mask_function
from ...viz.util import center_of_mass, find_nearby_dot_group
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

########################################################
# Helper functions for reverse affine transform
########################################################


def enforce_transformation_boundary(coord, radius, max_val) -> float:
    """Helper function to adjust a coordinate to stay within bounds

    Args:
        coord (float): The coordinate value to adjust
        radius (int): The radius to check bounds against
        max_val (int): The maximum allowed value

    Returns:
        float: The adjusted coordinate value
    """
    if coord - radius < 0:
        return coord - (coord - radius)
    if coord + radius > max_val:
        return coord - ((coord + radius) - max_val)
    return coord


def get_coordinate_range(coord: float, radius: int) -> tuple[int, int]:
    """Calculate start and end coordinates for a given center coordinate and radius

    Args:
        coord (float): Center coordinate
        radius (int): Radius to extend from center

    Returns:
        tuple[int, int]: Start and end coordinates
    """
    return (int(coord - radius), int(coord + radius))


def crop_single_diffraction_spot(
    center_coordinates: torch.Tensor, radius: int = 50, max_: int = 200
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

    return masked_image


def reverse_affine_transform_gpu(
    image,
    mask_positions,
    batch_size,
    theta,
    device,
    intensity_adjustment_factor=None,
    radius=12,
    coef=1.5,
    divide_by_intensity_adjustment=False,
    affine_mode="bicubic",
    intensity_adjustment_radius=4,
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


def create_square_mask(device, radius, dot_size):
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
            1,
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


class conv_block(nn.Module):
    """
    A convolutional block that implements a Residual Neural Network (ResNet) module.

    This class inherits from nn.Module and defines a residual block with three convolutional layers,
    each followed by a ReLU activation function. The input tensor is added to the output tensor
    after passing through the convolutional layers to form the residual connection.

    Attributes:
        cov1d_1 (nn.Conv2d): The first convolutional layer.
        cov1d_2 (nn.Conv2d): The second convolutional layer.
        cov1d_3 (nn.Conv2d): The third convolutional layer.
        norm_3 (nn.LayerNorm): The normalization layer applied after the third convolutional layer.
        relu_1 (nn.ReLU): The ReLU activation function applied after the first convolutional layer.
        relu_2 (nn.ReLU): The ReLU activation function applied after the second convolutional layer.
        relu_3 (nn.ReLU): The ReLU activation function applied after the normalization layer.
    """

    def __init__(self, num_channels, spatial_dims):
        """Initializes the convolutional block.

        This constructor sets up three convolutional layers, a normalization layer, and three ReLU activation functions.
        The convolutional layers use a kernel size of 3x3, a stride of 1, and zero padding.

        Args:
            num_channels (int): The number of input and output channels for the convolutional layers.
            spatial_dims (tuple): The shape of the input tensor for the normalization layer.
        """
        super(conv_block, self).__init__()

        # Convolutional layer 1
        self.cov1d_1 = nn.Conv2d(
            num_channels, num_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )

        # Convolutional layer 2
        self.cov1d_2 = nn.Conv2d(
            num_channels, num_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )

        # Convolutional layer 3
        self.cov1d_3 = nn.Conv2d(
            num_channels, num_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )

        # Normalization layer
        self.norm_3 = nn.LayerNorm(spatial_dims)

        # ReLU activation functions
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        """Performs the forward pass of the convolutional block.

        This method takes an input tensor and passes it through three convolutional layers,
        each followed by a ReLU activation function. After the third convolutional layer,
        the output is normalized and another ReLU activation is applied. The input tensor
        is then added to the output tensor to form the residual connection.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: The output tensor after applying the convolutional block, with the same shape as the input tensor.
        """

        # Use Residual structure to concatenate input tensor to output of 3 convolutional layers
        x_input = x
        out = self.cov1d_1(x)
        out = self.relu_1(out)
        out = self.cov1d_2(out)
        out = self.relu_2(out)
        out = self.cov1d_3(out)
        out = self.norm_3(out)
        out = self.relu_3(out)
        out = out.add(x_input)

        return out


class identity_block(nn.Module):
    """
    Identity Block for a Neural Network.

    This class defines an identity block, which is a fundamental component of residual networks.
    It consists of a single convolutional layer followed by a normalization layer and a ReLU activation function.
    The identity block helps in training deep neural networks by allowing the gradient to flow through the network
    without vanishing or exploding.

    Attributes:
        cov1d_1 (nn.Conv2d): The first convolutional layer.
        norm_1 (nn.LayerNorm): The normalization layer.
        relu (nn.ReLU): The ReLU activation function.
    """

    def __init__(self, num_channels, spatial_dims):
        """Initializes the identity block.

        This method sets up the identity block by initializing its convolutional layer,
        normalization layer, and ReLU activation function.

        Args:
            num_channels (int): The number of input and output channels for the convolutional layer.
            spatial_dims (tuple): The shape of the input tensor for the normalization layer.
        """

        super(identity_block, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            num_channels, num_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_1 = nn.LayerNorm(spatial_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Performs the forward pass of the identity block.

        This method takes an input tensor, applies a convolutional layer,
        followed by a normalization layer and a ReLU activation function,
        and returns the output tensor.

        Args:
            x (torch.Tensor): The input tensor with shape (N, C, H, W), where
                N is the batch size, C is the number of channels, H is the height,
                and W is the width of the input feature map.

        Returns:
            torch.Tensor: The output tensor after applying the identity block,
            with the same shape as the input tensor.
        """
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


class AffineTransformationBlock(nn.Module):
    """
    nn.Module class to return 3 type of affine transformation matrices (scale and shear, rotation, translation) and
    a adjust parameter to change pixel intensity in mask region.
    """

    def __init__(self, **kwargs):
        """

        Args:
            kwargs (dict): Dictionary containing the following parameters:
                device (torch.device): Specifies the device on which the model will run.
                scale (bool): Indicates if the model includes a scale affine transformation.
                shear (bool): Indicates if the model includes a shear affine transformation.
                rotation (bool): Indicates if the model includes a rotation affine transformation.
                rotate_clockwise (bool): Indicates if the image should be rotated in a clockwise direction.
                translation (bool): Indicates if the model includes a translation affine transformation.
                shear_symmetric (bool): Indicates if the shear affine transformation is symmetric.
                mask_intensity (bool): Indicates if the intensity of the mask region is learnable.
                scale_limit (float, optional): Limits the range of the scale parameter. Defaults to 0.05.
                shear_limit (float, optional): Limits the range of the shear parameter. Defaults to 0.1.
                rotation_limit (float, optional): Limits the range of the rotation parameter. Defaults to 0.1.
                trans_limit (float, optional): Limits the range of the translation parameter. Defaults to 0.15.
                adj_mask_para (int, optional): Limits the range of learnable intensity in the mask region. Defaults to 0.
                verbose (bool, optional): Indicates if the count of affine transformations should be printed. Defaults to False.

        """

        super(AffineTransformationBlock, self).__init__()

        # initialize the parameters from the kwargs
        self.scale = kwargs.get("scale", True)
        self.shear = kwargs.get("shear", True)
        self.rotation = kwargs.get("rotation", True)
        self.rotate_clockwise = kwargs.get("rotate_clockwise", True)
        self.translation = kwargs.get("translation", False)
        self.shear_symmetric = kwargs.get("shear_symmetric", True)
        self.scale_limit = kwargs.get("scale_limit", 0.05)
        self.shear_limit = kwargs.get("shear_limit", 0.1)
        self.rotation_limit = kwargs.get("rotation_limit", 0.1)
        self.trans_limit = kwargs.get("trans_limit", 0.15)
        self.adj_mask_para = kwargs.get("adj_mask_para", 0)
        self.mask_intensity = kwargs.get("mask_intensity", True)
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.count = 0
        self.verbose = kwargs.get("verbose", False)

    def apply_scale(self, embedding_layer):
        """Apply scale transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.

        Returns:
            tuple: A tuple containing scale_x and scale_y tensors.
        """
        if self.scale:
            scale_x = self.scale_limit * nn.Tanh()(embedding_layer[:, self.count]) + 1
            scale_y = (
                self.scale_limit * nn.Tanh()(embedding_layer[:, self.count + 1]) + 1
            )

            # Update count value to switch index for affine parameter calculation
            self.count += 2

        # If there's no scale transformation, scale_x and scale_y should be 1
        else:
            scale_x = torch.ones([embedding_layer.shape[0]]).to(self.device)
            scale_y = torch.ones([embedding_layer.shape[0]]).to(self.device)

        return scale_x, scale_y

    def apply_shear(self, embedding_layer):
        """Apply shear transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.

        Returns:
            tuple: A tuple containing shear_x and shear_y tensors.
        """
        if self.shear:
            shear_x = self.shear_limit * nn.Tanh()(embedding_layer[:, self.count])
            if self.shear_symmetric:
                shear_y = shear_x
                # TODO: late add check that works
                self.count += 1
            else:
                shear_y = self.shear_limit * nn.Tanh()(
                    embedding_layer[:, self.count + 1]
                )
                self.count += 2
        else:
            shear_x = torch.zeros([embedding_layer.shape[0]]).to(self.device)
            shear_y = torch.zeros([embedding_layer.shape[0]]).to(self.device)
        return shear_x, shear_y

    def apply_rotation(self, embedding_layer, fixed_major_rotation=None):
        """Apply rotation transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.
            fixed_major_rotation (torch.Tensor, optional): Predefined major rotation values. Defaults to None.

        Returns:
            torch.Tensor: The rotation tensor.
        """
        if self.rotation:
            if fixed_major_rotation is not None:
                rotate = self.apply_predetermined_rotation(
                    embedding_layer, fixed_major_rotation
                )
            elif self.rotate_clockwise:
                rotate = self.apply_ring_rotation(embedding_layer)
            elif self.rotation_limit is not None:
                rotate = self.apply_bounded_rotation(embedding_layer)
            else:
                raise ValueError(
                    "No rotation transformation found in the model structure"
                )
            self.count += 1
        else:
            rotate = torch.zeros([embedding_layer.shape[0]]).to(self.device)
        return rotate

    def apply_bounded_rotation(self, embedding_layer):
        """Apply a bounded rotation transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.

        Returns:
            torch.Tensor: The bounded rotation tensor.
        """
        return self.rotation_limit * nn.Tanh()(embedding_layer[:, self.count])

    def apply_predetermined_rotation(self, embedding_layer, fixed_major_rotation):
        """Apply a predetermined rotation transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.
            fixed_major_rotation (torch.Tensor): Predefined major rotation values.

        Returns:
            torch.Tensor: The predetermined rotation tensor.
        """
        return fixed_major_rotation.reshape(
            embedding_layer[:, self.count].shape
        ) + self.rotation_limit * nn.Tanh()(embedding_layer[:, self.count])

    def apply_ring_rotation(self, embedding_layer):
        """Apply a ring rotation transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.

        Returns:
            torch.Tensor: The ring rotation tensor.
        """
        return nn.ReLU()(embedding_layer[:, self.count])

    def apply_translation(self, embedding_layer):
        """Apply a translation transformation to the embedding layer.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.

        Returns:
            tuple: A tuple containing the x and y translation tensors.
        """
        if self.translation:
            translation_x = self.trans_limit * nn.Tanh()(embedding_layer[:, self.count])
            translation_y = self.trans_limit * nn.Tanh()(
                embedding_layer[:, self.count + 1]
            )
            self.count += 2
        else:
            translation_x = torch.zeros([embedding_layer.shape[0]]).to(self.device)
            translation_y = torch.zeros([embedding_layer.shape[0]]).to(self.device)
        return translation_x, translation_y

    def apply_mask_intensity(self, embedding_layer):
        """Apply intensity adjustment to the mask region.

        Args:
            embedding_layer (torch.Tensor): The input tensor containing embedding values.

        Returns:
            torch.Tensor: The mask intensity adjustment parameter.
        """
        if self.mask_intensity:
            mask_parameter = (
                self.adj_mask_para
                * nn.Tanh()(embedding_layer[:, self.count : self.count + 1])
                + 1
            )
        else:
            mask_parameter = torch.ones([embedding_layer.shape[0], 1]).to(self.device)
        return mask_parameter

    def forward(self, embedding_layer, rotate_value=None):
        """Forward pass of the affine transform

        Args:
            out (Tensor): Input tensor
            rotate_value (tensor, optional): pretrained rotation if have. Defaults to None.

        Returns:
            Tensor: affine matrix and adjust parameters
        """

        # if there's scale transformation, scale x and scale y should be corresponding index of the tensor out
        scale_x, scale_y = self.apply_scale(embedding_layer)

        # if there's rotation transformation, rotation value should be corresponding index of the tensor out
        rotation = self.apply_rotation(embedding_layer, rotate_value)

        # if there's shear transformation, shear parameter should be corresponding index of the tensor out
        shear_xy, shear_yx = self.apply_shear(embedding_layer)

        # if there's translation transformation, translation x and translation y should be corresponding index of the tensor out
        translation_x, translation_y = self.apply_translation(embedding_layer)

        # add one additional learnable parameter to adjust intensity of value in mask region
        mask_parameter = self.apply_mask_intensity(embedding_layer)

        # reset count to 0 for next mini-batch
        self.count = 0

        transformation_matrix = (
            scale_x,
            scale_y,
            rotation,
            shear_xy,
            shear_yx,
            translation_x,
            translation_y,
        )

        # calculate the affine transformation matrix
        scale_shear, rotation, translation = (
            self.calculate_affine_transformation_matrix(
                embedding_layer, *transformation_matrix
            )
        )

        return scale_shear, rotation, translation, mask_parameter

    def calculate_affine_transformation_matrix(
        self,
        embedding_layer,
        scale_1,
        scale_2,
        rotate,
        shear_1,
        shear_2,
        trans_1,
        trans_2,
    ):
        a_1 = torch.cos(rotate)
        a_2 = torch.sin(rotate)
        a_4 = torch.ones([embedding_layer.shape[0]]).to(self.device)
        a_5 = torch.zeros([embedding_layer.shape[0]]).to(self.device)

        # combine shear and strain together
        c1 = torch.stack((scale_1, shear_1), dim=1)
        c2 = torch.stack((shear_2, scale_2), dim=1)
        c3 = torch.stack((a_5, a_5), dim=1)
        scale_shear = torch.stack((c1, c2, c3), dim=2)

        # Add the rotation after the shear and strain
        b1 = torch.stack((a_1, a_2), dim=1)
        b2 = torch.stack((-a_2, a_1), dim=1)
        b3 = torch.stack((a_5, a_5), dim=1)
        rotation = torch.stack((b1, b2, b3), dim=2)

        # add translation after rotation
        d1 = torch.stack((a_4, a_5), dim=1)
        d2 = torch.stack((a_5, a_4), dim=1)
        d3 = torch.stack((trans_1, trans_2), dim=1)
        translation = torch.stack((d1, d2, d3), dim=2)

        return scale_shear, rotation, translation


# narrow the range of the adjust parameter for the mask region, since it is not the noise free dataset,
# this will increase the background noise's influence to the MSE loss
class Encoder(nn.Module):
    """
    Encoder class for neural networks, incorporating affine transformations and base classification.

    This class defines the structure of an encoder that processes input images through a series of
    convolutional and pooling layers, while also applying affine transformations such as scaling,
    shearing, rotation, and translation. It supports various configurations for these transformations
    and allows for the adjustment of pixel intensities in specified mask regions.

    Returns:
        torch.Tensor: The processed tensor after passing through the encoder structure.
    """

    def __init__(self, input_image_dim, pool_list, number_channels, **kwargs):
        """
        Initializes the Encoder with specified parameters for image processing and transformation.

        Args:
            original_step_size (list of int): Dimensions [x, y] of the input image.
            pool_list (list of int): Parameters for each 2D MaxPool layer.
            conv_size (int): Number of filters in each convolutional block.
            device (torch.device): Device on which the model will run.
            scale (bool): If True, includes scale affine transformation.
            shear (bool): If True, includes shear affine transformation.
            rotation (bool): If True, includes rotation affine transformation.
            rotate_clockwise (bool): If True, rotates the image in a clockwise direction.
            translation (bool): If True, includes translation affine transformation.
            Symmetric (bool): If True, applies symmetric shear transformation.
            mask_intensity (bool): If True, allows learnable intensity in the mask region.
            num_base (int, optional): Number of base elements. Defaults to 2.
            fixed_mask (list of torch.Tensor, optional): List of binary tensors for masking. Defaults to None.
            interpolate (bool): If True, calculates loss in interpolated version. Defaults to False.
            revise_affine (bool): If True, applies revised affine transformations. Defaults to False.
            up_size (int, optional): Image size for MSE loss calculation. Defaults to 800.
            scale_limit (float): Range limit for scaling. Defaults to 0.05.
            shear_limit (float): Range limit for shearing. Defaults to 0.1.
            rotation_limit (float): Range limit for rotation. Defaults to 0.1.
            trans_limit (float): Range limit for translation. Defaults to 0.15.
            adj_mask_para (float): Range for adjusting pixel values in mask region. Defaults to 0.
            radius (int): Radius for cropping small square images. Defaults to 60.
            coef (float): Threshold for center of mass (COM) operation. Defaults to 1.5.
            reduced_size (int): Input length for the K-top layer. Defaults to 20.
            interpolate_mode (str, optional): Interpolation mode for F.interpolate(). Defaults to 'bicubic'.
            affine_mode (str): Affine mode for F.affine_grid(). Defaults to 'bicubic'.
        """
        
        self.input_image_dim = input_image_dim
        self.pool_list = pool_list
        self.num_channels = number_channels

        self.initialize_variables(kwargs)
        super(Encoder, self).__init__()
        
        self.model_layers = []        
    
        # set number of blocks depends on length of pool list, each block includes one conv_block and one identity_block
        self.input_block()
        self.build_conv_block()
        self.nn_module_list = nn.ModuleList(self.model_layers)

        # update input_size for linear layer
        self.build_flatten_block()
        self.calculate_embedding_size()

        
        self.build_mask()

        # if mask_intensity is true, give an extra index of learnable parameter for adjusting pixel intensity
        self.add_intensity_learnable_parameter()

        # set the number of base (number of cluster)
        self.build_classification_block()        

        # initialize affine matrix
        self.affine_matrix = AffineTransformationBlock(
            device=self.device,
            scale=self.scale,
            shear=self.shear,
            rotation=self.rotation,
            rotate_clockwise=self.rotate_clockwise,
            translation=self.translation,
            symmetric=self.symmetric,
            mask_intensity=self.mask_intensity_flag,
            scale_limit=self.scale_limit,
            shear_limit=self.shear_limit,
            rotation_limit=self.rotation_limit,
            trans_limit=self.trans_limit,
            adj_mask_para=self.adj_mask_para,
            **kwargs,
        ).to(self.device)

    def build_classification_block(self):
        self.for_k = nn.Linear(self.dense_layer_size, self.num_base)
        self.norm = nn.LayerNorm(self.num_base)
        self.softmax = nn.Softmax()

    def add_intensity_learnable_parameter(self):
        if self.mask_intensity_flag:
            self.dense = nn.Linear(self.dense_layer_size + self.num_base, self.embedding_size + 1)
        else:
            # Set the all the adj parameter to be the same
            self.dense = nn.Linear(self.dense_layer_size + self.num_base, self.embedding_size)

    def build_mask(self):
        if self.fixed_mask_flag is not None:
            # Set the mask_ to upscale mask if the interpolate mode is True
            if self.interpolate_flag:
                mask_with_inp = []

                for mask_ in self.fixed_mask_flag:
                    # switch mask type into tensor
                    temp_mask = torch.tensor(
                        mask_.reshape(1, 1, self.input_size_0, self.input_size_1),
                        dtype=torch.float,
                    )
                    
                    # add the same interpolate as input images to mask
                    temp_mask = F.interpolate(
                        temp_mask,
                        size=(self.up_size, self.up_size),
                        mode=self.interpolate_mode,
                    )

                    # make the interpolated mask binary ahead to avoid distortion
                    temp_mask[temp_mask < self.interpolation_threshold] = 0
                    temp_mask[temp_mask >= self.interpolation_threshold] = 1
                    temp_mask = torch.tensor(temp_mask.squeeze(), dtype=torch.bool)
                    mask_with_inp.append(temp_mask)

                self.mask = mask_with_inp

            else:
                self.mask = self.fixed_mask_flag
        else:
            self.mask = None

    def build_flatten_block(self):
        flattened_image_size = self.reduced_image_size[0] * self.reduced_image_size[1]
        
        
        self.cov2d = nn.Conv2d(
            1, self.num_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            self.num_channels, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dense_before_embedding = nn.Linear(flattened_image_size, self.dense_layer_size)

    def calculate_embedding_size(self):
        self.embedding_size = 0

        # determine number of embedding channels depends on affine transformation type
        if self.scale:
            self.embedding_size += 2
            
        if self.shear:
            if self.symmetric:
                self.embedding_size += 1
            else:
                self.embedding_size += 2

        if self.rotation:
            self.embedding_size += 1

        if self.translation:
            self.embedding_size += 2

        if self.embedding_size == 0:
            print(" No affine transformation found in the model structure")

    def input_block(self):
        self.model_layers.append(
            conv_block(num_channels=self.num_channels, spatial_dims=self.input_image_dim)
        )
        self.model_layers.append(
            identity_block(num_channels=self.num_channels, spatial_dims=self.input_image_dim)
        )
        self.model_layers.append(nn.MaxPool2d(self.pool_list[0], stride=self.pool_list[0]))
        
    @property
    def num_layers(self):
        return len(self.model_layers)

    def build_conv_block(self):
        
        self.reduced_image_size = self.input_image_dim
        number_of_blocks = len(self.pool_list)
        
        for i in range(1, number_of_blocks):
        
            # update value of step size for each block
            self.reduced_image_size = self.calculate_image_size(self.reduced_image_size, self.pool_list[i-1])
            
            self.model_layers.append(
                conv_block(num_channels=self.num_channels, spatial_dims=self.reduced_image_size)
            )
            self.model_layers.append(
                identity_block(num_channels=self.num_channels, spatial_dims=self.reduced_image_size)
            )
            # add MaxPool layer after each block
            self.model_layers.append(nn.MaxPool2d(self.pool_list[i], stride=self.pool_list[i]))
            
        # update image size to to each convolutional block and identity block
        self.reduced_image_size = [
            self.reduced_image_size[0] // self.pool_list[-1],
            self.reduced_image_size[1] // self.pool_list[-1],
        ]

    def calculate_image_size(self, layer_norm_step_size, pool_size):
        image_size = [
                layer_norm_step_size[0] // pool_size,
                layer_norm_step_size[1] // pool_size,
            ]
        return image_size

    def initialize_variables(self, kwargs):
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scale = kwargs.get("scale", True)
        self.shear = kwargs.get("shear", True)
        self.rotation = kwargs.get("rotation", True)
        self.rotate_clockwise = kwargs.get("rotate_clockwise", True)
        self.translation = kwargs.get("translation", False)
        self.symmetric = kwargs.get("symmetric", True)
        self.mask_intensity_flag = kwargs.get("mask_intensity", True)
        self.num_base = kwargs.get("num_base", 2)
        self.fixed_mask_flag = kwargs.get("fixed_mask", None)
        self.interpolate_flag = kwargs.get("interpolate", False)
        self.revise_affine = kwargs.get("revise_affine", False)
        self.up_size = kwargs.get("up_size", 800)
        self.scale_limit = kwargs.get("scale_limit", 0.05)
        self.shear_limit = kwargs.get("shear_limit", 0.1)
        self.rotation_limit = kwargs.get("rotation_limit", 0.1)
        self.trans_limit = kwargs.get("trans_limit", 0.15)
        self.adj_mask_para = kwargs.get("adj_mask_para", 0)
        self.radius = kwargs.get("radius", 60)
        self.coef = kwargs.get("coef", 1.5)
        self.dense_layer_size = kwargs.get("dense_layer_size", 20)
        self.interpolate_mode = kwargs.get("interpolate_mode", "bicubic")
        self.affine_mode = kwargs.get("affine_mode", "bicubic")
        self.num_k_sparse = kwargs.get("num_k_sparse", 1)
        self.interpolation_threshold = kwargs.get("interpolation_threshold", 0.5)

    # create K-sparse strategy for classification
    def ktop(self, x):
        """ktop function

        Args:
            x (torch.tensor): 1D vector

        Returns:
            torch.tensor: binary vector
        """
        x = self.for_k(x)
        x = self.norm(x)
        x = self.softmax(x)
        
        
        # determine input images belongs to which base cluster by top k algorithm, k=1
        k_top_output = ktop_layer(x, self.num_k_sparse)

        return k_top_output

    def forward(self, x, rotate_value=None):
        """forward function for nn.Module class

        Args:
            x (torch.tensor): input torch.tensor image
            rotate_value (float, optional): float value represents pretrained rotation angle. Defaults to None.
        """

        # reshape the input into (mini-batch, 1 , image_size)
        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.num_layers):
            out = self.nn_module_list[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        kout = self.dense_before_embedding(out)
        k_out = self.ktop(kout)

        # concatenate reduced dimensional vector and output vector of k-sparse function
        out = torch.cat((kout, k_out), dim=1).to(self.device)
        out = self.dense(out)

        # generate affine matrix by tensor out
        scale_shear, rotation, translation, mask_parameter = self.affine_matrix(
            out, rotate_value
        )

        # add affine transformation to input image
        if not self.interpolate_flag:
            grid_1 = F.affine_grid(scale_shear.to(self.device), x.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x, grid_1)

            grid_2 = F.affine_grid(rotation.to(self.device), x.size()).to(self.device)
            out_rotate = F.grid_sample(out_sc_sh, grid_2)

            grid_3 = F.affine_grid(translation.to(self.device), x.size()).to(
                self.device
            )
            output = F.grid_sample(out_rotate, grid_3)

        else:
            x_inp = x.view(-1, 1, self.input_size_0, self.input_size_1)

            # image interpolation before affine transformation
            x_inp = F.interpolate(
                x_inp, size=(self.up_size, self.up_size), mode=self.interpolate_mode
            )

            grid_1 = F.affine_grid(scale_shear.to(self.device), x_inp.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x_inp, grid_1, mode=self.affine_mode)

            grid_2 = F.affine_grid(rotation.to(self.device), x_inp.size()).to(
                self.device
            )
            out_rotate = F.grid_sample(out_sc_sh, grid_2, mode=self.affine_mode)

            grid_3 = F.affine_grid(translation.to(self.device), x_inp.size()).to(
                self.device
            )
            output = F.grid_sample(out_rotate, grid_3)

        if self.interpolate_flag:
            # apply inverse affine to each diffraction spot if revise_affine is True
            if self.revise_affine:
                # Test 1.5 is good for 5%-45% background noise, add to 2 for larger noise and rot512x512 4dstem
                out_revise = reverse_affine_transform_gpu(
                    output,
                    self.mask,
                    x.shape[0],
                    scale_shear,
                    self.device,
                    intensity_adjustment_factor=mask_parameter,
                    radius=self.radius,
                    coef=self.coef,
                    affine_mode=self.affine_mode,
                )
            else:
                out_revise = output

            return (
                out_revise,
                k_out,
                scale_shear,
                rotation,
                translation,
                mask_parameter,
                x_inp,
            )

        else:
            return output, k_out, scale_shear, rotation, translation, mask_parameter


class Encoder_FPGA(nn.Module):
    """
        nn.Module class of Encoder structure of the model to fpga
    Returns:
        tensor: torch.tensor
    """

    def __init__(
        self,
        original_step_size,
        pool_list,
        embedding_size,
        conv_size,
        device,
        first_stride=1,
        kernel_size=3,
        fixed_mask=None,
        interpolate=False,
        up_size=800,
        scale_limit=0.05,
        shear_limit=0.1,
        to_onnx=False,
    ):
        """_summary_

        Args:
            original_step_size (list of int): the x and y size of input image
            pool_list (list of int): the list of parameter for each 2D MaxPool layer
            embedding_size (int): the value for number of channels
            conv_size (int): the value of filters number goes to each block
            device (torch.device): set the device to run the model
            scale (bool): set to True if the model include scale affine transform
            shear (bool): set to True if the model include shear affine transform
            rotation (bool): set to True if the model include rotation affine transform
            rotate_clockwise (bool): set to True if the image should be rotated along one direction
            translation (bool): set to True if the model include translation affine transform
            Symmetric (bool): set to True if the shear affine transform is symmetric
            mask_intensity (bool):set to True if the intensity of the mask region is learnable
            num_base(int, optional): the value for number of base. Defaults to 2.
            fixed_mask (list of tensor, optional): The list of tensor with binary type. Defaults to None.
            interpolate (bool): set to determine if need to calculate loss value in interpolated version. Defaults to False.
            revise_affine (bool): set to determine if need to add revise affine to image with affine transformation. Default to False.
            up_size (int, optional): the size of image to set for calculating MSE loss. Defaults to 800.
            scale_limit (float): set the range of scale. Defaults to 0.05.
            shear_limit (float): set the range of shear. Defaults to 0.1.
            rotation_limit (float): set the range of shear. Defaults to 0.1.
            trans_limit (float): set the range of translation. Defaults to 0.15.
            adj_mask_para (float): set the range of learnable parameter used to adjust pixel value in mask region. Defaults to 0.
            radius (int): set the radius of small square image for cropping. Defaults to 60.
            coef (float): set the threshold for COM operation. Defaults to 1.5.
            reduced_size (int): set the input length of K-top layer. Defaults 20.
            interpolate_size (string, optional): set the interpolate mode to function F.interpolate(). Defaults 'bicubic'.
            affine_mode (int): set the affine mode to function F.affine_grid(). Defaults 'bicubic'.
        """
        super(Encoder_FPGA, self).__init__()

        self.fixed_mask = fixed_mask
        self.device = device
        blocks = []
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        number_of_blocks = len(pool_list)
        original_step_size = [
            original_step_size[0] // first_stride,
            original_step_size[1] // first_stride,
        ]
        blocks.append(conv_block_fpga(t_size=conv_size))
        blocks.append(nn.MaxPool2d(pool_list[0], stride=pool_list[0]))
        for i in range(1, number_of_blocks):
            original_step_size = [
                original_step_size[0] // pool_list[i - 1],
                original_step_size[1] // pool_list[i - 1],
            ]
            blocks.append(conv_block_fpga(t_size=conv_size))
            blocks.append(identity_block_fpga(t_size=conv_size))
            blocks.append(nn.MaxPool2d(pool_list[i], stride=pool_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)
        original_step_size = [
            original_step_size[0] // pool_list[-1],
            original_step_size[1] // pool_list[-1],
        ]

        input_size = original_step_size[0] * original_step_size[1]
        self.cov2d = nn.Conv2d(
            1,
            conv_size,
            kernel_size,
            stride=first_stride,
            padding=1,
            padding_mode="zeros",
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.tanh = nn.Tanh()
        self.before = nn.Linear(input_size, 20)
        self.embedding_size = embedding_size

        self.to_onnx = to_onnx
        self.interpolate = interpolate
        self.up_size = up_size
        # initialize affine matrix
        self.dense = nn.Linear(20, self.embedding_size)
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.create_mask()

    def create_mask(self):
        if self.fixed_mask is not None:
            # Set the mask_ to upscale mask if the interpolate set True
            if self.interpolate:
                mask_with_inp = []
                for mask_ in self.fixed_mask:
                    temp_mask = torch.tensor(
                        mask_.reshape(1, 1, self.input_size_0, self.input_size_1),
                        dtype=torch.float,
                    )
                    temp_mask = F.interpolate(
                        temp_mask, size=(self.up_size, self.up_size), mode="bicubic"
                    )
                    temp_mask[temp_mask < 0.5] = 0
                    temp_mask[temp_mask >= 0.5] = 1
                    temp_mask = torch.tensor(temp_mask.squeeze(), dtype=torch.bool)
                    mask_with_inp.append(temp_mask)

                self.mask = mask_with_inp

            else:
                self.mask = self.fixed_mask
        else:
            self.mask = None

    def rotate_mask(self):
        return self.mask

    def check_inp(self):
        return self.interpolate

    def check_upsize(self):
        return self.up_size

    def forward(self, x):
        #        x = x.view(-1,1,self.input_size_0,self.input_size_1)

        out = self.cov2d(x)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.before(out)
        out = self.dense(out)
        # determine if the model goes to onnx
        if self.to_onnx:
            return out
        else:
            # generate affine matrix by tensor out
            ################# out of inference ###################
            scale_1 = self.scale_limit * nn.Tanh()(out[:, 0]) + 1
            scale_2 = self.scale_limit * nn.Tanh()(out[:, 1]) + 1

            rotate = nn.ReLU()(out[:, 2])
            shear_1 = self.shear_limit * nn.Tanh()(out[:, 3])
            a_1 = torch.cos(rotate)
            a_2 = torch.sin(rotate)
            a_5 = rotate * 0

            # combine shear and strain together
            c1 = torch.stack((scale_1, shear_1), dim=1).squeeze()
            c2 = torch.stack((shear_1, scale_2), dim=1).squeeze()
            c3 = torch.stack((a_5, a_5), dim=1).squeeze()
            scale_shear = torch.stack((c1, c2, c3), dim=2)

            # Add the rotation after the shear and strain
            b1 = torch.stack((a_1, a_2), dim=1).squeeze()
            b2 = torch.stack((-a_2, a_1), dim=1).squeeze()
            b3 = torch.stack((a_5, a_5), dim=1).squeeze()
            rotation = torch.stack((b1, b2, b3), dim=2)

            grid_1 = F.affine_grid(scale_shear.to(self.device), x.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x, grid_1)

            grid_2 = F.affine_grid(rotation.to(self.device), x.size()).to(self.device)
            output = F.grid_sample(out_sc_sh, grid_2)

            return output, scale_shear, rotation, out


class Decoder(nn.Module):
    """
        nn.Module class of Decoder (Generator for generating base)

    Returns:
        tensor: torch.tensor
    """

    def __init__(self, original_step_size, up_list, conv_size, device, num_base=2):
        """

        Args:
            original_step_size (list of int): the x and y size of input image
            up_list (list of int): the list of parameter for each 2D Upsample layer
            conv_size (int): the value of filters number goes to each block
            device (torch.device): set the device to run the model
            num_base (int): the value for number of base. Defaults to 2.
        """

        super(Decoder, self).__init__()

        self.device = device

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(num_base, original_step_size[0] * original_step_size[1])

        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        # set number of blocks depends on length of pool list, each block includes one conv_block and one identity_block
        blocks = []
        number_of_blocks = len(up_list)
        blocks.append(
            conv_block(num_channels=conv_size, spatial_dims=original_step_size)
        )
        blocks.append(
            identity_block(num_channels=conv_size, spatial_dims=original_step_size)
        )
        for i in range(number_of_blocks):
            # add UpSample layer before each block
            blocks.append(
                nn.Upsample(
                    scale_factor=up_list[i], mode="bilinear", align_corners=True
                )
            )
            # update value of step size for each block
            original_step_size = [
                original_step_size[0] * up_list[i],
                original_step_size[1] * up_list[i],
            ]
            blocks.append(
                conv_block(num_channels=conv_size, spatial_dims=original_step_size)
            )
            blocks.append(
                identity_block(num_channels=conv_size, spatial_dims=original_step_size)
            )

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        # self.output_size_0 = original_step_size[0]
        # self.output_size_1 = original_step_size[1]

        self.relu_1 = nn.LeakyReLU(0.001)

    def forward(self, x):
        """Forward pass of the decoder

        Args:
            x (tensor): input tensor

        Returns:
            tensor: output tensor
        """

        # generator to reconstruct image into original size
        out = self.dense(x)
        out = out.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = self.relu_1(out)

        return out


class Decoder_FPGA(nn.Module):
    def __init__(
        self,
        embedding_size,
        device,
        scale_limit=0.05,
        shear_limit=0.1,
        rotate_limit=0.1,
        to_onnx=False,
    ):
        super(Decoder_FPGA, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.linear_1 = nn.Linear(self.embedding_size, 20)
        self.linear_2 = nn.Linear(20, self.embedding_size)
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.rotate_limit = rotate_limit
        self.to_onnx = to_onnx

    def forward(self, vec, rotate_value=None):
        out = self.linear_1(vec)
        out = self.linear_2(out)
        if self.to_onnx:
            return out
        else:
            # create strain and rotation matrix
            scale_1 = self.scale_limit * nn.Tanh()(out[:, 0]) + 1
            scale_2 = self.scale_limit * nn.Tanh()(out[:, 1]) + 1
            rotate = rotate_value.reshape(
                out[:, 2].shape
            ) + self.rotate_limit * nn.Tanh()(out[:, 2])
            shear_1 = self.shear_limit * nn.Tanh()(out[:, 3])
            a_1 = torch.cos(rotate)
            a_2 = torch.sin(rotate)
            a_5 = rotate * 0

            # combine shear and strain together
            c1 = torch.stack((scale_1, shear_1), dim=1).squeeze()
            c2 = torch.stack((shear_1, scale_2), dim=1).squeeze()
            c3 = torch.stack((a_5, a_5), dim=1).squeeze()
            scale_shear = torch.stack((c1, c2, c3), dim=2)

            # Add the rotation after the shear and strain
            b1 = torch.stack((a_1, a_2), dim=1).squeeze()
            b2 = torch.stack((-a_2, a_1), dim=1).squeeze()
            b3 = torch.stack((a_5, a_5), dim=1).squeeze()
            rotation = torch.stack((b1, b2, b3), dim=2)

            return scale_shear, rotation, out


class CC_ST_AE(nn.Module):
    """
        nn.Module class of VAE, which includes both encoder and decoder
    Returns:
        tensor: torch.tensor, affine matrix, interpolated tensor, updated mask list
    """

    def __init__(
        self,
        encoder,
        decoder,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        radius=60,
        coef=1.5,
        interpolate_mode="bicubic",
        affine_mode="bicubic",
    ):
        """Initializes the CC_ST_AE class, which combines an encoder and decoder for a VAE model.

        Args:
            encoder (torch.Module): The encoder component of the neural network.
            decoder (torch.Module): The decoder component of the neural network.
            device (torch.device): The device on which the model will run. Defaults to CUDA if available, otherwise CPU.
            radius (int): The radius for cropping small square images. Defaults to 60.
            coef (float): The threshold coefficient for the Center of Mass (COM) operation. Defaults to 1.5.
            interpolate_mode (str): The interpolation mode used in F.interpolate(). Defaults to 'bicubic'.
            affine_mode (str): The affine transformation mode used in F.affine_grid(). Defaults to 'bicubic'.
        """
        super(CC_ST_AE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Load variable from encoder
        self.mask = encoder.mask
        self.interpolate = encoder.interpolate
        self.revise_affine = encoder.revise_affine
        self.up_size = encoder.up_size
        self.radius = radius
        self.coef = coef
        self.interpolate_mode = interpolate_mode
        self.affine_mode = affine_mode

    def rotate_mask(self):
        """function return the mask list

        Returns:
            list: list of torch.bool array
        """

        return self.mask

    # if have pretrained rotation, add to forward function, and keep optimizing on it
    def forward(self, x, rotate_value=None):
        """

        Args:
            x (torch.tensor): input torch.tensor image
            rotate_value (float, optional): float value represents pretrained rotation angle. Defaults to None.
        """

        if self.interpolate:
            (
                predicted_revise,
                k_out,
                scale_shear,
                rotation,
                translation,
                adj_mask,
                x_inp,
            ) = self.encoder(x, rotate_value)

        else:
            predicted_revise, k_out, scale_shear, rotation, translation, adj_mask = (
                self.encoder(x, rotate_value)
            )
        # create identity matrix for computing inverse affine matrix
        identity = (
            torch.tensor([0, 0, 1], dtype=torch.float)
            .reshape(1, 1, 3)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )
        # add identity matrix to affine matrix
        new_theta_1 = torch.cat((scale_shear, identity), axis=1).to(self.device)
        new_theta_2 = torch.cat((rotation, identity), axis=1).to(self.device)
        new_theta_3 = torch.cat((translation, identity), axis=1).to(self.device)

        # generate inverse affine matrix
        inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
        inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)
        inver_theta_3 = torch.linalg.inv(new_theta_3)[:, 0:2].to(self.device)

        predicted_base = self.decoder(k_out)

        # Up grid image when interpolate mode is True
        if self.interpolate:
            predicted_base_inp = F.interpolate(
                predicted_base,
                size=(self.up_size, self.up_size),
                mode=self.interpolate_mode,
            )
            # add inverse affine transform to generated base
            grid_1 = F.affine_grid(
                inver_theta_1.to(self.device), predicted_base_inp.size()
            ).to(self.device)
            grid_2 = F.affine_grid(
                inver_theta_2.to(self.device), predicted_base_inp.size()
            ).to(self.device)
            grid_3 = F.affine_grid(
                inver_theta_3.to(self.device), predicted_base_inp.size()
            ).to(self.device)

            predicted_translation = F.grid_sample(
                predicted_base_inp, grid_3, mode=self.affine_mode
            )
            predicted_rotate = F.grid_sample(
                predicted_translation, grid_2, mode=self.affine_mode
            )
            predicted_input = F.grid_sample(
                predicted_rotate, grid_1, mode=self.affine_mode
            )

        else:
            # add inverse affine transform to generated base
            grid_1 = F.affine_grid(inver_theta_1.to(self.device), x.size()).to(
                self.device
            )
            grid_2 = F.affine_grid(inver_theta_2.to(self.device), x.size()).to(
                self.device
            )
            grid_3 = F.affine_grid(inver_theta_3.to(self.device), x.size()).to(
                self.device
            )

            predicted_translation = F.grid_sample(predicted_base, grid_3)

            predicted_rotate = F.grid_sample(predicted_translation, grid_2)

            predicted_input = F.grid_sample(predicted_rotate, grid_1)

        # create new mask list to save updated mask region with inverse affine transform
        new_list = []
        if self.mask is not None:
            for mask_ in self.mask:
                # repeat number of mask to size of mini-batch
                batch_mask = (
                    mask_.reshape(1, 1, mask_.shape[-2], mask_.shape[-1])
                    .repeat(x.shape[0], 1, 1, 1)
                    .to(self.device)
                )

                batch_mask = torch.tensor(batch_mask, dtype=torch.float).to(self.device)

                rotated_mask = F.grid_sample(batch_mask, grid_2)

                if self.interpolate:
                    # Add reverse affine transform of scale and shear to make all spots in the mask region, crucial when mask region small
                    rotated_mask = F.grid_sample(rotated_mask, grid_1)

                # maintain the correct size of mask region after affine transformation
                rotated_mask[rotated_mask < 0.5] = 0
                rotated_mask[rotated_mask >= 0.5] = 1

                rotated_mask = (
                    torch.tensor(rotated_mask, dtype=torch.bool)
                    .reshape(-1, rotated_mask.shape[-2], rotated_mask.shape[-1])
                    .to(self.device)
                )

                new_list.append(rotated_mask)

        if self.interpolate:
            # apply inverse affine transform to recreate input image
            if self.revise_affine:
                predicted_input_revise = reverse_affine_transform_gpu(
                    predicted_input,
                    new_list,
                    x.shape[0],
                    inver_theta_1,
                    self.device,
                    intensity_adjustment_factor=adj_mask,
                    radius=self.radius,
                    coef=self.coef,
                    divide_by_intensity_adjustment=True,
                    affine_mode=self.affine_mode,
                )
            else:
                predicted_input_revise = predicted_input

            # change predicted_base to predicted_base_inp, add new_list when interpolate mode is True
            return (
                predicted_revise,
                predicted_base_inp,
                predicted_input_revise,
                k_out,
                scale_shear,
                rotation,
                translation,
                adj_mask,
                new_list,
                x_inp,
            )

        else:
            return (
                predicted_revise,
                predicted_base,
                predicted_input,
                k_out,
                scale_shear,
                rotation,
                translation,
                adj_mask,
                new_list,
            )


class Joint_FPGA(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        device,
        disturb=-15,
    ):
        super(Joint_FPGA, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.disturb = disturb
        self.input_size_0 = encoder.input_size_0
        self.input_size_1 = encoder.input_size_1
        self.up_size = encoder.up_size
        self.mask = encoder.mask
        self.to_onnx = self.encoder.to_onnx

    def forward(self, x):
        if self.to_onnx:
            vec = self.encoder(x)
            vec_2 = self.decoder(vec)
            return vec, vec_2
        else:
            out_1st_affine, s_c, rotation_1, vec = self.encoder(x)
            # switch angle to particular direction
            rot = rotation_1[:, :, 0]
            angle = torch.remainder(
                self.disturb * torch.pi / 180
                + torch.atan2(rot[:, 1].reshape(-1), rot[:, 0].reshape(-1)),
                torch.pi / 3,
            )

            scale_shear, rotation_2, vec_2 = self.decoder(vec, angle)
            x_inp = x.view(-1, 1, self.input_size_0, self.input_size_1)
            x_inp = F.interpolate(
                x_inp, size=(self.up_size, self.up_size), mode="bicubic"
            )

            grid_1 = F.affine_grid(scale_shear.to(self.device), x_inp.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x_inp, grid_1, mode="bicubic")

            grid_2 = F.affine_grid(rotation_2.to(self.device), x_inp.size()).to(
                self.device
            )
            out_affine = F.grid_sample(out_sc_sh, grid_2, mode="bicubic")

            # For clean dataset we assume there's no background noise, so the coefficient on ReLU(VALUE-coef*MEAN_VALUE) is 0,
            # The dictionary compared with out_2nd_affine is interpolated, so do not need to recover the size back to origin.
            out_2nd_affine = reverse_affine_transform_gpu(
                out_affine,
                self.mask,
                x.shape[0],
                scale_shear,
                self.device,
                radius=60,
                coef=1.5,
            )

            #            out_revise = F.interpolate(out_revise,size=(self.input_size_0,self.input_size_1),mode = 'bicubic')

            return (
                out_2nd_affine,
                out_1st_affine,
                scale_shear,
                rotation_1,
                rotation_2,
                x_inp,
                vec,
                vec_2,
            )


# class CC_ST_AE(Encoder, Decoder, Joint):
#     def __init__(self, device, **kwargs):
#         super(CC_ST_AE, self).__init__(device, **kwargs)


def make_model_fn(
    device,
    learning_rate=3e-5,
    en_original_step_size=[200, 200],
    de_original_step_size=[5, 5],
    pool_list=[5, 4, 2],
    up_list=[2, 4, 5],
    conv_size=128,
    scale=True,
    shear=True,
    rotation=True,
    rotate_clockwise=True,
    translation=False,
    Symmetric=True,
    mask_intensity=True,
    num_base=1,
    up_size=800,
    scale_limit=0.05,
    shear_limit=0.1,
    rotation_limit=0.1,
    trans_limit=0.15,
    adj_mask_para=0,
    radius=60,
    coef=1.5,
    reduced_size=20,
    interpolate_mode="bicubic",
    affine_mode="bicubic",
    fixed_mask=None,
    interpolate=True,
    revise_affine=False,
):
    """function for creating autoencoder and optimizer

    Args:
        device (torch.device): set the device initialize model
        learning_rate (float): learning rate to optimizer. Defaults to 3e-5.
        en_original_step_size (list of int): the x and y size of input image to encoder
        de_original_step_size (list of int): the x and y size of input image to decoder
        pool_list (list of int): the list of parameter for each 2D MaxPool layer
        embedding_size (int): the value for number of channels
        conv_size (int): the value of filters number goes to each block
        device (torch.device): set the device to run the model
        scale (bool): set to True if the model include scale affine transform
        shear (bool): set to True if the model include shear affine transform
        rotation (bool): set to True if the model include rotation affine transform
        rotate_clockwise (bool): set to True if the image should be rotated along one direction
        translation (bool): set to True if the model include translation affine transform
        Symmetric (bool): set to True if the shear affine transform is symmetric
        mask_intensity (bool):set to True if the intensity of the mask region is learnable
        num_base(int, optional): the value for number of base. Defaults to 2.
        fixed_mask (list of tensor, optional): The list of tensor with binary type. Defaults to None.
        interpolate (bool): set to determine if need to calculate loss value in interpolated version. Defaults to False.
        up_size (int, optional): the size of image to set for calculating MSE loss. Defaults to 800.
        scale_limit (float): set the range of scale. Defaults to 0.05.
        shear_limit (float): set the range of shear. Defaults to 0.1.
        rotation_limit (float): set the range of shear. Defaults to 0.1.
        trans_limit (float): set the range of translation. Defaults to 0.15.
        adj_mask_para (float): set the range of learnable parameter used to adjust pixel value in mask region. Defaults to 0.
        radius (int): set the radius of small square image for cropping. Defaults to 60.
        coef (float): set the threshold for COM operation. Defaults to 1.5.
        reduced_size (int): set the input length of K-top layer. Defaults 20.
        interpolate_size (string, optional): set the interpolate mode to function F.interpolate(). Defaults 'bicubic'.
        affine_mode (int): set the affine mode to function F.affine_grid(). Defaults 'bicubic'.
        revise_affine (bool): set to determine if need to add revise affine to image with affine transformation. Default to False.
    Returns:
        torch.Module: pytorch model and optimizer
    """

    encoder = Encoder(
        en_original_step_size,
        pool_list,
        conv_size,
        device,
        scale,
        shear,
        rotation,
        rotate_clockwise,
        translation,
        Symmetric,
        mask_intensity,
        num_base,
        fixed_mask,
        interpolate,
        revise_affine,
        up_size,
        scale_limit,
        shear_limit,
        rotation_limit,
        trans_limit,
        adj_mask_para,
        radius,
        coef,
        reduced_size,
        interpolate_mode,
        affine_mode,
    ).to(device)

    decoder = Decoder(de_original_step_size, up_list, conv_size, device, num_base).to(
        device
    )

    join = CC_ST_AE(
        encoder, decoder, device, radius, coef, interpolate_mode, affine_mode
    ).to(device)

    optimizer = optim.Adam(join.parameters(), lr=learning_rate)

    if device == torch.device("cuda"):
        join = torch.nn.parallel.DataParallel(join)

    return encoder, decoder, join, optimizer


def make_model_fpga(
    device,
    en_original_step_size=[200, 200],
    pool_list=[4, 2, 2],
    embedding_size=4,
    conv_size_encoder=14,
    fixed_mask=None,
    learning_rate=3e-5,
    interpolate=False,
    up_size=800,
    disturb=-15,
    first_stride=4,
    kernel_size=8,
    scale_limit=0.05,
    shear_limit=0.1,
    rotate_limit=0.1,
    to_onnx=False,
):
    """_summary_

    Args:
        device (_type_): _description_
        en_original_step_size (list, optional): _description_. Defaults to [200,200].
        pool_list (list, optional): _description_. Defaults to [4,2,2].
        embedding_size (int, optional): _description_. Defaults to 4.
        conv_size_encoder (int, optional): _description_. Defaults to 14.
        fixed_mask (_type_, optional): _description_. Defaults to None.
        learning_rate (_type_, optional): _description_. Defaults to 3e-5.
        interpolate (bool, optional): _description_. Defaults to False.
        up_size (int, optional): _description_. Defaults to 800.
        disturb (int, optional): _description_. Defaults to -15.
        first_stride (int, optional): _description_. Defaults to 4.
        kernel_size (int, optional): _description_. Defaults to 8.
        scale_limit (float, optional): _description_. Defaults to 0.05.
        shear_limit (float, optional): _description_. Defaults to 0.1.
        rotate_limit (float, optional): _description_. Defaults to 0.1.
        to_onnx (bool, optional): _description_. Defaults to False.
    """

    encoder = Encoder_FPGA(
        original_step_size=en_original_step_size,
        pool_list=pool_list,
        embedding_size=embedding_size,
        conv_size=conv_size_encoder,
        device=device,
        first_stride=first_stride,
        kernel_size=kernel_size,
        fixed_mask=fixed_mask,
        interpolate=interpolate,
        up_size=up_size,
        scale_limit=scale_limit,
        shear_limit=shear_limit,
        to_onnx=to_onnx,
    ).to(device)

    decoder = Decoder_FPGA(
        embedding_size=embedding_size,
        device=device,
        scale_limit=scale_limit,
        shear_limit=shear_limit,
        rotate_limit=rotate_limit,
        to_onnx=to_onnx,
    ).to(device)

    join = Joint_FPGA(encoder, decoder, device, disturb=disturb)

    optimizer = optim.Adam(join.parameters(), lr=learning_rate)

    #    encoder = torch.nn.parallel.DataParallel(encoder)

    return encoder, decoder, join, optimizer
