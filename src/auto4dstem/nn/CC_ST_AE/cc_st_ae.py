import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from auto4dstem.nn.CC_ST_AE.decoder import Decoder
from auto4dstem.nn.CC_ST_AE.encoder import Encoder
from auto4dstem.nn.CC_ST_AE.transforms import (
    apply_affine_transformation_to_image,
    reverse_affine_transform,
    generate_inverse_affine,
)


# TODO: make this inherit structure to base class
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
        reverse_affine_transform_crop_radius=60,
        COM_threshold_coef=1.5,
        upsampling_interpolation_mode="bicubic",
        **kwargs,
    ):
        """Initializes the CC_ST_AE class, which combines an encoder and decoder for a VAE model.

        Args:
            encoder (torch.Module): The encoder component of the neural network.
            decoder (torch.Module): The decoder component of the neural network.
            device (torch.device): The device on which the model will run. Defaults to CUDA if available, otherwise CPU.
            reverse_affine_transform_crop_radius (int): The radius for cropping small square images. Defaults to 60.
            COM_threshold_coef (float): The threshold coefficient for the Center of Mass (COM) operation. Defaults to 1.5.
            upsampling_interpolation_mode (str): The interpolation mode used in F.interpolate(). Defaults to 'bicubic'.
            affine_interpolation_mode (str): The affine transformation mode used in F.affine_grid(). Defaults to 'bicubic'.
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
        self.reverse_affine_transform_crop_radius = reverse_affine_transform_crop_radius
        self.COM_threshold_coef = COM_threshold_coef
        self.upsampling_interpolation_mode = upsampling_interpolation_mode

        if self.interpolate:
            self.affine_interpolation_mode = kwargs.get("affine_interpolation_mode", "bicubic")
        else:
            self.affine_interpolation_mode = kwargs.get("affine_interpolation_mode", "bilinear")

    def rotate_mask(self):
        """function return the mask list

        Returns:
            list: list of torch.bool array
        """

        return self.encoder.mask

    # if have pretrained rotation, add to forward function, and keep optimizing on it
    def forward(self, x, rotate_value=None):
        """

        Args:
            x (torch.tensor): input torch.tensor image
            rotate_value (float, optional): float value represents pretrained rotation angle. Defaults to None.
        """

        (
            predicted_revise,
            k_out,
            scale_shear,
            rotation,
            translation,
            adj_mask,
            x_interpolated,
        ) = self.encoder(x, rotate_value)

        # create identity matrix for computing inverse affine matrix
        identity = (
            torch.tensor([0, 0, 1], dtype=torch.float)
            .reshape(1, 1, 3)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )

        # add identity matrix to affine matrix
        inverse_scale_shear, inverse_rotation, inverse_translation = (
            generate_inverse_affine(scale_shear, rotation, translation, identity)
        )

        predicted_base = self.decoder(k_out)

        # Up grid image when interpolate mode is True
        if self.interpolate:
            predicted_base = F.interpolate(
                predicted_base,
                size=(self.up_size, self.up_size),
                mode=self.upsampling_interpolation_mode,
            )

        predicted_input, scale_shear_grid, rotation_grid, translation_grid = (
            apply_affine_transformation_to_image(
                predicted_base,
                inverse_scale_shear,
                inverse_rotation,
                inverse_translation,
                inverse_affine=True,
                device=self.device,
                affine_mode=self.affine_interpolation_mode,
            )
        )

        new_list = self.create_list_of_masks(x, scale_shear_grid, rotation_grid)

        if self.interpolate:
            # apply inverse affine transform to recreate input image
            if self.revise_affine:
                predicted_input = reverse_affine_transform(
                    predicted_input,
                    new_list,
                    inverse_scale_shear,
                    self.device,
                    intensity_adjustment_factor=adj_mask,
                    radius=self.reverse_affine_transform_crop_radius,
                    coef=self.COM_threshold_coef,
                    divide_by_intensity_adjustment=True,
                    affine_mode=self.affine_interpolation_mode,
                )

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
            x_interpolated,
        )

    def create_list_of_masks(self, x, scale_shear_grid, rotation_grid):
        new_list = []
        if self.encoder.mask is not None:
            for mask_ in self.encoder.mask:
                # repeat number of mask to size of mini-batch
                batch_mask = (
                    mask_.reshape(1, 1, mask_.shape[-2], mask_.shape[-1])
                    .repeat(x.shape[0], 1, 1, 1)
                    .to(self.device)
                )

                batch_mask = torch.tensor(batch_mask, dtype=torch.float).to(self.device)

                rotated_mask = F.grid_sample(batch_mask, rotation_grid)

                if self.interpolate:
                    # Add reverse affine transform of scale and shear to make all spots in the mask region, crucial when mask region small
                    rotated_mask = F.grid_sample(rotated_mask, scale_shear_grid)

                # maintain the correct size of mask region after affine transformation
                rotated_mask[rotated_mask < 0.5] = 0
                rotated_mask[rotated_mask >= 0.5] = 1

                rotated_mask = (
                    torch.tensor(rotated_mask, dtype=torch.bool)
                    .reshape(-1, rotated_mask.shape[-2], rotated_mask.shape[-1])
                    .to(self.device)
                )

                new_list.append(rotated_mask)
        return new_list

def build_cc_st_ae(
    input_image_dim,
    pool_list,
    number_channels,
    learning_rate=3e-5,
    decoder_input_dimensions=[5, 5],
    upsample_list=[2, 4, 5],
    reverse_affine_transform_crop_radius=60,
    COM_threshold_coef=1.5,
    upsampling_interpolation_mode="bicubic",
    affine_interpolation_mode="bicubic",
    **kwargs,
):
    """Create an autoencoder and optimizer.

    Args:
        input_image_dim (list of int): Dimensions [x, y] of the input image.
        pool_list (list of int): Parameters for each 2D MaxPool layer.
        number_channels (int): Number of filters in each convolutional block.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 3e-5.
        first_layer_output_size (list of int, optional): Output size of the first layer. Defaults to [5, 5].
        upsample_list (list of int, optional): Parameters for each 2D Upsample layer. Defaults to [2, 4, 5].
        radius (int, optional): Radius for cropping small square images. Defaults to 60.
        coef (float, optional): Threshold for center of mass (COM) operation. Defaults to 1.5.
        interpolate_mode (str, optional): Interpolation mode for F.interpolate(). Defaults to 'bicubic'.
        affine_mode (str, optional): Affine mode for F.affine_grid(). Defaults to 'bicubic'.
        **kwargs: Additional keyword arguments for encoder and decoder configuration, including:
            - device (torch.device): Device on which the model will run.
            - scale (bool): If True, includes scale affine transformation.
            - shear (bool): If True, includes shear affine transformation.
            - rotation (bool): If True, includes rotation affine transformation.
            - rotate_clockwise (bool): If True, rotates the image in a clockwise direction.
            - translation (bool): If True, includes translation affine transformation.
            - symmetric (bool): If True, applies symmetric shear transformation.
            - mask_intensity (bool): If True, allows learnable intensity in the mask region.
            - num_base (int, optional): Number of base elements. Defaults to 2.
            - fixed_mask (list of torch.Tensor, optional): List of binary tensors for masking. Defaults to None.
            - interpolate (bool): If True, calculates loss in interpolated version. Defaults to False.
            - revise_affine (bool): If True, applies revised affine transformations. Defaults to False.
            - up_size (int, optional): Image size for MSE loss calculation. Defaults to 800.
            - scale_limit (float): Range limit for scaling. Defaults to 0.05.
            - shear_limit (float): Range limit for shearing. Defaults to 0.1.
            - rotation_limit (float): Range limit for rotation. Defaults to 0.1.
            - trans_limit (float): Range limit for translation. Defaults to 0.15.
            - adj_mask_para (float): Range for adjusting pixel values in mask region. Defaults to 0.
            - reduced_size (int): Input length for the K-top layer. Defaults to 20.

    Returns:
        tuple: A tuple containing the encoder, decoder, autoencoder model, and optimizer.
    """
    
    device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = Encoder(
        input_image_dim, # input image
        pool_list, # pool list
        number_channels,
        **kwargs,
    ).to(device)

    decoder = Decoder(decoder_input_dimensions, upsample_list, number_channels, **kwargs).to(
        device
    )

    join = CC_ST_AE(
        encoder=encoder, 
        decoder=decoder, 
        device=device, 
        reverse_affine_transform_crop_radius=reverse_affine_transform_crop_radius, 
        COM_threshold_coef=COM_threshold_coef, 
        upsampling_interpolation_mode=upsampling_interpolation_mode, 
        affine_interpolation_mode=affine_interpolation_mode
    ).to(device)

    optimizer = optim.Adam(join.parameters(), lr=learning_rate)

    if device == torch.device("cuda"):
        join = torch.nn.parallel.DataParallel(join)

    return encoder, decoder, join, optimizer
