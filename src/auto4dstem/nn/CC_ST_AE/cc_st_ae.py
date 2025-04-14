
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from auto4dstem.nn.CC_ST_AE.decoder import Decoder
from auto4dstem.nn.CC_ST_AE.encoder import Encoder
from auto4dstem.nn.CC_ST_AE.transforms import (
    apply_affine_transformation_to_image,
    reverse_affine_transform_gpu,
)


#TODO: make this inherit structure to base class
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
        **kwargs,
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
            x_inp,
        ) = self.encoder(x, rotate_value)

        # create identity matrix for computing inverse affine matrix
        identity = (
            torch.tensor([0, 0, 1], dtype=torch.float)
            .reshape(1, 1, 3)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )
                
        
        # add identity matrix to affine matrix
        inverse_scale_shear, inverse_rotation, inverse_translation = self.generate_inverse_affine(scale_shear, rotation, translation, identity)

        predicted_base = self.decoder(k_out)

        # Up grid image when interpolate mode is True
        if self.interpolate:
            predicted_base_inp = F.interpolate(
                predicted_base,
                size=(self.up_size, self.up_size),
                mode=self.interpolate_mode,
            )
            
                        
        predicted_input = apply_affine_transformation_to_image(predicted_base_inp, inverse_scale_shear, inverse_rotation, inverse_translation, inverse_affine=True, device=self.device, affine_mode=self.affine_mode)            
            

        else:
            # add inverse affine transform to generated base
            grid_1 = F.affine_grid(inverse_scale_shear.to(self.device), x.size()).to(
                self.device
            )
            grid_2 = F.affine_grid(inverse_rotation.to(self.device), x.size()).to(
                self.device
            )
            grid_3 = F.affine_grid(inverse_translation.to(self.device), x.size()).to(
                self.device
            )

            predicted_translation = F.grid_sample(predicted_base, grid_3)

            predicted_rotate = F.grid_sample(predicted_translation, grid_2)

            predicted_input = F.grid_sample(predicted_rotate, grid_1)

        # create new mask list to save updated mask region with inverse affine transform
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
                    inverse_scale_shear,
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

    def generate_inverse_affine(self, scale_shear, rotation, translation, identity):
        scale_shear_affine = torch.cat((scale_shear, identity), axis=1).to(self.device)
        rotation_affine = torch.cat((rotation, identity), axis=1).to(self.device)
        translation_affine = torch.cat((translation, identity), axis=1).to(self.device)

        # generate inverse affine matrix
        inverse_scale_shear = torch.linalg.inv(scale_shear_affine)[:, 0:2].to(self.device)
        inverse_rotation = torch.linalg.inv(rotation_affine)[:, 0:2].to(self.device)
        inverse_translation = torch.linalg.inv(translation_affine)[:, 0:2].to(self.device)
        return inverse_scale_shear,inverse_rotation,inverse_translation


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
