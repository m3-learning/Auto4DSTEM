import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from .transforms import reverse_affine_transform


class conv_block_fpga(nn.Module):
    """_summary_

    Args:
        nn.Module class of Residual Neural Network for distilled model to fpga
    """

    def __init__(self, t_size):
        """_summary_

        Args:
            t_size (int): Size of the convolution kernel
        """
        super(conv_block_fpga, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_2 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_3 = nn.BatchNorm2d(t_size)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        """Forward pass of the convolutional block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        x_input = x
        out = self.cov1d_1(x)
        out = self.relu_1(out)
        out = self.cov1d_2(out)
        out = self.norm_3(out)
        out = self.relu_2(out)
        out = out.add(x_input)

        return out


class identity_block_fpga(nn.Module):
    """
    nn.Module class of Identity Neural Network
    """

    def __init__(self, t_size):
        """Initializes the identity block

        Args:
            t_size (int): Size of the convolution kernel
        """
        super(identity_block_fpga, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_1 = nn.BatchNorm2d(t_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


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
            out_2nd_affine = reverse_affine_transform(
                out_affine,
                self.mask,
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