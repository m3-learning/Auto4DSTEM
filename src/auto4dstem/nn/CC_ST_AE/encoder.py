import torch
import torch.nn as nn
import torch.nn.functional as F

from auto4dstem.nn.CC_ST_AE.ktop import ktop_layer
from auto4dstem.nn.CC_ST_AE.network_blocks import (
    AffineTransformationBlock,
    conv_block,
    identity_block,
)
from auto4dstem.nn.CC_ST_AE.transforms import apply_affine_transformation_to_image, reverse_affine_transform


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
            self.dense = nn.Linear(
                self.dense_layer_size + self.num_base, self.embedding_size + 1
            )
        else:
            # Set the all the adj parameter to be the same
            self.dense = nn.Linear(
                self.dense_layer_size + self.num_base, self.embedding_size
            )

    def build_mask(self):
        if self.fixed_mask_flag is not None:
            # Set the mask_ to upscale mask if the interpolate mode is True
            if self.interpolate_flag:
                mask_with_inp = []

                for mask_ in self.fixed_mask_flag:
                    # switch mask type into tensor
                    temp_mask = torch.tensor(
                        mask_.reshape(1, 1, self.input_image_dim[0], self.input_image_dim[1]),
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
        self.dense_before_embedding = nn.Linear(
            flattened_image_size, self.dense_layer_size
        )

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
            conv_block(
                num_channels=self.num_channels, spatial_dims=self.input_image_dim
            )
        )
        self.model_layers.append(
            identity_block(
                num_channels=self.num_channels, spatial_dims=self.input_image_dim
            )
        )
        self.model_layers.append(
            nn.MaxPool2d(self.pool_list[0], stride=self.pool_list[0])
        )

    @property
    def num_layers(self):
        return len(self.model_layers)

    def build_conv_block(self):
        self.reduced_image_size = self.input_image_dim
        number_of_blocks = len(self.pool_list)

        for i in range(1, number_of_blocks):
            # update value of step size for each block
            self.reduced_image_size = self.calculate_image_size(
                self.reduced_image_size, self.pool_list[i - 1]
            )

            self.model_layers.append(
                conv_block(
                    num_channels=self.num_channels, spatial_dims=self.reduced_image_size
                )
            )
            self.model_layers.append(
                identity_block(
                    num_channels=self.num_channels, spatial_dims=self.reduced_image_size
                )
            )
            # add MaxPool layer after each block
            self.model_layers.append(
                nn.MaxPool2d(self.pool_list[i], stride=self.pool_list[i])
            )

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
        self.mask_intensity_flag = kwargs.get("mask_intensity_flag", True)
        self.num_base = kwargs.get("num_base", 2)
        self.fixed_mask_flag = kwargs.get("fixed_mask_flag", None)
        self.interpolate_flag = kwargs.get("interpolate_flag", False)
        self.reverse_affine_transform_flag = kwargs.get(
            "reverse_affine_transform_flag", False
        )
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
        self.affine_mode = kwargs.get("affine_mode", "bilinear")
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
        k_top_output = ktop_layer(x, self.num_k_sparse, self.device)

        return k_top_output

    def forward(self, x, rotate_value=None):
        """forward function for nn.Module class

        Args:
            x (torch.tensor): input torch.tensor image
            rotate_value (float, optional): float value represents pretrained rotation angle. Defaults to None.
        """
        x = x.view(-1, 1, self.input_image_dim[0], self.input_image_dim[1])

        # reshape the input into (mini-batch, 1 , image_size)
        out, k_out = self.network_forward_pass(x)

        # generate affine matrix by tensor out
        output, scale_shear, rotation, translation, intensity_adjustment_factor, x = (
            self.apply_affine_transformations(x, rotate_value, out)
        )
            
        result = (
            output,
            k_out,
            scale_shear,
            rotation,
            translation,
            intensity_adjustment_factor,
            x,
        )

        return result

    def network_forward_pass(self, x):
        out = self.cov2d(x)
        for i in range(self.num_layers):
            out = self.nn_module_list[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        kout = self.dense_before_embedding(out)
        k_out = self.ktop(kout)
        # concatenate reduced dimensional vector and output vector of k-sparse function
        out = torch.cat((kout, k_out), dim=1).to(self.device)
        out = self.dense(out)
        return out, k_out

    def apply_affine_transformations(self, x, rotate_value, out):
        scale_shear, rotation, translation, intensity_adjustment_factor = (
            self.affine_matrix(out, rotate_value)
        )

        if self.interpolate_flag:
            # image interpolation before affine transformation
            x = F.interpolate(
                x, size=(self.up_size, self.up_size), mode=self.interpolate_mode
            )

        cumulative_transformed_image, _, _, _ = apply_affine_transformation_to_image(x, scale_shear, rotation, translation, device=self.device, affine_mode=self.affine_mode)

        if self.interpolate_flag:
            # apply inverse affine to each diffraction spot if revise_affine is True
            if self.reverse_affine_transform_flag:
                # Test 1.5 is good for 5%-45% background noise, add to 2 for larger noise and rot512x512 4dstem
                cumulative_transformed_image = reverse_affine_transform(
                    cumulative_transformed_image,
                    self.mask,
                    scale_shear,
                    device=self.device,
                    intensity_adjustment_factor=intensity_adjustment_factor,
                    radius=self.radius,
                    coef=self.coef,
                    affine_mode=self.affine_mode,
                )

        return (
            cumulative_transformed_image,
            scale_shear,
            rotation,
            translation,
            intensity_adjustment_factor,
            x,
        )

