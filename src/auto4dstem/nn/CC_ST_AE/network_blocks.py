import torch
import torch.nn as nn

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
                symmetric (bool): Indicates if the shear affine transformation is symmetric.
                mask_intensity_flag (bool): Indicates if the intensity of the mask region is learnable.
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
        self.symmetric = kwargs.get("symmetric", True)
        self.scale_threshold = kwargs.get("scale_threshold", 0.05)
        self.shear_threshold = kwargs.get("shear_threshold", 0.1)
        self.rotation_threshold = kwargs.get("rotation_threshold", 0.1)
        self.translation_threshold = kwargs.get("translation_threshold", 0.15)
        self.learnable_mask_intensity = kwargs.get("learnable_mask_intensity", 0)
        self.learnable_mask = kwargs.get("learnable_mask", True)
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
            scale_x = self.scale_threshold * nn.Tanh()(embedding_layer[:, self.count]) + 1
            scale_y = (
                self.scale_threshold * nn.Tanh()(embedding_layer[:, self.count + 1]) + 1
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
            shear_x = self.shear_threshold * nn.Tanh()(embedding_layer[:, self.count])
            if self.symmetric:
                shear_y = shear_x
                # TODO: late add check that works
                self.count += 1
            else:
                shear_y = self.shear_threshold * nn.Tanh()(
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
            elif self.rotation_threshold is not None:
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
        return self.rotation_threshold * nn.Tanh()(embedding_layer[:, self.count])

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
        ) + self.rotation_threshold * nn.Tanh()(embedding_layer[:, self.count])

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
            translation_x = self.translation_threshold * nn.Tanh()(embedding_layer[:, self.count])
            translation_y = self.translation_threshold * nn.Tanh()(
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
        if self.learnable_mask:
            mask_parameter = (
                self.learnable_mask_intensity
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