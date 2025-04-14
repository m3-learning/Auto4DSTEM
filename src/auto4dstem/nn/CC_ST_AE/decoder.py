from auto4dstem.nn.CC_ST_AE.network_blocks import identity_block
from auto4dstem.nn.CC_ST_AE.network_blocks import conv_block


class Decoder(nn.Module):
    """
        nn.Module class of Decoder (Generator for generating base)

    Returns:
        tensor: torch.tensor
    """

    def __init__(self, first_layer_output_size, upsample_list, number_channels, num_base=2, **kwargs):
        """

        Args:
            original_step_size (list of int): the x and y size of input image
            up_list (list of int): the list of parameter for each 2D Upsample layer
            conv_size (int): the value of filters number goes to each block
            device (torch.device): set the device to run the model
            num_base (int): the value for number of base. Defaults to 2.
        """

        super(Decoder, self).__init__()

        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.first_layer_output_size = first_layer_output_size
        self.dense = nn.Linear(num_base, self.first_layer_output_size[0] * self.first_layer_output_size[1])
        self.cov2d = nn.Conv2d(
            1, number_channels, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            number_channels, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        # set number of blocks depends on length of pool list, each block includes one conv_block and one identity_block
        blocks = []
        number_of_blocks = len(upsample_list)
        blocks.append(
            conv_block(num_channels=number_channels, spatial_dims=first_layer_output_size)
        )
        blocks.append(
            identity_block(num_channels=number_channels, spatial_dims=first_layer_output_size)
        )
        for i in range(number_of_blocks):
            # add UpSample layer before each block
            blocks.append(
                nn.Upsample(
                    scale_factor=upsample_list[i], mode="bilinear", align_corners=True
                )
            )
            # update value of step size for each block
            first_layer_output_size = [
                self.first_layer_output_size[0] * upsample_list[i],
                self.first_layer_output_size[1] * upsample_list[i],
            ]
            blocks.append(
                conv_block(num_channels=number_channels, spatial_dims=first_layer_output_size)
            )
            blocks.append(
                identity_block(num_channels=number_channels, spatial_dims=first_layer_output_size)
            )

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)
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
        out = out.view(-1, 1, self.first_layer_output_size[0], self.first_layer_output_size[1])
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = self.relu_1(out)
        return out