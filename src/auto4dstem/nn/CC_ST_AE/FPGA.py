import torch.nn as nn


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