from auto4dstem.nn.CC_ST_AE.network_blocks import conv_block
import pytest
import torch
import torch.nn as nn

from auto4dstem.nn.CC_ST_AE.network_blocks import identity_block  

@pytest.fixture
def sample_input():
    return torch.rand(1, 64, 32, 32)  # A random tensor with shape (batch_size, channels, height, width)

def test_conv_block(sample_input):
    model = conv_block(num_channels=64, spatial_dims=[32, 32])
    output = model(sample_input)
    assert output.shape == sample_input.shape, "conv_block output shape should match input shape"

def test_identity_block(sample_input):
    model = identity_block(num_channels=64, spatial_dims=[32, 32])
    output = model(sample_input)
    assert output.shape == sample_input.shape, "identity_block output shape should match input shape"
