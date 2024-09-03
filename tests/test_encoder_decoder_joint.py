import pytest
import torch
import torch.nn as nn
from auto4dstem.nn.CC_ST_AE import (
    Encoder,
    Decoder,
    Joint,
    make_model_fn
)

@pytest.fixture
def sample_input():
    return torch.rand(1, 1, 32, 32)  # A random tensor with shape (batch_size, channels, height, width)

def test_encoder(sample_input):
    device = torch.device('cpu')
    encoder = Encoder(
        original_step_size=[32, 32],
        pool_list=[2, 2],
        conv_size=64,
        device=device
    )
    output, *_ = encoder(sample_input)
    assert output.shape[2:] == sample_input.shape[2:], "Encoder output spatial dimensions should match input"

def test_decoder():
    device = torch.device('cpu')
    decoder = Decoder(
        original_step_size=[5, 5],
        up_list=[2, 4],
        conv_size=64,
        device=device
    )
    sample_input = torch.rand(1, 2)  # Assuming num_base=2
    output = decoder(sample_input)
    assert output.shape == (1, 1, 40, 40), "Decoder output shape should be (batch_size, 1, height, width)"

def test_joint():
    device = torch.device('cpu')
    encoder = Encoder(
        original_step_size=[40, 40],
        pool_list=[4, 2],
        conv_size=64,
        device=device
    )
    decoder = Decoder(
        original_step_size=[5, 5],
        up_list=[2, 4],
        conv_size=64,
        device=device
    )
    joint_model = Joint(
        encoder=encoder,
        decoder=decoder,
        device=device
    )
    
    sample_input = torch.rand(1, 1, 40, 40)
    joint_model.mask = [torch.ones(40,40,dtype=torch.bool)]
    output = joint_model(sample_input)
    assert isinstance(output, tuple), "Joint model should return a tuple"
    assert output[0].shape == sample_input.shape, "Joint model output shape should match input shape"

def test_make_model_fn():
    device = torch.device('cpu')
    encoder, decoder, join, optimizer = make_model_fn(device=device)
    assert isinstance(encoder, Encoder), "encoder should be an instance of Encoder"
    assert isinstance(decoder, Decoder), "decoder should be an instance of Decoder"
    assert isinstance(join, nn.Module), "join should be an instance of nn.Module"
    assert isinstance(optimizer, torch.optim.Optimizer), "optimizer should be an instance of torch.optim.Optimizer"
