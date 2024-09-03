import pytest
import torch
import torch.nn as nn

from auto4dstem.nn.CC_ST_AE import Affine_Transform


def test_affine_transform():
    device = torch.device('cpu')
    model = Affine_Transform(device=device, scale=True, shear=True, rotation=True, translation=True)
    
    sample_input = torch.rand(1, 7)
    scale_shear, rotation, translation, mask_param = model(sample_input)
    
    assert scale_shear.shape == (1, 2, 3), "scale_shear matrix should be of shape (batch_size, 2, 3)"
    assert rotation.shape == (1, 2, 3), "rotation matrix should be of shape (batch_size, 2, 3)"
    assert translation.shape == (1, 2, 3), "translation matrix should be of shape (batch_size, 2, 3)"
    assert mask_param.shape == (1, 1), "mask_param should be of shape (batch_size, 1)"
