import pytest
import torch
import h5py
import os
from unittest import mock
from torch.utils.data import DataLoader
from auto4dstem.Viz.util import (
    Show_Process,
    upsample_mask
)

# Assuming the Show_Process function is imported from the appropriate module
# from your_module import Show_Process, upsample_mask

class MockModel(torch.nn.Module):
    def __init__(self,up_inp = False):
        super(MockModel,self).__init__()
        self.up_inp = up_inp
    def forward(self, x, y=None):
        # Return mock values similar to what the real model might return
#        batch_size, channels, height, width = x.size()
        mock_tensor = torch.zeros_like(x)
        if self.up_inp:
            return (
                mock_tensor,  # predicted_x
                mock_tensor,  # predicted_base
                mock_tensor,  # predicted_input
                mock_tensor,  # kout
                torch.zeros(1),  # theta_1
                torch.zeros(1),  # theta_2
                torch.zeros(1),  # theta_3
                torch.zeros_like(x),  # adj_mask
                [torch.zeros_like(x)],  # new_list
                torch.zeros_like(x)  # x_inp (only if up_inp is True)
            )
        else:
            return(
                mock_tensor,  # predicted_x
                mock_tensor,  # predicted_base
                mock_tensor,  # predicted_input
                mock_tensor,  # kout
                torch.zeros(1),  # theta_1
                torch.zeros(1),  # theta_2
                torch.zeros(1),  # theta_3
                torch.zeros_like(x),  # adj_mask
                [torch.zeros_like(x)],  # new_list
            )

@pytest.fixture
def mock_dataloader():
    # Create a mock DataLoader with one batch of data
    data = torch.rand(1, 3, 64, 64)  # batch_size=1, channels=3, height=64, width=64
    return DataLoader([data])

@pytest.fixture
def mock_mask_list():
    # Create a mock mask list
    return [torch.ones(1, 64, 64)]

@pytest.fixture
def mock_device():
    return torch.device('cpu')


def test_Show_Process_with_up_inp(tmpdir,mock_dataloader, mock_mask_list, mock_device):
    # Initialize a mock model
    model = MockModel(up_inp=True).to(mock_device)

    # Run the Show_Process function and save to a temporary directory
    file_path = os.path.join(tmpdir, "test_file")
    Show_Process(
        model=model,
        test_iterator=mock_dataloader,
        mask_list=mock_mask_list,
        name_of_file=file_path,
        device=mock_device,
        up_inp=True,
    )

    # Check if the file was saved correctly
    saved_file = f"{file_path}.h5"
    assert os.path.exists(saved_file), "The HDF5 file was not saved correctly."

    # Verify the contents of the saved HDF5 file
    with h5py.File(saved_file, "r") as h5f:
        assert "base" in h5f, "The 'base' dataset was not saved in the HDF5 file."
        assert "mask_list" in h5f, "The 'mask_list' dataset was not saved in the HDF5 file."

def test_Show_Process_file_saving(tmpdir, mock_dataloader, mock_mask_list, mock_device):
    # Initialize a mock model
    model = MockModel().to(mock_device)

    # Run the Show_Process function and save to a temporary directory
    file_path = os.path.join(tmpdir, "test_file")
    Show_Process(
        model=model,
        test_iterator=mock_dataloader,
        mask_list=mock_mask_list,
        name_of_file=file_path,
        device=mock_device,
        up_inp=False,
    )

    # Check if the file was saved correctly
    saved_file = f"{file_path}.h5"
    assert os.path.exists(saved_file), "The HDF5 file was not saved correctly."

    # Verify the contents of the saved HDF5 file
    with h5py.File(saved_file, "r") as h5f:
        assert "base" in h5f, "The 'base' dataset was not saved in the HDF5 file."
        assert "mask_list" in h5f, "The 'mask_list' dataset was not saved in the HDF5 file."

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_Show_Process.py
