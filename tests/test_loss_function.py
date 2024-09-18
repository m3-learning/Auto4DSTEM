import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock
from tqdm import tqdm
from auto4dstem.nn.Loss_Function import AcumulatedLoss 
from torch.utils.data import DataLoader

@pytest.fixture
def mock_acumulated_loss():
    """Fixture to initialize an AcumulatedLoss instance."""
    """
    device: torch.device = torch.device("cpu")
    reg_coef: float = 0
    scale_coef: float = 0
    shear_coef: float = 0
    norm_order: float = 1
    scale_penalty: float = 0.04
    shear_penalty: float = 0.03
    mask_list: list = None
    weighted_mse: bool = True
    reverse_mse: bool = True
    weight_coef: float = 2
    interpolate: bool = True
    batch_para: int = 1
    cycle_consistent: bool = True
    dynamic_mask_region: bool = False
    soft_threshold: float = 1.5
    hard_threshold: float = 3
    con_div: int = 15
    """
    return AcumulatedLoss(
        device=torch.device('cpu'),
        reg_coef=0.01,
        scale_coef=0.1,
        shear_coef=0.1,
        norm_order=1,
        scale_penalty=0.04,
        shear_penalty=0.03,
        mask_list=[torch.ones([10,10],dtype = torch.bool)],  # Example mask
        weighted_mse=True,
        reverse_mse=True,
        weight_coef=2,
        interpolate=False,
        batch_para=1,
        cycle_consistent=True,
        dynamic_mask_region=False,
        soft_threshold=1.5,
        hard_threshold=3,
        con_div=15
    )

@pytest.fixture
def mock_model(interpolate = False):
    """Fixture to mock a model that returns predicted values."""
    """
    predicted_x,
    predicted_base,
    predicted_input,
    kout,
    theta_1,
    theta_2,
    theta_3,
    adj_mask,
    new_list,
    x_inp,
    """
    model = MagicMock()
    if interpolate:
        model.return_value = (
            torch.rand(1, 1, 20, 20),  # predicted_x
            torch.rand(1, 1, 20, 20),  # predicted_base
            torch.rand(1, 1, 20, 20),  # predicted_input
            torch.ones(1,1),                 # kout
            torch.rand(1, 2, 3),  # theta_1 (for scale and shear loss)
            torch.rand(1, 2, 3),  # theta_2
            torch.rand(1, 2, 3),  # theta_3
            torch.ones(1,1),                 # adj_mask
            [torch.ones(1,20,20)],                 # new_list
            torch.rand(1, 1, 20, 20)                  # x_inp
        )
    else:
        model.return_value = (
            torch.rand(1, 1, 10, 10),  # predicted_x
            torch.rand(1, 1, 10, 10),  # predicted_base
            torch.rand(1, 1, 10, 10),  # predicted_input
            torch.ones(1,1),                 # kout
            torch.rand(1, 2, 3),  # theta_1 (for scale and shear loss)
            torch.rand(1, 2, 3),  # theta_2
            torch.rand(1, 2, 3),  # theta_3
            torch.ones(1,1),                # adj_mask
            [torch.ones(1,10,10)],                 # new_list
        )
    return model

@pytest.fixture
def mock_optimizer():
    """Fixture to mock an optimizer."""
    return MagicMock()

@pytest.fixture
def mock_data_iterator():
    """Fixture to mock a data iterator (DataLoader)."""
    dataloader = DataLoader(
                torch.rand(1,1,10,10), 
                batch_size=4, 
                shuffle=True, num_workers=0
            )
    return  dataloader # Example batch with data and target

# def test_loss_computation(mock_acumulated_loss, mock_model, mock_data_iterator, mock_optimizer):
#     """Test the main __call__ function for loss computation."""
#     loss_dict = mock_acumulated_loss(mock_model, mock_data_iterator, mock_optimizer)
    
#     # Check if the loss dictionary is returned correctly
#     assert "train_loss" in loss_dict
#     assert "l2_loss" in loss_dict
#     assert "scale_loss" in loss_dict
#     assert "shear_loss" in loss_dict

#     # Ensure that the values are floats (from torch tensors)
#     assert isinstance(loss_dict["train_loss"], float)
#     assert isinstance(loss_dict["l2_loss"], float)
#     assert isinstance(loss_dict["scale_loss"], float)
#     assert isinstance(loss_dict["shear_loss"], float)