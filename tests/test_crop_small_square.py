import pytest
import torch
from auto4dstem.nn.CC_ST_AE import crop_small_square

# Assuming the crop_small_square function is imported from the appropriate module
# from your_module import crop_small_square

@pytest.mark.parametrize(
    "center_coordinates, radius, max_, expected_output",
    [
        # Case: Center in the middle
        (torch.tensor([100, 100]), 50, 200, ((50, 150), (50, 150))),
        
        # Case: Center near top-left corner
        (torch.tensor([30, 30]), 50, 200, ((0, 100), (0, 100))),
        
        # Case: Center near bottom-right corner
        (torch.tensor([180, 180]), 50, 200, ((100, 200), (100, 200))),
        
        # Case: Center near the top-right corner
        (torch.tensor([180, 20]), 50, 200, ((100, 200), (0, 100))),
        
        # Case: Center near the bottom-left corner
        (torch.tensor([20, 180]), 50, 200, ((0, 100), (100, 200))),
        
        # Case: Center exactly on the edge
        (torch.tensor([200, 200]), 50, 200, ((100, 200), (100, 200))),
    ],
)
def test_crop_small_square(center_coordinates, radius, max_, expected_output):
    result = crop_small_square(center_coordinates, radius, max_)
    assert result == expected_output, f"Expected {expected_output}, but got {result}"

# To run the test, you would typically run `pytest` in the command line
# pytest -q test_crop_small_square.py
