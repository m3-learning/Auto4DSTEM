import numpy as np
import torch


import os
import random


def set_seed(**kwargs):
    """Sets the seed for reproducibility across various libraries.

    This method sets the seed for Python's built-in random module, NumPy, and PyTorch to ensure that the results
    are reproducible. It also sets the environment variable 'PYTHONHASHSEED' to ensure consistent hashing.

    Args:
        seed (int, optional): The seed value to set. If not provided, it uses the default seed value from the instance attribute.
    """
    seed = kwargs.get("seed", None)
    if seed is None:
        seed = self.seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)