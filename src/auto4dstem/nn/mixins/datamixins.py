import os
from dataclasses import dataclass, field
import torch

@dataclass
class IOMixin:
    """class of the IOMixin process, including load and preprocess the dataset and initialize loss class.

    Attributes:
        data_file (string): directory of the dataset
    """

    data_file: str = field(default="data")


@dataclass
class DeviceMixin:
    """class of the DeviceMixin process, including set the device and seed.

    Attributes:
        device: torch.device = torch.device("cpu") Set the device to run the model. Defaults to torch.device('cpu')
        seed: int = 42 Set the seed to make the training reproducible. Defaults to 42.
    """

    device: torch.device = torch.device("cpu")
    seed: int = 42


@dataclass
class DataPropertyMixin:
    """class of the DataPropertyMixin process, including set the data property.

    Attributes:
        simulated_data (bool): determine if the input dataset is simulated data or not. Defaults to True.
    """

    simulated_data: bool = True


@dataclass
class NoisyMixin:
    """class of the NoisyMixin process, including set the noise parameters.

    Attributes:
        background_weight (float, optional): set the intensity of background noise for simulated dataset. Defaults to 0.2.
        counts_per_probe (float, optional): Counts per probe, can be None or float, defaulting to 1e5.
    """

    background_weight: float = 0.2
    counts_per_probe: float = 1e5

    # @property
    # def background_weight(self) -> float:  # noqa: F811
    #     return self._background_weight

    # @background_weight.setter
    # def background_weight(self, value: float) -> None:
    #     if value > 1:
    #         raise ValueError("background_weight cannot be greater than 1.")
    #     self._background_weight = value


@dataclass
class DataMixin(IOMixin, DeviceMixin, DataPropertyMixin, NoisyMixin):
    """Class for managing the data properties during training."""