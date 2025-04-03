from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CenterBeamAlignMixin:
    """class of the CenterBeamAlignMixin process, including set the center beam align parameters.

    Attributes:
        align_center_beam_sobel (bool): determine if the dataset needs to be center beam aligned with a sobel filter. Defaults to False.
    """

    align_center_beam_sobel: bool = False


@dataclass
class ImageThresholdMixin:
    """class of the ImageThresholdMixin process, including set the image threshold parameters.

    Attributes:
        max_threshold (float): determine the value of max threshold of dataset. Defaults to 1000.
        min_threshold (float): determine the value of min threshold of dataset. Default to 0.
    """

    max_threshold: float = 1000
    min_threshold: float = 0


@dataclass
class ImageTransformMixin:
    """class of the ImageTransformMixin process, including set the image transformation parameters.

    Attributes:
        crop (tuple): A tuple of tuples specifying the crop dimensions. Defaults to ((28, 228), (28, 228)).
        transpose (tuple): A tuple specifying the order of axes for transposing the image. Defaults to (2, 3, 0, 1).
        intensity_scaler (float): A coefficient to scale the intensity of the image. Defaults to 1e5 / 4.
        standard_scaler (float, optional): Precomputed standard scaler for the dataset. If provided, the dataset will be scaled using this scaler. Defaults to None.
        upsampling_interpolation_mode (str): The interpolation mode to use for the image transformation in the affine transform. Defaults to "bicubic".
        affine_interpolation_mode (str): The interpolation mode to use for the image transformation in the affine transform. Defaults to "bicubic".
    """

    crop: tuple = field(default_factory=lambda: ((28, 228), (28, 228)))
    transpose: tuple = field(default_factory=lambda: (2, 3, 0, 1))
    intensity_scaler: float = 1e5 / 4
    standard_scaler: Optional[float] = None
    upsampling_interpolation_mode: str = "bicubic"
    affine_interpolation_mode: str = "bicubic"


@dataclass
class ImageMixin(CenterBeamAlignMixin, ImageThresholdMixin, ImageTransformMixin):
    """class of the ImageMixin process, including set the image parameters."""