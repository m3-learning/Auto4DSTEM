from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from skimage import filters
from dataclasses import dataclass, field
import h5py


import argparse
import logging
import sys

from auto4dstem import __version__

__author__ = "Shuyu Qin, Joshua Agar"
__copyright__ = "Joshua Agar"
__license__ = "BSD-3-Clause"

_logger = logging.getLogger(__name__)


@dataclass
class STEM4D_DataSet:
    """
    Represents a dataset for STEM 4D data processing.

    Attributes:
        data_dir (str): Directory of the dataset.
        background_weight (float): Weight for the background, defaulting to 0.10.
        crop (tuple): Tuple for cropping, defaulting to ((28, 228), (28, 228)).
        transpose (tuple): Tuple for transposing, defaulting to (1, 0, 3, 2).
        background_intensity (bool): Determine if needed adding background noise or not.
        counts_per_probe (Optional[float]): Counts per probe, can be None or float, defaulting to 1e5.
        intensity_coefficient (float): The intensity coefficient for scaling the noise, defaulting to 1e5/4.
        rotation (Optional[float]): Rotation angle, can be None or float.
        standard_scale: Optional[float] = None
        up_threshold (float): determine the value of up threshold of dataset. Defaults to 1000.
        down_threshold (float): determine the value of down threshold of dataset. Default to 0.
        x_size (int): Computed x size from crop values, not provided during initialization.
        y_size (int): Computed y size from crop values, not provided during initialization.
    """

    data_dir: str
    background_weight: float = 0.10
    crop: tuple = ((28, 228), (28, 228))
    transpose: tuple = (0, 1, 2, 3)
    background_intensity: bool = False
    counts_per_probe: float = 1e5
    intensity_coefficient: float = 1e5 / 4
    rotation: Optional[float] = None
    standard_scale: Optional[float] = None
    up_threshold: float = 1000
    down_threshold: float = 0
    boundary_filter: bool = False
    x_size: int = field(init=False)
    y_size: int = field(init=False)

    def __post_init__(self):
        """Post-initialization method to compute derived attributes and perform additional setup.

        This method calculates the x_size and y_size based on the crop values, calls the load_data method to load the dataset,
        and optionally calls methods to generate background noise and rotate the data based on the given parameters.

        No additional arguments are required as it uses the attributes initialized in the constructor.
        """
        # Crops the size of the images based on the crop values
        self.x_size = self.crop[0][1] - self.crop[0][0]
        self.y_size = self.crop[1][1] - self.crop[1][0]

        # Load the data from the specified directory
        self.load_data()

        # option to apply sobel filter to the dataset
        # Used to determine the center diffraction spot postion
        if self.boundary_filter:
            self.filter_sobel(self.stem4d_data)

        # used for simulated dataset to add background noise
        if self.background_intensity:
            self.generate_background_noise(
                self.stem4d_data,
                self.background_weight,
                self.counts_per_probe,
                intensity_coefficient=self.intensity_coefficient,
            )

        # Reshape the data to the correct format
        self.stem4d_data = self.stem4d_data.reshape(-1, 1, self.x_size, self.y_size)

        # Rotate the data based on the specified rotation angles if provided
        if self.rotation is not None:
            self.rotate_data(self.stem4d_data, self.rotation)

    def load_data(self):
        """
        Loads the 4D STEM data from the specified directory.

        This method supports loading data from '.h5', '.mat', and '.npy' files.
        It also calls the format_data method to apply any necessary formatting to the loaded data.

        Returns:
            str: An error message if no legal format is found for the data path or if an exception occurs.
        """
        try:
            # Check if the data directory ends with '.h5' or '.mat' extension
            if self.data_dir.endswith(".h5") or self.data_dir.endswith(".mat"):
                print(self.data_dir)  # Printing the data directory for logging purposes
                with h5py.File(self.data_dir, "r") as f:  # Open the file in read mode
                    stem4d_data = f["output4D"]  # Extract the data
                    self.format_data(
                        stem4d_data
                    )  # Call format_data to format the loaded data

            # Check if the data directory ends with '.npy' extension
            elif self.data_dir.endswith(".npy"):
                stem4d_data = np.load(self.data_dir)  # Load the data using NumPy
                self.format_data(
                    stem4d_data
                )  # Call format_data to format the loaded data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while loading the data: {e}")
            return f"An error occurred: {e}"

    def format_data(self, stem4d_data):
        """
        Formats the loaded 4D STEM data according to the specified crop and transpose parameters.

        Args:
            stem4d_data (numpy.ndarray): The 4D STEM data to be formatted.

        Returns:
            str: An error message if an exception occurs during formatting.
        """
        try:
            # Transpose the data according to the specified transpose values
            stem4d_data = np.transpose(stem4d_data, self.transpose)

            # Apply the cropping according to the specified crop values
            if len(stem4d_data.shape) == 3:
                stem4d_data = stem4d_data[
                    :,
                    self.crop[0][0] : self.crop[0][1],
                    self.crop[1][0] : self.crop[1][1],
                ]
            else:
                stem4d_data = stem4d_data[
                    :,
                    :,
                    self.crop[0][0] : self.crop[0][1],
                    self.crop[1][0] : self.crop[1][1],
                ]

            # Reshape the data using the computed x_size and y_size
            stem4d_data = stem4d_data.reshape(-1, self.x_size, self.y_size)

            # Standard scale the data with pre-set up and bottom bound
            if self.standard_scale is not None:

                stem4d_data[stem4d_data > self.up_threshold] = self.up_threshold
                stem4d_data[stem4d_data < self.down_threshold] = self.down_threshold
                stem4d_data = (
                    self.standard_scale
                    * (stem4d_data - self.down_threshold)
                    / (self.up_threshold - self.down_threshold)
                )

            # Assign the formatted data to the class attribute
            self.stem4d_data = stem4d_data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while formatting the data: {e}")
            return f"An error occurred: {e}"

    def generate_background_noise(
        self,
        stem4d_data,
        background_weight,
        counts_per_probe,
        intensity_coefficient=1e5 / 4,
    ):
        """
        Generates background noise for the 4D STEM data based on the specified parameters.

        Args:
            stem4d_data (numpy.ndarray): The 4D STEM data to add noise to.
            background_weight (float): The weight for the background noise.
            counts_per_probe (float): The number of counts per probe for scaling the noise.
            intensity_coefficient (float): The intensity coefficient for scaling the noise, defaulting to 1e5/4.

        Returns:
            str: An error message if an exception occurs during noise generation.
        """
        try:
            # If the background_weight is zero, simply scale the data
            if background_weight == 0:
                self.stem4d_data = stem4d_data * intensity_coefficient
                self.stem4d_data = self.stem4d_data.reshape(
                    -1, 1, self.x_size, self.y_size
                )

            else:
                noisy_data = np.zeros(stem4d_data.shape)
                im = np.zeros(stem4d_data.shape[1:])

                # Loop through each frame and apply the noise generation algorithm
                print("add Poison distributed background noise to whole dataset")
                for i in tqdm(
                    range(stem4d_data.shape[0]), leave=True, total=stem4d_data.shape[0]
                ):
                    # creates a local deep copy of the data
                    test_img = np.copy(stem4d_data[i])

                    #### Generates the Poisson distributed background noise ####
                    qx = np.fft.fftfreq(im.shape[0], d=1)
                    qy = np.fft.fftfreq(im.shape[1], d=1)
                    qya, qxa = np.meshgrid(qy, qx)
                    qxa = np.fft.fftshift(qxa)
                    qya = np.fft.fftshift(qya)
                    qra2 = qxa**2 + qya**2
                    im_bg = 1.0 / (1 + qra2 / 1e-2**2)
                    im_bg = im_bg / np.sum(im_bg)
                    int_comb = (
                        test_img * (1 - background_weight) + im_bg * background_weight
                    )
                    int_noisy = (
                        np.random.poisson(int_comb * counts_per_probe)
                        / counts_per_probe
                    )
                    int_noisy = int_noisy * intensity_coefficient
                    noisy_data[i] = int_noisy

                self.stem4d_data = noisy_data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while generating background noise: {e}")
            return f"An error occurred: {e}"

    def rotate_data(self, stem4d_data, rotation):
        """
        Rotates the 4D STEM data according to the specified rotation angles.

        Args:
            stem4d_data (numpy.ndarray): The 4D STEM data to be rotated.
            rotation (numpy.ndarray): The rotation angles to be applied.

        Raises:
            ValueError: If the rotation size and image size do not match each other.
        """
        try:
            # Compute the angles based on the rotation parameter
            self.angle = np.mod(
                np.arctan2(rotation[:, 1], rotation[:, 0]), np.pi / 3
            ).reshape(-1)

            # Check if the size of the angle array matches the size of the stem4d_data
            if self.angle.shape[0] != stem4d_data.shape[0]:
                raise ValueError(
                    "The rotation size and image size do not match each other"
                )
            else:
                # Combine the data and rotation angle for each frame
                whole_data_with_rotation = []
                print("add image-rotation pair to whole dataset")
                for i in tqdm(
                    range(stem4d_data.shape[0]), leave=True, total=stem4d_data.shape[0]
                ):
                    whole_data_with_rotation.append([stem4d_data[i], self.angle[i]])

                # Assign the rotated data to the class attribute
                self.stem4d_rotation = whole_data_with_rotation

        except Exception as e:
            # Log the exception and re-raise to allow for additional handling if needed
            print(f"An error occurred while rotating the data: {e}")
            raise e

    def filter_sobel(
        self,
        stem4d_data,
        upscale_factor=2,
    ):
        """

        Args:
            stem4d_data (torch.Tensor): _description_
            upscale_factor (float): factor which to upscale the edge. Defaults to 2.

        Returns:
            _type_: replace input image by filtered edges.
        """
        try:
            print("add sobel filter to whole dataset")
            for i in tqdm(range(stem4d_data.shape[0])):

                # standard scale each image individually divided by largest value
                max_ = np.max(stem4d_data[i])
                stem4d_data[i] = stem4d_data[i] / max_

                # use sobel filter for boundary detecting
                img_ = filters.sobel(stem4d_data[i])

                # upscale the intensity of boundary by 2 for potential easily training
                stem4d_data[i] = upscale_factor * img_

        except Exception as e:
            # Log the exception and re-raise to allow for additional handling if needed
            print(f"An error occurred while doing sobel detection: {e}")
            raise e

    @property
    def stem4d_data(self):
        """function to call the preprocessed input data

        Returns:
            tensor: preprocessed input data
        """
        return self._stem4d_data

    @stem4d_data.setter
    def stem4d_data(self, stem4d_data):
        """function to call the preprocessed input data

        Args:
            stem4d_data (tensor): preprocessed input data
        """
        self._stem4d_data = stem4d_data


def data_translated(
    data_path,
    translation,
    crop=((2, 122), (2, 122)),
    transpose=(0, 1, 2, 3),
    save_path="",
):
    # TODO: DOCSTRING

    # import dataset from directory
    if data_path.endswith(".h5") or data_path.endswith(".mat"):
        print(data_path)  # Printing the data directory for logging purposes
        with h5py.File(data_path, "r") as f:  # Open the file in read mode
            stem4d_data = f["output4D"][:]  # Extract the data

    # check if the data directory ends with '.npy' extension
    elif data_path.endswith(".npy"):
        stem4d_data = np.load(data_path)  # Load the data using NumPy

    # raise error when no correct format
    else:
        print("no correct format of dataset detected")

    # transpose dataset
    stem4d_data = np.transpose(stem4d_data, transpose)

    # crop dataset
    if len(stem4d_data.shape) == 3:
        stem4d_data = stem4d_data[:, crop[0][0] : crop[0][1], crop[1][0] : crop[1][1]]
    # crop dataset
    else:
        stem4d_data = stem4d_data[
            :, :, crop[0][0] : crop[0][1], crop[1][0] : crop[1][1]
        ]
    # calculate x and y size
    x_size = int(crop[0][1] - crop[0][0])
    y_size = int(crop[1][1] - crop[1][0])

    # reshape dataset
    stem4d_data = stem4d_data.reshape(-1, x_size, y_size)

    # generate translated version of dataset
    for i in tqdm(range(stem4d_data.shape[0])):
        # turn each image into torch.tensor version
        test_img = torch.tensor(stem4d_data[i], dtype=torch.float32).reshape(
            1, 1, x_size, y_size
        )
        # interpolate image to decrease artifact broken
        test_up = F.interpolate(test_img, size=(4 * x_size, 4 * y_size), mode="bicubic")
        # create affine matrix
        trans_ = torch.tensor(
            [[1, 0, translation[i, 0]], [0, 1, translation[i, 1]]], dtype=torch.float
        ).unsqueeze(0)
        # apply affine transformation to image
        grid_1 = F.affine_grid(trans_, test_up.size())
        after_trans = F.grid_sample(test_up, grid_1, mode="bicubic")
        # down sample image into original size
        test_down = F.interpolate(after_trans, size=(x_size, y_size), mode="bicubic")
        # replace with translated image
        stem4d_data[i] = np.array(test_down.squeeze(), dtype=np.float32)

    # save translated image
    np.save(f"{save_path}_translated_version.npy", stem4d_data)
