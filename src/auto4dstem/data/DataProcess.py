from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from skimage import filters
from dataclasses import dataclass, field
from ..calculations.noise import PoissonNoise
import h5py


@dataclass
class STEM4D_DataSet:
    """
    Represents a dataset for STEM 4D data processing.

    This class is designed to handle and process STEM 4D datasets, providing functionalities such as cropping, transposing,
    adding background noise, and applying rotation. It also computes derived attributes like x_size and y_size based on the crop values.

    Attributes:
        data_file (str): Directory of the dataset.
        background_weight (float): Weight for the background, defaulting to 0.10.
        crop (tuple): Tuple for cropping, defaulting to ((28, 228), (28, 228)).
        transpose (tuple): Tuple for transposing, defaulting to (0, 1, 2, 3).
        simulated_data (bool): Determines if background noise should be added.
        counts_per_probe (Optional[float]): Counts per probe, can be None or float, defaulting to 1e5.
        intensity_scaler (float): The intensity coefficient for scaling the noise, defaulting to 1e5/4.
        rotation (Optional[float]): Rotation angle, can be None or float.
        standard_scaler (Optional[float]): Standard scale factor, default is None.
        max_threshold (float): Upper threshold value for the dataset. Defaults to 1000.
        min_threshold (float): Lower threshold value for the dataset. Defaults to 0.
        align_center_beam_sobel (bool): determine if the dataset needs to be center beam aligned with a sobel filter. Defaults to False.
        x_size (int): Computed x size from crop values, not provided during initialization.
        y_size (int): Computed y size from crop values, not provided during initialization.

    Methods:
        __post_init__(): Computes derived attributes and performs additional setup after initialization.
        load_data(): Loads the dataset from the specified directory.
        filter_sobel(data): Applies a Sobel filter to the dataset for edge detection.
        generate_background_noise(data, weight, counts, intensity_scaler): Adds background noise to the dataset.
        reshape_data(): Reshapes the dataset to the correct format.
        rotate_data(): Rotates the dataset based on the specified rotation angles.

    Examples:
        >>> dataset = STEM4D_DataSet(data_dir="/path/to/data")
        >>> print(dataset.x_size, dataset.y_size)
        200 200

        >>> dataset_with_rotation = STEM4D_DataSet(data_dir="/path/to/data", rotation=45.0)
        >>> print(dataset_with_rotation.rotation)
        45.0

        >>> dataset_with_background = STEM4D_DataSet(data_dir="/path/to/data", background_intensity=True)
        >>> print(dataset_with_background.background_intensity)
        True
    """

    data_file: str = field(default="data")
    background_weight: float = 0.10
    crop: tuple = ((28, 228), (28, 228))
    transpose: tuple = (0, 1, 2, 3)
    simulated_data: bool = False
    counts_per_probe: float = 1e5
    intensity_scaler: float = 1e5 / 4
    learned_rotation: Optional[float] = None
    standard_scaler: Optional[float] = None
    max_threshold: float = 1000
    min_threshold: float = 0
    align_center_beam_sobel: bool = False
    x_size: int = field(init=False)
    y_size: int = field(init=False)
    verbose: bool = False
    kwargs: dict = field(default_factory=dict)

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
        # Used to determine the center diffraction spot position
        self._data_transformations()

    def _data_transformations(self):
        """
        Applies various data transformations to the 4D STEM dataset.

        This method performs the following transformations:
        1. Applies a Sobel filter to align the center beam if `align_center_beam_sobel` is True.
        2. Generates background noise for simulated datasets if `simulated_data` is True.
        3. Reshapes the data to the format (N, 1, x_size, y_size).
        4. Rotates the data based on the specified rotation angles if `learned_rotation` is provided.

        No additional arguments are required as it uses the attributes initialized in the constructor.
        """
        if self.align_center_beam_sobel:
            self.filter_sobel()

        # used for simulated dataset to add background noise
        if self.simulated_data:
            self.generate_background_noise()

        # Reshape the data to (N, 1, x_size, y_size)
        self.stem4d_data = self.stem4d_data.reshape(-1, 1, self.x_size, self.y_size)

        # Rotate the data based on the specified rotation angles if provided
        if self.learned_rotation is not None:
            self.apply_precomputed_rotation()

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
            if self.data_file.endswith(".h5") or self.data_file.endswith(".mat"):
                # Printing the data directory for logging purposes
                stem4d_data = self._load_h5()

            # Check if the data directory ends with '.npy' extension
            elif self.data_file.endswith(".npy"):
                stem4d_data = self._load_npy()

            stem4d_data = self.format_data(
                stem4d_data
            )  # Call format_data to format the loaded data

            # Assign the formatted data to the class attribute
            self.raw_data = stem4d_data
            self.stem4d_data = stem4d_data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while loading the data: {e}")
            return f"An error occurred: {e}"

    def _load_npy(self):
        """
        Loads 4D STEM data from a NumPy (.npy) file.

        This method reads the specified NumPy file and loads the data into a numpy array.

        Returns:
            numpy.ndarray: The loaded 4D STEM data.

        Raises:
            IOError: If the file cannot be opened or read.
        """
        if self.verbose:
            print(f"Loading data from {self.data_file}")
        stem4d_data = np.load(self.data_file)

        return stem4d_data

    def _load_h5(self):
        """
        Loads 4D STEM data from an HDF5 (.h5) file.

        This method reads the specified HDF5 file and extracts the 'output4D' dataset.

        Returns:
            numpy.ndarray: The loaded 4D STEM data.

        Raises:
            OSError: If the file cannot be opened or read.
        """
        if self.verbose:
            print(f"Loading data from {self.data_file}")

        with h5py.File(self.data_file, "r") as f:
            stem4d_data = f["output4D"][:]

        return stem4d_data

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
            stem4d_data = self.reshape_stem4d_data(stem4d_data)

            # Standard scale the data with pre-set up and bottom bound
            if self.standard_scaler is not None:
                stem4d_data = self._standard_scaler(stem4d_data)

            return stem4d_data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while formatting the data: {e}")
            return f"An error occurred: {e}"

    def _standard_scaler(self, stem4d_data):
        stem4d_data[stem4d_data > self.max_threshold] = self.max_threshold
        stem4d_data[stem4d_data < self.min_threshold] = self.min_threshold
        stem4d_data = (
            self.standard_scaler
            * (stem4d_data - self.min_threshold)
            / (self.max_threshold - self.min_threshold)
        )

        return stem4d_data

    def reshape_stem4d_data(self, stem4d_data):
        """
        Reshapes and crops the 4D STEM data according to specified parameters.

        This function transposes the input data, applies cropping based on the specified crop values,
        and reshapes the data to the desired dimensions.

        Args:
            stem4d_data (numpy.ndarray): The 4D STEM data to be reshaped and cropped.

        Returns:
            numpy.ndarray: The reshaped and cropped 4D STEM data.
        """
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
        return stem4d_data

    def generate_background_noise(
        self
    ):
        """
        Generates background noise for the 4D STEM data based on the specified parameters.

        Returns:
            str: An error message if an exception occurs during noise generation.
        """

        noise_generator = PoissonNoise(
            background_weight=self.background_weight,
            counts_per_probe=self.counts_per_probe,
            intensity_scaler=self.intensity_scaler
        )

        try:
            # If the background_weight is zero, simply scale the data
            if self.background_weight == 0:
                self.stem4d_data = self.stem4d_data * self.intensity_scaler
                self.stem4d_data = self.stem4d_data.reshape(
                    -1, 1, self.x_size, self.y_size
                )

            else:
                noisy_data = np.zeros(self.stem4d_data.shape)

                # Loop through each frame and apply the noise generation algorithm
                if self.verbose:
                    print("add Poison distributed background noise to whole dataset")
                
                #TODO: Feature to parallelize the noise generation
                for i in tqdm(
                    range(self.stem4d_data.shape[0]),
                    leave=True,
                    total=self.stem4d_data.shape[0],
                ):
                    noisy_data[i] = noise_generator.generate(self.stem4d_data[i])

                self.stem4d_data = noisy_data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while generating background noise: {e}")
            return f"An error occurred: {e}"

    def apply_precomputed_rotation(self):
        """
        Rotates the 4D STEM data according to the specified rotation angles.

        Raises:
            ValueError: If the rotation size and image size do not match each other.
        """

        try:
            # Compute the angles based on the rotation parameter
            self.angle = np.mod(
                np.arctan2(self.learned_rotation[:, 1], self.learned_rotation[:, 0]), np.pi / 3
            ).reshape(-1)

            # Check if the size of the angle array matches the size of the stem4d_data
            if self.angle.shape[0] != self.stem4d_data.shape[0]:
                raise ValueError(
                    "The rotation size and image size do not match each other"
                )
            else:
                # Combine the data and rotation angle for each frame
                self.stem4d_rotation = []
                
                if self.verbose:
                    print("add image-rotation pair to whole dataset")
                    
                for i in tqdm(
                    range(self.stem4d_data.shape[0]), leave=True, total=self.stem4d_data.shape[0]
                ):
                    self.stem4d_rotation.append([self.stem4d_data[i], self.angle[i]])

        except Exception as e:
            # Log the exception and re-raise to allow for additional handling if needed
            print(f"An error occurred while rotating the data: {e}")
            raise e

    def filter_sobel(self):
        """
        Applies a Sobel filter to the 4D STEM dataset for edge detection.

        This method processes each image in the dataset by first normalizing it and then applying a Sobel filter
        to detect boundaries. The intensity of the detected edges is then upscaled by a specified factor to enhance
        visibility for training purposes.

        The upscale factor can be specified in the class's kwargs attribute. If not provided, it defaults to 2.

        Raises:
            Exception: If an error occurs during the Sobel filtering process, it logs the error and re-raises it.
        """

        upscale_factor = self.kwargs.get("upscale_factor", 2)

        try:
            if self.verbose:
                print("Applying Sobel filter to the entire dataset.")

            for i in tqdm(range(self.stem4d_data.shape[0]), desc="Filtering Sobel"):
                # Normalize each image by dividing by its maximum value
                max_value = np.max(self.stem4d_data[i])
                self.stem4d_data[i] = self.stem4d_data[i] / max_value

                # Apply Sobel filter for edge detection
                edge_detected_image = filters.sobel(self.stem4d_data[i])

                # Upscale the intensity of the detected edges
                self.stem4d_data[i] = upscale_factor * edge_detected_image

        except Exception as e:
            # Log the exception and re-raise to allow for additional handling if needed
            print(f"An error occurred while applying Sobel detection: {e}")
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
        """function to set the preprocessed input data

        Args:
            stem4d_data (tensor): preprocessed input data
        """
        try:
            self._stem4d_data = stem4d_data
        except Exception as e:
            print(f"An error occurred while setting the stem4d_data: {e}")
            raise e
    
    @property
    def raw_data(self):
        """function to call the raw data

        Returns:
            tensor: raw data
        """
        return self._raw_data
    
    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = raw_data


def data_translated(
    data_file,
    translation,
    crop=((2, 122), (2, 122)),
    transpose=(0, 1, 2, 3),
    save_path="",
):
    """
    Apply translation on a dataset.

    This function reads a dataset from the specified path, applies cropping and transposing,
    and then performs a translation operation on the dataset using the provided translation matrix.
    The processed dataset can optionally be saved to a specified path.

    Args:
        data_file (str): Path to the dataset file. Supported formats are .h5, .mat, and .npy.
        translation (np.array): Translation matrix to be applied to the dataset.
        crop (tuple, optional): Tuple specifying the cropping dimensions. Defaults to ((2, 122), (2, 122)).
        transpose (tuple, optional): Tuple specifying the transposing order. Defaults to (0, 1, 2, 3).
        save_path (str, optional): Path to save the processed dataset. If empty, the dataset is not saved.

    Raises:
        ValueError: If the dataset format is not supported.
    """

    # import dataset from directory
    if data_file.endswith(".h5") or data_file.endswith(".mat"):
        print(data_file)  # Printing the data directory for logging purposes
        with h5py.File(data_file, "r") as f:  # Open the file in read mode
            stem4d_data = f["output4D"][:]  # Extract the data

    # check if the data directory ends with '.npy' extension
    elif data_file.endswith(".npy"):
        stem4d_data = np.load(data_file)  # Load the data using NumPy

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
