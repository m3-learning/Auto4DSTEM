from tqdm import tqdm
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import h5py

@dataclass
class STEM4D_DataSet:
    """
    Represents a dataset for STEM 4D data processing.

    Attributes:
        data_dir (str): Directory of the dataset.
        background_weight (float): Weight for the background, defaulting to 0.10.
        crop (tuple): Tuple for cropping, defaulting to ((28, 228), (28, 228)).
        transpose (tuple): Tuple for transposing, defaulting to (1, 0, 3, 2).
        background_intensity (Optional[float]): Intensity of background noise, can be None or float.
        counts_per_probe (Optional[float]): Counts per probe, can be None or float, defaulting to 1e5.
        rotation (Optional[float]): Rotation angle, can be None or float.
        x_size (int): Computed x size from crop values, not provided during initialization.
        y_size (int): Computed y size from crop values, not provided during initialization.
    """
    data_dir: str
    background_weight: float = 0.10
    crop: tuple = ((28, 228), (28, 228))
    transpose: tuple = (1, 0, 3, 2)
    background_intensity: Optional[float] = None  
    counts_per_probe: float = 1e5       
    rotation: Optional[float] = None              
    x_size: int = field(init=False)
    y_size: int = field(init=False)
        
        
        
    def __post_init__(self):
        """ Post-initialization method to compute derived attributes and perform additional setup.

        This method calculates the x_size and y_size based on the crop values, calls the load_data method to load the dataset,
        and optionally calls methods to generate background noise and rotate the data based on the given parameters.

        No additional arguments are required as it uses the attributes initialized in the constructor.
        """
        self.x_size = self.crop[0][1]-self.crop[0][0]
        self.y_size = self.crop[1][1]-self.crop[1][0]
        self.load_data()
        if self.background_intensity is not None:
            self.generate_background_noise(self.stem4d_data, self.background_weight, self.counts_per_probe)
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
            if self.data_dir.endswith('.h5') or self.data_dir.endswith('.mat'):
                print(self.data_dir)  # Printing the data directory for logging purposes
                with h5py.File(self.data_dir, 'r') as f:  # Open the file in read mode
                    stem4d_data = f['output4D']           # Extract the data
                    self.format_data(stem4d_data)         # Call format_data to format the loaded data

            # Check if the data directory ends with '.npy' extension
            elif self.data_dir.endswith('.npy'):
                self.stem4d_data = np.load(self.data_dir) # Load the data using NumPy
                self.format_data(stem4d_data)             # Call format_data to format the loaded data

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
            # Apply the cropping according to the specified crop values
            stem4d_data = stem4d_data[:, :, self.crop[0][0]
                :self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
            
            # Transpose the data according to the specified transpose values
            stem4d_data = np.transpose(stem4d_data, self.transpose)
            
            # Reshape the data using the computed x_size and y_size
            stem4d_data = stem4d_data.reshape(-1, self.x_size, self.y_size)
            
            # Assign the formatted data to the class attribute
            self.stem4d_data = stem4d_data

        except Exception as e:
            # Log and return a generic error message along with the specific exception
            print(f"An error occurred while formatting the data: {e}")
            return f"An error occurred: {e}"

        
    def generate_background_noise(self, stem4d_data, background_weight, counts_per_probe):
        """
        Generates background noise for the 4D STEM data based on the specified parameters.

        Args:
            stem4d_data (numpy.ndarray): The 4D STEM data to add noise to.
            background_weight (float): The weight for the background noise.
            counts_per_probe (float): The number of counts per probe for scaling the noise.

        Returns:
            str: An error message if an exception occurs during noise generation.
        """
        try:
            # If the background_weight is zero, simply scale the data
            if background_weight == 0:
                self.stem4d_data = stem4d_data * 1e5 / 4
                self.stem4d_data = self.stem4d_data.reshape(-1, 1, self.x_size, self.y_size)

            else:
                noisy_data = np.zeros(stem4d_data.shape)
                im = np.zeros(stem4d_data.shape[1:])

                # Loop through each frame and apply the noise generation algorithm
                for i in tqdm(range(stem4d_data.shape[0]), leave=True, total=stem4d_data.shape[0]):
                    test_img = np.copy(stem4d_data[i])
                    qx = np.fft.fftfreq(im.shape[0], d=1)
                    qy = np.fft.fftfreq(im.shape[1], d=1)
                    qya, qxa = np.meshgrid(qy, qx)
                    qxa = np.fft.fftshift(qxa)
                    qya = np.fft.fftshift(qya)
                    qra2 = qxa ** 2 + qya ** 2
                    im_bg = 1. / (1 + qra2 / 1e-2 ** 2)
                    im_bg = im_bg / np.sum(im_bg)
                    int_comb = test_img * (1 - background_weight) + im_bg * background_weight
                    int_noisy = np.random.poisson(int_comb * counts_per_probe) / counts_per_probe
                    int_noisy = int_noisy * 1e5 / 4
                    noisy_data[i] = int_noisy

                self.stem4d_data = noisy_data.reshape(-1, 1, self.x_size, self.y_size)

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
            self.angle = np.mod(np.arctan2(
                                rotation[:, 1],
                                rotation[:, 0]),
                                np.pi / 3).reshape(-1)

            # Check if the size of the angle array matches the size of the stem4d_data
            if self.angle.shape[0] != stem4d_data.shape[0]:
                raise ValueError('The rotation size and image size do not match each other')
            else:
                # Combine the data and rotation angle for each frame
                whole_data_with_rotation = []
                for i in tqdm(range(stem4d_data.shape[0]), leave=True, total=stem4d_data.shape[0]):
                    whole_data_with_rotation.append([stem4d_data[i], self.angle[i]])

                # Assign the rotated data to the class attribute
                self.stem4d_rotation = whole_data_with_rotation

        except Exception as e:
            # Log the exception and re-raise to allow for additional handling if needed
            print(f"An error occurred while rotating the data: {e}")
            raise e


        
    @property
    def stem4d_data(self):
        return self._stem4d_data
    
    @stem4d_data.setter
    def stem4d_data(self, stem4d_data):
        self._stem4d_data = stem4d_data

        
        