from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

class NoiseClass(ABC):
    """
    Abstract base class for noise generation.
    
    This class serves as a blueprint for creating different types of noise
    generation classes. Any subclass must implement the `generate` method.
    """
 
    @abstractmethod
    def generate(self):
        """
        Abstract method to generate noise.

        This method should be implemented by subclasses to generate noise
        based on specific algorithms or parameters.
        """
        pass

@dataclass
class PoissonNoise(NoiseClass):
    """
    Class for generating Poisson distributed noise.

    Attributes:
        background_weight (float): The weight of the background noise.
        counts_per_probe (float): The number of counts per probe for scaling the noise.
        intensity_coefficient (float): The intensity coefficient for scaling the noise.

    Methods:
        generate(data): Generates Poisson distributed noise for the given data.
    """
    background_weight: float = 0
    counts_per_probe: float
    intensity_coefficient: float
        
    @property
    def background_weight(self):
        return self._background_weight
    
    @property
    def counts_per_probe(self):
        return self._counts_per_probe
    
    @property
    def intensity_coefficient(self):
        return self._intensity_coefficient
    
    @background_weight.setter
    def background_weight(self, value):
        self._background_weight = value
        
    @counts_per_probe.setter
    def counts_per_probe(self, value):
        self._counts_per_probe = value
        
    @intensity_coefficient.setter
    def intensity_coefficient(self, value):
        self._intensity_coefficient = value
        
    def generate(self, data):
        """
        Generates Poisson distributed noise for the given data.

        Args:
            data (numpy.ndarray): The input data to which noise will be added.

        Returns:
            numpy.ndarray: The noisy data with Poisson distributed noise applied.

        The method creates a background noise pattern using a frequency domain
        approach, combines it with the input data based on the background weight,
        and applies Poisson noise. The result is scaled by the intensity coefficient.
        """
        
        if data is None:
            raise ValueError("data must be a numpy array")
        
        if data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")
        
        if self.background_weight == 0:
            return data * self.intensity_coefficient
        
        test_img = np.copy(data)
        
        # Generates the Poisson distributed background noise
        qx = np.fft.fftfreq(data.shape[0], d=1)
        qy = np.fft.fftfreq(data.shape[1], d=1)
        qya, qxa = np.meshgrid(qy, qx)
        qxa = np.fft.fftshift(qxa)
        qya = np.fft.fftshift(qya)
        qra2 = qxa**2 + qya**2
        im_bg = 1.0 / (1 + qra2 / 1e-2**2)
        im_bg = im_bg / np.sum(im_bg)
        int_comb = (
            test_img * (1 - self.background_weight) + im_bg * self.background_weight
        )
        int_noisy = (
            np.random.poisson(int_comb * self.counts_per_probe)
            / self.counts_per_probe
        )
        
        int_noisy = int_noisy * self.intensity_coefficient
        
        return int_noisy
