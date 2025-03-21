

import cv2
import numpy as np
import torch


def mask_function(img, radius=7, center_coordinates=(100, 100)):
    """Function for make mask

    Args:
        img (numpy.array): blank image with the same size of input data
        radius (int): radius of the circle in the mask
        center_coordinates (tuple): center coordinates of the circle

    Returns:
        numpy.array: mask
    """
    image = np.copy(img.squeeze())
    # set coefficient of cv2.circle function
    thickness = -1
    color = 100
    # create binary circle mask img
    image_2 = cv2.circle(image, center_coordinates, radius, color, thickness)
    image_2 = np.array(image_2)
    mask = image_2 == 100
    mask = np.array(mask)

    return mask


class mask_class:
    """
    A class to initialize mask and mask list.

    Attributes:
        img_size (list): Size of the image to add mask on.
        img (ndarray): Image array initialized with zeros.
        center_coordinates (tuple): Coordinates of the center of the image.

    Methods:
        __init__(img_size): Initializes the mask_class with the given image size.
        mask_single(radius): Creates a single circular mask.
        mask_ring(radius_1, radius_2): Creates a ring mask with specified inner and outer radii.
    """

    def __init__(self, img_size=[200, 200]):
        """
        Initializes the mask_class with the given image size.

        Args:
            img_size (list): Size of the image to add mask on.
        """
        # set image for mask function
        self.img_size = img_size
        self.img = np.zeros(self.img_size)
        # calculate center coordinate
        self.center_coordinates = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))

    def mask_single(self, radius):
        """make circle mask

        Args:
            radius (int): radius of the circle

        Returns:
            tensor, list: tensor of boolean mask, list of mask
        """

        # load the mask function to create inner and outer circle mask
        mask_ = mask_function(
            self.img, radius=radius, center_coordinates=self.center_coordinates
        )

        # make the mask into tensor version
        mask_tensor = torch.tensor(mask_)
        # put mask into list
        mask_list = [mask_tensor]

        return mask_tensor, mask_list

    def mask_ring(self, radius_1, radius_2):
        """make ring mask

        Args:
            radius_1 (int): radius of inner circle
            radius_2 (int): radius of outer circle

        Returns:
            tensor, list: tensor of boolean mask, list of mask
        """

        # load the mask function to create inner and outer circle mask
        mask_0 = mask_function(
            self.img, radius=radius_1, center_coordinates=self.center_coordinates
        )
        mask_1 = mask_function(
            self.img, radius=radius_2, center_coordinates=self.center_coordinates
        )

        # combine masks together
        mask_combine = ~mask_0 * mask_1
        # make the mask into tensor version
        mask_tensor = torch.tensor(mask_combine)
        # put mask into list
        mask_list = [mask_tensor]

        return mask_tensor, mask_list

    def mask_round(self, radius, center_list):
        """make round mask

        Args:
            radius (int): radius of each circle
            center_list (list): list of tuple for each center coordinate

        Returns:
            tensor, list: tensor of boolean mask, list of mask
        """

        # initial bool image for to add all mask
        mask_combine = np.array(self.img, dtype=bool)
        # initial mask list
        mask_list = []

        for cor in center_list:

            # add each mask into mask list
            temp_mask = mask_function(self.img, radius=radius, center_coordinates=cor)
            # combine all components together
            mask_combine += temp_mask
            # make the mask into tensor version
            temp_mask = torch.tensor(temp_mask)
            # add temp_mask into list
            mask_list.append(temp_mask)

        # make the mask into tensor version
        mask_tensor = torch.tensor(mask_combine)

        return mask_tensor, mask_list