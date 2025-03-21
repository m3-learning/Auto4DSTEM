import numpy as np
import cv2
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import h5py
import time
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from m3util.util.IO import make_folder
from m3util.viz.text import labelfigs

from ..masks.masks import mask_function

def add_disturb(rotation, dist=20):
    """function to add additional angles to pretrained rotation

    Args:
        rotation (numpy.array): pretrained rotation value in numpy format ([batch, cos, sin])
        dist (float): additional angle in degree to rotate. Default to 20

    Returns:
        numpy.array: rotation value in numpy format
    """
    # extract rotation value from radians to degree
    angles = np.rad2deg(np.arctan2(rotation[:, 1], rotation[:, 0]))
    angles = angles.reshape(-1)

    # add additional degree to all
    angles = angles + dist

    # turn degree back to radians
    angles = np.deg2rad(angles)

    # set format to output
    new_rotation = np.zeros([angles.shape[0], 2])

    # calculate cosine and sine value of updated radians
    cos_ = np.cos(angles)
    sin_ = np.sin(angles)

    new_rotation[:, 0] = cos_
    new_rotation[:, 1] = sin_

    return new_rotation


def select_points(data, 
                mask, 
                threshold=0, 
                clim=None, 
                img_size=None, 
                cmap="viridis",
                add_label = True,
                label_style = 'wb'
                ):
    """function to extract data index according to threshold

    Args:
        data (tensor/numpy): input dataset
        mask (tensor/numpy): boolean type mask in tensor or numpy format
        threshold (float, optional): threshold for target data. Defaults to 0.
        clim (list, optional): color range of plt.imshow and histogram. Defaults to None.
        img_size (list, optional): size of the image. Defaults to None.
        cmap (str, optional): type of color map. Defaults to viridis.
        add_label (bool, optional): determine if add label to figure. Defaults to True.
        label_style (str, optional): determine label style. Defaults to 'wb'

    Returns:
        numpy: index of target data
    """
    # initial image size if not predefined
    if img_size == None:
        x_size = y_size = int(np.sqrt(data.shape[0]))
    else:
        x_size = img_size[0]
        y_size = img_size[1]
    # initial loss map with zeros value
    loss_map = np.zeros([data.shape[0]])
    # generate loss map
    for i in tqdm(range(data.shape[0])):
        # select image
        temp_img = np.copy(data[i].squeeze())
        # add mask on it
        temp_img[~mask] = 0
        # calculate sum
        loss_map[i] = np.sum(temp_img)

    # initial color range if not predefined
    if clim == None:
        clim = [np.min(loss_map), np.max(loss_map)]

    # visualize the loss map
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].set_xticklabels("")
    ax[0].set_yticklabels("")
    ax[0].imshow(loss_map.reshape(x_size, y_size), cmap=cmap, clim=clim)
    counts, bins, _ = ax[1].hist(loss_map, 200, range=clim)
    # add red lines on the threshold
    ax[1].vlines(x=threshold, ymin=0, ymax=max(counts) / 2, color="r", linestyle="-")
    # generate index
    sample_index = np.argwhere(loss_map > threshold).squeeze()
    if add_label:
        for i in range(2):
            labelfigs(ax[i],
                    number = i,
                    style = label_style,
                    loc ='tl',
                    size=20,
                    inset_fraction=(0.1, 0.1)
                    )
    fig.tight_layout()

    return sample_index


def generate_classification(
    sample_index, sample_position=0, img_size=[512, 512], save_file=True, file_path=""
):
    """function to create correct format of classification array

    Args:
        sample_index (np.array): index of sample position
        sample_position (int, optional): sample position. Defaults to 0
        img_size (list, optional): image size. Defaults to [512,512].
        save_file (bool, optional): determine save or not. Defaults to True.
        file_path (str, optional): determine the file path. Defaults to ''.

    Returns:
        np.array: classification
    """
    # create background index according to sample index
    bkg_index = [e for e in range(img_size[0] * img_size[1]) if e not in sample_index]
    # change type to int
    bkg_index = np.array(bkg_index, dtype=int)
    # create correct format of classification
    classification = np.zeros([img_size[0] * img_size[1], 2])
    # change sample index to 1
    classification[sample_index, sample_position] = 1
    # change background index to 1
    classification[bkg_index, int(1 - sample_position)] = 1
    # save file
    if save_file:
        np.save(f"{file_path}_classification.npy", classification)

    # return correct format of classification
    return classification

def custom_formatter(value, pos):
    """_summary_

    Args:
        value (_type_): _description_

    Returns:
        _type_: _description_
    """
    if value == 0:
        return '0'  # For zero, just return "0"
    else:
        return f'{value:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')

def get_strain_parameter_by_given_vec(x,y,dist=-15):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        dist (int, optional): _description_. Defaults to -15.
    """
        
    # generate the first rotation value from localization vector
    x = torch.tensor(x,dtype=torch.float)
    rotate = nn.ReLU()(x[:,2])

#        print(rotate)
    a_1 = torch.cos(rotate)
#        a_2 = -torch.sin(selection)
    a_2 = torch.sin(rotate)    
    a_4 = torch.ones(rotate.shape)
    a_5 = rotate*0

    # Add the rotation after the shear and strain
    b1 = torch.stack((a_1,a_2), dim=1).squeeze()
    b2 = torch.stack((-a_2,a_1), dim=1).squeeze()
    b3 = torch.stack((a_5,a_5), dim=1).squeeze()
    rotation_1 = torch.stack((b1, b2, b3), dim=2)
    
    # generate the angle by rotation1
    rot = rotation_1[:,:,0]
    angle = torch.remainder(dist * torch.pi/180+torch.atan2(
                            rot[:,1].reshape(-1),
                            rot[:,0].reshape(-1)),torch.pi/3)
    
    # generate scale and shear from y and angle
    
    y = torch.tensor(y,dtype = torch.float)
    
    scale_1 = 0.05*nn.Tanh()(y[:,0])+1
    scale_2 = 0.05*nn.Tanh()(y[:,1])+1
    rotate = angle.reshape(y[:,2].shape) + 0.1 * nn.Tanh()(y[:,2])
    shear_1 = 0.1*nn.Tanh()(y[:,3])
    a_1 = torch.cos(rotate)
    a_2 = torch.sin(rotate)    
    a_4 = torch.ones(rotate.shape)
    a_5 = rotate*0

    # combine shear and strain together
    c1 = torch.stack((scale_1,shear_1), dim=1).squeeze()
    c2 = torch.stack((shear_1,scale_2), dim=1).squeeze()
    c3 = torch.stack((a_5,a_5), dim=1).squeeze()
    scale_shear = torch.stack((c1, c2, c3), dim=2) 

    # Add the rotation after the shear and strain
    b1 = torch.stack((a_1,a_2), dim=1).squeeze()
    b2 = torch.stack((-a_2,a_1), dim=1).squeeze()
    b3 = torch.stack((a_5,a_5), dim=1).squeeze()
    rotation_2 = torch.stack((b1, b2, b3), dim=2)
    
    return scale_shear[:,:,0:2].cpu().detach().numpy().reshape(-1,4),rotation_2[:,:,0].cpu().detach().numpy()

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


class find_nearby_dot_group:
    """class to find nearby dot group

    Methods:
        __init__(img): Initializes the class with an input image.
        set_cluster(threshold=0.95, eps=20, min_samples=2, cmap="viridis", marker="o"): Classifies pixels in the image based on the given parameters.
    """
    
    def __init__(self, img):
        """class to capture the center coordinates for creating mask

        Args:
            img (np.array): input image to crop cluster
        """
        self.img = img

    def set_cluster(
        self,
        threshold=0.95,
        eps=20,
        min_samples=2,
        cmap="viridis",
        marker="o",
    ):
        """function to classify pixels in image

        Args:
            threshold (float, optional): threshold to filter pixels to classify. Defaults to 0.95.
            eps (int, optional): distance of the pixels belongs to same cluster. Defaults to 20.
            min_samples (int, optional): smallest value for a cluster. Defaults to 2.
            cmap (str, optional): color map to show. Defaults to 'viridis'.
            marker (str, optional): marker type to show. Defaults to 'o'.
        """
        # create x and y coordinates for each pixels exceeds threshold
        x_cor = np.where(self.img > threshold)[1]
        y_cor = np.where(self.img > threshold)[0]
        # put coordinate into array
        new_array = []
        for i in range(len(x_cor)):
            new_array.append([x_cor[i], y_cor[i]])
        self.new_array = np.array(new_array)
        if len(self.new_array) ==0:
            return('No value exceeds the threshold, please decrease threshold and try again')
        # set class
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # Fit the model
        self.clusters = dbscan.fit_predict(self.new_array)
        plt.scatter(
            self.new_array[:, 0],
            self.new_array[:, 1],
            c=self.clusters,
            cmap=cmap,
            marker=marker,
        )
        plt.title("Image Clustering")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.colorbar(label="Cluster Label")
        plt.show()

    def center_cor_list(self, clim=[0, 2], cmap="viridis", dot_col="b."):
        """function to extract center coordinates for each cluster

        Args:
            clim (list, optional): color range to visualize. Defaults to [0,2].
            cmap (str, optional): color map. Defaults to 'viridis'.
            dot_col (str, optional): type of dot for each coordinates. Defaults to 'b.'.

        Returns:
            list: list of center coordinates for each cluster
        """
        # set number of cluster
        max_ = np.max(self.clusters) + 1
        # initial coordinates list
        cor_list = []
        plt.imshow(self.img, clim=clim, cmap=cmap, alpha=0.9, interpolation="none")
        # updates coordinates
        for i in range(max_):
            cor_ = np.round(
                np.mean(self.new_array[np.where(self.clusters == i)], axis=0)
            )
            cor_list.append([int(cor_[0]), int(cor_[1])])
            plt.plot(int(cor_[0]), int(cor_[1]), dot_col)

        return cor_list
