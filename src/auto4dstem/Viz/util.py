import numpy as np
import cv2
import os
import torch.nn.functional as F
import torch
import h5py
import time
import subprocess
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from m3util.util.IO import make_folder


def center_of_mass(img, mask, coef=1.5):
    """function for COM operation

    Args:
        img (torch.tensor): Input tensor
        mask (torch.tensor): binary tensor added to img
        coef (float, optional): the parameter to control the value of threshold for COM operation. Defaults to 1.5.

    Returns:
        Tensor: coordinates of center point
    """
    # extract coordinate of mask region
    cor_x, cor_y = torch.where(mask != 0)
    # compute mean value in mask region
    mean_mass = torch.mean(img[mask])
    # subtract mean to each element in mask
    mass = F.relu(img[mask] - coef * mean_mass)
    img_after = torch.clone(img)
    img_after[mask] = mass

    sum_mass = torch.sum(mass)
    # compute COM coordinate
    if sum_mass == 0:
        weighted_x = torch.sum(cor_x) / len(cor_x)
        weighted_y = torch.sum(cor_y) / len(cor_y)
    else:
        weighted_x = torch.sum(cor_x * mass) / sum_mass

        weighted_y = torch.sum(cor_y * mass) / sum_mass

    return weighted_x, weighted_y


def translate_base(add_x, add_y, img, mask_, coef=0.5):
    """Function for visualize translated coordinate

    Args:
        add_x (float): translation parameter of x direction
        add_y (float): translation parameter of y direction
        img (tensor): Input tensor
        mask_ (tensor): binary tensor added to img
        coef (float, optional): the parameter to control the value of threshold for COM operation. Defaults to 0.5

    Returns:
        _type_: coordinates of center point
    """

    test_img = torch.clone(img).unsqueeze(0).unsqueeze(1)
    add_trans = torch.tensor(
        [[1.0000, 0.0000, add_x], [0.0000, 1.0000, add_y]], dtype=torch.float
    ).unsqueeze(0)

    grid_ = F.affine_grid(add_trans, test_img.size())

    after_trans = F.grid_sample(test_img, grid_, mode="bicubic").squeeze()

    weight_x, weight_y = center_of_mass(after_trans, mask_, coef)

    return weight_x, weight_y


def mask_function(img, radius=7, center_coordinates=(100, 100)):
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


# TODO: import from Util
# def make_folder(folder, **kwargs):
#     """function to generate folder

#     Args:
#         folder (string): dictionary of folder

#     Returns:
#         string: dictionary of folder
#     """

#     # Makes folder
#     os.makedirs(folder, exist_ok=True)

#     return folder

# TODO: Import from Util
# def download_files_from_txt(url_file,
#                             download_path):
#     """Download files from URLs listed in a text file.

#     Args:
#     url_file (str): Path to the text file containing URLs, each on a new line.
#     download_path (str): Directory to save the downloaded files. The directory must exist.

#     """
#     # create folder if not yet
#     make_folder(download_path)
#     abs_path = os.path.abspath(download_path)

#     # set delay
#     delay = 1

#     # Open the text file containing URLs
#     with open(url_file, 'r') as file:
#         urls = file.readlines()

#     # Iterate over each URL
#     for url in tqdm(urls):
#         url = url.strip()  # Remove any extraneous whitespace or newline characters
#         if url:  # Ensure the URL is not empty
#             while True:
#                 try:
#                     # Make HTTP GET request to the URL
#                     response = requests.get(url, stream=True)
#                     response.raise_for_status()  # Check if the request was successful

#                     # Extract filename from URL if possible, or default to a name with its index
#                     filename = url.split('/')[-1]
#                     # skip download if file exists
#                     if os.path.exists(f'{abs_path}/{filename}'):
#                         print(f"File already exists: {filename}")
#                         break
#                     file_path = os.path.join(abs_path, filename)

#                     # Save the content to a file in the specified download path
#                     with open(file_path, 'wb') as f:
#                         for chunk in response.iter_content(chunk_size=8192):
#                             f.write(chunk)
#                     print(f"Downloaded: {filename}")
#                     break

#                 except requests.exceptions.HTTPError as e:
#                     if response.status_code == 429:  # Too Many Requests
#                         print("Rate limit reached, waiting to retry...")
#                         time.sleep(delay)
#                         delay *= 2  # Exponential backoff
#                         if delay > 1024:
#                             print(f"Failed to download {url}: time exceeds limit")
#                             break

#                 except requests.exceptions.RequestException as e:
#                     print(f"Failed to download {url}: {str(e)}")
#                     break  # exit the loop if a different HTTP error occurred

# TODO: check if we can delete this function
# def config_folders(folder_name,
#                 file_download):
#     """function to create folder to save the weights and datasets

#     Args:
#         folder_name (str): folder name
#         file_download (str): file name
#     """
#     #
#     make_folder(folder_name)
#     abs_path = os.path.abspath(folder_name)
#     subprocess.run(["wget", "-nc", "-i",file_download, "-P",abs_path])


def Show_Process(
    model,
    test_iterator,
    mask_list,
    name_of_file,
    device,
    up_inp,
):
    """function to generate and save updated base and mask list

    Args:
        model (torch.Module): pytorch model
        test_iterator (torch.util.data.dataloader): dataloader of dataset without shuffle
        mask_list (list of tensor): The list of tensor with binary type
        name_of_file (string): file name
        device (torch.device): set the device to run the model
        up_inp (bool): determine whether generate interpolated mask
    """

    model.eval()

    for i, x_value in enumerate(
        tqdm(test_iterator, leave=True, total=len(test_iterator))
    ):
        # stop gradient updating
        with torch.no_grad():
            # check type of input
            if type(x_value) != list:
                x = x_value.to(device, dtype=torch.float)
                y = None
            else:
                x, y = x_value
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)
            # generate results of trained weights
            if up_inp:
                (
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    kout,
                    theta_1,
                    theta_2,
                    theta_3,
                    adj_mask,
                    new_list,
                    x_inp,
                ) = model(x, y)
                # upgrid mask size
                mask_list = upsample_mask(mask_list, x.shape[-1], x_inp.shape[-1])

            else:
                (
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    kout,
                    theta_1,
                    theta_2,
                    theta_3,
                    adj_mask,
                    new_list,
                ) = model(x, y)

        if i == 0:
            break
    # move base image to cpu and numpy version
    predicted_base = predicted_base[0].cpu().detach().numpy()
    # create h5 file
    h5f = h5py.File(name_of_file + ".h5", "w")
    # save base image
    h5f.create_dataset("base", data=predicted_base)

    # stack the list to torch tensor for saving in the h5 format
    gou_list = torch.cat(mask_list)
    gou_list = gou_list.reshape(
        len(mask_list), mask_list[0].shape[-2], mask_list[0].shape[-1]
    )

    h5f.create_dataset("mask_list", data=gou_list)
    h5f.close()


def inverse_base(name_of_file, input_mask_list, coef=2, radius=7):
    """generate updated mask list that center the spots

    Args:
        name_of_file (string): file directory
        input_mask_list (list of tensor): mask list used for updating
        coef (float): threshold for center the spots. Defaults to 2.
        radius (int): radius of updated mask. Defaults to 7.

    Returns:
        list of tensor, tensor: mask list and mask
    """
    # load h5 file
    load_file = h5py.File(name_of_file + ".h5", "r")
    # load base image
    load_base = load_file["base"][0].squeeze()
    # reshape base
    base_ = torch.tensor(load_base, dtype=torch.float).reshape(
        1, 1, load_base.shape[-1], load_base.shape[-2]
    )
    # update mask list region using center_mask_list_function
    center_mask_list, rotate_center = center_mask_list_function(
        base_, input_mask_list, coef, radius=radius
    )

    return center_mask_list, rotate_center


def upsample_single_mask(mask, up_size):
    """function to interpolate mask size

    Args:
        mask (tensor): mask list used for updating
        up_size (list/np.array/torch.tensor): updated size of mask tensor (x size and y size)

    Returns:
        tensor: updated mask
    """
    # reshape and change type of mask
    up_mask = torch.tensor(mask.unsqueeze(0).unsqueeze(0), dtype=torch.float)
    # upgrid image size
    up_mask = F.interpolate(up_mask, size=(up_size[0], up_size[1]), mode="bicubic")
    # make mask region correct
    up_mask[up_mask < 0.5] = 0
    up_mask[up_mask >= 0.5] = 1
    # switch type back to boolean
    up_mask = torch.tensor(up_mask.squeeze(), dtype=torch.bool)

    return up_mask


def upsample_mask(mask_list, input_size, up_size):
    """function to interpolate mask size

    Args:
        mask_list (list of tensor): mask list used for updating
        input_size (int): size of each mask tensor
        up_size (int): updated size of mask tensor

    Returns:
        list of tensor: updated mask list
    """
    # check if upgrid size is needed
    if mask_list[0].shape[-1] == up_size:
        return mask_list
    # create list
    mask_with_inp = []
    for mask_ in mask_list:
        temp_mask = torch.tensor(
            mask_.reshape(1, 1, input_size, input_size), dtype=torch.float
        )
        # upgrid image size
        temp_mask = F.interpolate(temp_mask, size=(up_size, up_size), mode="bicubic")
        # make mask region correct
        temp_mask[temp_mask < 0.5] = 0
        temp_mask[temp_mask >= 0.5] = 1
        temp_mask = torch.tensor(temp_mask.squeeze(), dtype=torch.bool)
        mask_with_inp.append(temp_mask)

    return mask_with_inp


def rotate_mask_list(mask_list, theta_):
    """function to rotate mask region

    Args:
        mask_list (list): list of tensor
        theta_ (float): radius to rotate

    Returns:
        _type_: rotated mask list, a tensor with all mask region
    """
    modified_mask_list_2 = []
    a_1 = torch.cos(theta_).reshape(1, 1)
    a_2 = torch.sin(theta_).reshape(1, 1)
    a_5 = torch.zeros([1, 1])
    b1 = torch.stack((a_1, a_2), dim=1)
    b2 = torch.stack((-a_2, a_1), dim=1)
    b3 = torch.stack((a_5, a_5), dim=1)
    rotation = torch.stack((b1, b2, b3), dim=2)
    rotation = rotation.reshape(1, 2, 3)
    zero_tensor = torch.zeros(mask_list[0].shape)
    print(zero_tensor.shape)
    zero_tensor = zero_tensor.reshape(
        1, 1, zero_tensor.shape[-2], zero_tensor.shape[-1]
    )
    grid_2 = F.affine_grid(rotation, zero_tensor.size())

    for mask_ in mask_list:

        tmp = torch.clone(mask_).reshape(1, 1, mask_.shape[-2], mask_.shape[-1])
        tmp = torch.tensor(tmp, dtype=torch.float)
        rotate_tmp = F.grid_sample(tmp, grid_2)
        rotate_tmp = torch.tensor(rotate_tmp, dtype=torch.bool).squeeze()
        modified_mask_list_2.append(rotate_tmp)

    rotate_mask_up = torch.clone(modified_mask_list_2[0])

    for i in range(1, len(mask_list)):
        rotate_mask_up += modified_mask_list_2[i]

    return modified_mask_list_2, rotate_mask_up


def center_mask_list_function(image, mask_list, coef, radius=7):
    """function to update mask list

    Args:
        image (tensor): torch.tensor
        mask_list (list of tensor): mask list used for updating
        coef (float): threshold for center the spots. Defaults to 2.
        radius (int): radius of updated mask. Defaults to 7.

    Returns:
        list of tensor, tensor: mask list and mask
    """
    # create mask list
    center_mask_list = []
    # create image with zero value
    mean_ = np.zeros([image.shape[-2], image.shape[-1]])

    input_size = mask_list[0].shape[-1]
    up_size = image.shape[-1]
    # upgrid image if necessary
    if input_size != up_size:
        mask_list = upsample_mask(mask_list, input_size, up_size)

    for j, mask in enumerate(mask_list):
        mask_ = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])

        new_image = image * mask_
        # compute coordinate with center of mass
        center_x, center_y = center_of_mass(new_image.squeeze(), mask_.squeeze(), coef)

        center_x = int(np.round(np.array(center_x)))
        center_y = int(np.round(np.array(center_y)))
        print(center_x, center_y)
        # create small mask region using center coordinate
        small_mask = mask_function(
            mean_, radius=radius, center_coordinates=(center_y, center_x)
        )
        # switch type into tensor
        small_mask = torch.tensor(small_mask, dtype=torch.bool)

        center_mask_list.append(small_mask)
    # change mask size if necessary
    if input_size != up_size:
        center_mask_list = upsample_mask(center_mask_list, up_size, input_size)
    # create whole mask region in one image
    rotate_mask_up = torch.clone(center_mask_list[0])

    for i in range(1, len(center_mask_list)):
        rotate_mask_up += center_mask_list[i]

    return center_mask_list, rotate_mask_up


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


def select_points(data, mask, threshold=0, clim=None, img_size=None, cmap="viridis"):
    """function to extract data index according to threshold

    Args:
        data (tensor/numpy): input dataset
        mask (tensor/numpy): boolean type mask in tensor or numpy format
        threshold (float, optional): threshold for target data. Defaults to 0.
        clim (list, optional): color range of plt.imshow and histogram. Defaults to None.
        img_size (list, optional): size of the image. Defaults to None.

    Returns:
        numpy: index of target data
    """
    # initial image size if not predefined
    if img_size == None:
        x_size = y_size = int(np.sqrt(data.shape[0]))

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
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_xticklabels("")
    ax[0].set_yticklabels("")
    ax[0].imshow(loss_map.reshape(x_size, y_size), cmap=cmap, clim=clim)
    counts, bins, _ = ax[1].hist(loss_map, 200, range=clim)
    # add red lines on the threshold
    ax[1].vlines(x=threshold, ymin=0, ymax=max(counts) / 2, color="r", linestyle="-")
    # generate index
    sample_index = np.argwhere(loss_map > threshold).squeeze()
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


class mask_class:

    def __init__(self, img_size=[200, 200]):
        """class to initialize mask and mask list

        Args:
            img_size (list): size of image to add mask on
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
    """class to find nearby dot group"""

    # TODO: add docstring

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
