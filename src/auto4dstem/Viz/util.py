import numpy as np
import cv2
import os
import torch.nn.functional as F
import torch
import h5py
from tqdm import tqdm


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

def translate_base(add_x,
                    add_y,
                    img,
                    mask_,
                    coef = 0.5):
    """ Function for visualize translated coordinate

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
    add_trans = torch.tensor([[1.0000, 0.0000, add_x],
                [0.0000, 1.0000, add_y]],dtype=torch.float).unsqueeze(0)
    
    grid_ = F.affine_grid(add_trans, test_img.size())
            
    after_trans = F.grid_sample(test_img, grid_, mode = 'bicubic').squeeze()
    
    weight_x,weight_y = center_of_mass(after_trans,mask_,coef)
    
    return weight_x,weight_y


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


def make_folder(folder, **kwargs):
    """function to generate folder

    Args:
        folder (string): dictionary of folder

    Returns:
        string: dictionary of folder
    """

    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return folder


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


def add_disturb(rotation,
                dist = 20):
    """function to add additional angles to pretrained rotation

    Args:
        rotation (numpy.array): pretrained rotation value in numpy format ([batch, cos, sin])
        dist (float): additional angle in degree to rotate. Default to 20

    Returns:
        numpy.array: rotation value in numpy format
    """
    # extract rotation value from radians to degree 
    angles = np.rad2deg(np.arctan2(
                        rotation[:,1].reshape(256,256),
                        rotation[:,0].reshape(256,256)))
    angles = angles.reshape(-1)
    
    # add additional degree to all 
    angles = angles+dist
    
    # turn degree back to radians
    angles = np.deg2rad(angles)
    
    # set format to output
    new_rotation = np.zeros([angles.shape[0],2])
    
    # calculate cosine and sine value of updated radians
    cos_ = np.cos(angles)
    sin_ = np.sin(angles)
    
    new_rotation[:,0] = cos_
    new_rotation[:,1] = sin_
    
    return new_rotation