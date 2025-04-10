import numpy as np

def add_rotation(rotation, dist=20, **kwargs):
    """function to add additional angles to pretrained rotation

    Args:
        rotation (numpy.array): pretrained rotation value in numpy format ([batch, cos, sin])
        dist (float): additional angle in degree to rotate the rotation is in degree. Default to 20

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
    modified_rotation = np.zeros([angles.shape[0], 2])

    modified_rotation[:, 0] = np.cos(angles)
    modified_rotation[:, 1] = np.sin(angles)

    return modified_rotation