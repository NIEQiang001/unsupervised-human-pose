""""This file is used to generate some geometry transformation function for
the 3D human pose."""

import numpy as np
import os

def translation_matrix(dx, dy, dz):
    """Generate translation matrix for a given axis
    Parameters
    ----------
    dx: translation in x domain
    dy: translation in y domain
    dz: translation in z domain
    Returns
    -------
    numpy.ndarray
    3x3 translation matrix
    """
    R_trans = np.array([[1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])
    return R_trans

def rotation_matrix(theta, axis, active=False):
    """Generate rotation matrix for a given axis
       Parameters
    ----------
    theta: numeric, optional
        The angle (degrees) by which to perform the rotation.  Default is
        0, which means return the coordinates of the vector in the rotated
        coordinate system, when rotate_vectors=False.
    axis: int, optional
        Axis around which to perform the rotation (x=0; y=1; z=2)
    active: bool, optional
        Whether to return active transformation matrix.
    Returns
    -------
    numpy.ndarray
    3x3 rotation matrix
    """
    theta = np.radians(theta)
    if axis == 0:
        R_theta = np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        R_theta = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
    else:
        R_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
    if active:
        R_theta = np.transpose(R_theta)
    return R_theta

def rotation_3d(orientation):
    """Generate rotation matrix for a joint with three angles.
    ---------
       orientation: joint Euler angles, (alpha, beta, garma)
    ---------
    Return:
    numpy.ndarray
    3x3 rotation matrix
    """
    R = np.matmul(rotation_matrix(orientation[0], axis=0), rotation_matrix(orientation[1], axis=1))
    R = np.matmul(R, rotation_matrix(orientation[2], axis=2))
    return R

def random_rotation():
    """ generate a random rotation matrix in 3 dimension """
    rotation_angle = np.random.rand(3) * 360 - 180
    rot_mat_rand = rotation_3d(rotation_angle)
    return rot_mat_rand
