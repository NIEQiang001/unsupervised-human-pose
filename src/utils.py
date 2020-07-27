from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import axes3d

def loadJasondata(filepath):
    assert os.path.exists(filepath), (
        'Can not find data at given directory!!')
    with open(filepath) as f:
        data = json.load(f)
    return data


# To find the parent joint of a joint
def findParentJoint(joint):
    if joint == 0:
        return 0
    elif joint == 16:
        return 1
    elif joint == 2:
        return 16
    elif joint == 4:
        return 16
    elif joint == 7:
        return 16
    elif joint == 10:
        return 0
    elif joint == 13:
        return 0
    else:
        return joint-1



def draw_skeleton(action, R_action):
    """
    This function is build for the 17-joint skeleton model.
    inputs: action is the ground truth pose
              R_action is the recovered pose
    """
    draw_line = np.array([
        [3, 2, 16, 1, 0],
        [16, 4, 5, 6],
        [16, 7, 8, 9],
        [0, 10, 11, 12],
        [0, 13, 14, 15],
    ])

    parser = OptionParser()
    parser.add_option("-e", "--axes_equal", action="store_true", dest="axes_equal",
                      default="", help="Make the plot axes equal.")
    (options, args) = parser.parse_args()

    # Read the original and optimized poses files.
    poses_original_dw = np.reshape(np.array(action[0]), [-1, 3])
    poses_rotation_dw = np.reshape(np.array(R_action[0]), [-1, 3])
    # Plots the results for the specified poses.
    figure = plot.figure()
    ax = plot.axes(projection='3d')
    for i in range(draw_line.shape[0]):
        ax.plot3D(poses_original_dw[draw_line[i], 0], poses_original_dw[draw_line[i], 2],
                  poses_original_dw[draw_line[i], 1], 'green')
        ax.scatter(poses_original_dw[draw_line[i], 0], poses_original_dw[draw_line[i], 2],
                   poses_original_dw[draw_line[i], 1], s=10, c='green')
        ax.plot3D(poses_rotation_dw[draw_line[i], 0], poses_rotation_dw[draw_line[i], 2],
                  poses_rotation_dw[draw_line[i], 1], 'red')
        ax.scatter(poses_rotation_dw[draw_line[i], 0], poses_rotation_dw[draw_line[i], 2],
                   poses_rotation_dw[draw_line[i], 1], s=10, c='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # figure.suptitle(label)
    plot.show()
    # time.sleep(10)
    plot.close()
    return