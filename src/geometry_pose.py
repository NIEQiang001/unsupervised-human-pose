""""This file is used to generate human poses with input of bone length
    based on physical skeleton model"""

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

class Joint:
    def __init__(self, label):
        self.position = [None] * 3      # (x, y, z)
        self.orientation = [None] * 3   # alpha, beta, garma
        self.label = label
        self.limit_inf = [] #inferior limits corresponding to alpha, beta, garma
        self.limit_sup = [] #super limits corresponding to alpha, beta, garma
        self.parent = None
        self.Children = []

class HumanSkeleton:
    def __init__(self, JointNum, Bonelengths):
        self.JointNum = JointNum  # As spine shoulder joint is split to three joints, 17-> 19, 21->23
        self.Joints = []
        # self.rootJoint = Joint(0)
        for i in range(self.JointNum):
            self.Joints.append(Joint(i))
        self.Bonelengths = Bonelengths # defined refer to child joint
        # Joint alias name
        self.SpineShoulder = 16 if self.JointNum == 19 else 20
        self.LeftShoulder = 4
        self.RightShoulder = 7 if self.JointNum == 19 else 8
        self.LeftHip = 10 if self.JointNum == 19 else 12
        self.RightHip = 13 if self.JointNum == 19 else 16
        self.LeftElbow = self.LeftShoulder + 1
        self.LeftWaist = self.LeftElbow + 1
        self.RightElbow = self.RightShoulder + 1
        self.RightWaist = self.RightElbow + 1
        self.LeftKnee = self.LeftHip + 1
        self.LeftAnkle = self.LeftKnee + 1
        self.RightKnee = self.RightHip + 1
        self.RightAnkle = self.RightKnee + 1
        self.SpineShoulderLeft = self.SpineShoulder + 1
        self.SpineShoulderRight = self.SpineShoulder + 1
        self.Head = 3
        self.Neck = 2
        if JointNum == 23:
            self.RightHand = self.RightWaist + 1  # exist only when JointNum = 23
            self.LeftHand = self.LeftWaist + 1    # exist only when JointNum = 23
            self.LeftFoot = self.LeftAnkle + 1    # exist only when JointNum = 23
            self.RightFoot = self.RightAnkle + 1  # exist only when JointNum = 23
        #std skeleton pose
        self.Joints[0].position = [0, 0, 0]
        self.Joints[1].position = [0, 0, self.Bonelengths[1]]
        self.Joints[self.Neck].position = [0, 0, self.Bonelengths[1] + self.Bonelengths[self.SpineShoulder] + self.Bonelengths[self.Neck]]
        self.Joints[self.Head].position = [0, 0, self.Joints[self.Neck].position + self.Bonelengths[self.Head]]
        self.Joints[4].position = [0, -self.Bonelengths[4], self.Bonelengths[1] + self.Bonelengths[self.SpineShoulder]]
        self.Joints[5].position = self.Joints[4].position - [0, 0, self.Bonelengths[self.LeftElbow]]
        self.Joints[6].position = self.Joints[5].position - [0, 0, self.Bonelengths[self.LeftWaist]]
        self.Joints[self.SpineShoulder].position = self.Joints[1].position + [0, 0, self.Bonelengths[self.SpineShoulder]]
        self.Joints[self.RightShoulder].position = self.Joints[self.SpineShoulder].position + [0, self.Bonelengths[self.RightShoulder], 0]
        self.Joints[self.RightElbow].position = self.Joints[self.RightShoulder].position - [0, 0, self.Bonelengths[self.RightElbow]]
        self.Joints[self.RightWaist].position = self.Joints[self.RightElbow].position - [0, 0, self.Bonelengths[self.RightWaist]]
        self.Joints[self.LeftHip].position = [0.259 * self.Bonelengths[self.LeftHip], -0.966 * self.Bonelengths[self.LeftHip], 0]
        self.Joints[self.RightHip].position = [0.259 * self.Bonelengths[self.LeftHip], 0.966 * self.Bonelengths[self.LeftHip], 0]
        self.Joints[self.LeftKnee].position = self.Joints[self.LeftHip].position - [0, 0, -1.15 * self.Bonelengths[self.LeftKnee]]
        self.Joints[self.LeftAnkle].position = self.Joints[self.LeftKnee].position - [0, 0, -self.Bonelengths[self.LeftAnkle]]
        self.Joints[self.RightKnee].position = self.Joints[self.RightHip].position - [0, 0, -1.15 * self.Bonelengths[self.RightKnee]]
        self.Joints[self.RightAnkle].position = self.Joints[self.RightKnee].position - [0, 0, -self.Bonelengths[self.RightAnkle]]
        if JointNum == 23:
            self.Joints[self.LeftHand].position = self.Joints[self.LeftWaist].position - [0, 0, self.Bonelengths[self.LeftHand]]
            self.Joints[self.RightHand].position = self.Joints[self.RightWaist].position - [0, 0, self.Bonelengths[self.RightHand]]
            self.Joints[self.LeftFoot].position = self.Joints[self.LeftAnkle].position + [self.Bonelengths[self.LeftFoot], 0, 0]
            self.Joints[self.RightFoot].position = self.Joints[self.RightAnkle].position + [self.Bonelengths[self.RightFoot], 0, 0]
        # Joints motion limitation
        self.Joints[1].limit_inf = [-30, -30, -30] # degree
        self.Joints[1].limit_sup = [30, 90, 30]
        self.Joints[self.SpineShoulder].limit_inf = [-45, 0, 0]
        self.Joints[self.SpineShoulder].limit_sup = [45, 0, 0]
        self.Joints[self.SpineShoulderLeft].limit_inf = [-30, 0, -15]
        self.Joints[self.SpineShoulderLeft].limit_sup = [30, 0, 15]
        self.Joints[self.SpineShoulderRight].limit_inf = [-30, 0, -15]
        self.Joints[self.SpineShoulderRight].limit_sup = [30, 0, 15]
        self.Joints[self.Neck].limit_inf = [0, -45, -80]
        self.Joints[self.Neck].limit_sup = [0, 45, 80]
        self.Joints[self.LeftShoulder].limit_inf = [-180, -170, -70]
        self.Joints[self.LeftShoulder].limit_sup = [40, 40, 70]
        self.Joints[self.LeftElbow].limit_inf = [0, -140, -90]
        self.Joints[self.LeftElbow].limit_sup = [0, 10, 90]
        self.Joints[self.LeftWaist].limit_inf = [-40, -60, 60]
        self.Joints[self.LeftWaist].limit_sup = [30, 60, 0]
        self.Joints[self.RightShoulder].limit_inf = [-40, -170, -70]
        self.Joints[self.RightShoulder].limit_sup = [180, 40, 70]
        self.Joints[self.RightElbow].limit_inf = [0, -140, -90]
        self.Joints[self.RightElbow].limit_sup = [0, 10, 90]
        self.Joints[self.RightWaist].limit_inf = [-30, -60, 0]
        self.Joints[self.RightWaist].limit_sup = [40, 60, 0]
        self.Joints[self.LeftHip].limit_inf = [-45, -140, -40]
        self.Joints[self.LeftHip].limit_sup = [30, 15, 50]
        self.Joints[self.LeftKnee].limit_inf = [0, -10, 0]
        self.Joints[self.LeftKnee].limit_sup = [0, 170, 0]
        self.Joints[self.LeftAnkle].limit_inf = [-10, -30, -35]
        self.Joints[self.LeftAnkle].limit_sup = [10, 50, 30]
        self.Joints[self.RightHip].limit_inf = [-30, -140, -50]
        self.Joints[self.RightHip].limit_sup = [45, 15, 40]
        self.Joints[self.RightKnee].limit_inf = [0, -10, 0]
        self.Joints[self.RightKnee].limit_sup = [0, 170, 0]
        self.Joints[self.RightAnkle].limit_inf = [-10, -30, -30]
        self.Joints[self.RightAnkle].limit_sup = [10, 50, 35]



    def findparent(self, joint_label):
        if joint_label == 0:
            return 0
        if joint_label == self.LeftHip:
            return 0
        if joint_label == self.RightHip:
            return 0
        if joint_label == self.SpineShoulder:
            return 1
        if joint_label == self.SpineShoulderLeft:
            return 1
        if joint_label == self.SpineShoulderRight:
            return 1
        if joint_label == self.Neck:
            return self.SpineShoulder
        if joint_label == self.LeftShoulder:
            return self.SpineShoulderLeft
        if joint_label == self.RightShoulder:
            return self.SpineShoulderRight
        elif joint_label > self.SpineShoulderRight:
            raise ("Joint label larger than the joint number!")
        else:
            return joint_label-1

    def cal_bonelength(self, joint):
        parent = self.findparent(joint.label)
        bonelength = np.norm(joint.position - self.Joints[parent].position)
        return bonelength

def generate_poses(JointNum, Bonelengths, type='valid', corruptedJoints=0):
    """
    Generate a human pose based on the basic human structure.
    :param JointNum: joint number in the defined skeleton, 19 means hands and feet are not included, 23 means hands
                      and feet are included.
    :param Bonelengths: bone lengths for initializing the skeleton
    :param type: whether to generate a valid human pose or invalid human pose. Default is valid
    :param corruptedJoints: if type='invalid', how many joints are going to corrupted must be given.
    :return: a human pose represented by the positions of joints
    """
    skeleton = HumanSkeleton(JointNum, Bonelengths)
    if type == 'invalid':
        crpt_Joints = np.random.randint(0, JointNum, corruptedJoints)

    def generate_randomOrientation(Joint, type='valid'):
        if type == 'valid':
            high = Joint.limit_sup
            low = Joint.limit_inf
            if high == low:
                high = high + 1
                low = low - 1
        else:
            a = np.random.randint(2)
            if a == 0:
                high = Joint.limit_inf
                low = -360 + Joint.limit_sup
            else:
                high = 360 + Joint.limit_inf
                low = Joint.limit_sup
        orientation = np.random.uniform(low, high, 3)
        return orientation

    for i in range(JointNum):
        if i in crpt_Joints:
            skeleton.Joints[i].orientation = generate_randomOrientation(skeleton.Joints[i], type='invalid')
        skeleton.Joints[i].orientation = generate_randomOrientation(skeleton.Joints[i], type='valid')
    # calculating rotation matrix
    R = [None] * JointNum
    for i in range(JointNum):
        # the rotation matrix is the final rotation matrix multiplied in a kinematics chain and can be used
        # directly for calculating the position of the joint
        # rotation 0 represents the general rotation of human pose for all joints
        R[i] = rotation_combination(skeleton.Joints[i].orientation) * R[skeleton.findparent(skeleton.Joints[i].label)]

# print(rotation_combination([180, 180, 90]))
# print(random_rotation())