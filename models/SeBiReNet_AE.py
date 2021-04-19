"""
The following codes define the SeBiReNet, the encoder and the decoder based on the SeBiReNet.
The tensorflow version we use is 1.14.
Author: Qiang Nie

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


class Node:  # a node in the tree
    def __init__(self, label, unit_num, cell="GRU"):
        self.label = label
        if cell=="RNN":
            self.cell = tf.nn.rnn_cell.RNNCell(unit_num)
        elif cell=="LSTM":
            self.cell = tf.nn.rnn_cell.LSTMCell(unit_num)
        else:
            self.cell = tf.nn.rnn_cell.GRUCell(unit_num)

        self.parent = None  # reference to parent
        self.child = []  # reference to child
        self.childnum = 0  # number of child
        self.state = None
        # true if it's a leaf joint
        self.isLeaf = False
        # true if it's a root joint
        self.isRoot = False


    def addChild(self, child_nodes):
        for _ in range(len(child_nodes)):
            child_nodes[_].parent = self
        self.child.extend(child_nodes)

class JointTree:
    """The information flows from leaf joints to root joint.

        When standard is True, the skeleton is a standard skeleton in which
        the SpineShoulder joint is not splitted to 3 joints. Otherwise, it is
        splitted for convenience of kinematics calculation.

    """

    def __init__(self, unit_num, cell="GRU", sktype="standard"):
        self.unit_num = unit_num
        self.cell = cell
        # define joint nodes
        self.SpineBase = Node(0, unit_num, cell=cell)
        self.HipLeft = Node(10, unit_num, cell=cell)
        self.HipRight = Node(13, unit_num, cell=cell)
        self.SpineMid = Node(1, unit_num, cell=cell)
        self.SpineShoulderMid = Node(16, unit_num, cell=cell)
        self.SpineShoulderLeft = Node(17, unit_num, cell=cell) # for unstandard skeleton
        self.SpineShoulderRight = Node(18, unit_num, cell=cell) # for unstandard skeleton
        self.Neck = Node(2, unit_num, cell=cell)
        self.Head = Node(3, unit_num, cell=cell)
        self.ShoulderLeft = Node(4, unit_num, cell=cell)
        self.ElbowLeft = Node(5, unit_num, cell=cell)
        self.WristLeft = Node(6, unit_num, cell=cell)
        # self.HandLeft = Node(7, unit_num, cell=cell)
        self.ShoulderRight = Node(7, unit_num, cell=cell)
        self.ElbowRight = Node(8, unit_num, cell=cell)
        self.WristRight = Node(9, unit_num, cell=cell)
        # self.HandRight = Node(11, unit_num, cell=cell)
        self.KneeLeft = Node(11, unit_num, cell=cell)
        self.AnkleLeft = Node(12, unit_num, cell=cell)
        # self.FootLeft = Node(15, unit_num, cell=cell)
        self.KneeRight = Node(14, unit_num, cell=cell)
        self.AnkleRight = Node(15, unit_num, cell=cell)
        # self.FootRight = Node(19, unit_num, cell=cell)
        self.Nodes = [self.SpineBase, self.SpineMid, self.Neck, self.Head, self.ShoulderLeft, self.ElbowLeft,
                      self.WristLeft, self.ShoulderRight, self.ElbowRight, self.WristRight, self.HipLeft, self.KneeLeft,
                      self.AnkleLeft, self.HipRight, self.KneeRight, self.AnkleRight, self.SpineShoulderMid]

        # define the tree structure
        self.SpineBase.addChild([self.SpineMid, self.HipLeft, self.HipRight])
        self.SpineBase.isRoot = True
        self.SpineMid.addChild([self.SpineShoulderMid])
        self.SpineShoulderMid.addChild([self.Neck])
        self.Neck.addChild([self.Head])
        self.Head.isLeaf = True
        self.ShoulderLeft.addChild([self.ElbowLeft])
        self.ElbowLeft.addChild([self.WristLeft])
        self.WristLeft.isLeaf = True
        # self.WristLeft.addChild([self.HandLeft])
        # self.HandLeft.isLeaf = True
        self.ShoulderRight.addChild([self.ElbowRight])
        self.ElbowRight.addChild([self.WristRight])
        self.WristRight.isLeaf = True
        # self.WristRight.addChild([self.HandRight])
        # self.HandRight.isLeaf = True
        self.HipLeft.addChild([self.KneeLeft])
        self.KneeLeft.addChild([self.AnkleLeft])
        self.AnkleLeft.isLeaf = True
        # self.AnkleLeft.addChild([self.FootLeft])
        # self.FootLeft.isLeaf = True
        self.HipRight.addChild([self.KneeRight])
        self.KneeRight.addChild([self.AnkleRight])
        self.AnkleRight.isLeaf = True
        # self.AnkleRight.addChild([self.FootRight])
        # self.FootRight.isLeaf = True
        if sktype == "standard":
            self.SpineShoulderMid.addChild([self.ShoulderLeft, self.ShoulderRight])
        else:
            self.SpineShoulderMid.addChild([self.SpineShoulderLeft, self.SpineShoulderRight])
            self.SpineShoulderLeft.addChild([self.ShoulderLeft])
            self.SpineShoulderRight.addChild([self.ShoulderRight])

class SequentialBiRecursiveNN:
    """
        The information flows from forward tree to backward tree or from backward tree to forward tree.
        And hidden states are shared between forward tree and backward tree.
        repeat: recurrent times
        Input: should has a shape of [batchsize, jointNum, 3]

    """
    def __init__(self, hidden_size, batch_size, jointNum_):
        # super(BasicRecursiveNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.jointNum = jointNum_

    def forward(self, root_node_fw, root_node_bw, inputs, repeat=1):
        outputs_fw = [0] * self.jointNum
        outputs_bw = [0] * self.jointNum
        node_states = [None] * self.jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = tf.unstack(inputs, self.jointNum, 1)
        for i in range(repeat):
            final_state = self.recursive_forward(root_node_fw, x, outputs_fw, node_states)
            self.recursive_backward(root_node_bw, x, outputs_bw, node_states)
        return [outputs_bw, outputs_fw], node_states

    def backward(self, root_node_fw, root_node_bw, inputs):
        outputs_fw = [0] * self.jointNum
        outputs_bw = [0] * self.jointNum
        node_states = [None] * self.jointNum
        inputs = tf.cast(inputs, dtype=tf.float32)
        x = tf.unstack(inputs, self.jointNum, 1)
        # print(tf.shape(x))
        self.recursive_backward(root_node_fw, x, outputs_bw, node_states)
        final_state = self.recursive_forward(root_node_bw, x, outputs_fw, node_states)
        return outputs_fw, node_states

    def recursive_forward(self, node, inputs, outputs, node_states):
        # get states from children
        local_batchsize = tf.shape(inputs[0])[0]
        if node_states[node.label] == None:
            node_states[node.label] = node.cell.zero_state(local_batchsize, dtype=tf.float32)
        if len(node.child) == 0:
            output, hstate = node.cell(inputs[node.label], node_states[node.label])
            outputs[node.label] = output
            node_states[node.label] = hstate
            return hstate
        else:
            assert len(node.child) <= 3
            child_states = []
            for idx in range(len(node.child)):
                child_state = self.recursive_forward(node.child[idx], inputs, outputs, node_states)
                child_states.append(child_state)
            if len(child_states) == 1:
                node_inputs = tf.concat([inputs[node.label], child_states[0]], 1)
                output, hstate = node.cell(node_inputs, node_states[node.label])
                outputs[node.label] = output
                node_states[node.label] = hstate
                return hstate
            elif len(child_states) == 2:
                node_inputs = tf.concat([inputs[node.label], child_states[0], child_states[1]], 1)
                output, hstate = node.cell(node_inputs, node_states[node.label])
                outputs[node.label] = output
                node_states[node.label] = hstate
                return hstate
            else:
                node_inputs = tf.concat([inputs[node.label], child_states[0], child_states[1], child_states[2]], 1)
                output, hstate = node.cell(node_inputs, node_states[node.label])
                outputs[node.label] = output
                node_states[node.label] = hstate
                # outputs.append(node_state)
                return hstate

    def recursive_backward(self, node, inputs, outputs, node_states):
        # transmit the parent state to children joints
        local_batchsize = tf.shape(inputs[0])[0]
        if node_states[node.label] == None:
            node_states[node.label] = node.cell.zero_state(local_batchsize, dtype=tf.float32)
        if node.parent == None:
            node_input = inputs[node.label]
            output, hstate = node.cell(node_input, node_states[node.label])
            # node.state = hstate
            outputs[node.label] = output
            node_states[node.label] = hstate
        else:
            parent_state = node_states[node.parent.label]
            node_input = tf.concat([inputs[node.label], parent_state], 1)
            output, hstate = node.cell(node_input, node_states[node.label])
            # node.state = hstate
            outputs[node.label] = output
            node_states[node.label] = hstate
        if len(node.child) > 0:
            for idx in range(len(node.child)):
                self.recursive_backward(node.child[idx], inputs, outputs, node_states)
        return

def SkeletonAutoEncoder_encoder(inputs, batch_size, unitnum, JointNum):
    """ the output is a tuple of recovered pose, view-variant feature, view-invariant feature"""
    with tf.variable_scope('skEncoder_rot'):
        jointTree_fw_en = JointTree(unitnum)
        jointTree_bw_en = JointTree(unitnum)
        basicmodel = SequentialBiRecursiveNN(unitnum, batch_size, JointNum)
        output_tensor, _ = basicmodel.forward(jointTree_fw_en.SpineBase, jointTree_bw_en.SpineBase, inputs, 1)
        # turn the output into a feature representation
        features = tf.transpose(tf.convert_to_tensor(output_tensor[0]), [1, 0, 2])
        features = tf.reshape(features, [-1, unitnum])
        features_vi = tf.layers.dense(features, unitnum / 2)
        features_vi = tf.nn.tanh(features_vi, name='features_vi_1')
        features_v = tf.reshape(features, [-1, JointNum * unitnum])
        features_v = tf.layers.dense(features_v, unitnum * unitnum / 4)
        features_v = tf.reshape(features_v, [-1, int(unitnum/2), int(unitnum/2)])
        features_vi = tf.reshape(features_vi, [-1, JointNum, int(unitnum / 2)])
        features_en = tf.matmul(features_vi, features_v, name='encoder_out')
        return [features_en, features_vi, features_v]

def SkeletonAutoEncoder_decoder(inputs, batch_size, unitnum, JointNum):
    """ the output is a tuple of recovered pose, view-variant feature, view-invariant feature"""
    with tf.variable_scope('skDecoder_rot'):
        inputs = tf.reshape(inputs, [-1, int(unitnum /2)])
        inputs = tf.layers.dense(inputs, unitnum / 2, name='decoded_v_1')
        inputs = tf.nn.tanh(inputs)
        inputs = tf.layers.dense(inputs, unitnum)
        inputs = tf.nn.tanh(inputs)
        inputs = tf.reshape(inputs, [-1, JointNum, unitnum])
        jointTree_fw_de = JointTree(3)
        jointTree_bw_de = JointTree(3)
        basicmodel = SequentialBiRecursiveNN(3, batch_size, JointNum)
        output_tensor, _ = basicmodel.forward(jointTree_fw_de.SpineBase, jointTree_bw_de.SpineBase, inputs, 1)
        output = tf.transpose(tf.convert_to_tensor(output_tensor[0]), [1, 0, 2])
        output = tf.reshape(output, [-1, 3])
        output = tf.layers.dense(output, 3)
        output = tf.reshape(output, [-1, JointNum, 3])
        return output

