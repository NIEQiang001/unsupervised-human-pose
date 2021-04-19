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
from optparse import OptionParser

sys.path.append(r'../models/')
import SeBiReNet_AE as JointNet
import geometry_pose as gp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sys.setrecursionlimit(100000)
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./SeBi_autoencoder_models/',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=64,
                    help='The number of images per batch.')

Joint_num = 17   # joints
n_hidden = 32    # hidden layer num of features
n_classes = 10
_WEIGHT_DECAY = 1e-4
orthogonal = 5
perceptweights = 1
epsilon = 1e-5
Num_samples = {'train': 10500*5 , 'test': 2800*5}
data_mean = {'train': np.array([-4.908, 74.542, -91.972]), 'test': None}
data_std = {'train': np.array([199.715, 456.308, 98.219]), 'test': None}


def loadJasondata(filepath):
    assert os.path.exists(filepath), (
        'Can not find data at given directory!!')
    with open(filepath) as f:
        data = json.load(f)
    return data

def mean_std(pose_array):
    mean_pos = np.zeros([3])
    std_pos = np.zeros([3])
    mean_pos[0] = np.mean(pose_array[:, :, 0])
    mean_pos[1] = np.mean(pose_array[:, :, 1])
    mean_pos[2] = np.mean(pose_array[:, :, 2])
    std_pos[0] = np.std(pose_array[:, :, 0])
    std_pos[1] = np.std(pose_array[:, :, 1])
    std_pos[2] = np.std(pose_array[:, :, 2])
    return mean_pos, std_pos

def draw_skeleton(action, R_action):
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

def Norm(train_pose, mean, stddev):
    Norm_pos = np.zeros([train_pose.shape[0], 17, 3])
    for i in range(train_pose.shape[0]):
        for j in range(17):
            Norm_pos[i, j, 0] = (train_pose[i, j, 0] - mean[0])/stddev[0]
            Norm_pos[i, j, 1] = (train_pose[i, j, 1] - mean[1])/stddev[1]
            Norm_pos[i, j, 2] = (train_pose[i, j, 2] - mean[2])/stddev[2]
    return Norm_pos

def DeNorm(normpos):
    """The input pos must be normalized position."""
    # if is_training == True:
    if data_mean['train'].any() == None:
        raise ValueError("Pls. calculate the mean position of training data first!")
        return
    if data_std['train'].any() == None:
        raise ValueError("Pls. calculate the standard deviation of training data first!")
        return
    denormPos = normpos * data_std['train'] + data_mean['train']
    return denormPos

def input_fn(is_training, num_epochs, batch_size):
    dataPath = {
        'train': ['../data/random_remove/remove1/train/APE_train.json',
                  '../data/random_remove/remove2/train/APE_train.json',
                  '../data/random_remove/remove3/train/APE_train.json',
                  '../data/random_remove/remove4/train/APE_train.json',
                  '../data/random_remove/remove5/train/APE_train.json'],
        'train_gt': ['../data/random_remove/remove1/train/APE_train_gt.json',
                     '../data/random_remove/remove1/train/APE_train_gt.json',
                     '../data/random_remove/remove3/train/APE_train_gt.json',
                     '../data/random_remove/remove4/train/APE_train_gt.json',
                     '../data/random_remove/remove5/train/APE_train_gt.json'
                     ],
        'test': ['../data/random_remove/remove1/test/APE_test.json',
                 '../data/random_remove/remove2/test/APE_test.json',
                 '../data/random_remove/remove3/test/APE_test.json',
                 '../data/random_remove/remove4/test/APE_test.json',
                 '../data/random_remove/remove5/test/APE_test.json'
                 ],
        'test_gt': ['../data/random_remove/remove1/test/APE_test_gt.json',
                    '../data/random_remove/remove2/test/APE_test_gt.json',
                    '../data/random_remove/remove3/test/APE_test_gt.json',
                    '../data/random_remove/remove4/test/APE_test_gt.json',
                    '../data/random_remove/remove5/test/APE_test_gt.json']
    }

    APE_train_load = []
    APE_train_gt_load = []
    APE_test_load = []
    APE_test_gt_load = []
    for i in range(5):
        train_load = np.reshape(np.asarray(loadJasondata(dataPath['train'][i])), [-1, 17, 3])
        train_gt_load = np.reshape(np.asarray(loadJasondata(dataPath['train_gt'][i])), [-1, 17, 3])
        test_load = np.reshape(np.asarray(loadJasondata(dataPath['test'][i])), [-1, 17, 3])
        test_gt_load = np.reshape(np.asarray(loadJasondata(dataPath['test_gt'][i])), [-1, 17, 3])
        APE_train_load.append(train_load)
        APE_train_gt_load.append(train_gt_load)
        APE_test_load.append(test_load)
        APE_test_gt_load.append(test_gt_load)

    APE_train = np.concatenate(APE_train_load, axis=0)
    APE_train_gt = np.concatenate(APE_train_gt_load, axis=0)
    APE_test = np.concatenate(APE_test_load, axis=0)
    APE_test_gt = np.concatenate(APE_test_gt_load, axis=0)

    # convert the joint coordinates to relative coordinates
    APE_train_relative = np.transpose(np.transpose(APE_train, (1, 0, 2)) - APE_train_gt[:, 0, :], (1, 0, 2))
    APE_train_gt_relative = np.transpose(np.transpose(APE_train_gt, (1, 0, 2)) - APE_train_gt[:, 0, :], (1, 0, 2))
    APE_test_relative = np.transpose(np.transpose(APE_test, (1, 0, 2)) - APE_test_gt[:, 0, :], (1, 0, 2))
    APE_test_gt_relative = np.transpose(np.transpose(APE_test_gt, (1, 0, 2)) - APE_test_gt[:, 0, :], (1, 0, 2))
    # root_position['test'] = APE_test_load[:, 0, :]
    # gt_train_mean, gt_train_stddev = mean_std(APE_train_gt_relative)
    gt_train_mean = data_mean['train']
    gt_train_stddev = data_std['train']
    APE_train = Norm(APE_train_relative, gt_train_mean, gt_train_stddev)
    APE_train_gt = Norm(APE_train_gt_relative, gt_train_mean, gt_train_stddev)
    APE_test_gt = Norm(APE_test_gt_relative, gt_train_mean, gt_train_stddev)
    APE_test = Norm(APE_test_relative, gt_train_mean, gt_train_stddev)

    if is_training == True:
        pos = APE_train
        pos_gt = APE_train_gt
        # draw_skeleton(test_pos, R_pose)
    else:
        pos = APE_test
        pos_gt = APE_test_gt
    dataset = tf.data.Dataset.from_tensor_slices((pos, pos_gt))
    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance. Because the scores
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(Num_samples['train'])
    dataset = dataset.prefetch(2 * batch_size)
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    # features, gtpos, dropout_labels = iterator.get_next()
    pos, gtpos = iterator.get_next()
    return pos, gtpos

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

def Cal_Bonelengths(pos):
    # for structure with 17 joints
    # input pos must be a tensor with a shape of jointNum * batchsize * 3
    Bone_lens = [None] * (Joint_num-1)
    for i in range(Joint_num-1):
        Bone_lens[i] = tf.norm(pos[i+1] - pos[findParentJoint(i+1)], axis=1)
        # Bone_lens[i] = Bone_lens[i] + epsilon
    return Bone_lens

def model_fn(features, labels, mode, params):
    pos = features
    # generate random rotated pos
    rand_rotationM = gp.random_rotation()
    R_pos = tf.matmul(tf.reshape(pos, [-1, 3]), rand_rotationM)
    R_pos = tf.reshape(R_pos, [-1, Joint_num, 3])
    R_pos_gt = tf.matmul(tf.reshape(labels, [-1, 3]), rand_rotationM)
    R_pos_gt = tf.reshape(R_pos_gt, [-1, Joint_num, 3])
    pos_gt = tf.cast(labels, dtype=tf.float32)
    pos_gt = tf.unstack(pos_gt, Joint_num, 1)
    R_pos_gt = tf.cast(R_pos_gt, dtype=tf.float32)
    R_pos_gt = tf.unstack(R_pos_gt, Joint_num, 1)

    # stop calculating the gradient about ground truth position
    pos_gt = tf.stop_gradient(pos_gt, name="gtpos_stop_gradient")
    R_pos_gt = tf.stop_gradient(R_pos_gt, name="Rgtpos_stop_gradient")
    with tf.variable_scope('skautoencoder_rot') as vars_scope:
        outputs_encoder = JointNet.SkeletonAutoEncoder_encoder(pos, FLAGS.batch_size, n_hidden, Joint_num)
        outputs_decoder = JointNet.SkeletonAutoEncoder_decoder(outputs_encoder[0], FLAGS.batch_size, n_hidden, Joint_num)
        vars_scope.reuse_variables()
        R_outputs_encoder = JointNet.SkeletonAutoEncoder_encoder(R_pos, FLAGS.batch_size, n_hidden, Joint_num)
        R_outputs_decoder = JointNet.SkeletonAutoEncoder_decoder(R_outputs_encoder[0], FLAGS.batch_size, n_hidden, Joint_num)
        # cross_pose
        pos_vi = outputs_encoder[1]
        pos_v = outputs_encoder[2]
        R_pos_vi = R_outputs_encoder[1]
        R_pos_v = R_outputs_encoder[2]
        RO_pose_feature = tf.matmul(pos_vi, R_pos_v)
        OR_pose_feature = tf.matmul(R_pos_vi, pos_v)
        RO_pose_regression = JointNet.SkeletonAutoEncoder_decoder(RO_pose_feature, FLAGS.batch_size, n_hidden, Joint_num)
        OR_pose_regressioin = JointNet.SkeletonAutoEncoder_decoder(OR_pose_feature, FLAGS.batch_size, n_hidden, Joint_num)

    pred_pos = tf.unstack(outputs_decoder, Joint_num, 1)
    pred_R_pos = tf.unstack(R_outputs_decoder, Joint_num, 1)
    RO_pose_regression = tf.unstack(RO_pose_regression, Joint_num, 1)
    OR_pose_regressioin = tf.unstack(OR_pose_regressioin, Joint_num, 1)

    pred_pos_denorm = [DeNorm(pred_pos[i]) for i in range(Joint_num)]
    # pred_R_pos_denorm = [DeNorm(pred_R_pos[i]) for i in range(Joint_num)]

    predictions = {
        'predicted_pos': tf.transpose(tf.convert_to_tensor(pred_pos_denorm), [1, 0, 2])
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    def posloss(pred_pos, pos_gt):
        posloss = 0
        for i in range(Joint_num):
            regression_error = tf.nn.l2_loss(tf.subtract(pred_pos[i], pos_gt[i]))   # l2_loss is (x^2 + y^2 + z^2)/2
            posloss = posloss + regression_error   # regression_error_fw
        return posloss

    def boneloss(pred_pos, pos_gt):
        pred_boneLens = tf.convert_to_tensor(Cal_Bonelengths(pred_pos))
        gt_boneLens = tf.convert_to_tensor(Cal_Bonelengths(pos_gt))
        boneloss = tf.nn.l2_loss(tf.subtract(pred_boneLens, gt_boneLens))
        return boneloss

    pos_loss = posloss(pred_pos, pos_gt)
    tf.identity(pos_loss, name='pos_loss')
    tf.summary.scalar('pos_loss', pos_loss)
    R_pos_loss = posloss(pred_R_pos, R_pos_gt)
    tf.identity(R_pos_loss, name='R_pos_loss')
    tf.summary.scalar('R_pos_loss', R_pos_loss)
    # cross pos loss
    RO_pos_loss = posloss(RO_pose_regression, R_pos_gt)
    tf.identity(RO_pos_loss, name='RO_pos_loss')
    tf.summary.scalar('RO_pos_loss', RO_pos_loss)
    OR_pos_loss = posloss(OR_pose_regressioin, pos_gt)
    tf.identity(OR_pos_loss, name='OR_pos_loss')
    tf.summary.scalar('OR_pos_loss', OR_pos_loss)
    # calculating bone length loss
    pos_bone_loss = boneloss(pred_pos, pos_gt)
    tf.identity(pos_bone_loss, name='pos_bonelength_loss')
    tf.summary.scalar('pos_bonelength_loss', pos_bone_loss)
    R_pos_bone_loss = boneloss(pred_R_pos, R_pos_gt)
    tf.identity(R_pos_bone_loss, name='R_pos_bonelength_loss')
    tf.summary.scalar('R_pos_bonelength_loss', R_pos_bone_loss)
    # calculate perceptual loss
    perceptual_loss = tf.nn.l2_loss(tf.subtract(pos_vi, R_pos_vi))
    tf.identity(perceptual_loss, name='perceptual_loss')
    tf.summary.scalar('perceptual_loss', perceptual_loss)

    # regulerizer for the view variant feature
    K = pos_v.get_shape()[1].value
    pos_mat_diff = tf.matmul(pos_v, tf.transpose(pos_v, perm=[0, 2, 1]))
    pos_mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    R_pos_mat_diff = tf.matmul(R_pos_v, tf.transpose(R_pos_v, perm=[0, 2, 1]))
    R_pos_mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(pos_mat_diff) + tf.nn.l2_loss(R_pos_mat_diff)
    tf.identity(mat_diff_loss, name='mat_loss')
    tf.summary.scalar('mat_loss', mat_diff_loss)


    # Define loss and optimizer  #rot_esti_loss +\
    loss_op = pos_loss + pos_bone_loss + R_pos_loss + R_pos_bone_loss + 0.01 * RO_pos_loss + 0.01 * OR_pos_loss + \
              perceptweights * perceptual_loss + orthogonal * mat_diff_loss +\
              _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    if mode == tf.estimator.ModeKeys.TRAIN:
        initial_learning_rate = 0.00005
        global_step = tf.train.get_or_create_global_step()
        batches_per_epoch = Num_samples['train'] / params['batch_size']
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [120, 180, 240]]
        values = [initial_learning_rate * decay for decay in [1, 0.5, 0.1, 0.05]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step)
    else:
        train_op = None

    pos_gt_denorm = [DeNorm(pos_gt[i]) for i in range(Joint_num)]
    MPJPE = 0
    for i in range(Joint_num):
        regression_error_denorm = tf.norm(tf.subtract(pred_pos_denorm[i], pos_gt_denorm[i]), axis=1)
        MPJPE = MPJPE + regression_error_denorm
        MPJPE = MPJPE/Joint_num

    MPJPE = tf.metrics.mean(MPJPE)
    metrics = {'accuracy': MPJPE}
    tf.identity(MPJPE[1], name='train_MPJPE')
    tf.summary.scalar('train_MPJPE', MPJPE[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    motionrcg_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={'batch_size': FLAGS.batch_size})

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'pos_loss': 'pos_loss',
            'mat_loss': 'mat_loss',
            'R_pos_loss': 'R_pos_loss',
            'train_MPJPE': 'train_MPJPE',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        motionrcg_classifier.train(
            input_fn=lambda: input_fn(
                True, FLAGS.epochs_per_eval, FLAGS.batch_size),
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = motionrcg_classifier.evaluate(
            input_fn=lambda: input_fn(False, FLAGS.epochs_per_eval, FLAGS.batch_size))
        print(eval_results)
    # prediction, pls. comment the training part and using the following code for prediction of the reconstructed pose
    # predictions = list(motionrcg_classifier.predict(
    #     input_fn=lambda: input_fn(False, FLAGS.epochs_per_eval, FLAGS.batch_size),
    #     checkpoint_path='./SeBi_autoencoder_models4/model.ckpt-472517'))
    # np.save("./Nucla_JsonData/representation_modelV_full_version/with rotloss_without matloss/estimated_rotM.npy",
    #         predictions)


if __name__ == '__main__':
    # main()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
