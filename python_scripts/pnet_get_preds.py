import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

## Configuration
#======================================
mixed = True

if mixed:
    pi0_file_num = 13
    pipm1_file_num = 11
    pipm2_file_num = 12
    len_file = 6000
    i_low = 0
    i_high = 9
    pion_file_dir = "/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify/test_mixed_thresh_0.787/"
    test_files = [pion_file_dir + "pi0_" + str(pi0_file_num) + "_pipm_" + str(pipm1_file_num) + "_" + str(pipm2_file_num) + "_len_" + str(len_file) + "_i_" + str(i) + "_cartesian_event_grouped.npz" for i in range(i_low, i_high + 1)]

else:
    pi0_file_num = 16
    pipm1_file_num = 14
    pipm2_file_num = 15
    len_file = 6000
    i_low = 20
    i_high = 29
    pion_file_dir = "/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify/test/"
    test_files = [pion_file_dir + "pipm_" + str(pipm1_file_num) + "_" + str(pipm2_file_num) + "_" + str(pi0_file_num) + "_len_" + str(len_file) + "_i_" + str(i) + "_cartesian_event_grouped_charged.npz" for i in range(i_low, i_high + 1)]

data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify'
model_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify/pnet_part_seg_no_tnets_charged_events_thresh_0.787_tr_312_val_78_tst_10_lr_1e-2_BN_timedist' # best charged trained so far
load_epoch = 13 # start at 0

save_preds_file = model_dir + "/tests/mixed_preds_" + str(load_epoch) + ".npy"
save_labels_file = model_dir + "/tests/mixed_labels.npy"

BATCH_SIZE = 75
LEARNING_RATE = 1e-2


## General imports
#======================================
import numpy as np
import pickle
import time as t
from time import perf_counter as cput
import logging
import glob
import math
import csv

## local ML Packages
#sys.path.append(module_path)
from pnet_models_updated import pnet_part_seg_no_tnets, PointNet_gamma_no_tnet, part_segmentation_model, part_segmentation_model_propagate_mask, PointNet_omicron, PointNet_delta, PointNet_gamma

## TensorFlow
#======================================
import tensorflow as tf
from tensorflow import keras
import keras.backend as K


## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()

print("TensorFlow version {}".format(tf.__version__))
print("Keras version {}".format(keras.__version__))
print()


## Load Data
#======================================
def batched_data_generator(file_names, batch_size, max_num_points, loop_infinite=True):
    while True:
        for file in file_names:
            point_net_data = np.load(file)
            cluster_data = point_net_data['X']
            Y = point_net_data['Y']
            print("loaded file")

            # pad X data to have y dimension of max_num_points
            X_padded = np.negative(np.ones((cluster_data.shape[0], max_num_points, cluster_data.shape[2]))) # pad X data with 0's instead of -1's to have less influence on BN stats??
            Y_padded = np.negative(np.ones((cluster_data.shape[0], max_num_points, 1)))
            for i, cluster in enumerate(cluster_data):
                X_padded[i, :cluster.shape[0], :] = cluster
                Y_padded[i, :cluster.shape[0], :] = Y[i]
            
            # split into batch_size groups of clusters
            for i in range(1, math.ceil(cluster_data.shape[0]/batch_size)):
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]
        if not loop_infinite:
            break

CUT_ONE_CLASS_EVENTS = 1
def batched_data_generator_cut(file_names, batch_size, max_num_points, loop_infinite=True):
    while True:
        for file in file_names:
            point_net_data = np.load(file)
            cluster_data = point_net_data['X']
            Y = point_net_data['Y']
            print("loaded file")

            # pad X data to have y dimension of max_num_points
            X_padded = []#np.empty((0, max_num_points, cluster_data.shape[2])) # pad X data with 0's instead of -1's to have less influence on BN stats??
            Y_padded = []#np.empty((0, max_num_points, 1)) # NOTE: update for weighted cells
            clus_idx = 0
            for i, cluster in enumerate(cluster_data):
                frac_em_class = np.sum(Y[i][Y[i] != -1]) / len(Y[i][Y[i] != -1])
                if max(frac_em_class, 1 - frac_em_class) < CUT_ONE_CLASS_EVENTS:
                    X_padded.append(np.zeros((max_num_points, cluster_data.shape[2])))
                    Y_padded.append(np.negative(np.ones((max_num_points, 1))))
                    X_padded[clus_idx][:cluster.shape[0], :] = cluster
                    Y_padded[clus_idx][:cluster.shape[0], :] = Y[i]
                    clus_idx += 1
            X_padded = np.asarray(X_padded)
            Y_padded = np.asarray(Y_padded)
            # split into batch_size groups of clusters
            for i in range(1, math.ceil(X_padded.shape[0]/batch_size)):
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]
        if not loop_infinite:
            break

# load the max number of points (N) - saved to data dir
with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())

# LOSS
def masked_bce_pointwise_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    return K.sum(K.binary_crossentropy(tf.multiply(y_pred, mask),
                tf.multiply(y_true, mask)), axis=None) / K.sum(mask, axis=None)


## Compile Model
#======================================
model = pnet_part_seg_no_tnets(N)
model.compile(loss=masked_bce_pointwise_loss, optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
model.summary()
print()
print()


# if resuming training load saved weights
model.load_weights(model_dir + "/weights/weights_" + str(load_epoch) + ".h5")


# GET PREDS
test_generator_one_file = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False)
predictions = []
labels = []
for X_test, Y_test in test_generator_one_file:
    predictions.extend(model.predict(X_test))
    labels.extend(Y_test)

np.save(save_preds_file, predictions)

if save_labels_file:
    np.save(save_labels_file, labels)

print("Done getting preds")