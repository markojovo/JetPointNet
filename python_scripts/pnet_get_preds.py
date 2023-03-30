import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

## Configuration
#======================================
train_charged = False

if train_charged:
    data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_3'
    model_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_3/gamma_charged_325_tr_40_tst_38_val_BN_cartesian_stacked_output_weight_points_equal' 
    save_preds_file = '/home/jbohm/start_tf/PointNet_Segmentation/nbs/pi0_14_pipm_13_14_len_6000_i_0_cartesian_preds_train_charged.npy'
else:
    data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2'
    model_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2/gamma_325_tr_40_tst_40_val_10_epoch_BN_cartesian_stacked_output_weight_points_equal' 
    save_preds_file = '/home/jbohm/start_tf/PointNet_Segmentation/nbs/pi0_14_pipm_13_14_len_6000_i_0_cartesian_preds_train_mixed.npy'

test_files = ['/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2/train2/pi0_14_pipm_13_14_len_6000_i_0_cartesian.npz']
save_labels_file = None
load_epoch = 6

BATCH_SIZE = 64
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
from pnet_models import part_segmentation_model, part_segmentation_model_propagate_mask, PointNet_omicron, PointNet_delta, PointNet_gamma

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



# load the max number of points (N) - saved to data dir
with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())


## Compile Model
#======================================
pnet = PointNet_gamma(shape=(N, 4),
    name='PointNet_gamma')


pnet.compile(loss="mse", optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))

pnet.summary()
print()
print()

# if resuming training load saved weights
pnet.load_weights(model_dir + "/weights/weights_" + str(load_epoch) + ".h5")


# GET PREDS
test_generator_one_file = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False)
predictions = []
labels = []
for X_test, Y_test in test_generator_one_file:
    predictions.extend(pnet.predict(X_test))
    labels.extend(Y_test)

np.save(save_preds_file, predictions)

if save_labels_file:
    np.save(save_labels_file, labels)

print("Done getting preds")