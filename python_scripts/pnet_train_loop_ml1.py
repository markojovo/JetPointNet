'''
##+===========================##
### TIME DISTRIBUTED PointNet ##
##============================##

author: Russell Bate
russellbate@phas.ubc.ca

2023-03-06
eidited by: Jessica Bohm
j2bohm@uwaterloo.ca
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

## Configuration
#======================================

data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2'
output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2/gamma_325_tr_40_tst_5_val_cor_cartesian_stacked_output_weight_points_equal_log_E_round_mean_no_BN'
#output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_3/gamma_charged_325_tr_40_tst_38_val_BN_cartesian_stacked_output_weight_points_equal' 
num_train_files = 325
num_val_files = 40
num_test_files = 5
#test_file = data_dir + "/val/pi0_14_pipm_13_14_len_6000_i_10_cartesian.npz"
clusters_per_file = 9000 # approx since it varries
start_at_epoch = 0

EPOCHS = 10
BATCH_SIZE = 200
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
import sys
#sys.path.append(module_path)
import deep_set_util as dsu
import pnet_models
from pnet_models_updated import PointNet_gamma_no_tnet, part_segmentation_model, part_segmentation_model_propagate_mask, PointNet_omicron, PointNet_delta, PointNet_gamma

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

def batch_normalize(X, Y, epsilon=1e-8):
    not_masked = np.squeeze(Y != -1)
    # for each value x, y, x, E - normalize it
    for i in range(X.shape[2]):
        mean = np.mean(X[:, :, i][not_masked], axis=None)
        std = np.std(X[:, :, i][not_masked], axis=None)
        X[:, :, i][not_masked] = (X[:, :, i][not_masked] - mean) / (std + epsilon)
    return X, Y # expect Y data to be (BATCH_SIZE, NUM_POINTS, 1)


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
                X_padded[i, :cluster.shape[0], 0:] = cluster[:, 0:]
                X_padded[i, :len(cluster[:, 0][cluster[:, 0] != -1]), 0] = np.log10(cluster[:, 0][cluster[:, 0] != -1]) + 1 # scale by log(energy) - mean(log(energy))
                Y_padded[i, :cluster.shape[0], :] = Y[i]
            
            # split into batch_size groups of clusters
            for i in range(1, math.ceil(cluster_data.shape[0]/batch_size)):
                #yield batch_normalize(X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]) # pre normalize
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]
        if not loop_infinite:
            break


## Set Up Data generators
#======================================

train_output_dir = data_dir + '/train2/'
val_output_dir = data_dir + '/val2/'
test_output_dir = data_dir + '/test2/'

train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_output_dir+'*.npz'))[:num_test_files]

num_batches_train = (len(train_files) * clusters_per_file) / BATCH_SIZE 
num_batches_val = (len(val_files) * clusters_per_file) / BATCH_SIZE
num_batches_test = (len(test_files) * clusters_per_file) / BATCH_SIZE


# load the max number of points (N) - saved to data dir
with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())
N = 935

train_generator = batched_data_generator(train_files, BATCH_SIZE, N)
val_generator = batched_data_generator(val_files, BATCH_SIZE, N)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False)


## Compile Model
#======================================

pnet = PointNet_gamma(shape=(N, 4), # 7 feat w lawrence data
    name='PointNet_gamma')

#pnet = part_segmentation_model_propagate_mask(N, 1)

def loss_func(y_true, y_pred):
    # find out which timesteps in `y_true` are not the padding character '#'
    mask = K.equal(y_true, -1)
    mask = 1 - K.cast(mask, K.floatx())

    # multiply categorical_crossentropy with the mask
    loss = K.square(y_true - y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    out = K.sum(loss, axis=1) / K.sum(mask, axis=1)
    
    return out

def default_loss_func(y_true, y_pred):
    # find out which timesteps in `y_true` are not the padding character '#'
    mask = K.equal(y_true, -1)
    mask = 1 - K.cast(mask, K.floatx())

    # multiply categorical_crossentropy with the mask
    loss = K.square(y_true - y_pred) * mask
    
    return loss # the default reduction will be applied

def pointwise_loss(y_true, y_pred):
    mask = (y_pred != -1)
    return K.mse_loss(y_pred[mask], y_true[mask])

pnet.compile(loss=default_loss_func, optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))
pnet.summary()
print()
print()

# if resuming training load saved weights
if start_at_epoch:
    pnet.load_weights(output_dir + "/weights/weights_" + str(start_at_epoch - 1) + ".h5")

## Set up callbacks
#======================================

# mkdirs if not present
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_dir + "/weights"):
    os.makedirs(output_dir + "/weights")
if not os.path.exists(output_dir + "/tests"):
    os.makedirs(output_dir + "/tests")


# save loss after each batch
batch_end_loss = list()
class SaveBatchLoss(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        batch_end_loss.append(logs['loss'])

# save preds, model weights, and train/val loss after each epoch
class SaveEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # save preds on new test file
        per_epoch_test_generator = batched_data_generator(test_files[:5], BATCH_SIZE, N, loop_infinite=False)

        predictions = []
        labels = []
        for X_test, Y_test in per_epoch_test_generator:
            predictions.extend(pnet.predict(X_test))
            labels.extend(Y_test)
        np.save(output_dir + "/tests/preds_5_files_" + str(start_at_epoch + epoch) + ".npy", predictions)
        if epoch == 0:
            # save the labels for 5 test files
            np.save(output_dir + "/tests/labels_5_files.npy", labels)

        # save model weights
        pnet.save_weights(output_dir + "/weights/weights_" + str(start_at_epoch + epoch) + ".h5")
        # save loss
        with open(output_dir + "/log_loss.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch , logs["loss"], logs["val_loss"]])


## Train Model
#======================================

t0 = cput()

history = pnet.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        verbose=1,
                        steps_per_epoch=num_batches_train,
                        validation_steps=num_batches_val,
                        callbacks=[SaveBatchLoss(), SaveEpoch()])

t1 = cput()

print()
print()
print('Time to train: {} (s)'.format(t1-t0))
print('{} (min)'.format((t1-t0)/60))
print('{} (hour)'.format((t1-t0)/3600))
print()

## Predictions
#======================================
"""
print("start testing")
t0 = cput()
predictions = []
labels = []

for X_test, Y_test in test_generator:
    predictions.extend(pnet.predict(X_test))
    labels.extend(Y_test)
"""
# save train history - loss after each batch, predictions/labels on test data set, train/val loss after each epoch
np.savez(output_dir + "/history_epochs_" + str(start_at_epoch) + "_to_" + str(start_at_epoch + EPOCHS) + ".npz", loss=batch_end_loss, history=history.history) #preds=predictions, labels=labels,)

t1 = cput()
print('Time to make predictions: {} (s)'.format(t1-t0))
print()


