'''
##+===========================##
### TIME DISTRIBUTED PointNet ##
##============================##

author: Russell Bate
russellbate@phas.ubc.ca
'''

print()
print('CUDA Environment Info')
## Configuration
#======================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2'
output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2/delta_grouped_25_tr_5_val_1_tst_cartesian_cor'
num_train_files = 26
num_val_files = 5
num_test_files = 1
events_per_file = 4000 # approx since it varries
start_at_epoch = 0

EPOCHS = 5
BATCH_SIZE = 200
LEARNING_RATE = 1e-2


print()
print('loading packages..')
## General imports
#======================================
import numpy as np
import pickle
import time as t
from time import perf_counter as cput
import logging
import math
import glob
import csv


## local ML Packages
import deep_set_util as dsu
import pnet_models
from pnet_models import PointNet_gamma, PointNet_delta


## TensorFlow
#======================================
import tensorflow as tf
from tensorflow import keras

## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()

print("TensorFlow version {}".format(tf.__version__))
print("Keras version {}".format(keras.__version__))
print()

"""
## Load Data
#======================================

def batched_data_generator(file_names, batch_size, max_num_points, loop_infinite=True):
    while True:
        for file in file_names:
            point_net_data = np.load(file)
            cluster_data = point_net_data['X']
            Y = point_net_data['Y']

            # pad X data to have y dimension of max_num_points
            X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2])) # pad X data with 0's instead of -1's to have less influence on BN stats??
            Y_padded = np.zeros(((cluster_data.shape[0], max_num_points, 1)))
            for i, cluster in enumerate(cluster_data):
                X_padded[i, :len(cluster[:, 1][cluster[:, 1] != 0]), 1] = cluster[:, 1][cluster[:, 1] != 0]
                X_padded[i, :len(cluster[:, 2][cluster[:, 2] != 0]), 2] = cluster[:, 2][cluster[:, 2] != 0]
                X_padded[i, :len(cluster[:, 3][cluster[:, 3] != 0]), 3] = cluster[:, 3][cluster[:, 3] != 0]
                X_padded[i, :len(cluster[:, 0][cluster[:, 0] != 0]), 0] = np.log10(cluster[:, 0][cluster[:, 0] != 0]) + 1 # scale by log(energy) - mean(log(energy))
                Y_padded[i, :len(Y[i][Y[i] != 0]), 0] = Y[i][Y[i] != 0]
            
            #print("X_padded shape:", X_padded.shape)
            #print("Y_padded shape:", Y_padded.shape)

            #print("X data:", X_padded[1000, 40:43, :])
            #print("Y data:", Y_padded[1000, 40:43, :])

            # split into batch_size groups of clusters
            for i in range(1, math.ceil(cluster_data.shape[0]/batch_size)):
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]
        if not loop_infinite:
            break

## Set Up Data generators
#======================================

train_output_dir = data_dir + '/train_events_cor/'
val_output_dir = data_dir + '/val_events_cor/'
test_output_dir = data_dir + '/test_events_cor/'

train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_output_dir+'*.npz'))[:num_test_files]

num_batches_train = (len(train_files) * events_per_file) / BATCH_SIZE 
num_batches_val = (len(val_files) * events_per_file) / BATCH_SIZE
num_batches_test = (len(test_files) * events_per_file) / BATCH_SIZE


# load the max number of points (N) - saved to data dir
with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())

# hardcode N
#N = 1053

train_generator = batched_data_generator(train_files, BATCH_SIZE, N)
val_generator = batched_data_generator(val_files, BATCH_SIZE, N)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False)

"""

## load dataset
# load the max number of points (N) - saved to data dir
with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())

# first load just one file
train_output_dir = data_dir + '/train_events_cor/'
train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
file_names = train_files
for file in file_names:
    point_net_data = np.load(file)
    cluster_data = point_net_data['X']
    Y = point_net_data['Y']

    # pad X data to have y dimension of max_num_points
    X_padded = np.zeros((cluster_data.shape[0], N, cluster_data.shape[2])) # pad X data with 0's instead of -1's to have less influence on BN stats??
    Y_padded = np.zeros(((cluster_data.shape[0], N, 1)))
    for i, cluster in enumerate(cluster_data):
        X_padded[i, :len(cluster[:, 1][cluster[:, 1] != 0]), 1] = cluster[:, 1][cluster[:, 1] != 0]
        X_padded[i, :len(cluster[:, 2][cluster[:, 2] != 0]), 2] = cluster[:, 2][cluster[:, 2] != 0]
        X_padded[i, :len(cluster[:, 3][cluster[:, 3] != 0]), 3] = cluster[:, 3][cluster[:, 3] != 0]
        X_padded[i, :len(cluster[:, 0][cluster[:, 0] != 0]), 0] = np.log10(cluster[:, 0][cluster[:, 0] != 0]) + 1 # scale by log(energy) - mean(log(energy))
        Y_padded[i, :len(Y[i][Y[i] != 0]), 0] = Y[i][Y[i] != 0]

## Compile Model
#======================================
pnet = PointNet_delta(shape=(N, 4),
    name='PointNet_delta')

pnet.compile(loss='mse', optimizer=keras.optimizers.Adam(
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

"""
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
"""


## Train Model
#======================================
"""
t0 = cput()
history = pnet.fit(train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1,
        steps_per_epoch=num_batches_train,
        validation_steps=num_batches_val,
        callbacks=[SaveBatchLoss(), SaveEpoch()])
t1 = cput()
"""

t0 = cput()
history = pnet.fit(x=X_padded, y=Y_padded,
                           epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           verbose=1)
t1 = cput()

print()
print()
print('Time to train: {} (s)'.format(t1-t0))
print('{} (min)'.format((t1-t0)/60))
print('{} (hour)'.format((t1-t0)/3600))
print()


