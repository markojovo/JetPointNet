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
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

## Configuration
#======================================

data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_1'
output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_1/history_gamma_1_tr_1_tst_1_val_5_epoch_BN_cartesian_test' 
num_train_files = 1
num_val_files = 1
num_test_files = 1
test_file = data_dir + "/val/pi0_14_pipm_13_14_len_6000_i_10_cartesian.npz" # TODO: don't use a valuation file
clusters_per_file = 9000 # approx since it varries
start_at_epoch = 5

EPOCHS = 1
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
import sys
#sys.path.append(module_path)
import deep_set_util as dsu
import pnet_models
from pnet_models import segmentation_model, segmentation_model_propagate_mask, PointNet_omicron, PointNet_delta, PointNet_gamma

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
            X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2]))
            Y_padded = np.zeros((cluster_data.shape[0], max_num_points, 1))
            for i, cluster in enumerate(cluster_data):
                X_padded[i, :cluster.shape[0], :] = cluster
                Y_padded[i, :cluster.shape[0], :] = Y[i]
            
            # split into batch_size groups of clusters
            for i in range(1, math.ceil(cluster_data.shape[0]/batch_size)):
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]
        if not loop_infinite:
            break


## Set Up Data generators
#======================================

train_output_dir = data_dir + '/train/'
val_output_dir = data_dir + '/val/'
test_output_dir = data_dir + '/test/'

train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_output_dir+'*.npz'))[:num_test_files]

num_batches_train = (len(train_files) * clusters_per_file) / BATCH_SIZE 
num_batches_val = (len(val_files) * clusters_per_file) / BATCH_SIZE
num_batches_test = (len(test_files) * clusters_per_file) / BATCH_SIZE


# load the max number of points (N) - saved to val and train data folders
with open(train_output_dir + 'max_points.txt') as f:
    N = int(f.readline())
with open(val_output_dir + 'max_points.txt') as f:
    N = max(int(f.readline()), N)

train_generator = batched_data_generator(train_files, BATCH_SIZE, N)
val_generator = batched_data_generator(val_files, BATCH_SIZE, N)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False)


## Compile Model
#======================================

pnet = PointNet_gamma(shape=(N, 4),
    name='PointNet_gamma')

#pnet = segmentation_model(N, 1)

# mse
#def loss_func(y_true, y_pred):
#   return K.mean(K.square(y_pred - y_true))

pnet.compile(loss="mse", optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))
pnet.summary()
print()
print()

# if resuming training load saved weights
if start_at_epoch:
    pnet.load_weights(output_dir + "/weights/weights_" + str(start_at_epoch - 1) + ".h5")

## Set up callbacks
#======================================

# Load the test file data
test_data = np.load(test_file, allow_pickle=True)
X_test = test_data["X"]
Y_test = test_data["Y"]
# pad X data to have y dimension of max_num_points
X_test_padded = np.zeros((X_test.shape[0], N, X_test.shape[2]))
Y_test_padded = np.zeros((X_test.shape[0], N, 1))
for i, cluster in enumerate(X_test):
    X_test_padded[i, :cluster.shape[0], :] = cluster
    Y_test_padded[i, :cluster.shape[0], :] = Y_test[i]

# save the labels for test file
np.save(output_dir + "/tests/labels.npy", Y_test_padded)

# save loss after each batch
batch_end_loss = list()
class SaveBatchLoss(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        batch_end_loss.append(logs['loss'])

# save preds, model weights, and train/val loss after each epoch
class SaveEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # save preds on test file
        preds = pnet.predict(X_test_padded)
        np.save(output_dir + "/tests/preds_" + str(start_at_epoch + epoch) + ".npy", preds)
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
print("start testing")
t0 = cput()
predictions = []
labels = []

for X_test, Y_test in test_generator:
    predictions.extend(pnet.predict(X_test))
    labels.extend(Y_test)

# save train history - loss after each batch, predictions/labels on test data set, train/val loss after each epoch
np.savez(output_dir + "/history_epochs_" + str(start_at_epoch) + "_to_" + str(start_at_epoch + EPOCHS) + ".npz", loss=batch_end_loss, preds=predictions, labels=labels, history=history.history)

t1 = cput()
print('Time to make predictions: {} (s)'.format(t1-t0))
print()


