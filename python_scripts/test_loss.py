import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

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


data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_2'
output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_1/gamma_1_tr_1_tst_1_val_1_epoch_BN_cartesian_test_loss' 
test_file = data_dir + "/val/pi0_13_pipm_11_12_len_6000_i_1_cartesian.npz"

LEARNING_RATE = 1e-2

with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())

# Load the test file data
# old per epoch test - one file
test_data = np.load(test_file, allow_pickle=True)
X_test = test_data["X"]
Y_test = test_data["Y"]
# pad X data to have y dimension of max_num_points
X_test_padded = np.negative(np.ones((X_test.shape[0], N, X_test.shape[2])))
Y_test_padded = np.negative(np.ones((X_test.shape[0], N, 1)))
for i, cluster in enumerate(X_test):
    X_test_padded[i, :cluster.shape[0], :] = cluster
    Y_test_padded[i, :cluster.shape[0], :] = Y_test[i]

k_X = K.constant(X_test_padded)
k_Y = K.constant(Y_test_padded)

pnet = PointNet_gamma(shape=(N, 4),
    name='PointNet_gamma')

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

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def mask(point_mse, y_true):
    y_true = tf.squeeze(y_true)
    mask = K.equal(y_true, -1)
    mask = 1 - K.cast(mask, K.floatx())

    return point_mse*mask

def mean_over_points(masked_point_mse):
    return K.mean(masked_point_mse, axis=None)

def mean_over_clus(masked_point_mse):
    return K.mean(K.mean(masked_point_mse, axis=1))

def loss_b(y_true, y_pred):
    return mean_over_points(mask(mse(y_true, y_pred), y_true))

def loss_c(y_true, y_pred):
    return mean_over_clus(mask(mse(y_true, y_pred), y_true))

pnet.compile(loss=default_loss_func, optimizer=keras.optimizers.Adam(
    learning_rate=LEARNING_RATE))

pnet.load_weights(output_dir + "/weights/weights_0.h5")

print("model evaluate (all default mse + mask + reduce):", pnet.evaluate(X_test_padded, Y_test_padded, batch_size=100))
preds = pnet.predict(X_test_padded)
print("pred shape:", preds.shape)

loss_b = loss_b(k_Y, K.constant(preds)).eval(session=tf.compat.v1.Session())
loss_c = loss_c(k_Y, K.constant(preds)).eval(session=tf.compat.v1.Session())

print("loss_b:", loss_b)
print("loss_c:", loss_c)