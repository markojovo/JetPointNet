import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import math
import numpy as np
import glob
import csv
import sys
sys.path.append('/home/mjovanovic/Work/PointNet_Segmentation')
from utils.pnet_models import pnet_part_seg_no_tnets, pnet_part_seg
import awkward as ak

# for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import genfromtxt


# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# disable eager execution with tensorflow (since can't execute lambda functions with eager execution)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# DATA AND OUTPUT DIRS
data_dir = '/data/mjovanovic/cell_particle_deposit_learning/delta_train' # parent directory that holds train, val, and test files
train_data_dir = data_dir + '/train_2_tracks/'
val_data_dir = data_dir + '/val_2_tracks/'
test_data_dir = data_dir + '/test_2_tracks/'

output_dir = "/data/mjovanovic/cell_particle_deposit_learning/delta_train/tr_50_val_5_tst_5_delta_2_tracks_eventwise_loss_lr_1e-2_BS_100_no_tnets_add_min_dist" # save model and predictions to this dir
max_points_file = '/max_points_2_tracks.txt' # load the max number of points in a sample

num_train_files = 50
num_val_files = 5
num_test_files = 5
events_per_file = 2200 # approx number of samples in a file
start_at_epoch = 0 # to load a pre trained model from this epoch

EPOCHS = 100
BATCH_SIZE = 80
LEARNING_RATE = 1e-2
add_min_dist = True

num_feat = 7
num_classes = 4

num_tracks = 2


# VALIDATE ONLY (load a trained model and save predictions)
validate_only = False
if validate_only:
    data_dir = '/data/mjovanovic/cell_particle_deposit_learning/delta_train/'
    model_dir = '/data/mjovanovic/cell_particle_deposit_learning/rho_train/'
    model = "tr_25_val_3_tst_5_rho_2_class_lr_1e-2_BS_100_no_tnets_add_min_dist"
    test_set = "test_1_track"
    output_dir = model_dir + model
    start_at_epoch = 49 # state to load + 1
    BATCH_SIZE = 100
    num_validate_files = 5
    validate_files_dir = data_dir + test_set + "/"
    validate_files = np.sort(glob.glob(validate_files_dir+'*.npz'))[:num_validate_files]
    max_points_file = '/max_points_1_track.txt'
    save_preds_file = model_dir + model + "/tests/" + test_set + "_preds_" + str(start_at_epoch - 1) + ".npy"
    save_labels_file = model_dir + model + "/tests/" + test_set + "_labels.npy"


# DATA GENERATORS
def batched_data_generator(file_names, batch_size, max_num_points, loop_infinite=True, add_min_track_dist=False, add_weights=False):
    while True:
        for file in file_names:
            point_net_data = np.load(file, allow_pickle=True)
            event_data = point_net_data['X']
            Y = point_net_data['Y']
            Y_int = Y.astype(np.int32)

            if(add_weights):
                cell_weights = point_net_data['cell_weights']
                #cell_weights = np.ones((event_data.shape[0], max_num_points))

            # for a test overwrite class 2 and 3 to be class 1
            #Y_int[Y_int == 2] = 1 # overwrite pi0 to be rho->pi0
            #Y_int[Y_int == 3] = 1 # overwrite other neutral deposit to be rho->pi0

            identity = np.eye(num_classes)
            one_hot = np.concatenate([identity, np.negative(np.ones((1, num_classes)))])

            # pad X and Y data to have y dimension of max_num_points
            X_padded = np.zeros((event_data.shape[0], max_num_points, num_feat)) # pad with zeros

            if(add_weights):
                Y_padded = np.negative(np.ones(((event_data.shape[0], max_num_points, num_classes + 1)))) # pad with -1s
            else:
                Y_padded = np.negative(np.ones(((event_data.shape[0], max_num_points, num_classes)))) # pad with -1s
            
            for i, event in enumerate(event_data): 
                X_padded[i, :len(event), :5] = event
                Y_padded[i, :len(event), :num_classes] = one_hot[np.squeeze(Y_int[i])]

                if(add_weights):
                    Y_padded[i, :max_num_points, num_classes] = np.zeros(max_num_points)
                    Y_padded[i, :len(cell_weights[i]), num_classes] = cell_weights[i]

                # add a feature of min distance from cell point to track point (for track points this is 0)
                if add_min_track_dist:
                    track_points_idx = np.arange(len(event))[event[:, 4] == 1] # for track of interest
                    dists = np.zeros((len(event), len(track_points_idx)))

                    for j, track_point_idx in enumerate(track_points_idx):
                        dists[:, j] = np.sqrt((event[:, 1] - event[track_point_idx, 1])**2 + (event[:, 2] - event[track_point_idx, 2])**2 + (event[:, 3] - event[track_point_idx, 3])**2)

                    dist_feat_idx = 5
                    if np.any(track_points_idx):
                        min_dists = np.min(dists, axis=1)
                        # recast padding to 0 (padding is where the point is not a track and the label is -1)
                        min_dists[(event[:, 4] == 0) & (Y[i, :, 0] == -1)] = 0#13500
                        X_padded[i, :len(event), dist_feat_idx] = min_dists
                    
                    # only overwrite second track min dist values of 0 if this is an event with a second track
                    if num_tracks == 2 and np.count_nonzero(event[:, 4] == 2) != 0:
                        track_points_idx = np.arange(len(event))[event[:, 4] == 2] # for other track
                        dists = np.zeros((len(event), len(track_points_idx)))

                        for j, track_point_idx in enumerate(track_points_idx):
                            dists[:, j] = np.sqrt((event[:, 1] - event[track_point_idx, 1])**2 + (event[:, 2] - event[track_point_idx, 2])**2 + (event[:, 3] - event[track_point_idx, 3])**2)

                        dist_feat_idx = 6
                        if np.any(track_points_idx):
                            min_dists = np.min(dists, axis=1)

                            # recast padding to 0 (padding is where the point is not a track and the label is -1)
                            min_dists[(event[:, 4] == 0) & (Y[i, :, 0] == -1)] = 0

                            # also make sure the events with only one track have 0 for all min dist to second track


                            X_padded[i, :len(event), dist_feat_idx] = min_dists

    
            # split into batch_size groups of events
            for i in range(1, math.ceil(event_data.shape[0]/batch_size)):
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]

        if not loop_infinite:
            break


# LOSS
def weighted_masked_ce_pointwise_loss(y_true, y_pred):
    cell_weights = y_true[:, :, num_classes] # the fifth element of every cells class labels is the weight
    y_true = y_true[:, :, :num_classes] # cut y_true to only be the 4 classes
    cell_ce = K.categorical_crossentropy(y_pred, y_true, axis=2)
    return K.sum(cell_ce * cell_weights, axis=None) / K.sum(cell_weights, axis=None)

# NOTE: unfinished function
def eventwise_masked_ce_pointwise_loss(y_true, y_pred):

    # Create a binary mask indicating whether each element is non-zero
    mask = tf.reduce_all(tf.equal(y_true, -1), axis=2)
    
    # Count the number of non-zero cells along the second axis
    num_non_zero_cells = tf.reduce_sum(tf.cast(mask != 0, dtype=tf.float32), axis=2, keepdims=True)
    
    # Avoid division by zero by setting the denominator to 1 where it's zero
    divisor = tf.where(tf.not_equal(num_non_zero_cells, 0), num_non_zero_cells, 1)
    
    # Compute the weights using the reciprocal of the divisor
    weights = 1 / tf.cast(divisor, dtype=tf.float32)
    
    # Apply the binary mask to set weights to zero where the mask is zero
    weights *= tf.cast(non_zero_mask, dtype=tf.float32)
    
    # Apply mask to y_true for categorical cross-entropy
    masked_y_true = tf.where(mask, -1, y_true)
    
    cell_ce = K.categorical_crossentropy(masked_y_true, y_pred, axis=2)
    
    return K.sum(cell_ce * weights, axis=None) / K.sum(weights, axis=None)


def masked_ce_pointwise_loss(y_true, y_pred):
    mask = tf.reduce_all(tf.equal(y_true, -1), axis=2)
    cell_ce = K.categorical_crossentropy(y_pred, y_true, axis=2)
    return K.sum(tf.where(mask, tf.zeros_like(cell_ce), cell_ce)) / K.sum(tf.cast(tf.logical_not(mask), tf.float32), axis=None)

def masked_bce_pointwise_loss(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:,:,0], -1)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    return K.sum(K.binary_crossentropy(tf.multiply(y_pred, mask), tf.multiply(y_true, mask)), axis=None) / K.sum(mask, axis=None)


# SETUP TRAIN, VAL, AND TEST GENERATORS
train_files = np.sort(glob.glob(train_data_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_data_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_data_dir+'*.npz'))[:num_test_files]


num_batches_train = (len(train_files) * events_per_file) / BATCH_SIZE 
num_batches_val = (len(val_files) * events_per_file) / BATCH_SIZE
num_batches_test = (len(test_files) * events_per_file) / BATCH_SIZE

# load the max number of points (N) - saved to data dir
with open(data_dir + max_points_file) as f:
    N = int(f.readline())

train_generator = batched_data_generator(train_files, BATCH_SIZE, N, add_min_track_dist=add_min_dist)
val_generator = batched_data_generator(val_files, BATCH_SIZE, N, add_min_track_dist=add_min_dist)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=add_min_dist)


# COMPILE MODEL
model = pnet_part_seg_no_tnets(N, num_feat, num_classes)

# if resuming training load saved weights
if start_at_epoch:
    model.load_weights(output_dir + "/weights/weights_" + str(start_at_epoch - 1) + ".h5")

if validate_only:
    validate_gernerator = batched_data_generator(validate_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=add_min_dist)
    predictions = []
    labels = []
    for X_test, Y_test in validate_gernerator:
        print("preds for file")
        predictions.extend(model.predict(X_test))
        labels.extend(Y_test)

    np.save(save_preds_file, predictions)

    if save_labels_file:
        np.save(save_labels_file, labels)

    print("Done getting preds")
    assert(False)

model.compile(loss=eventwise_masked_ce_pointwise_loss, optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
model.summary()



# CALLBACKS
# make directories if not present
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir + "/weights"):
    os.makedirs(output_dir + "/weights")
if not os.path.exists(output_dir + "/tests"):
    os.makedirs(output_dir + "/tests")

# save preds, model weights, and train/val loss after each epoch
class SaveEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # save preds on new test file
        per_epoch_test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=add_min_dist)

        predictions = []
        labels = []
        for X_test, Y_test in per_epoch_test_generator:
            predictions.extend(model.predict(X_test))
            labels.extend(Y_test)
        np.save(output_dir + "/tests/preds_" + str(start_at_epoch + epoch) + ".npy", predictions)
        if epoch == 0:
            # save the labels for up to 5 test files
            np.save(output_dir + "/tests/labels.npy", labels)

        # save model weights
        model.save_weights(output_dir + "/weights/weights_" + str(start_at_epoch + epoch) + ".h5")
        # save loss
        with open(output_dir + "/log_loss.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch , logs["loss"], logs["val_loss"]])



history = model.fit(train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1,
    steps_per_epoch=num_batches_train,
    validation_steps=num_batches_val,
    callbacks=[SaveEpoch()])