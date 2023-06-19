import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3" # GPU
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import math
import numpy as np
import glob
import csv
from pnet_models_updated import pnet_part_seg_no_tnets, pnet_part_seg

## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()

MEAN_TRACK_LOG_ENERGY = 2.4


# LOSS
def masked_bce_weighted_pointwise_loss(y_true, y_pred):
    weights = tf.expand_dims(y_true[:,:,1], -1)
    y_true = tf.expand_dims(y_true[:,:,0], -1)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    return K.sum(tf.multiply(tf.multiply(weights, mask), K.binary_crossentropy(tf.multiply(y_pred, mask), tf.multiply(y_true, mask))), axis=None) / K.sum(mask, axis=None)

def masked_bce_pointwise_loss(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:,:,0], -1)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    return K.sum(K.binary_crossentropy(tf.multiply(y_pred, mask), tf.multiply(y_true, mask)), axis=None) / K.sum(mask, axis=None)

# DATA AND OUTPUT DIRS
data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_tracks_cor'
#output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify/pnet_part_seg_no_tnets_charged_events_thresh_0.787_tr_707_val_210_tst_10_lr_1e-6'
output_dir = "/fast_scratch_1/jbohm/train_testing_data/pointnet_train_tracks_cor/pnet_charged_events_1_track_add_min_dist_tr_50_val_5_tst_5_lr_1e-2_masked_BN_SGD"
max_points_file = '/max_points_1_track.txt'

num_train_files = 50 #707
num_val_files = 5 #210
num_test_files = 5 #10
events_per_file = 3000
start_at_epoch = 0 # load start_at_epoch - 1

EPOCHS = 5
BATCH_SIZE = 50
LEARNING_RATE = 1e-2

# VALIDATE ONLY
validate_only = False
if validate_only:
    model = "pnet_charged_events_1_track_add_min_dist_and_energy_ratio_tr_650_val_198_tst_10_lr_1e-2_masked_BN"
    test_set = "test_1_track"
    data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_evaluate_tracks/'
    output_dir = "/fast_scratch_1/jbohm/train_testing_data/pointnet_train_tracks_cor/" + model
    start_at_epoch = 13 # state to load + 1
    BATCH_SIZE = 100
    num_validate_files = 10
    validate_files_dir = data_dir + test_set + "/"
    validate_files = np.sort(glob.glob(validate_files_dir+'*.npz'))[:num_validate_files]
    max_points_file = 'max_points_mixed_1_track.txt'
    save_preds_file = data_dir + model + "_" + test_set + "_preds_" + str(start_at_epoch - 1) + ".npy"
    save_labels_file = data_dir + test_set + "_labels.npy"
    

# DATA GENERATORS
def batched_data_generator(file_names, batch_size, max_num_points, loop_infinite=True, add_energy_ratio=False, add_min_track_dist=False, add_both=False):
    while True:
        for file in file_names:
            point_net_data = np.load(file)
            cluster_data = point_net_data['X']
            Y = point_net_data['Y']

            # pad X data to have y dimension of max_num_points
            if add_energy_ratio or add_min_track_dist:
                X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2] + 1))
            elif add_both:
                X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2] + 2))
            else:
                X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2]))
            Y_padded = np.negative(np.ones(((cluster_data.shape[0], max_num_points, 1)))) # NOTE: update for weighted cells
            
            for i, cluster in enumerate(cluster_data):
                # update track to be energy # NOTE: errrr - used logged momentum for this calculation :/
                #track_momentum = cluster[:, 0][cluster[:, 4] == 1] # assume MeV/c ?? ** TODO: check this
                #charged_pion_mass = 134.9768 # MeV/c^2
                #track_energy = np.sqrt(track_momentum**2 + charged_pion_mass**2)
                #cluster[:, 0][cluster[:, 4] == 1] = np.log10(cluster[:, 0][cluster[:, 4] == 1]) - MEAN_TRACK_LOG_MOMENTUM
                
                if add_energy_ratio or add_both:
                   track_values = cluster[:,0][cluster[:,4] == 1]
                   track_value = track_values[0] if np.any(track_values) else 0
                   if track_value or add_both:
                       X_padded[i, :len(cluster), 5] = X_padded[i, :len(cluster), 0] / track_value
                elif add_min_track_dist:
                    track_points_idx = np.arange(len(cluster))[cluster[:, 4] == 1]
                    dists = np.zeros((len(cluster), len(track_points_idx)))

                    for j, track_point_idx in enumerate(track_points_idx):
                        dists[:, j] = np.sqrt((cluster[:, 1] - cluster[track_point_idx, 1])**2 + (cluster[:, 2] - cluster[track_point_idx, 2])**2 + (cluster[:, 3] - cluster[track_point_idx, 3])**2)

                    dist_feat_idx = 5
                    if add_both:
                        dist_feat_idx = 6

                    if np.any(track_points_idx):
                        min_dists = np.min(dists, axis=1)
                        # recast padding to 0 (padding is where the point is not a track and the label is -1)
                        min_dists[(cluster[:, 4] == 0) & (Y[i, :, 0] == -1)] = 0#13500
                        X_padded[i, :len(cluster), dist_feat_idx] = min_dists

                X_padded[i, :len(cluster), :5] = cluster
                Y_padded[i, :len(cluster), :] = Y[i]

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
            #print("loaded file")

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

train_output_dir = data_dir + '/train_1_track_larger/'
val_output_dir = data_dir + '/val_1_track_larger/'
test_output_dir = data_dir + '/test_1_track_larger/'


train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_output_dir+'*.npz'))[:num_test_files]

num_batches_train = (len(train_files) * events_per_file) / BATCH_SIZE 
num_batches_val = (len(val_files) * events_per_file) / BATCH_SIZE
num_batches_test = (len(test_files) * events_per_file) / BATCH_SIZE

# load the max number of points (N) - saved to data dir
with open(data_dir + max_points_file) as f:
    N = int(f.readline())

train_generator = batched_data_generator(train_files, BATCH_SIZE, N, add_min_track_dist=True)
val_generator = batched_data_generator(val_files, BATCH_SIZE, N, add_min_track_dist=True)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=True)

# COMPILE MODEL
model = pnet_part_seg(N)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

decay_rate = 0.1/5
#opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
opt = keras.optimizers.SGD(lr=0.1, momentum=0.8, decay=decay_rate)
model.compile(loss=masked_bce_pointwise_loss, optimizer=opt) # default bce is sum over batch size and doesn't include masking => superrrr low numbers ie. -20678613091047.0508 
model.summary()

# if resuming training load saved weights
if start_at_epoch:
    model.load_weights(output_dir + "/weights/weights_" + str(start_at_epoch - 1) + ".h5")


if validate_only:
    validate_gernerator = batched_data_generator(validate_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=True)
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

# CALLBACKS
# mkdirs if not present
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
        per_epoch_test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=True)

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