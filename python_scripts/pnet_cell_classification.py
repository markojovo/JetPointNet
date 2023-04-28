import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4" # GPU
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import math
import numpy as np
import glob
import csv
from pnet_models_updated import pnet_part_seg_no_tnets

## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()


# LOSS
def masked_bce_pointwise_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    return K.sum(K.binary_crossentropy(tf.multiply(y_pred, mask),
                tf.multiply(y_true, mask)), axis=None) / K.sum(mask, axis=None)

# DATA AND OUTPUT DIRS
data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify'
output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify/pnet_part_seg_no_tnets_events_tr_472_val_118_tst_10_lr_1e-6'
num_train_files = 472
num_val_files = 118
num_test_files = 10
events_per_file = 6000
start_at_epoch = 6

EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 1e-6

# DATA GENERATORS
def batched_data_generator(file_names, batch_size, max_num_points, loop_infinite=True):
    while True:
        for file in file_names:
            point_net_data = np.load(file)
            cluster_data = point_net_data['X']
            Y = point_net_data['Y']

            # pad X data to have y dimension of max_num_points
            X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2])) # pad X data with 0's instead of -1's to have less influence on BN stats??
            Y_padded = np.negative(np.ones(((cluster_data.shape[0], max_num_points, 1))))
            for i, cluster in enumerate(cluster_data):
                X_padded[i, :len(cluster), :] = cluster
                Y_padded[i, :len(cluster), :] = Y[i] 

            # split into batch_size groups of clusters
            for i in range(1, math.ceil(cluster_data.shape[0]/batch_size)):
                yield X_padded[(i-1)*batch_size:i*batch_size], Y_padded[(i-1)*batch_size:i*batch_size]
        if not loop_infinite:
            break

train_output_dir = data_dir + '/train/'
val_output_dir = data_dir + '/val/'
test_output_dir = data_dir + '/test/'

train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_output_dir+'*.npz'))[:num_test_files]

num_batches_train = (len(train_files) * events_per_file) / BATCH_SIZE 
num_batches_val = (len(val_files) * events_per_file) / BATCH_SIZE
num_batches_test = (len(test_files) * events_per_file) / BATCH_SIZE

# load the max number of points (N) - saved to data dir
with open(data_dir + '/max_points.txt') as f:
    N = int(f.readline())

train_generator = batched_data_generator(train_files, BATCH_SIZE, N)
val_generator = batched_data_generator(val_files, BATCH_SIZE, N)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False)

# COMPILE MODEL
model = pnet_part_seg_no_tnets(N)
model.compile(loss=masked_bce_pointwise_loss, optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE)) # default bce is sum over batch size and doesn't include masking => superrrr low numbers ie. -20678613091047.0508 
model.summary()

# if resuming training load saved weights
if start_at_epoch:
    model.load_weights(output_dir + "/weights/weights_" + str(start_at_epoch - 1) + ".h5")

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
        per_epoch_test_generator = batched_data_generator(test_files[:5], BATCH_SIZE, N, loop_infinite=False)

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