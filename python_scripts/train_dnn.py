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

# MODEL
def dnn(shape):
    inputs = keras.Input(shape=shape, name="input")
    dense64 = layers.Dense(64)
    x1 = layers.Activation('relu')(layers.TimeDistributed(dense64)(inputs))
    dense32 = layers.Dense(32)
    x2 = layers.Activation('relu')(layers.TimeDistributed(dense32)(x1))
    dense16 = layers.Dense(32)
    x3 = layers.Activation('relu')(layers.TimeDistributed(dense16)(x2))
    dense1 = layers.Dense(1)
    out = layers.Activation('sigmoid')(layers.TimeDistributed(dense1)(x3))
    return keras.Model(inputs=inputs, outputs=out)

# LOSS
def masked_mae_pointwise_loss(y_true, y_pred):
    return K.abs(y_true[y_true != -1] - y_pred[y_true != -1]) # taking cut flattens the data giving pointwise ae to be averages over on reduction

# DATA AND OUTPUT DIRS
data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_4'
output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_4/dnn_4_layers'#_clus_mean_EM_below_0.5_lr_1e-6'
num_train_files = 50
num_val_files = 10
num_test_files = 5
events_per_file = 2600 # approx since it varries # for filtered low energy events - 2600
start_at_epoch = 0

EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 1e-2

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
model = dnn(shape=(N, 4))
model.compile(loss=masked_mae_pointwise_loss, optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
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