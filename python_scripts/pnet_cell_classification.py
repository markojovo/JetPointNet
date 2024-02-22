import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import numpy as np
import glob
import csv
import sys
sys.path.append('/home/mjovanovic/Work/PointNet_Segmentation')
from utils.pnet_models import pnet_part_seg_no_tnets, pnet_part_seg
import awkward as ak


os.environ['CUDA_VISIBLE_DEVICES'] = "0" # SET GPU


## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()

MEAN_TRACK_LOG_ENERGY = 2.4

# Not these?
#LOG_ENERGY_MEAN = -1 # unrounded mean is ~ -0.93
#LOG_MEAN_TRACK_MOMETUM = 2

# DATA AND OUTPUT DIRS
data_dir = '/data/mjovanovic/cell_particle_deposit_learning/rho/rho_processed_train_files/'
#data_dir = '/fast_scratch_1/jbohm/cell_particle_deposit_learning/rho_train/train/'
output_dir = "/data/mjovanovic/cell_particle_deposit_learning/rho_train/KF_regression_Loss_test/"
max_points_file = '../max_points.txt'

num_train_files = 100 #707
num_val_files = 3 #210
num_test_files = 5 #10
events_per_file = 3800
start_at_epoch = False # load start_at_epoch - 1
add_min_track_dist = True

EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 1e-2
train_output_dir = data_dir# + '/train_1_track/'
val_output_dir = data_dir + '/val_1_track/'
test_output_dir = data_dir + '/test_1_track/'

class PrintSamplePredictions(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\nSample Predictions vs. Labels at Epoch: {}".format(epoch + 1))
        sample_generator = batched_data_generator(train_files, 1, N, validation_split=0.1, is_validation=False, loop_infinite=False, add_min_track_dist=add_min_track_dist)
        num_samples_to_display = 50  # Number of samples to display

        for i, (X_sample, Y_sample) in enumerate(sample_generator):
            if i >= num_samples_to_display:
                break
            prediction = model.predict(X_sample)
            print("\nSample {}: Prediction = {}, Label = {}".format(i + 1, prediction[:20].flatten(), Y_sample[:20].flatten()))
            # Flatten and show only the first 10 elements to keep the output readable


    

# DATA GENERATORS (E, X, Y, Z, TYPE)
def batched_data_generator(file_names, batch_size, max_num_points, validation_split=0.1, is_validation=False, loop_infinite=True, add_min_track_dist=False):
    while True:
        for file in file_names:
            point_net_data = np.load(file)
            cluster_data = point_net_data['X']
            Y = point_net_data['Y']

            # Calculate the number of data points to be used for validation and adjust the dataset accordingly
            num_validation_samples = int(len(cluster_data) * validation_split)
            if is_validation:
                cluster_data = cluster_data[-num_validation_samples:]
                Y = Y[-num_validation_samples:]
            else:
                cluster_data = cluster_data[:-num_validation_samples]
                Y = Y[:-num_validation_samples]

            # Pad and process data as before, now adjusted for training/validation split
            X_padded = np.zeros((len(cluster_data), max_num_points, cluster_data.shape[2]))# + (1 if add_min_track_dist else 0)))
            Y_padded = np.negative(np.ones((len(cluster_data), max_num_points, 1)))

            for i, cluster in enumerate(cluster_data):
                if add_min_track_dist:
                    # Your logic to calculate and add minimum track distance or any other feature
                    pass

                # Adjust padding based on actual cluster size
                actual_size = min(len(cluster), max_num_points)
                X_padded[i, :actual_size, :cluster.shape[1]] = cluster[:actual_size]
                #Y_padded[i, :actual_size, 0] = Y[i][:actual_size]
                Y_padded[i, :actual_size, 0] = Y[i][:actual_size].flatten()


            # Yield batches
            total_batches = math.ceil(len(cluster_data) / batch_size)
            for batch_num in range(total_batches):
                start_index = batch_num * batch_size
                end_index = start_index + batch_size
                yield X_padded[start_index:end_index], Y_padded[start_index:end_index]

        if not loop_infinite:
            break


def masked_mae_loss(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:, :, 0], -1)  # Adjusting shape if necessary
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)  # Mask for values not equal to -1
    absolute_difference = tf.abs(y_true - y_pred)  # Calculate absolute difference
    masked_absolute_difference = absolute_difference * mask  # Apply mask
    return tf.reduce_sum(masked_absolute_difference) / tf.reduce_sum(mask)  # Normalize by the sum of the mask


def masked_bce_pointwise_loss(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:,:,0], -1)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    return tf.reduce_sum(tf.keras.losses.binary_crossentropy(tf.multiply(y_pred, mask), tf.multiply(y_true, mask)), axis=None) / tf.reduce_sum(mask, axis=None)

def masked_mse_pointwise_loss(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:,:,0], -1)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    masked_squared_difference = squared_difference * mask
    masked_loss = tf.reduce_sum(masked_squared_difference) / tf.reduce_sum(mask)
    return masked_loss

def masked_kl_divergence_loss(y_true, y_pred):
    y_true = tf.expand_dims(y_true[:, :, 0], -1)  # Adjusting shape if necessary
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)  # Mask for values not equal to -1
    y_true_masked = tf.multiply(y_true, mask)  # Apply mask to y_true
    y_pred_masked = tf.multiply(y_pred, mask)  # Apply mask to y_pred
    
    # Calculate KL divergence
    kl_divergence = y_true_masked * tf.math.log(y_true_masked / (y_pred_masked + 1e-15) + 1e-15)
    masked_kl_divergence = kl_divergence * mask  # Apply mask
    combined_loss = tf.reduce_sum(masked_kl_divergence)
    
    # Normalize by the sum of the mask to account for the masked values
    return tf.abs(combined_loss / tf.reduce_sum(mask))



train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
test_files = np.sort(glob.glob(test_output_dir+'*.npz'))[:num_test_files]

num_batches_train = (len(train_files) * events_per_file * (1 - 0.1)) // BATCH_SIZE
num_batches_val = (len(train_files) * events_per_file * 0.1) // BATCH_SIZE

num_batches_test = (len(test_files) * events_per_file) / BATCH_SIZE

# load the max number of points (N) - saved to data dir
with open(data_dir + max_points_file) as f:
    N = int(f.readline())
N = 1180

train_generator = batched_data_generator(train_files, BATCH_SIZE, N, validation_split=0.1, is_validation=False, loop_infinite=True, add_min_track_dist=add_min_track_dist)
val_generator = batched_data_generator(train_files, BATCH_SIZE, N, validation_split=0.1, is_validation=True, loop_infinite=True, add_min_track_dist=add_min_track_dist)
test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=add_min_track_dist)



# COMPILE MODEL
model = pnet_part_seg_no_tnets(N, 5 , 1)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

decay_rate = 0.1/5
#opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
#opt = tf.optimizers.SGD(lr=0.1, momentum=0.8)
opt = tf.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=opt,
              loss= masked_kl_divergence_loss,  # or any other loss function suitable for your problem
              metrics=[masked_mae_loss,masked_bce_pointwise_loss, masked_mse_pointwise_loss])  # Add MAE and binary_crossentropy here




# Create a new generator with batch size of 1 for prediction
single_sample_generator = batched_data_generator(train_files, 1, N, validation_split=0.1, is_validation=False, loop_infinite=False, add_min_track_dist=add_min_track_dist)

# Use the new generator for predictions
for i, (X_sample, Y_sample) in enumerate(single_sample_generator):
    output = model.predict(X_sample)
    print(output)
    print(X_sample.shape)  # This should now show a batch size of 1
    print(output.shape)
    print(output[0])
    if i >= 0:  # Just to demonstrate with the first sample, remove or adjust this condition as needed
        break



model.summary()

# if resuming training load saved weights
#if start_at_epoch:
#    model.load_weights(output_dir + "/weights/weights_" + str(start_at_epoch - 1) + ".h5")


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
        per_epoch_test_generator = batched_data_generator(test_files, BATCH_SIZE, N, loop_infinite=False, add_min_track_dist=add_min_track_dist)

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
print(f"Validation Steps: {num_batches_val}")


# Add the PrintSamplePredictions callback to your model training
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=val_generator,
                    verbose=1,
                    steps_per_epoch=num_batches_train,
                    validation_steps=num_batches_val,
                    callbacks=[SaveEpoch(), PrintSamplePredictions()]
                    )
