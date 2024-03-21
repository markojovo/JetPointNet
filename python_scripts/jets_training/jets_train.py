import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import os
import glob
import sys
sys.path.append('/home/mjovanovic/Work/PointNet_Segmentation')
from utils.pnet_models import pnet_part_seg_no_tnets, pnet_part_seg

os.environ['CUDA_VISIBLE_DEVICES'] = "" # SET GPU

## IMPORTANT ## ====== ## DISABLE EAGER EXECUTION WITH TensorFlow!! ##
print()
print('disabling eager execution..')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
print()


MAX_SAMPLE_LENGTH=859

def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data['feats']  # Shape: (num_samples, 859, 6)
    labels = data['labels']  # Shape: (num_samples, 859)
    return feats, labels

def data_generator(data_dir, batch_size):
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    while True:
        for npz_file in npz_files:
            feats, labels = load_data_from_npz(npz_file)
            dataset_size = feats.shape[0]
            for i in range(0, dataset_size, batch_size):
                end_index = i + batch_size
                # Ensure we don't exceed the number of samples by slicing till the end
                batch_feats = feats[i:end_index]
                batch_labels = labels[i:end_index]
                yield batch_feats, batch_labels.reshape(*batch_labels.shape, 1)  # Reshape labels to (batch_size, 859, 1)


def custom_training_loss(y_true, y_pred):
    # Mask to exclude points with a label of -1
    mask = tf.not_equal(y_true, -1)
    # Apply the mask to both y_true and y_pred
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    # Calculate the loss for the masked values
    loss = tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss)



BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIR = '/data/mjovanovic/jets/processed_files/firstAttempts/SavedNpz/train'
VAL_DIR = '/data/mjovanovic/jets/processed_files/firstAttempts/SavedNpz/val'

train_generator = data_generator(TRAIN_DIR, BATCH_SIZE)
val_generator = data_generator(VAL_DIR, BATCH_SIZE)

model = pnet_part_seg_no_tnets(MAX_SAMPLE_LENGTH, 6, 1)

# Compute steps per epoch for training and validation
train_steps = sum(1 for _ in glob.glob(os.path.join(TRAIN_DIR, '*.npz'))) * (10000 // BATCH_SIZE)
val_steps = sum(1 for _ in glob.glob(os.path.join(VAL_DIR, '*.npz'))) * (10000// BATCH_SIZE)

model.fit(train_generator, steps_per_epoch=train_steps, epochs=EPOCHS, validation_data=val_generator, validation_steps=val_steps)
