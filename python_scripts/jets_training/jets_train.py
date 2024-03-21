import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import os
import glob
import sys
from models.JetPointNet import PointNetRegression, masked_training_loss, masked_evaluation_loss, StreamlineLoggingCallback

os.environ['CUDA_VISIBLE_DEVICES'] = "" # SET GPU



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





BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIR = '/data/mjovanovic/jets/processed_files/firstAttempts/SavedNpz/train'
VAL_DIR = '/data/mjovanovic/jets/processed_files/firstAttempts/SavedNpz/val'

train_generator = data_generator(TRAIN_DIR, BATCH_SIZE)
val_generator = data_generator(VAL_DIR, BATCH_SIZE)


# Compute steps per epoch for training and validation
train_steps = sum(1 for _ in glob.glob(os.path.join(TRAIN_DIR, '*.npz'))) * (10000 // BATCH_SIZE)
val_steps = sum(1 for _ in glob.glob(os.path.join(VAL_DIR, '*.npz'))) * (10000// BATCH_SIZE)

model = PointNetRegression(MAX_SAMPLE_LENGTH, 1)
model.compile(optimizer='adam',
              loss=masked_training_loss,
              metrics=[masked_evaluation_loss])  

model.fit(train_generator,
          steps_per_epoch=train_steps,
          epochs=EPOCHS,
          validation_data=val_generator,
          validation_steps=val_steps,
          callbacks=[StreamlineLoggingCallback()])


print("Done!")