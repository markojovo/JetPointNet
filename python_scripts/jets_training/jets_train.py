import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, LambdaCallback
import keras
import math
import os
import glob
import sys
import time
from models.JetPointNet import PointNetRegression, masked_bce_loss, masked_mae_loss, masked_mse_loss

os.environ['CUDA_VISIBLE_DEVICES'] = "7" # SET GPU



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


def calculate_steps(data_dir, batch_size):
    total_samples = 0
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    for npz_file in npz_files:
        data = np.load(npz_file)
        total_samples += len(data['feats'])  # Assuming 'feats' holds the features for each sample
    steps_per_epoch = math.ceil(total_samples / batch_size)  # Use math.ceil to round up to ensure covering all samples
    return steps_per_epoch

def save_model_on_epoch_end(epoch, logs):
    model.save(f"saved_model/PointNetModel.keras")  # epoch+1 because epochs are 0-indexed


learning_rate = 0.00001 #0.0001  
BATCH_SIZE = 128
EPOCHS = 100
TRAIN_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/train'
VAL_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/val'

train_steps = calculate_steps(TRAIN_DIR, BATCH_SIZE)
val_steps = calculate_steps(VAL_DIR, BATCH_SIZE)
train_generator = data_generator(TRAIN_DIR, BATCH_SIZE)
val_generator = data_generator(VAL_DIR, BATCH_SIZE)


# Compute steps per epoch for training and validation
#train_steps = sum(1 for _ in glob.glob(os.path.join(TRAIN_DIR, '*.npz'))) * (10000 // BATCH_SIZE)
#val_steps = sum(1 for _ in glob.glob(os.path.join(VAL_DIR, '*.npz'))) * (10000// BATCH_SIZE)

model = PointNetRegression(MAX_SAMPLE_LENGTH, 1)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer,
              loss=masked_bce_loss, #masked_kl_divergence_loss,
              metrics=[masked_mae_loss])


model.summary()
model_params = model.count_params()
model_size_in_bytes = model_params * 4  
model_size_in_megabytes = model_size_in_bytes / (1024 ** 2)

print(f"Model Parameters: {model_params}")
print(f"Model Size: {model_size_in_megabytes:.2f} MB")
print("Training on Dataset: ", TRAIN_DIR)
csv_logger = CSVLogger('saved_model/training_log.csv', append=True, separator=';')
save_model_callback = LambdaCallback(on_epoch_end=save_model_on_epoch_end)

start_time = time.time()
model.fit(train_generator,
          steps_per_epoch=train_steps,
          epochs=EPOCHS,
          validation_data=val_generator,
          validation_steps=val_steps,
          callbacks=[csv_logger, save_model_callback])
end_time = time.time()


print(f"Training Done! Took {(end_time - start_time) / 60 / 60} hours!")