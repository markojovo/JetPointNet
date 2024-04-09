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
from models.JetPointNet import PointNetSegmentation, masked_bce_loss, masked_mae_loss, masked_mse_loss

os.environ['CUDA_VISIBLE_DEVICES'] = "4" # SET GPU

MAX_SAMPLE_LENGTH=859

class BatchNormalizationMomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_momentum=0.5, final_momentum=0.99, total_epochs=20):
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.total_epochs = total_epochs
        self.delta = (final_momentum - initial_momentum) / total_epochs

    def on_epoch_end(self, epoch, logs=None):
        new_momentum = self.initial_momentum + self.delta * epoch
        new_momentum = min(new_momentum, self.final_momentum)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = new_momentum
        print(f"\nUpdated BatchNormalization momentum to: {new_momentum:.4f}")

def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data['feats']  # Shape: (num_samples, 859, 6)
    frac_labels = data['frac_labels']  # Shape: (num_samples, 859)
    tot_labels = data['tot_labels']  # Shape: (num_samples, 859)
    tot_truth_e = data['tot_truth_e']  # Shape: (num_samples, 859) (This is the true total energy deposited by particles into this cell)
    return feats, frac_labels, tot_labels, tot_truth_e

def data_generator(data_dir, batch_size):
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    while True:
        for npz_file in npz_files:
            feats, frac_labels, tot_labels, tot_truth_e = load_data_from_npz(npz_file)

            #Think of a better way to handle the different label types
            labels = tot_labels # predicting the absolute truth energy from focused particle
            # labels = frac_labels # predicting the fraction of truth energy from focused particle (absolute / total)

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
        total_samples += len(data['feats']) 
    steps_per_epoch = math.ceil(total_samples / batch_size)  # Use math.ceil to round up to ensure covering all samples
    return steps_per_epoch

def save_model_on_epoch_end(epoch, logs):
    model.save(f"saved_model/PointNetModel.keras") 

def scheduler(epoch, lr):
    if epoch > 0 and epoch % 2 == 0: 
        return lr * 0.5
    else:
        return lr

initial_learning_rate = 0.001 / 1000000 
BATCH_SIZE = 48
EPOCHS = 20
TRAIN_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/train'
VAL_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/val'


train_steps = calculate_steps(TRAIN_DIR, BATCH_SIZE)
val_steps = calculate_steps(VAL_DIR, BATCH_SIZE)
train_generator = data_generator(TRAIN_DIR, BATCH_SIZE)
val_generator = data_generator(VAL_DIR, BATCH_SIZE)

model = PointNetSegmentation(MAX_SAMPLE_LENGTH, 1)

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Compile the model with loss=None due to custom loss within the model
model.compile(optimizer=optimizer, loss=masked_mse_loss, metrics=[masked_mae_loss])  # Consider updating or customizing metrics as necessary


model.summary()
model_params = model.count_params()
model_size_in_bytes = model_params * 4  
model_size_in_megabytes = model_size_in_bytes / (1024 ** 2)

print(f"Model Parameters: {model_params}")
print(f"Model Size: {model_size_in_megabytes:.2f} MB")
print("Training on Dataset: ", TRAIN_DIR)


# Define your callbacks
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
csv_logger = CSVLogger('saved_model/training_log.csv', append=True, separator=';')
save_model_callback = LambdaCallback(on_epoch_end=save_model_on_epoch_end)
# batch_norm_scheduler = BatchNormalizationMomentumScheduler(total_epochs=EPOCHS) # Was used in the pointnet paper, can decide if you'd like to


start_time = time.time()
model.fit(train_generator,
          steps_per_epoch=train_steps,
          epochs=EPOCHS,
          validation_data=val_generator,
          validation_steps=val_steps,
          callbacks=[csv_logger, save_model_callback, lr_scheduler])  # Include lr_scheduler in callbacks
end_time = time.time()

print(f"Training Done! Took {(end_time - start_time) / 60 / 60} hours!")