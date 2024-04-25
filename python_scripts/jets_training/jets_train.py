import numpy as np
import tensorflow as tf
import os
import glob
import math
import time
from models.JetPointNet import PointNetSegmentation, masked_weighted_bce_loss, masked_regular_accuracy, masked_weighted_accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = "3"  # Set GPU

MAX_SAMPLE_LENGTH = 278
BATCH_SIZE = 480
EPOCHS = 120
TRAIN_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/train'
VAL_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/val'

def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data['feats']
    frac_labels = data['frac_labels']
    energy_weights = data['tot_truth_e']
    return feats, frac_labels, energy_weights

def data_generator(data_dir, batch_size):
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    while True:
        np.random.shuffle(npz_files)
        for npz_file in npz_files:
            feats, frac_labels, e_weights = load_data_from_npz(npz_file)
            dataset_size = feats.shape[0]
            for i in range(0, dataset_size, batch_size):
                end_index = i + batch_size
                batch_feats = feats[i:end_index]
                batch_labels = frac_labels[i:end_index]
                batch_e_weights = e_weights[i:end_index]
                yield batch_feats, batch_labels.reshape(*batch_labels.shape, 1), batch_e_weights.reshape(*batch_e_weights.shape, 1)

def calculate_steps(data_dir, batch_size):
    total_samples = 0
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    for npz_file in npz_files:
        data = np.load(npz_file)
        total_samples += data['feats'].shape[0]
    return math.ceil(total_samples / batch_size)

train_steps = calculate_steps(TRAIN_DIR, BATCH_SIZE)
val_steps = calculate_steps(VAL_DIR, BATCH_SIZE)

model = PointNetSegmentation(MAX_SAMPLE_LENGTH, 1)
optimizer = tf.keras.optimizers.Adam(learning_rate=(0.001))

@tf.function
def train_step(x, y, energy_weights, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = masked_weighted_bce_loss(y, predictions, energy_weights)
        reg_acc = masked_regular_accuracy(y, predictions, energy_weights)
        weighted_acc = masked_weighted_accuracy(y, predictions, energy_weights)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, reg_acc, weighted_acc

@tf.function
def val_step(x, y, energy_weights, model):
    predictions = model(x, training=False)
    v_loss = masked_weighted_bce_loss(y, predictions, energy_weights)
    reg_acc = masked_regular_accuracy(y, predictions, energy_weights)
    weighted_acc = masked_weighted_accuracy(y, predictions, energy_weights)
    return v_loss, reg_acc, weighted_acc

train_loss_tracker = tf.metrics.Mean(name='train_loss')
val_loss_tracker = tf.metrics.Mean(name='val_loss')
train_reg_acc = tf.metrics.Mean(name='train_regular_accuracy')
train_weighted_acc = tf.metrics.Mean(name='train_weighted_accuracy')
val_reg_acc = tf.metrics.Mean(name='val_regular_accuracy')
val_weighted_acc = tf.metrics.Mean(name='val_weighted_accuracy')

for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    train_loss_tracker.reset_states()
    val_loss_tracker.reset_states()
    train_reg_acc.reset_states()
    train_weighted_acc.reset_states()
    val_reg_acc.reset_states()
    val_weighted_acc.reset_states()


    for step, (x_batch_train, y_batch_train, e_weight_train) in enumerate(data_generator(TRAIN_DIR, BATCH_SIZE)):
        if step >= train_steps:
            break
        loss_value, reg_acc_value, weighted_acc_value = train_step(x_batch_train, y_batch_train, e_weight_train, model, optimizer)
        train_loss_tracker.update_state(loss_value)
        train_reg_acc.update_state(reg_acc_value)
        train_weighted_acc.update_state(weighted_acc_value)
        print(f"\rEpoch {epoch + 1}, Step {step + 1}/{train_steps}, Training Loss: {train_loss_tracker.result().numpy():.4e}, Reg Acc: {train_reg_acc.result().numpy():.4f}, Weighted Acc: {train_weighted_acc.result().numpy():.4f}", end='')

    print(f"\nTraining loss over epoch: {train_loss_tracker.result():.4f}")
    print(f"Time taken for training: {time.time() - start_time:.2f} sec")

    for step, (x_batch_val, y_batch_val, e_weight_val) in enumerate(data_generator(VAL_DIR, BATCH_SIZE)):
        if step >= val_steps:
            break
        val_loss_value, val_reg_acc_value, val_weighted_acc_value = val_step(x_batch_val, y_batch_val, e_weight_train, model)
        val_loss_tracker.update_state(val_loss_value)
        val_reg_acc.update_state(val_reg_acc_value)
        val_weighted_acc.update_state(val_weighted_acc_value)
        print(f"\rEpoch {epoch + 1}, Step {step + 1}/{val_steps}, Validation Loss: {val_loss_tracker.result().numpy():.4f}, Reg Acc: {val_reg_acc.result().numpy():.4f}, Weighted Acc: {val_weighted_acc.result().numpy():.4f}", end='')

    print(f"Validation loss: {val_loss_tracker.result():.4f}")
    print(f"Time taken for validation: {time.time() - start_time:.2f} sec")

    model.save(f"saved_model/PointNetModel.keras")
    print("Model saved.")

print("Training completed!")
