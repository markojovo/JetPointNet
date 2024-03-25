import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import os

from models.JetPointNet import PointNetRegression

os.environ['CUDA_VISIBLE_DEVICES'] = "" # SET GPU

def load_data_from_npz(npz_file):
    data = np.load(npz_file)
    feats = data['feats']  # Shape: (num_samples, 859, 6)
    labels = data['labels']  # Shape: (num_samples, 859)
    return feats, labels


model = PointNetRegression(num_points=859, n_classes=1) 
model.load_weights("saved_model/PointNetModel.keras") 
TEST_DIR = '/data/mjovanovic/jets/processed_files/2000_events_w_fixed_hits/SavedNpz/test'
all_labels = []

npz_files = glob.glob(os.path.join(TEST_DIR, '*.npz'))
for npz_file in npz_files:
    _, labels = load_data_from_npz(npz_file)
    labels = labels[:1] 
    indices = np.where(labels != -1)
    all_labels.extend(labels[indices])

overall_mean_label = np.mean(all_labels)

mae_predicted_vals = []
mae_basic_model_vals = []

for i, npz_file in enumerate(npz_files):
    feats, labels = load_data_from_npz(npz_file)
    feats = feats[:]
    labels = labels[:]
    indices = np.where(labels != -1)

    predicted_labels = np.squeeze(model.predict(feats), axis=-1)
    mae_predicted = np.mean(np.abs(predicted_labels[indices] - labels[indices]))
    mae_predicted_vals.append(mae_predicted)

    if i == 0:
        np.set_printoptions(suppress=True,precision=3)
        print("Features: \n",feats[0][3:9],"\n", feats[0][-5:])
        print("Labels: \n",labels[0])
        print("Predictions: \n", predicted_labels[0])

    mae_basic_model = np.mean(np.abs(labels[indices] - overall_mean_label))
    mae_basic_model_vals.append(mae_basic_model)

    # print(f"Processed file {i+1}/{len(npz_files)}")

mean_mae_predicted = np.mean(mae_predicted_vals)
mean_mae_basic_model = np.mean(mae_basic_model_vals)



print(f"Mean MAE (Predicted vs. Actual): {mean_mae_predicted}")
print(f"Mean MAE (Basic Model vs. Actual): {mean_mae_basic_model}")