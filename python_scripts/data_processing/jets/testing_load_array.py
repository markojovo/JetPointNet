from util_functs import print_events, calculate_max_sample_length, build_labels_array, build_input_array
from both import SAVE_LOC, NPZ_SAVE_LOC
import awkward as ak
import pyarrow.parquet as pq
import numpy as np
import os
import time

DATA_FOLDERS =  ['train', 'test', 'val']

def read_parquet(filename):
    table = pq.read_table(filename)
    ak_array = ak.from_arrow(table)
    return ak_array

def find_global_max_sample_length():
    global_max_sample_length = 0
    for folder in DATA_FOLDERS:
        folder_path = os.path.join(SAVE_LOC, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".parquet"):
                full_path = os.path.join(folder_path, filename)
                ak_array = read_parquet(full_path)
                max_sample_length = calculate_max_sample_length(ak_array)
                print("Max sample length found: ",max_sample_length)
                global_max_sample_length = max(global_max_sample_length, max_sample_length)
    print(f"Global Max Sample Length: {global_max_sample_length}")
    return global_max_sample_length

global_max_sample_length = 859 #placeholder for now 


for data_folder in DATA_FOLDERS:
    npz_data_folder_path = os.path.join(NPZ_SAVE_LOC, data_folder)
    os.makedirs(npz_data_folder_path, exist_ok=True)  # Ensure the directory exists
    print(f"Processing data for: {data_folder}")

    for i in range(4):
        print(f"  Working on chunk {i} in {data_folder}...", end="", flush=True)
        start_time = time.time()
        # Replace 'read_parquet', 'build_labels_array', and 'build_input_array' with your actual functions
        ak_array = read_parquet(os.path.join(SAVE_LOC, data_folder, f'chunk_{i}_{data_folder}.parquet'))

        labels = build_labels_array(ak_array, global_max_sample_length)
        feats = build_input_array(ak_array, global_max_sample_length)

        # Save the feats and labels arrays to an NPZ file for each chunk
        npz_save_path = os.path.join(npz_data_folder_path, f'chunk_{i}_{data_folder}.npz')
        np.savez(npz_save_path, feats=feats, labels=labels)
        end_time = time.time()
        print(f"    Saved {data_folder} chunk {i} to {npz_save_path}...", end="")
        print(f"    Chunk processing took: ", end_time - start_time, "seconds")

    print(f"Completed processing data for: {data_folder}")