from util_functs import print_events, calculate_max_sample_length, build_labels_array, build_input_array
from preprocessing_header import SAVE_LOC, NPZ_SAVE_LOC
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

# Make sure this happens after SAVE_LOC is defined and created if necessary
for folder in DATA_FOLDERS:
    folder_path = os.path.join(SAVE_LOC, folder)
    os.makedirs(folder_path, exist_ok=True)  # This line ensures the SAVE_LOC directories exist


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

find_global_max_sample_length()
global_max_sample_length = 278 #placeholder for now 


for data_folder in DATA_FOLDERS:
    npz_data_folder_path = os.path.join(NPZ_SAVE_LOC, data_folder)
    os.makedirs(npz_data_folder_path, exist_ok=True)  # Ensure the directory exists
    print(f"Processing data for: {data_folder}")

    data_folder_path = os.path.join(SAVE_LOC, data_folder)
    chunk_files = [f for f in os.listdir(data_folder_path) if f.startswith('chunk_') and f.endswith('.parquet')]
    num_chunks = len(chunk_files)

    for i in range(num_chunks):
        print(f"  Working on chunk {i} of {num_chunks - 1} in {data_folder}...", end="", flush=True)
        start_time = time.time()
        # Note: The file name is directly obtained from chunk_files list based on the sorted order
        chunk_file_name = sorted(chunk_files)[i]
        ak_array = read_parquet(os.path.join(data_folder_path, chunk_file_name))

        frac_labels = build_labels_array(ak_array, global_max_sample_length, "Fraction_Label")
        tot_labels = build_labels_array(ak_array, global_max_sample_length, "Total_Label", label_scale = 1)
        tot_truth_e = build_labels_array(ak_array, global_max_sample_length, "Total_Truth_Energy", label_scale = 1)

        feats = build_input_array(ak_array, global_max_sample_length, energy_scale = 1)

        # Save the feats and labels arrays to an NPZ file for each chunk
        npz_save_path = os.path.join(npz_data_folder_path, f'chunk_{i}_{data_folder}.npz')
        np.savez(npz_save_path, feats=feats, frac_labels=frac_labels, tot_labels=tot_labels, tot_truth_e = tot_truth_e)
        end_time = time.time()
        print(f"    Saved {data_folder} chunk {i} to {npz_save_path}...", end="")
        print(f"    Chunk processing took: ", end_time - start_time, "seconds")

    print(f"Completed processing data for: {data_folder}")

#ADD STUFF FOR STORING THE METADATA WHEN GENERATING THESE AWKWARD ARRAYS, THEN SAVE IT OR SOMETHINGG FOR MAX SAMPLE LEGNTH? IDK