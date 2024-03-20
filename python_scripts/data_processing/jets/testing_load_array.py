from util_functs import print_events, calculate_max_sample_length
from both import SAVE_LOC, NPZ_SAVE_LOC
import awkward as ak
import pyarrow.parquet as pq
import numpy as np
import os

DATA_FOLDERS = ['train', 'test', 'val']

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

def print_first_sample_fields_from_first_train_file():
    train_path = os.path.join(SAVE_LOC, 'train')
    train_files = [f for f in os.listdir(train_path) if f.endswith('.parquet')]
    if train_files:
        first_file_path = os.path.join(train_path, train_files[0])
        ak_array = read_parquet(first_file_path)
        first_sample = ak_array[0]
        print(f"Fields of the first sample: {first_sample.fields}")
        print("Fields of the subarrays:")
        f = first_sample["track_layer_intersections"].fields
        print(f"Track Layer Intersections: {f}")

        f = first_sample["associated_cells"].fields
        print(f"Associated Cells: {f}")
        f = first_sample["associated_tracks"].fields
        print(f"Associated Tracks: {f}")

        f = first_sample["associated_tracks"]["track_layer_intersections"].fields
        print(f"Associated layer intersections: {f}")

# Calculate the global maximum sample length across all files
#global_max_sample_length = find_global_max_sample_length()
global_max_sample_length = 859 #placeholder for now

# Print the fields of the first sample from the first train file
#print_first_sample_fields_from_first_train_file()

# Assuming ak_array is loaded with your Awkward Array data
ak_array = read_parquet(os.path.join(SAVE_LOC, 'train', 'chunk_0_train.parquet'))

print_events(ak_array, 1)

