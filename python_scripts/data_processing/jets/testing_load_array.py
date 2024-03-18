from util_functs import print_events
from both import SAVE_LOC
import awkward as ak
import pyarrow.parquet as pq


def read_parquet_and_print_fields(filename, num_events_to_print):
    # Load the Parquet file
    table = pq.read_table(filename)
    # Convert the PyArrow table to an Awkward array
    ak_array = ak.from_arrow(table)
    print(len(ak_array))
    # Print out the specified number of events
    #print_events(ak_array, num_events_to_print)

# Example usage
parquet_file = SAVE_LOC + "processed_chunk_0.parquet"
num_events_to_print = 5
read_parquet_and_print_fields(parquet_file, num_events_to_print)