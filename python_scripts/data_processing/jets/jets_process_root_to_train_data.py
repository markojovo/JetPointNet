import uproot
import awkward as ak
import numpy as np
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from both import *
from util_functs import *

def process_events(data, cellgeo):
    """
    Main event processing function.
    
    Parameters:
    - data: The events data to process.
    - cellgeo: The cell geometry data.
    """
    tracks_sample = ak.ArrayBuilder()  # Initialize the awkward array structure for track samples
    print("Going...")
    print(f"Processing a batch of {len(data)} events.")
    for event_idx, event in enumerate(data):
        #print(f"Processing event: {event_idx + 1}")
        if DEBUG_NUM_EVENTS_TO_USE is not None:
            if event_idx >= DEBUG_NUM_EVENTS_TO_USE:  # Limiting processing for demonstration
                break

        print("processing cells")
        event_cells, event_cell_truths, track_etas, track_phis = process_and_filter_cells(event, cell_ID_geo, cell_eta_geo, cell_phi_geo, cell_rPerp_geo):

        print("Done proc and filtering cells")
        tracks_sample.begin_list()  # Start a new list for each event to hold tracks
        for track_idx in range(event["nTrack"]):
            print(f"Processing track: {track_idx}")
            tracks_sample.begin_record()  # Each track is a record within the event list

            # Meta info
            fields = [
                ("eventID", "integer"),
                ("trackEta_EMB2", "real"),
                ("trackPhi_EMB2", "real"),
                ("trackEta_EME2", "real"),
                ("trackPhi_EME2", "real"),
                ("trackSubtractedCaloEnergy", "real"),
                ("trackPt", "real"),
                ("trackChiSquared/trackNumberDOF", "real"),
            ]
            track_eta_ref, track_phi_ref, track_part_Idx = add_track_meta_info(tracks_sample, event, event_idx, track_idx, fields)

            # Track intersections
            track_intersections = add_track_intersection_info(tracks_sample, track_idx, track_etas, track_phis)

            # Associated cell info
            process_associated_cell_info(event_cells, event_cell_truths, track_part_Idx, tracks_sample, track_eta_ref, track_phi_ref, track_intersections)

            # Associated tracks
            process_associated_tracks(event, tracks_sample, track_eta_ref, track_phi_ref, track_idx, event["nTrack"], track_etas, track_phis)

            tracks_sample.end_record()  # End the record for the current track

        tracks_sample.end_list()  # End the list for the current event
        print(f"Done event {event_idx}")
        #print()
    return tracks_sample.snapshot()  # Convert the ArrayBuilder to an actual Awkward array and return it

def save_to_disk(processed_data, filename):
    """
    Save the processed data to disk.
    """
    # Example: saving as a Parquet file (implementation depends on the desired format)
    ak.to_parquet(processed_data, filename)

def process_chunk(chunk, cellgeo):
    """
    Process a chunk of events using multiprocessing.
    """
    chunk_size = len(chunk)
    events_per_thread = chunk_size // NUM_THREAD_PER_CHUNK
    futures = []
    with ProcessPoolExecutor(max_workers=NUM_THREAD_PER_CHUNK) as executor:
        for i in range(NUM_THREAD_PER_CHUNK):
            start_idx = i * events_per_thread
            if i == NUM_THREAD_PER_CHUNK - 1:  # Last thread takes the remainder
                end_idx = chunk_size
            else:
                end_idx = start_idx + events_per_thread
            subset = chunk[start_idx:end_idx]
            # NOTE: The cellgeo object must be serializable or not necessary to pass between processes
            futures.append(executor.submit(process_events, subset, cellgeo))

    # Combine results from all processes
    combined_array = ak.concatenate([future.result() for future in futures])
    return combined_array


if __name__ == "__main__":
    events = uproot.open(FILE_LOC + ":EventTree")
    cellgeo = uproot.open(GEO_FILE_LOC + ":CellGeo")


    print("Events Keys:")
    for key in events.keys():
        print(key)
    print()
    print("\nGeometry Keys:")
    for key in cellgeo.keys():
        print(key)
    print()

    # Process events and tracks as before, with the following adjustments:
    track_layer_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers] # Getting all the cell layer points that the track hits (ie trackEta_EME2, trackPhi_EMB3, etc)

    jets_other_included_fields = ["trackSubtractedCaloEnergy", "trackPt", "nTrack", "cluster_cell_ID",
                            "trackNumberDOF","trackChiSquared","cluster_cell_E","cluster_cell_hitsTruthIndex","cluster_cell_hitsTruthE", "trackTruthParticleIndex"]

    fields_list = track_layer_branches + jets_other_included_fields

    # Load Cell Geometry Data
    cell_ID_geo = cellgeo["cell_geo_ID"].array(library="ak")[0]
    eta_geo = cellgeo["cell_geo_eta"].array(library="ak")[0]
    phi_geo = cellgeo["cell_geo_phi"].array(library="ak")[0]
    rPerp_geo = cellgeo["cell_geo_rPerp"].array(library="ak")[0]


    start_time = time.time()

    chunk_counter = 0
    for chunk in events.iterate(fields_list, library="ak", step_size=NUM_EVENTS_PER_CHUNK):  # Adjust entry_stop as needed
        print(f"Processing chunk {chunk_counter + 1}")
        processed_data = process_chunk(chunk, cellgeo)
        
        # Save the processed chunk to disk
        filename = f"processed_chunk_{chunk_counter}.parquet"
        save_to_disk(processed_data, filename)
        
        chunk_counter += 1

    end_time = time.time()
    print("Total Time Elapsed: ", (end_time - start_time) / 60 / 60, " Hours")