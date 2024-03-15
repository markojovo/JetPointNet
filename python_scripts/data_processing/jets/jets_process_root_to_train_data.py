import uproot
import awkward as ak
import numpy as np
import time

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
        print(f"Processing event: {event_idx + 1}")
        if DEBUG_NUM_EVENTS_TO_USE is not None:
            if event_idx >= DEBUG_NUM_EVENTS_TO_USE:  # Limiting processing for demonstration
                break

        event_cells, event_cell_truths, track_etas, track_phis = process_and_filter_cells(event, cellgeo)

        tracks_sample.begin_list()  # Start a new list for each event to hold tracks
        for track_idx in range(event["nTrack"]):
            print(f"\r    Processing Track: {track_idx + 1}", end='', flush=True)
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
        print()
    return tracks_sample.snapshot()  # Convert the ArrayBuilder to an actual Awkward array and return it

# Main code
if __name__ == "__main__":
    events = uproot.open(FILE_LOC + ":EventTree")
    cellgeo = uproot.open(GEO_FILE_LOC + ":CellGeo")

    print("Events Keys:")
    for key in events.keys():
        print(key)

    print("\nGeometry Keys:")
    for key in cellgeo.keys():
        print(key)


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

    print("Starting Timer...")
    start_time = time.time()
    for data in events.iterate(fields_list, library="ak", step_size="100MB"):
        processed_data = process_events(data, cellgeo)
        print_events(processed_data, 3)  # Assuming print_events is a function you have defined elsewhere

        max_sample_length = calculate_max_sample_length(processed_data)
        print("The maximum number of samples among these events is: ", max_sample_length)

    print("Ending Timer...")
    end_time = time.time()
    time_diff = end_time - start_time
    print("Total Time Elapsed: ", time_diff / 60 / 60, " Hours")
