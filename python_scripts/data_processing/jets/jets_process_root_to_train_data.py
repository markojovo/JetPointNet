import uproot
import awkward as ak
import numpy as np
import time

from both import *
from util_functs import *

# Open the ROOT file and access the EventTree
events = uproot.open(FILE_LOC + ":EventTree")
cellgeo = uproot.open(GEO_FILE_LOC + ":CellGeo")

print("Events Keys:")
for key in events.keys():
    print(key)

print("\nGeometry Keys:")
for key in cellgeo.keys():
    print(key)


# Load Cell Geometry Data
cell_ID_geo = cellgeo["cell_geo_ID"].array(library="ak")[0]
eta_geo = cellgeo["cell_geo_eta"].array(library="ak")[0]
phi_geo = cellgeo["cell_geo_phi"].array(library="ak")[0]
rPerp_geo = cellgeo["cell_geo_rPerp"].array(library="ak")[0]

# Before the loop, initialize the awkward array structure for track samples
tracks_sample = ak.ArrayBuilder()



print("Starting Timer...")
start_time = time.time()
for data in events.iterate(fields_list, library="ak", step_size="500MB"):
    print(f"Processing a batch of {len(data)} events.")
    for event_idx, event in enumerate(data):
        print(f"Processing event: {event_idx + 1}")

        if NUM_EVENTS_TO_USE is not None:
            if event_idx >= NUM_EVENTS_TO_USE:  # Limiting processing for demonstration
                break

        '''
        GRABBING ONLY CLUSTERED CELLS, SO WE CAN IGNORE ANY CELLS NOT IN ANY CLUSTER
        (THEN UNFLATTENING SO WE HAVE A MASTER LIST OF FILTERED CELLS FOR THIS EVENT)
        ============================================================
        '''
        event_cells, track_etas, track_phis = process_and_filter_cells(event, cellgeo)
        

        tracks_sample.begin_list()  # Start a new list for each event to hold tracks
        for track_idx in range(event["nTrack"]):
            print(f"\r    Processing Track: {track_idx + 1}", end='', flush=True)
            tracks_sample.begin_record()  # Each track is a record within the event list


            '''
            GET TRACK META INFO
            ============================================================
            '''
            fields = [
                ("eventID", "integer"),
                ("trackEta_EMB2", "real"),
                ("trackPhi_EMB2", "real"),
                ("trackEta_EME2", "real"),
                ("trackPhi_EME2", "real"),
                ("trackSubtractedCaloEnergy", "real"),
                ("trackPt", "real")
                #("trackNumberDOF", "integer"),
                #("trackChiSquared", "real")
            ]

            track_eta_ref, track_phi_ref = add_track_meta_info(tracks_sample, event, event_idx, track_idx, fields)
            '''
            ============================================================
            =======
            '''

            '''
            CALCULATING TRACK X, Y, Z PATH POINTS (INTERSECTIONS WITH CELL LAYERS)
            ============================================================
            '''
            track_intersections = add_track_intersection_info(tracks_sample, track_idx, track_etas, track_phis)
            '''
            ============================================================
            =======
            '''

            '''
            GET ASSOCIATED CELL INFO (Those within deltaR of track)
            ============================================================
            '''
            process_associated_cell_info(event_cells, tracks_sample, track_eta_ref, track_phi_ref, track_intersections)
            '''
            ============================================================
            =======
            '''

            '''
            GET ASSOCIATED TRACKS (Those within deltaR of the track's eta/phi)
            ============================================================
            '''
            process_associated_tracks(event, tracks_sample, track_eta_ref, track_phi_ref, track_idx, event["nTrack"], track_etas, track_phis)
            '''
            ============================================================
            =======
            '''
            tracks_sample.end_record()  # End the record for the current track


        tracks_sample.end_list()  # End the list for the current event
        print()

# After processing, convert the ArrayBuilder to an actual Awkward array and print it
tracks_sample_array = tracks_sample.snapshot()

print_events(tracks_sample_array, 1)


max_sample_length = calculate_max_sample_length(tracks_sample_array)

