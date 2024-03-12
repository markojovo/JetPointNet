import uproot
import awkward as ak
import numpy as np
import vector
import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

from util_functs import *

# Path to the ROOT file containing jet events
FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"

NUM_EVENTS_TO_USE = None

UPROOT_MASK_VALUE_THRESHOLD = -100000

# Maximum distance for cell and track identification
MAX_DISTANCE = 1.6

# Open the ROOT file and access the EventTree
events = uproot.open(FILE_LOC + ":EventTree")
cellgeo = uproot.open(GEO_FILE_LOC + ":CellGeo")

print("Events Keys:")
for key in events.keys():
    print(key)


print("\nGeometry Keys:")
for key in cellgeo.keys():
    print(key)

#print(events["trackTruthParticleIndex"].array()[0])
#print(events["cluster_cell_hitsTruthIndex"].array()[0])
#print(events["cluster_cell_hitsTruthE"].array()[0])
#exit()

'''
Done:
- Load in dataset
- Get track x, y, z values

Next:
- Outline what you need in outcome (what in labels, what in features, look at the feats/labels data you have already)
    - First, extract the tracks from the event (for each cell, get its ID, eta/phi in EMB2 and EME2), and its cell_E (measured)
    - Should be a track item, and in each track item:
        - the track info:
            - event number index
            - track particle ID (or index?)
            - x/y/z
            - EMB2 eta/phi (or EME2 if it doesnt hit that, can check metadata)
            - track point energy
            - include (as another single value info field) the total amount of energy (cell E) in the track window
            - Also the missing energy meme for testing metric
        - the cells within the deltaR distance (of the track eta/phi of the track) (include the fields from the "First, extract..." line)
            - include the minimum track distance (find the distance between the cell and the closest track point) for each cell
        - Indices of any tracks that fall within the focused track's EMB/EME eta/phi (so we can extract their x/y/z/E values later)
        

After:
- the features should be the x, y, z, energy, minimum distance_to_track and the point type (0 for cell, 1 for focused track, 2 for unfocused track)
- Now need to do the labels:
    - The label should be: for each cell (just 1d array, no xyz in label, but have it ordered with cells) how much of the cell's energy comes from the focused track
    - To do this, we grab focus particle index / ID, then grab the hitsTruthE for each cell and find the ratio of how much of the cell's energy is from the focused track particle ID
        Note: focused track will always be 1.0, unfocused track will always be 0.0 and the mask will be -1 (remember to find maximum number of cells hit by a track in dataset)

- Then, do final processing, format it into numpy files, and save!
- optimize for efficiency, file numbers and save space after 
'''



# Before the loop, initialize the awkward array structure for track samples
tracks_sample = ak.ArrayBuilder()

# Process events and tracks as before, with the following adjustments:
track_layer_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers] # Getting all the cell layer points that the track hits (ie trackEta_EME2, trackPhi_EMB3, etc)

jets_other_included_fields = ["trackSubtractedCaloEnergy", "trackPt", "trackID", "nTrack", "cluster_cell_ID", "cluster_cell_Eta", "cluster_cell_Phi",
                          "trackNumberDOF","trackChiSquared","cluster_cell_E", "cluster_cell_X","cluster_cell_Y","cluster_cell_Z","cluster_fullHitsTruthIndex","cluster_fullHitsTruthE"]

fields_list = track_layer_branches + jets_other_included_fields

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
        # Assuming `events` is your dataset, and it has cluster_cell_ID, cluster_cell_E, cluster_cell_eta, cluster_cell_phi
        cell_IDs = ak.flatten(event['cluster_cell_ID'])
        cell_Es = ak.flatten(event['cluster_cell_E'])
        cell_etas = ak.flatten(event['cluster_cell_Eta'])
        cell_phis = ak.flatten(event['cluster_cell_Phi'])
        cell_Xs = ak.flatten(event['cluster_cell_X'])  
        cell_Ys = ak.flatten(event['cluster_cell_Y'])  
        cell_Zs = ak.flatten(event['cluster_cell_Z'])  

        # To get the first occurrence indices of each unique ID
        _, unique_indices = np.unique(ak.to_numpy(cell_IDs), return_index=True)

        # Now use these indices to select the corresponding E, eta, and phi values
        unique_cell_IDs = cell_IDs[unique_indices]
        unique_cell_Es = cell_Es[unique_indices]
        unique_cell_etas = cell_etas[unique_indices]
        unique_cell_phis = cell_phis[unique_indices]
        unique_cell_Xs = cell_Xs[unique_indices]
        unique_cell_Ys = cell_Ys[unique_indices] 
        unique_cell_Zs = cell_Zs[unique_indices] 

        # Recombine into a new Awkward Array if needed
        event_cells = ak.zip({
            'ID': unique_cell_IDs,
            'E': unique_cell_Es,
            'eta': unique_cell_etas,
            'phi': unique_cell_phis,
            'X': unique_cell_Xs,  
            'Y': unique_cell_Ys,  
            'Z': unique_cell_Zs   
        })
        '''
        ============================================================
        =======
        '''

        tracks_sample.begin_list()  # Start a new list for each event to hold tracks


        # Prepare track eta and phi data for all layers (dictionary used for track X, Y, Z calculation)
        track_eta = {layer: event[f'trackEta_{layer}'] for layer in calo_layers}
        track_phi = {layer: event[f'trackPhi_{layer}'] for layer in calo_layers}


        # Initialize awkward array data structure for features
        for track_idx in range(event["nTrack"]):
            print(f"\r    Processing Track: {track_idx + 1}", end='', flush=True)
            tracks_sample.begin_record()  # Each track is a record within the event list


            '''
            GET TRACK META INFO
            ============================================================
            '''
            # List of field names and their types
            fields = [
                ("eventID", "integer"),
                ("trackID", "integer"),
                ("trackEta_EMB2", "real"),
                ("trackPhi_EMB2", "real"),
                ("trackEta_EME2", "real"),
                ("trackPhi_EME2", "real"),
                ("trackSubtractedCaloEnergy", "real"),
                ("trackPt", "real")
                #("trackNumberDOF", "integer"),
                #("trackChiSquared", "real")
            ]

            # Looped version instead of doing this (a 2-line example): tracks_sample.field("trackEta_EMB2") \ tracks_sample.real(event["trackEta_EMB2"][track_idx])  
            for field_name, field_type in fields:
                tracks_sample.field(field_name)
                if field_name in ["eventID", "trackID"]:  # Handle integer fields
                    if field_name == "eventID":
                        tracks_sample.integer(event_idx)  # Assuming event_idx is the ID
                    else:  # For trackID, fetch from the event dictionary
                        tracks_sample.integer(event[field_name][track_idx])
                else:  # Handle real number fields
                    if not event[field_name][track_idx] < UPROOT_MASK_VALUE_THRESHOLD:
                        tracks_sample.real(event[field_name][track_idx])

            track_eta_ref = event["trackEta_EMB2"][track_idx]  
            track_phi_ref = event["trackPhi_EMB2"][track_idx]  
            if track_eta_ref < UPROOT_MASK_VALUE_THRESHOLD:
                track_eta_ref = event["trackEta_EME2"][track_idx] 
                track_phi_ref = event["trackPhi_EME2"][track_idx] 
            tracks_sample.field("trackEta")
            tracks_sample.real(track_eta_ref)

            tracks_sample.field("trackPhi")
            tracks_sample.real(track_phi_ref)
            '''
            ============================================================
            =======
            '''

            '''
            CALCULATING TRACK X, Y, Z PATH POINTS (INTERSECTIONS WITH CELL LAYERS)
            ============================================================
            '''
            # Calculate intersections for each track
            track_intersections = calculate_track_intersections({layer: eta[track_idx] for layer, eta in track_eta.items()},
                                                                {layer: phi[track_idx] for layer, phi in track_phi.items()})
            
            # Add track intersection information
            tracks_sample.field("track_layer_intersections")
            tracks_sample.begin_list()  # Start list of intersection points for this track
            for layer, (x, y, z) in track_intersections.items():
                tracks_sample.begin_record()  # Each intersection point is a record
                tracks_sample.field("layer")
                tracks_sample.string(layer)
                tracks_sample.field("X")
                tracks_sample.real(x)
                tracks_sample.field("Y")
                tracks_sample.real(y)
                tracks_sample.field("Z")
                tracks_sample.real(z)
                tracks_sample.end_record()  # End the record for this intersection point
            tracks_sample.end_list()  # End list of intersection points
            '''
            ============================================================
            =======
            '''




            '''
            GET ASSOCIATED CELL INFO (Those within deltaR of track)
            ============================================================
            '''
            # Use cell eta and phi directly from the `cells` structured array
            cell_eta = event_cells['eta']
            cell_phi = event_cells['phi']

            # Vectorized calculation of delta R for all cells with respect to the track
            delta_r = calculate_delta_r(track_eta_ref, track_phi_ref, cell_eta, cell_phi)

            # Creating a mask for cells within the delta R threshold of 0.2
            mask = delta_r <= MAX_DISTANCE
            '''
            NOTE:
                This is returning some masks where NO cells (that are part of any cluster) are within the eta/phi range of the track.
                This is weird, since everywhere (except eta > ~5) should have at least some cells near it
                Check:
                    The cells are being filtered by clustering?
                    The delta_R calculation is weird (ie not considering rollover)
            '''

            # Apply the mask to filter cells directly using Awkward Array's boolean masking
            filtered_cells = event_cells[mask]
            tracks_sample.field("total_associated_cell_energy").real(sum(filtered_cells["E"]))

            # Preparing to add the filtered cells to the track sample
            tracks_sample.field("associated_cells")
            tracks_sample.begin_list()

            track_intersection_points = [(x, y, z) for layer, (x, y, z) in track_intersections.items()]
            # Iterate over filtered cells, now correctly filtered with delta_r <= 0.2
            for cell in filtered_cells:
                tracks_sample.begin_record()
                tracks_sample.field("ID").integer(cell["ID"])
                tracks_sample.field("E").real(cell["E"])
                tracks_sample.field("X").real(cell["X"])
                tracks_sample.field("Y").real(cell["Y"])
                tracks_sample.field("Z").real(cell["Z"])
                # Calculate distances to each track intersection point and find the minimum
                cell_x, cell_y, cell_z = cell["X"], cell["Y"], cell["Z"]
                min_distance = min(
                    np.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2 + (z - cell_z) ** 2)
                    for x, y, z in track_intersection_points
                )
                tracks_sample.field("distance_to_track").real(min_distance)
                tracks_sample.field("eta").real(cell["eta"])
                tracks_sample.field("phi").real(cell["phi"])


                tracks_sample.end_record()
                

            tracks_sample.end_list()


            '''
            ============================================================
            =======
            '''



            '''
            GET ASSOCIATED TRACKS (Those within deltaR of the track's eta/phi)
            ============================================================
            '''
            # Initialize the field for adjacent tracks
            tracks_sample.field("associated_tracks")
            tracks_sample.begin_list()  # Begin list of adjacent tracks for this track

            # Calculate eta/phi for the focal track
            focal_eta = track_eta_ref  # Defined in your existing code
            focal_phi = track_phi_ref  # Defined in your existing code

            # Retrieve focal track's intersection points for distance calculation
            focal_track_intersections = calculate_track_intersections({layer: eta[track_idx] for layer, eta in track_eta.items()},
                                                                    {layer: phi[track_idx] for layer, phi in track_phi.items()})
            focal_points = [(x, y, z) for _, (x, y, z) in focal_track_intersections.items()]

            # Iterate over all tracks in the event to find adjacent tracks
            for adj_track_idx in range(event["nTrack"]):
                if adj_track_idx == track_idx:  # Skip the focal track itself
                    continue
                
                # Determine reference eta/phi for the adjacent track
                adj_track_eta = event["trackEta_EMB2"][adj_track_idx] if event["trackEta_EMB2"][adj_track_idx] > UPROOT_MASK_VALUE_THRESHOLD else event["trackEta_EME2"][adj_track_idx]
                adj_track_phi = event["trackPhi_EMB2"][adj_track_idx] if event["trackPhi_EMB2"][adj_track_idx] > UPROOT_MASK_VALUE_THRESHOLD else event["trackPhi_EME2"][adj_track_idx]

                # Calculate delta R between focal and adjacent track
                delta_r_adj = calculate_delta_r(focal_eta, focal_phi, adj_track_eta, adj_track_phi)

                # Check if adjacent track is within MAX_DISTANCE
                if delta_r_adj <= MAX_DISTANCE:
                    # Add the adjacent track's information
                    tracks_sample.begin_record()
                    tracks_sample.field("trackIdx").integer(adj_track_idx)
                    tracks_sample.field("trackPt").real(event["trackPt"][adj_track_idx])
                    
                    # Calculate and add adjacent track's intersection points along with minimum distance to focal track
                    tracks_sample.field("track_layer_intersections")
                    tracks_sample.begin_list()
                    adj_track_intersections = calculate_track_intersections({layer: eta[adj_track_idx] for layer, eta in track_eta.items()},
                                                                            {layer: phi[adj_track_idx] for layer, phi in track_phi.items()})
                    
                    for layer, (x, y, z) in adj_track_intersections.items():
                        min_distance_to_focal = min(
                            np.sqrt((fx - x) ** 2 + (fy - y) ** 2 + (fz - z) ** 2)
                            for fx, fy, fz in focal_points
                        )
                        
                        tracks_sample.begin_record()
                        tracks_sample.field("layer").string(layer)
                        tracks_sample.field("X").real(x)
                        tracks_sample.field("Y").real(y)
                        tracks_sample.field("Z").real(z)
                        tracks_sample.field("distance_to_track").real(min_distance_to_focal)
                        tracks_sample.end_record()
                    tracks_sample.end_list()

                    tracks_sample.end_record()

            tracks_sample.end_list()
            '''
            ============================================================
            =======
            '''
            tracks_sample.end_record()  # End the record for the current track


        tracks_sample.end_list()  # End the list for the current event
        print()

# After processing, convert the ArrayBuilder to an actual Awkward array and print it
tracks_sample_array = tracks_sample.snapshot()


print_events(tracks_sample_array, NUM_EVENTS_TO_PRINT=1)
print(tracks_sample_array.fields)
print(tracks_sample_array["track_layer_intersections"].fields)
print(tracks_sample_array["associated_cells"].fields)
print(tracks_sample_array["associated_tracks"].fields)
print(tracks_sample_array["associated_tracks"]["track_layer_intersections"].fields)

max_sample_length = calculate_max_sample_length(tracks_sample_array)

feats = build_input_array(tracks_sample_array, max_sample_length)
labs = build_labels_array(tracks_sample_array, max_sample_length)
save_data = (feats, labs)

index = 9

print(feats[index])
print(labs[index])
print(len(feats))
print(len(labs))
print("Maximum sample size (original track + associated cells + associated track points): ",max_sample_length)

save_loc = '/data/mjovanovic/jets/mltree_JZ1_0_5000events_data/training_data.npz'
print("Saving to: ", save_loc)

np.savez(save_loc, features = feats, labels = labs)