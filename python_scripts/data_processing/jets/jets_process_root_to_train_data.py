import uproot
import awkward as ak
import numpy as np
import vector
import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

# Path to the ROOT file containing jet events
FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"

# Maximum distance for cell and track identification
MAX_DISTANCE = 1.6

# Open the ROOT file and access the EventTree
events = uproot.open(FILE_LOC + ":EventTree")

for key in events.keys():
    print(key)

print(events["trackID"].array()[0])
print(events["trackP"].array()[0])


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

# Define the function to convert eta and phi to cartesian coordinates
def eta_phi_to_cartesian(eta, phi, R=1):
    theta = 2 * np.arctan(np.exp(-eta))
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = R / np.tan(theta)
    return x, y, z

# Define the function to calculate the intersection with a fixed R layer
def intersection_fixed_r(eta, phi, fixed_r):
    x, y, z = eta_phi_to_cartesian(eta, phi, R=fixed_r)
    return x, y, z

# Define the function to calculate the intersection with a fixed Z layer
def intersection_fixed_z(eta, phi, fixed_z):
    x, y, z_unit = eta_phi_to_cartesian(eta, phi)
    scale_factor = fixed_z / z_unit
    x *= scale_factor
    y *= scale_factor
    z = fixed_z
    return x, y, z

# Helper function to calculate delta R using eta and phi directly
def calculate_delta_r(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi2 - phi1 + np.pi, 2 * np.pi) - np.pi
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)


# Define the function to calculate the intersection points for each track
def calculate_track_intersections(track_eta, track_phi):
    intersections = {}
    for layer in calo_layers:
        eta = track_eta[layer]
        phi = track_phi[layer]
        # Skip calculation for invalid eta, phi values
        if eta < -10000 or phi < -10000:
            continue
        # Calculate intersection based on layer type
        if HAS_FIXED_R.get(layer, False):
            x, y, z = intersection_fixed_r(eta, phi, FIXED_R[layer])
        elif layer in FIXED_Z:
            x, y, z = intersection_fixed_z(eta, phi, FIXED_Z[layer])
        else:
            raise Exception("Error: cell layers must either be fixed R or fixed Z, and not neither")
        intersections[layer] = (x, y, z)
    return intersections

# Before the loop, initialize the awkward array structure for track samples
tracks_sample = ak.ArrayBuilder()

# Process events and tracks as before, with the following adjustments:
track_layer_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers] # Getting all the cell layer points that the track hits (ie trackEta_EME2, trackPhi_EMB3, etc)
other_included_fields = ["trackSubtractedCaloEnergy", "trackP", "trackID", "nTrack", "cluster_cell_ID", "cluster_cell_Eta", "cluster_cell_Phi",
                          "cluster_cell_E", "cluster_cell_X","cluster_cell_Y","cluster_cell_Z","cluster_fullHitsTruthIndex","cluster_fullHitsTruthE"]

for data in events.iterate(track_layer_branches + other_included_fields, library="ak", step_size="100MB"):
    print(f"Processing a batch of {len(data)} events.")
    for event_idx, event in enumerate(data):
        if event_idx > 0:  # Limiting processing for demonstration
            break

        tracks_sample.begin_list()  # Start a new list for each event to hold tracks

        # Prepare track eta and phi data for all layers (dictionary used for track X, Y, Z calculation)
        track_eta = {layer: event[f'trackEta_{layer}'] for layer in calo_layers}
        track_phi = {layer: event[f'trackPhi_{layer}'] for layer in calo_layers}


        # Initialize awkward array data structure for features
        for track_idx in range(event["nTrack"]):
            tracks_sample.begin_record()  # Each track is a record within the event list


            '''
            GET TRACK META INFO
            ============================================================
            '''
            # Event ID (assuming event index can serve as an ID)
            tracks_sample.field("eventID")
            tracks_sample.integer(event_idx)
            
            # Track Particle ID (assuming a placeholder name, replace with actual field name)
            tracks_sample.field("trackID")
            tracks_sample.integer(event["trackID"][track_idx])  # Adjust field name as necessary
            
            # Track's eta in EMB2 (replace 'trackEta_EMB2' with actual field name if different)
            tracks_sample.field("trackEta_EMB2")
            tracks_sample.real(event["trackEta_EMB2"][track_idx])  # Adjust field name as necessary
            
            # Track's phi in EMB2 (replace 'trackPhi_EMB2' with actual field name if different)
            tracks_sample.field("trackPhi_EMB2")
            tracks_sample.real(event["trackPhi_EMB2"][track_idx])  # Adjust field name as necessary
            
            # Assuming similar fields exist for EME2, replace with actual names if different
            # Track's eta in EME2
            tracks_sample.field("trackEta_EME2")
            tracks_sample.real(event["trackEta_EME2"][track_idx])  # Adjust field name as necessary
            
            # Track's phi in EME2
            tracks_sample.field("trackPhi_EME2")
            tracks_sample.real(event["trackPhi_EME2"][track_idx])  # Adjust field name as necessary
            
            # Missing Energy Metric (assuming a placeholder name, replace with actual field name)
            tracks_sample.field("trackSubtractedCaloEnergy")
            tracks_sample.real(event["trackSubtractedCaloEnergy"][track_idx])  # Adjust field name as necessary

            # Track's phi in EME2
            tracks_sample.field("trackP")
            tracks_sample.real(event["trackP"][track_idx])  # Adjust field name as necessary
            '''
            ============================================================
            =======
            '''



            '''
            GET ASSOCIATED CELL INFO (Those within deltaR of track)
            ============================================================
            '''
            # Assuming track_eta and track_phi contain the eta and phi for EMB2/EME2 for each track
            track_eta_emb2 = event["trackEta_EMB2"][track_idx]  # Example for EMB2, adjust as necessary
            track_phi_emb2 = event["trackPhi_EMB2"][track_idx]  # Example for EMB2, adjust as necessary

            # Begin a list to hold cells associated with this track
            tracks_sample.field("associated_cells")
            tracks_sample.begin_list()

            # Iterate over each cell in the event
            for cell_idx in range(len(event["cluster_cell_Eta"])):
                cell_eta = event["cluster_cell_Eta"][cell_idx]
                cell_phi = event["cluster_cell_Phi"][cell_idx]
                
                # Calculate delta R between the track and cell
                delta_r = calculate_delta_r(track_eta_emb2, track_phi_emb2, cell_eta, cell_phi)
                
                # Check if the cell is within the 0.2 distance threshold
                if delta_r <= 0.2:
                    # If within threshold, add the cell's information
                    tracks_sample.begin_record()
                    tracks_sample.field("ID")
                    tracks_sample.integer(event["cluster_cell_ID"][cell_idx])
                    tracks_sample.field("E")
                    tracks_sample.real(event["cluster_cell_E"][cell_idx])
                    tracks_sample.field("Eta")
                    tracks_sample.real(cell_eta)
                    tracks_sample.field("Phi")
                    tracks_sample.real(cell_phi)
                    tracks_sample.field("X")
                    tracks_sample.real(event["cluster_cell_X"][cell_idx])
                    tracks_sample.field("Y")
                    tracks_sample.real(event["cluster_cell_Y"][cell_idx])
                    tracks_sample.field("Z")
                    tracks_sample.real(event["cluster_cell_Z"][cell_idx])
                    tracks_sample.field("fullHitsTruthIndex")
                    tracks_sample.integer(event["cluster_fullHitsTruthIndex"][cell_idx])
                    tracks_sample.field("fullHitsTruthE")
                    tracks_sample.real(event["cluster_fullHitsTruthE"][cell_idx])
                    tracks_sample.end_record()

            tracks_sample.end_list()  # End list of associated cells
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
                tracks_sample.field("x")
                tracks_sample.real(x)
                tracks_sample.field("y")
                tracks_sample.real(y)
                tracks_sample.field("z")
                tracks_sample.real(z)
                tracks_sample.end_record()  # End the record for this intersection point
            tracks_sample.end_list()  # End list of intersection points
            '''
            ============================================================
            =======
            '''



            '''
            GET (trackwise) CELL/TRACK POINT LABELS
            ============================================================
            '''


            '''
            ============================================================
            =======
            '''
            tracks_sample.end_record()  # End the record for the current track

        tracks_sample.end_list()  # End the list for the current event

# After processing, convert the ArrayBuilder to an actual Awkward array and print it
tracks_sample_array = tracks_sample.snapshot()

# Loop through each event in the array
for event in ak.to_list(tracks_sample_array):
    print("New event")
    # Each event can contain multiple tracks
    for track in event:
        print("  Track")
        # Now, print each field and its value for the track
        for field in track:
            value = track[field]
            print(f"    {field}: {value}")
        print()