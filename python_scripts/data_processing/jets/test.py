import uproot
import awkward as ak
import numpy as np
import vector
import sys
import time
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

from util_functs import *

# Path to the ROOT file containing jet events
FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"
#FILE_LOC = "/data/atlas/mltree.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"
save_loc = '/data/mjovanovic/jets/mltree_JZ1_0_5000events_data/training_data.npz'


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

# Load Cell Geometry Data
cell_ID_geo = cellgeo["cell_geo_ID"].array(library="ak")[0]
eta_geo = cellgeo["cell_geo_eta"].array(library="ak")[0]
phi_geo = cellgeo["cell_geo_phi"].array(library="ak")[0]
rPerp_geo = cellgeo["cell_geo_rPerp"].array(library="ak")[0]

# Before the loop, initialize the awkward array structure for track samples
tracks_sample = ak.ArrayBuilder()

# Process events and tracks as before, with the following adjustments:
track_layer_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers] # Getting all the cell layer points that the track hits (ie trackEta_EME2, trackPhi_EMB3, etc)

jets_other_included_fields = ["trackSubtractedCaloEnergy", "trackPt", "trackID", "nTrack", "cluster_cell_ID", "cluster_cell_Eta", "cluster_cell_Phi",
                          "trackNumberDOF","trackChiSquared","cluster_cell_E", "cluster_cell_X","cluster_cell_Y","cluster_cell_Z","cluster_fullHitsTruthIndex","cluster_fullHitsTruthE"]

fields_list = track_layer_branches + jets_other_included_fields

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
        # Assuming `events` is your dataset, and it has cluster_cell_ID, cluster_cell_E, cluster_cell_eta, cluster_cell_phi
        cell_IDs_with_multiples = ak.flatten(event['cluster_cell_ID'])
        cell_Es_with_multiples = ak.flatten(event['cluster_cell_E'])

        #cell_ID_geo
        #eta_geo
        #phi_geo
        #rPerp_geo

        # To get the first occurrence indices of each unique ID
        _, unique_indices = np.unique(ak.to_numpy(cell_IDs_with_multiples), return_index=True)

        # Now use these indices to select the corresponding E, eta, and phi values
        cell_IDs = cell_IDs_with_multiples[unique_indices]
        cell_Es = cell_Es_with_multiples[unique_indices]

        cell_IDs_array = np.array(cell_IDs)  # If cell_IDs is already a numpy array, this step can be skipped
        cell_ID_geo_array = np.array(cell_ID_geo)  # Same as above, skip if already a numpy array

        # Find which elements of cell_ID_geo are in cell_IDs
        mask = np.isin(cell_ID_geo_array, cell_IDs_array)

        # Find the indices of these elements
        indices_of_geo_that_contain_event_cells = np.where(mask)[0]

        print(cell_IDs)
        print(cell_ID_geo[indices_of_geo_that_contain_event_cells])

        print(len(cell_IDs))
        print(len(cell_ID_geo))
        exit()