import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z



NUM_EVENTS_TO_USE = 2
UPROOT_MASK_VALUE_THRESHOLD = -100000
MAX_DISTANCE = 0.2


# Path to the ROOT file containing jet events
#FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"
FILE_LOC = "/data/atlas/mltree.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"

SAVE_LOC = '/data/mjovanovic/jets/mltree_JZ1_0_5000events_data/training_data.npz'

# Process events and tracks as before, with the following adjustments:
track_layer_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers] # Getting all the cell layer points that the track hits (ie trackEta_EME2, trackPhi_EMB3, etc)

jets_other_included_fields = ["trackSubtractedCaloEnergy", "trackPt", "nTrack", "cluster_cell_ID",
                          "trackNumberDOF","trackChiSquared","cluster_cell_E","cluster_fullHitsTruthIndex","cluster_fullHitsTruthE"]

fields_list = track_layer_branches + jets_other_included_fields

''''
# old ending of the program (when saving to numpy)

feats = build_input_array(tracks_sample_array, max_sample_length)
labs = build_labels_array(tracks_sample_array, max_sample_length)
save_data = (feats, labs)

index = 9

print(feats[index])
print(labs[index])
print(len(feats))
print(len(labs))
print("Maximum sample size (original track + associated cells + associated track points): ",max_sample_length)

print("Stopping Timer...")
finish_time = time.time()

print("Saving to: ", SAVE_LOC)

np.savez(SAVE_LOC, features = feats, labels = labs)

duration = finish_time - start_time

print(f"Total time to run for {event_idx + 1} events took {duration/60/60} hours")

'''