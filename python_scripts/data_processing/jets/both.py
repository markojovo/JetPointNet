import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

NUM_EVENT_CHUNKS = 10
CHUNK_SIZE = 5000 // NUM_EVENT_CHUNKS
NUM_THREAD_PER_CHUNK = 10
NUM_EVENTS_TO_USE = None
UPROOT_MASK_VALUE_THRESHOLD = -100000
MAX_DISTANCE = 0.2


# Path to the ROOT file containing jet events
#FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"
FILE_LOC = "/data/atlas/mltree.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"

SAVE_LOC = '/data/mjovanovic/jets/mltree_JZ1_0_5000events_data/training_data.npz'


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