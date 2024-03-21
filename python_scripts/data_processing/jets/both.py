from track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z


# ===== FIELDS TO CHANGE =====
NUM_EVENTS_PER_CHUNK = 250
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1
# TEST_SPLIT_RATIO is implied to be the remaining percentage
NUM_THREAD_PER_CHUNK = 12
OUTPUT_DIRECTORY_NAME = "firstAttempts/"
# ============================


DEBUG_NUM_EVENTS_TO_USE = None
UPROOT_MASK_VALUE_THRESHOLD = -100000
MAX_DISTANCE = 0.2

# Path to the ROOT file containing jet events
FILE_LOC = "/data/atlas/mltree_1000.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"


SAVE_LOC = '/data/mjovanovic/jets/processed_files/' + OUTPUT_DIRECTORY_NAME + "AwkwardArrs/"
NPZ_SAVE_LOC = "/data/mjovanovic/jets/processed_files/" + OUTPUT_DIRECTORY_NAME + "SavedNpz/"

