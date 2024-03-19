import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

NUM_EVENTS_PER_CHUNK = 250
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1
# TEST_SPLIT_RATIO is implied to be the remaining percentage
NUM_THREAD_PER_CHUNK = 12

DEBUG_NUM_EVENTS_TO_USE = None
UPROOT_MASK_VALUE_THRESHOLD = -100000
MAX_DISTANCE = 0.2


# Path to the ROOT file containing jet events
FILE_LOC = "/data/atlas/mltree_1000.root"
#FILE_LOC = "/data/mjovanovic/cell_particle_deposit_learning/rho/rho_root_files/rho_full.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"

OUTPUT_DIRECTORY_NAME = "firstAttempts/"
SAVE_LOC = '/data/mjovanovic/jets/outputAwks/' + OUTPUT_DIRECTORY_NAME


