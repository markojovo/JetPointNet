import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

NUM_EVENTS_PER_CHUNK = 25
NUM_THREAD_PER_CHUNK = 10

DEBUG_NUM_EVENTS_TO_USE = None
UPROOT_MASK_VALUE_THRESHOLD = -100000
MAX_DISTANCE = 0.2


# Path to the ROOT file containing jet events
FILE_LOC = "/data/atlas/mltree_1000.root"
#FILE_LOC = "/data/mjovanovic/cell_particle_deposit_learning/rho/rho_root_files/rho_full.root"
GEO_FILE_LOC = "/data/atlas/data/rho_delta/rho_small.root"

SAVE_LOC = '/data/mjovanovic/jets/outputAwks/'


