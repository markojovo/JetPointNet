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
            # Skip layers without fixed R or Z for simplicity
            continue
        intersections[layer] = (x, y, z)
    return intersections

# Process events and tracks as before, with the following adjustments:
track_branches = [f'trackEta_{layer}' for layer in calo_layers] + [f'trackPhi_{layer}' for layer in calo_layers]

for data in events.iterate(track_branches + ["nTrack", "cluster_cell_ID", "cluster_cell_Eta", "cluster_cell_Phi", "cluster_cell_E"], library="ak", step_size="100MB"):
    print(f"Processing a batch of {len(data)} events.")
    for event_idx, event in enumerate(data):
        if event_idx > 0:  # Limiting processing for demonstration
            break
        
        # Prepare track eta and phi data for all layers
        track_eta = {layer: event[f'trackEta_{layer}'] for layer in calo_layers}
        track_phi = {layer: event[f'trackPhi_{layer}'] for layer in calo_layers}

        for track_idx in range(event["nTrack"]):
            # Calculate intersections for each track
            track_intersections = calculate_track_intersections({layer: eta[track_idx] for layer, eta in track_eta.items()},
                                                                {layer: phi[track_idx] for layer, phi in track_phi.items()})
            
            # Here, you can process track_intersections as needed, e.g., print or use them for further analysis
            print(f"Track {track_idx} intersects at (X, Y, Z):")
            for t in track_intersections:
                print(t, ": ", track_intersections[t])
            print()