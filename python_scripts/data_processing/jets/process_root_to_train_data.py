import uproot
import awkward as ak
import numpy as np
import vector
import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import *


# Path to the ROOT file containing jet events
FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"

# Maximum distance for cell and track identification
MAX_DISTANCE = 0.2

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
    x, y, z = eta_phi_to_cartesian(eta, phi, R=1)
    x *= fixed_r
    y *= fixed_r
    z *= fixed_r * np.sinh(eta)

    return x, y, z



# Define the function to calculate the intersection with a fixed Z layer

def intersection_fixed_z(eta, phi, fixed_z):
    x, y, z_unit = eta_phi_to_cartesian(eta, phi)
    scale_factor = fixed_z / z_unit
    x *= scale_factor
    y *= scale_factor
    z = fixed_z

    return x, y, z



'''
TODO:
1) Load root file for jets  ( /fast_scratch_1/atlas_images/jets )
2) set up the nested loops, there are 3 in total (outer to inner)
- Loop 1: Loop through events in root file
- Identify tracks in event
- Loop 2: Loop through tracks
- For each track, identify the tracked particle ID then get the eta and phi coordinates of the track (in EMB2 or EMEC2 if theyre in the end-caps (not sure how to tell yet))
- Loop 3: loop through all the cells and mark the cell IDs (add to an array?) of those that have a euclidian distance (in eta-phi coordinates) less than MAX_DISTANCE (< 0.2) from the track

We'll stop here then continue with stuff like cell filtering or dealing with overlaps or stuff like that. Make use of awkward arrays where appropriate (awkward library), otherwise use uproot
'''
# Printing available keys for diagnostics
print("Available keys in the dataset:\n ======================")
for key in events.keys():
    print(key)

print((events["trackEta_EMB2"].array()[0]))
print(events["nTrack"].array()[0])


trackEtaStr1 = "trackEta_EMB2"  # Barrel eta
trackPhiStr1 = "trackPhi_EMB2"  # Barrel phi
trackEtaStr2 = "trackEta_EME2"  # Endcap eta
trackPhiStr2 = "trackPhi_EME2"  # Endcap phi

# Helper function to calculate delta R using eta and phi directly
def calculate_delta_r(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi2 - phi1 + np.pi, 2 * np.pi) - np.pi
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)

for data in events.iterate([trackEtaStr1, trackPhiStr1, trackEtaStr2, trackPhiStr2, "nTrack",
                            "cluster_cell_ID", "cluster_cell_Eta", "cluster_cell_Phi", "cluster_cell_E"],
                           library="ak", step_size="100MB"):
    print(f"Processing a batch of {len(data)} events.")

    for event_idx, event in enumerate(data):
        tracks = ak.zip({
            "eta1": event[trackEtaStr1],
            "phi1": event[trackPhiStr1],
            "eta2": event[trackEtaStr2],
            "phi2": event[trackPhiStr2]
        })
        
        cells = ak.zip({
            "ID": event["cluster_cell_ID"],
            "eta": event["cluster_cell_Eta"],
            "phi": event["cluster_cell_Phi"],
            "energy": event["cluster_cell_E"]
        })

        # Loop over tracks
        for track_idx, track in enumerate(tracks):
            if track.eta1 > -10000:
                eta = track.eta1
                phi = track.phi1
                source = "Barrel"
            else:
                eta = track.eta2
                phi = track.phi2
                source = "Endcap"

            
            #if abs(eta) > 10000 or abs(phi) > 10000:  # Additional check for invalid EME2 values
            #    print(f"Skipped track {track_idx} due to invalid eta/phi")
            #    continue

            contained_cell_energies = []
            # Loop over cells
            for cell in cells:
                delta_r = calculate_delta_r(eta, phi, cell.eta, cell.phi)
                
                # Mask for cells within MAX_DISTANCE of track
                mask = delta_r < MAX_DISTANCE
                if ak.any(mask):
                    contained_cell_energies.extend(ak.to_list(cell.energy[mask]))
            
            # Print energies in a flattened list format
            print(f"Track {track_idx} Source: (Eta: {eta}, Phi: {phi})\ncontains {len(contained_cell_energies)} cells with energies: \n{np.round(contained_cell_energies,2)}\n\n")
