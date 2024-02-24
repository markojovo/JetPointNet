import uproot
import awkward as ak
import numpy as np
import vector



# Path to the ROOT file containing jet events
FILE_LOC = "/fast_scratch_1/atlas_images/jets/mltree_JZ1_0_5000events.root"

# Maximum distance for cell and track identification
MAX_DISTANCE = 1.6

# Open the ROOT file and access the EventTree
events = uproot.open(FILE_LOC + ":EventTree")

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


# Helper function to calculate delta R using eta and phi directly
def calculate_delta_r(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi2 - phi1 + np.pi, 2 * np.pi) - np.pi
    deta = eta2 - eta1
    return np.sqrt(deta**2 + dphi**2)

for data in events.iterate(["trackEta", "trackPhi", "nTrack",
                            "cluster_cell_ID", "cluster_cell_Eta", "cluster_cell_Phi", "cluster_cell_E"],
                           library="ak", step_size="100MB"):
    print(f"Processing a batch of {len(data)} events.")

    for event_idx, event in enumerate(data):
        print(f"\nEvent {event_idx+1} details:")
        
        track_eta = event["trackEta"]
        track_phi = event["trackPhi"]
        cells = ak.zip({
            "ID": event["cluster_cell_ID"],
            "eta": event["cluster_cell_Eta"],
            "phi": event["cluster_cell_Phi"],
            "energy": event["cluster_cell_E"]
        })

        # Loop over tracks
        for track_idx, (eta1, phi1) in enumerate(zip(track_eta, track_phi)):
            contained_cell_energies = []
            # Loop over cells
            for cell in cells:
                cell_eta = cell["eta"]
                cell_phi = cell["phi"]
                cell_energy = cell["energy"]
                
                # Calculate delta R for each cell with respect to the current track
                # Here, we assume eta and phi are single values for each cell
                # If they are arrays, you'll need additional handling
                delta_r = calculate_delta_r(eta1, phi1, cell_eta, cell_phi)
                
                # Now we have to use ak.any or ak.all since delta_r is an array
                if ak.any(delta_r < MAX_DISTANCE):
                    # This assumes we want to add the energy if any part of the cell is within range
                    contained_cell_energies.append(cell_energy)

            print(f"Track {track_idx} contains cells with energies: {contained_cell_energies}")

        if event_idx > 0:  # Process only the first event for demonstration
            break
