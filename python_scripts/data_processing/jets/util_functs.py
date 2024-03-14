import numpy as np
import awkward as ak
import sys
sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import calo_layers, has_fixed_r, fixed_r, fixed_z  # Assuming these are correctly defined
HAS_FIXED_R, FIXED_R, FIXED_Z = has_fixed_r, fixed_r, fixed_z

def calculate_cartesian_coordinates(eta, phi, rPerp):
    X = rPerp * np.cos(phi)
    Y = rPerp * np.sin(phi)
    Z = rPerp * np.sinh(eta)

    return X, Y, Z


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
        if eta < -100000 or phi < -100000:
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

def calculate_max_sample_length(tracks_array):
    max_length = 0
    for event in tracks_array:
        for track in event:
            length = len(track["track_layer_intersections"]) + len(track["associated_cells"])
            for associated_track in track["associated_tracks"]:
                length += len(associated_track["track_layer_intersections"])
            if length > max_length:
                max_length = length
    return max_length

def print_events(tracks_sample_array, NUM_EVENTS_TO_PRINT):
    # Weirdly structured. Is overcomplicated code-wise for more readable output
    for event_idx, event in enumerate(ak.to_list(tracks_sample_array)):
        if event_idx >= NUM_EVENTS_TO_PRINT:
            break
        print("New event")
        # Each event can contain multiple tracks
        for track in event:
            print("  Track")
            # Now, print each field and its value for the track
            for field in track:
                value = track[field]
                if field == "track_layer_intersections" or field == "associated_cells":
                    print(f"    {field}:")
                    for intpoint in value:
                        formatted_intpoint = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in intpoint.items()}
                        print(f"        {formatted_intpoint}")
                elif field == "associated_tracks":
                    print(f"    {field}:")
                    for adj_track in value:
                        for adj_field in adj_track:
                            adj_value = adj_track[adj_field]
                            if adj_field == "track_layer_intersections":
                                print(f"            {adj_field}:")
                                for layer_point in adj_value:
                                    formatted_layer_point = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in layer_point.items()}
                                    print(f"                {formatted_layer_point}")
                            else:
                                if isinstance(adj_value, float):
                                    print(f"            {adj_field}: {adj_value:.4f}")
                                else:
                                    print(f"            {adj_field}: {adj_value}")
                else:
                    if isinstance(value, float):  # Check if the value is a float and format it
                        print(f"    {field}: {value:.4f}")
                    else:  # If not a float, print the value as is
                        print(f"    {field}: {value}")
            print()


def build_input_array(tracks_sample_array, max_sample_length):
    # Initialize an empty list to hold all samples
    samples = []

    for event in tracks_sample_array:
        for track in event:
            # Check if the track has associated cells or tracks, skip if not
            if len(track['associated_cells']) == 0 and len(track['associated_tracks']) == 0:
                continue

            # Initialize a list to collect points for this track
            track_points = []

            # Add the focused track's intersection points
            for intersection in track['track_layer_intersections']:
                track_points.append([intersection['X'], intersection['Y'], intersection['Z'], 0, track['trackPt'], 1])

            # Add associated cells
            for cell in track['associated_cells']:
                track_points.append([cell['X'], cell['Y'], cell['Z'], cell['distance_to_track'], cell['E'], 0])

            # Add points from associated tracks
            for associated_track in track['associated_tracks']:
                for intersection in associated_track['track_layer_intersections']:
                    track_points.append([intersection['X'], intersection['Y'], intersection['Z'], intersection['distance_to_track'], associated_track['trackPt'], 2])

            # Check if padding is needed
            num_points = len(track_points)
            if num_points < max_sample_length:
                # Pad with zeros and -1 for class identity
                padding = [[0, 0, 0, 0, 0, -1] for _ in range(max_sample_length - num_points)]
                track_points.extend(padding)

            # Add to samples list
            samples.append(track_points[:max_sample_length])  # Ensure the sample does not exceed max_sample_length

    # Convert the list of samples to a NumPy array
    samples_array = np.array(samples, dtype=np.float32)
    return samples_array


def build_labels_array(tracks_sample_array, max_sample_length):
    # Initialize an empty list to hold all label arrays
    labels_list = []

    for event in tracks_sample_array:
        for track in event:
            # Check if the track has associated cells or tracks, skip if not
            if len(track['associated_cells']) == 0 and len(track['associated_tracks']) == 0:
                continue

            # Initialize the label array for this track, filled with -1 (for padding)
            label_array = np.full(max_sample_length, -1, dtype=np.float32)

            # Calculate the total number of points (focused track points, associated cells, associated track points)
            num_focused_track_points = len(track['track_layer_intersections'])
            num_associated_cells = len(track['associated_cells'])
            num_associated_track_points = sum(len(assoc_track['track_layer_intersections']) for assoc_track in track['associated_tracks'])

            # Set labels for focused track points to 1.0
            label_array[:num_focused_track_points] = 1.0

            # Set labels for associated cells to 0.5
            label_array[num_focused_track_points:num_focused_track_points+num_associated_cells] = 0.5

            # Set labels for associated track points to 0.0
            start_idx = num_focused_track_points + num_associated_cells
            end_idx = start_idx + num_associated_track_points
            label_array[start_idx:end_idx] = 0.0

            # Add to labels list
            labels_list.append(label_array)

    # Convert the list of label arrays into a NumPy array
    labels_array = np.array(labels_list, dtype=np.float32)
    return labels_array
