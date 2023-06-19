import argparse
import time
import numpy as np
import yaml
import sys
import uproot
import awkward as ak
import multiprocessing

sys.path.append('/home/jbohm/start_tf/LCStudies')

LOG_ENERGY_MEAN = -1 # unrounded mean is ~ -0.93
LOG_MEAN_TRACK_MOMETUM = 2

# track metadata
calo_layers = ['EMB1', 'EMB2', 'EMB3', 'EME1', 'EME2', 'EME3', 'HEC0', 'HEC1',
    'HEC2', 'HEC3', 'TileBar0', 'TileBar1', 'TileBar2', 'TileGap1', 'TileGap2',
    'TileGap3', 'TileExt0', 'TileExt1', 'TileExt2']

NUM_TRACK_POINTS = len(calo_layers)

has_fixed_r = {
    'EMB1': True,
    'EMB2': True,
    'EMB3': True,
    'EME1': False,
    'EME2': False,
    'EME3': False,
    'HEC0': False,
    'HEC1': False,
    'HEC2': False,
    'HEC3': False,
    'TileBar0': True,
    'TileBar1': True,
    'TileBar2': True,
    'TileGap1': True,
    'TileGap2': True,
    'TileGap3': True,
    'TileExt0': True,
    'TileExt1': True,
    'TileExt2': True
}

fixed_r = {
    'EMB1': 1532.18,
    'EMB2': 1723.89,
    'EMB3': 1923.02,
    'TileBar0': 2450.00,
    'TileBar1': 2995.00,
    'TileBar2': 3630.00,
    'TileGap1': 3215.00,
    'TileGap2': 3630.00,
    'TileGap3': 2246.50,
    'TileExt0': 2450.00,
    'TileExt1': 2870.00,
    'TileExt2': 3480.00
}

fixed_z = {
    'EME1': 3790.03,
    'EME2': 3983.68,
    'EME3': 4195.84,
    'HEC0': 4461.25,
    'HEC1': 4869.50,
    'HEC2': 5424.50,
    'HEC3': 5905.00,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, type=str)
    return parser.parse_args()

# by generic definition of spherical to cartesian coord conversion
def spherical_to_cartesian_prev_wrong(rs, thetas, phis):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.sin(phis[i])*np.cos(thetas[i]))
        y.append(r*np.sin(phis[i])*np.sin(thetas[i]))
        z.append(r*np.cos(phis[i]))

    return np.array(x), np.array(y), np.array(z)

def spherical_to_cartesian(rs, phis, thetas):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.cos(phis[i]))
        y.append(r*np.sin(phis[i]))
        z.append( r/np.tan(thetas[i]))

    return np.array(x), np.array(y), np.array(z)

def process_file(args):
    file_name, pion_dir, save_dir, node_feature_names, file_features, cell_geo_data, sorter, energy_threshold, group_events, max_points_queue, mixed_pions, weight_cells, em_frac_threshold, rho_events, include_tracks, left_from_include_tracks, add_3_min_dists_and_layer = args
    print(file_name)
    if rho_events:
        event_data = uproot.open("/data/atlas/data/rho_delta/rho_small.root:EventTree").arrays(library="np")
    else:
        event_data = np.load(pion_dir + file_name, allow_pickle=True).item()

    num_events = len(event_data["eventNumber"])

    # cluster data dict to look up data by feature name
    processed_event_data = {}
    processed_event_track_data = {}
    processed_event_track_flags = []

    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]

    samples_count = 0
    max_cells = 0
    num_cells = 0

    for feature in [*node_feature_names, 'x', 'y', 'z', 'truth_EM_frac', 'cluster_cell_E', 'cluster_cell_E_weight', 'min_dist', 'min_phi', 'min_eta', 'sampling_layer']:
        processed_event_data[feature] = []

    for feature in ['x', 'y', 'z', 'P', 'min_dist', 'min_eta', 'min_phi', 'sampling_layer']:
        processed_event_track_data[feature] = []
    for event_idx in range(num_events):
        num_clusters = event_data["nCluster"][event_idx]
        num_tracks = event_data["nTrack"][event_idx]

        if (include_tracks and num_tracks == 1) or (left_from_include_tracks and num_tracks != 1) or (not left_from_include_tracks and not include_tracks):
            if include_tracks:
                if num_tracks == 1:
                    # get track info
                    # considering 6 track hits, one in each section of calorimeter
                    eta_tracks = []
                    phi_tracks = []
                    rPerp_track = []
                    # get the eta, phi, and rPerp of each layers hit
                    for layer_name in calo_layers:
                        if has_fixed_r[layer_name]:
                            eta_tracks.append(event_data['trackEta_' + layer_name][event_idx][0])
                            phi_tracks.append(event_data['trackPhi_' + layer_name][event_idx][0])
                            rPerp_track.append(fixed_r[layer_name])
                        else:
                            eta = event_data['trackEta_' + layer_name][event_idx][0]
                            eta_tracks.append(eta)
                            phi_tracks.append(event_data['trackPhi_' + layer_name][event_idx][0])
                            
                            z = fixed_z[layer_name]
                            aeta = np.abs(eta)
                            rPerp = z*2*np.exp(aeta)/(np.exp(2*aeta) - 1)
                            rPerp_track.append(rPerp)
                    
                    # convert each hit to the cartesian coords
                    thetas = [2*np.arctan(np.exp(-eta)) for eta in eta_tracks]
                    x_tracks, y_tracks, z_tracks = spherical_to_cartesian(rPerp_track, phi_tracks, thetas)

                    non_null_tracks = np.full(NUM_TRACK_POINTS, True)

                    # if eta > 4.9 or phi > pi mark the track as null (why??)
                    non_null_tracks[(np.abs(eta_tracks) >= 2.5) | (np.abs(phi_tracks) > np.pi)] = False

                    eta_tracks = np.array(eta_tracks)
                    phi_tracks = np.array(phi_tracks)
                else:
                    non_null_tracks = np.full(NUM_TRACK_POINTS, 0)
                    x_tracks, y_tracks, z_tracks = np.zeros(NUM_TRACK_POINTS), np.zeros(NUM_TRACK_POINTS), np.zeros(NUM_TRACK_POINTS)

            if not include_tracks or np.count_nonzero(non_null_tracks) != 0:
                # Only keep event if there is at least one track hit
                # append empty array for all cells in an event
                if group_events:
                    for feature in [*node_feature_names, 'x', 'y', 'z', 'truth_EM_frac', 'cluster_cell_E', 'cluster_cell_E_weight', 'min_dist', 'min_eta', 'min_phi']:
                        processed_event_data[feature].append([])
                for cluster_idx in range(num_clusters):
                    cell_IDs = event_data['cluster_cell_ID'][event_idx][cluster_idx]

                    # get truth enery (EM/nonEM)
                    cell_hits_E_EM = np.array(event_data["cluster_cell_hitsE_EM"][event_idx][cluster_idx])
                    cell_hits_E_non_EM = np.array(event_data["cluster_cell_hitsE_nonEM"][event_idx][cluster_idx])
                    cell_hits_E = cell_hits_E_EM + cell_hits_E_non_EM

                    # filter out cells with truth energy of 0
                    threshold_E = cell_hits_E > energy_threshold
                    
                    cell_IDs = cell_IDs[threshold_E]
                    num_cells += len(cell_IDs)
                    cell_ID_map = sorter[np.searchsorted(cell_geo_ID, cell_IDs, sorter=sorter)]

                    # get truth energy fraction EM/EM + nonEM for each cell
                    truth_EM_frac = cell_hits_E_EM[threshold_E] / cell_hits_E[threshold_E]

                    # classify truth_EM_frac
                    truth_EM_frac = [0 if truth_EM_frac_i <= em_frac_threshold else 1 for truth_EM_frac_i in truth_EM_frac]

                    # get cluster cell energy
                    cluster_cell_E = event_data["cluster_cell_E"][event_idx][cluster_idx][threshold_E]
                    cluster_cell_E_weight = cluster_cell_E / np.sum(cluster_cell_E)
                    cluster_cell_E = np.log10(cluster_cell_E) - LOG_ENERGY_MEAN

                    # node features
                    node_features = {}
                    for feature in node_feature_names:
                        node_features[feature] = cell_geo_data[feature][0][cell_ID_map]

                    # get cartesian coords
                    thetas = [2*np.arctan(np.exp(-eta)) for eta in node_features["cell_geo_eta"]]
                    x, y, z = spherical_to_cartesian(node_features["cell_geo_rPerp"], node_features["cell_geo_phi"], thetas)

                    # if grouped by events extend the set of event points with the clusters points
                    if group_events and num_cells != 0:
                        processed_event_data["truth_EM_frac"][samples_count].extend(truth_EM_frac)
                        processed_event_data["cluster_cell_E"][samples_count].extend(cluster_cell_E)
                        processed_event_data["cluster_cell_E_weight"][samples_count].extend(cluster_cell_E_weight)

                        for feature in node_feature_names:
                            processed_event_data[feature][samples_count].extend(node_features[feature])

                        processed_event_data["x"][samples_count].extend(x)
                        processed_event_data["y"][samples_count].extend(y)
                        processed_event_data["z"][samples_count].extend(z)


                    # if grouped by clusters append the set of clusters points
                    elif num_cells != 0:
                        processed_event_data["truth_EM_frac"].append(truth_EM_frac)
                        processed_event_data["cluster_cell_E"].append(cluster_cell_E)
                        processed_event_data["cluster_cell_E_weight"].append(cluster_cell_E_weight)

                        for feature in node_feature_names:
                            processed_event_data[feature].append(np.array(node_features[feature]))

                        processed_event_data["x"].append(x)
                        processed_event_data["y"].append(y)
                        processed_event_data["z"].append(z)
                        
                        samples_count += 1

                        if num_cells > max_cells:
                            max_cells = num_cells
                        num_cells = 0
                
                # event has non zero energy hits    
                if group_events and num_cells > 0:
                    if include_tracks:
                        if num_tracks == 1:
                            # get tracks momentum readout
                            trackP = np.log10(event_data['trackP'][event_idx][0]) - LOG_MEAN_TRACK_MOMETUM
                            trackPs = np.full(NUM_TRACK_POINTS, trackP)
                        else:
                            trackPs = np.full(NUM_TRACK_POINTS, 0)
                        
                        # zero out all values in null tracks
                        x_tracks[~non_null_tracks] = 0
                        y_tracks[~non_null_tracks] = 0
                        z_tracks[~non_null_tracks] = 0
                        trackPs[~non_null_tracks] = 0
                        eta_tracks[~non_null_tracks] = 0
                        phi_tracks[~non_null_tracks] = 0

                        sampling_layer_num = np.arange(1,NUM_TRACK_POINTS + 1)
                        sampling_layer_num[~non_null_tracks] = 0

                        processed_event_track_data["x"].append(x_tracks)
                        processed_event_track_data["y"].append(y_tracks)
                        processed_event_track_data["z"].append(z_tracks)
                        processed_event_track_data["P"].append(trackPs)
                        processed_event_track_data["sampling_layer"].append(sampling_layer_num)
                        
                        processed_event_track_data["min_dist"].append(np.zeros(NUM_TRACK_POINTS))
                        processed_event_track_data["min_eta"].append(np.zeros(NUM_TRACK_POINTS))
                        processed_event_track_data["min_phi"].append(np.zeros(NUM_TRACK_POINTS))


                        processed_event_track_flags.append(np.zeros(num_cells + NUM_TRACK_POINTS))
                        processed_event_track_flags[samples_count][-NUM_TRACK_POINTS:] = 1*(non_null_tracks)

                        # add min dists from cells to closest strack point
                        # min euclidean dist from each point to the closest track point
                        track_points_idx = np.arange(NUM_TRACK_POINTS)[non_null_tracks]
                        num_track_points = len(track_points_idx)
                        dists = np.zeros((num_cells, num_track_points))
                        eta_dists = np.zeros((num_cells, num_track_points))
                        phi_dists = np.zeros((num_cells, num_track_points))

                        for j, track_point_idx in enumerate(track_points_idx):
                            dists[:, j] = np.sqrt((processed_event_data["x"][samples_count] - x_tracks[track_point_idx])**2 + (processed_event_data["y"][samples_count] - y_tracks[track_point_idx])**2 + (processed_event_data["z"][samples_count] - z_tracks[track_point_idx])**2)
                            eta_dists[:, j] = np.abs(processed_event_data["cell_geo_eta"][samples_count] - eta_tracks[track_point_idx])
                            phi_dists[:, j] = np.abs(processed_event_data["cell_geo_phi"][samples_count] - phi_tracks[track_point_idx])
                        
                        min_dists = np.min(dists, axis=1)
                        min_eta_dists = np.min(eta_dists, axis=1)
                        min_phi_dists = np.min(phi_dists, axis=1)
                        processed_event_data["min_dist"][samples_count][:NUM_TRACK_POINTS] = min_dists
                        processed_event_data["min_eta"][samples_count][:NUM_TRACK_POINTS] = min_eta_dists
                        processed_event_data["min_phi"][samples_count][:NUM_TRACK_POINTS] = min_phi_dists


                        if num_cells + NUM_TRACK_POINTS > max_cells:
                            max_cells = num_cells + NUM_TRACK_POINTS
                            
                    else:
                        if num_cells > max_cells:
                            max_cells = num_cells
                    
                    samples_count += 1
                    num_cells = 0

    
    for file_type in file_features:
        if include_tracks and not add_3_min_dists_and_layer:
            point_data = np.zeros((samples_count, max_cells, len(file_features[file_type]) + 1)) # pad X data with 0's
        elif add_3_min_dists_and_layer:
            point_data = np.zeros((samples_count, max_cells, 9)) # pad X data with 0's
        else:
            point_data = np.zeros((samples_count, max_cells, len(file_features[file_type]))) # pad X data with 0's

        if weight_cells:
            label_len = 2
        else:
            label_len = 1
        
        point_label = np.negative(np.ones((samples_count, max_cells, label_len))) # pad Y data with -1's, since some EM fracs are 0


        for idx in range(samples_count):
            #print("idx:", idx)
            len_cluster = len(processed_event_data["cell_geo_eta"][idx])
            if(len_cluster == 0):
                print("zero len clus!!!")

            if include_tracks and not add_3_min_dists_and_layer:
                len_cluster += NUM_TRACK_POINTS
                track_features = ["P", "x", "y", "z"]
                cluster_point_data = np.concatenate(([np.concatenate((processed_event_data[file_feature][idx], processed_event_track_data[track_features[i]][idx]), -1)  for i, file_feature in enumerate(file_features[file_type])], [processed_event_track_flags[idx]]), 0)
            elif add_3_min_dists_and_layer:
                len_cluster += NUM_TRACK_POINTS
                track_features = ["P", "x", "y", "z", 'min_dist', 'min_phi', 'min_eta', 'sampling_layer']
                file_features = ['cluster_cell_E', 'x', 'y', 'z', 'min_dist', 'min_phi', 'min_eta', 'cell_geo_sampling']
                cluster_point_data = np.concatenate(([np.concatenate((processed_event_data[file_feature][idx], processed_event_track_data[track_features[i]][idx]), -1)  for i, file_feature in enumerate(file_features)], [processed_event_track_flags[idx]]), 0)
            else:
                cluster_point_data = [processed_event_data[file_feature][idx] for file_feature in file_features[file_type]]
            
            #print("point data:", point_data.shape)
            #print("cluster point data:", cluster_point_data.shape)
            point_data[idx, :len_cluster] = np.transpose(cluster_point_data)
            if weight_cells:
                point_label[idx, :len_cluster] = np.transpose([processed_event_data["truth_EM_frac"][idx], processed_event_data["cluster_cell_E_weight"][idx]])
            else:
                if include_tracks:
                    point_label[idx, :len_cluster] = np.transpose([np.concatenate((processed_event_data["truth_EM_frac"][idx], np.full(NUM_TRACK_POINTS, -1)))]) # label all tracks as -1
                else:
                    point_label[idx, :len_cluster] = np.transpose([processed_event_data["truth_EM_frac"][idx]])

        if rho_events:
            file_path = "/fast_scratch_1/jbohm/train_testing_data/rho_files/rho_small.npz"
        else:
            file_path = save_dir + file_type + file_name.split(".")[0] + ('_charged' if not mixed_pions else '') + '_1_track.npz' #"/fast_scratch_1/jbohm/train_testing_data/rho_files/rho_small.npz"
        np.savez(file_path, X=point_data, Y=point_label)

    max_points_queue.put(max_cells)


if __name__ == "__main__":
    t_start = time.time()
    args = get_args()
    
    config_file = "./preprocess_events_to_pnet_npz_config.yaml"
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    save_dir = config["save_dir"]
    pi0_file_nums = config["pi0_file_nums"]
    pipm1_file_nums = config["pipm1_file_nums"]
    pipm2_file_nums = config["pipm2_file_nums"]
    len_file = config["len_file"]
    i_low = config["i_low"]
    i_high = config["i_high"]
    num_procs = config["num_procs"]
    energy_threshold = config["energy_threshold"]
    group_events = config["group_events"]
    mixed_pions = config["mixed_pions"]
    weight_cells = config["weight_cells"]
    em_frac_threshold = config["em_frac_threshold"]
    rho_events = config["rho_events"]
    new_format = config["new_format"]
    include_tracks = config["include_tracks"]
    left_from_include_tracks = config["left_from_include_tracks"]
    add_3_min_dists_and_layer = config["add_3_min_dists_and_layer"]

    if not new_format:
        pion_dir = "/fast_scratch_1/jbohm/train_testing_data/" + ("pion_files" if mixed_pions else "charged_pion_files")
    else:
        pion_dir = "/fast_scratch_1/jbohm/train_testing_data/charged_pion_files"

    # load cell geo tree dict
    file = uproot.open("/data/atlas/data/rho_delta/rho_small.root")
    cell_geo_tree = file["CellGeo"]

    node_feature_names = cell_geo_tree.keys()[1:7] # 'cell_geo_sampling', 'cell_geo_eta', 'cell_geo_phi', 'cell_geo_rPerp', 'cell_geo_deta', 'cell_geo_dphi'

    file_features = {
        #'pion_pointnet_spherical': ['cluster_cell_E', *(node_feature_names[1:4])],
        'pion_pointnet_cartesian': ['cluster_cell_E', 'x', 'y', 'z'],
        #'pion_pointnet_lawrence': ['cluster_cell_E', *node_feature_names]
    }

    cell_geo_data = cell_geo_tree.arrays(library='np')
    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]
    sorter = np.argsort(cell_geo_ID)

    # get list of paths to pion files to convert
    pion_files = []
    for j, pi0_num in enumerate(pi0_file_nums):
        if new_format:
            pion_files.extend(list(map(lambda i:  "/pipm_" + str(pipm1_file_nums[j]) + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1))))
        elif mixed_pions:
            pion_files.extend(list(map(lambda i:  "/pi0_" + str(pi0_num) + "_pipm_" + str(pipm1_file_nums[j]) + "_" + str(pipm2_file_nums[j]) + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1))))
        else: # charged pion files
            pion_files.extend(list(map(lambda i:  "/pipm_" + str(pipm1_file_nums[j]) + "_" + str(pipm2_file_nums[j]) + "_" + str(pi0_num)  + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1))))
    
    max_points = 0

    manager = multiprocessing.Manager()
    max_points_queue = manager.Queue()
    pool = multiprocessing.Pool(num_procs)

    pool.map(process_file, [(file_name, pion_dir, save_dir, node_feature_names, file_features, cell_geo_data, sorter, energy_threshold, group_events, max_points_queue, mixed_pions, weight_cells, em_frac_threshold, rho_events, include_tracks, left_from_include_tracks, add_3_min_dists_and_layer) for file_name in pion_files])

    while not max_points_queue.empty():
        q_max_points = max_points_queue.get()

        if q_max_points > max_points:
            max_points = q_max_points

    with open(save_dir + 'pion_pointnet_cartesian/max_points_1_track_larger.txt') as f:
        current_max_points = int(f.readline())
    
    if max_points > current_max_points:
        for file_type in file_features:
            with open(save_dir + file_type + '/max_points_1_track_larger.txt', 'w') as f:
                f.write('%d' % max_points)

    print("elapsed time:",  time.time() - t_start)
    
        