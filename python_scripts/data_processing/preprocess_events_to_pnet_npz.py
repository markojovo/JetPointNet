import argparse
import time
import numpy as np
import yaml
import sys
import uproot
import awkward as ak
import multiprocessing

sys.path.append('/home/jbohm/start_tf/LCStudies')

LOG_ENERGY_MEAN = -1 # unrounded mean is ~ 0.93

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

    return x,y,z

def spherical_to_cartesian(rs, phis, thetas):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.cos(phis[i]))
        y.append(r*np.sin(phis[i]))
        z.append( r/np.tan(thetas[i]))

    return x,y,z

def process_file(args):
    file_name, pion_dir, save_dir, node_feature_names, file_features, cell_geo_data, sorter, energy_threshold, group_events, max_points_queue, mixed_pions, weight_cells, em_frac_threshold, rho_events = args
    print(file_name)
    if rho_events:
        event_data = uproot.open("/data/atlas/data/rho_delta/rho_small.root:EventTree").arrays(library="np")
    else:
        event_data = np.load(pion_dir + file_name, allow_pickle=True).item()

    num_events = len(event_data["eventNumber"])

    # cluster data dict to look up data by feature name
    processed_event_data = {}

    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]

    samples_count = 0
    max_cells = 0
    num_cells = 0

    for feature in [*node_feature_names, 'x', 'y', 'z', 'truth_EM_frac', 'cluster_cell_E', 'cluster_cell_E_weight']:
        processed_event_data[feature] = []

    for event_idx in range(num_events):
        num_clusters = event_data["nCluster"][event_idx]

        # append empty array for all cells in an event
        if group_events:
            for feature in [*node_feature_names, 'x', 'y', 'z', 'truth_EM_frac', 'cluster_cell_E', 'cluster_cell_E_weight']:
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
                    processed_event_data[feature].append(node_features[feature])

                processed_event_data["x"].append(x)
                processed_event_data["y"].append(y)
                processed_event_data["z"].append(z)
                
                samples_count += 1

                if num_cells > max_cells:
                    max_cells = num_cells
                num_cells = 0
            
        if group_events and num_cells > 0:
            if num_cells > max_cells:
                max_cells = num_cells
            samples_count += 1
            num_cells = 0

    for file_type in file_features:
        point_data = np.zeros((samples_count, max_cells, len(file_features[file_type]))) # pad X data with 0's
        if weight_cells:
            label_len = 2
        else:
            label_len = 1
        point_label = np.negative(np.ones((samples_count, max_cells, label_len))) # pad Y data with -1's, since some EM fracs are 0

        for idx in range(samples_count):
            len_cluster = len(processed_event_data["cell_geo_eta"][idx])
            if(len_cluster == 0):
                print("zero len clus!!!")
            cluster_point_data = [processed_event_data[file_feature][idx] for file_feature in file_features[file_type]]

            point_data[idx, :len_cluster] = np.transpose(cluster_point_data)
            if weight_cells:
                point_label[idx, :len_cluster] = np.transpose([processed_event_data["truth_EM_frac"][idx], processed_event_data["cluster_cell_E_weight"][idx]])
            else:
                point_label[idx, :len_cluster] = np.transpose([processed_event_data["truth_EM_frac"][idx]])

        if rho_events:
            file_path = "/fast_scratch_1/jbohm/train_testing_data/rho_files/rho_small.npz"
        else:
            file_path = save_dir + file_type + file_name.split(".")[0] + "_" + "_".join(file_type.split("_")[2:]) + ('_event_grouped' if group_events else '') + ('_charged' if not mixed_pions else '') + '.npz' #"/fast_scratch_1/jbohm/train_testing_data/rho_files/rho_small.npz"
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

    pion_dir = "/fast_scratch_1/jbohm/train_testing_data/" + ("pion_files" if mixed_pions else "charged_pion_files")

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
        if mixed_pions:
            pion_files.extend(list(map(lambda i:  "/pi0_" + str(pi0_num) + "_pipm_" + str(pipm1_file_nums[j]) + "_" + str(pipm2_file_nums[j]) + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1))))
        else: # charged pion files
            pion_files.extend(list(map(lambda i:  "/pipm_" + str(pipm1_file_nums[j]) + "_" + str(pipm2_file_nums[j]) + "_" + str(pi0_num)  + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1))))
    
    max_points = 0

    manager = multiprocessing.Manager()
    max_points_queue = manager.Queue()
    pool = multiprocessing.Pool(num_procs)

    pool.map(process_file, [(file_name, pion_dir, save_dir, node_feature_names, file_features, cell_geo_data, sorter, energy_threshold, group_events, max_points_queue, mixed_pions, weight_cells, em_frac_threshold, rho_events) for file_name in pion_files])

    while not max_points_queue.empty():
        q_max_points = max_points_queue.get()

        if q_max_points > max_points:
            max_points = q_max_points

    with open(save_dir + 'pion_pointnet_cartesian/max_points.txt') as f:
        current_max_points = int(f.readline())
    
    if max_points > current_max_points:
        for file_type in file_features:
            with open(save_dir + file_type + '/max_points.txt', 'w') as f:
                f.write('%d' % max_points)

    print("elapsed time:",  time.time() - t_start)
    
        