import argparse
import time
import numpy as np
import yaml
import sys
import uproot
import awkward as ak
import multiprocessing

sys.path.append('/home/jbohm/start_tf/LCStudies')
from  util import graph_util as gu

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, type=str)
    return parser.parse_args()

# by generic definition of spherical to cartesian coord conversion
def spherical_to_cartesian(rs, thetas, phis):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.sin(phis[i])*np.cos(thetas[i]))
        y.append(r*np.sin(phis[i])*np.sin(thetas[i]))
        z.append(r*np.cos(phis[i]))

    return x,y,z

def process_file(args):
    file_name, pion_dir, save_dir, node_feature_names, file_features, cell_geo_data, sorter, energy_threshold,  max_points_queue = args
    print(file_name)
    event_data = np.load(pion_dir + file_name, allow_pickle=True).item()
    num_events = len(event_data["eventNumber"])

    # cluster data dict to look up data by feature name
    processed_event_data = {}

    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]

    cluster_count = 0
    max_cells = 0

    for feature in [*node_feature_names, 'x', 'y', 'z', 'truth_EM_frac', 'cluster_cell_E']:
        processed_event_data[feature] = []

    for event_idx in range(num_events):
        num_clusters = event_data["nCluster"][event_idx]
        for cluster_idx in range(num_clusters):
            cell_IDs = event_data['cluster_cell_ID'][event_idx][cluster_idx]

            # get truth enery (EM/nonEM)
            cell_hits_E_EM = np.array(event_data["cluster_cell_hitsE_EM"][event_idx][cluster_idx])
            cell_hits_E_non_EM = np.array(event_data["cluster_cell_hitsE_nonEM"][event_idx][cluster_idx])
            cell_hits_E = cell_hits_E_EM + cell_hits_E_non_EM

            # filter out cells with truth energy below the threshold
            threshold_E = (cell_hits_E > energy_threshold)
            cell_IDs = cell_IDs[threshold_E]
            cell_ID_map = sorter[np.searchsorted(cell_geo_ID, cell_IDs, sorter=sorter)]

            # get truth energy fraction EM/EM + nonEM for each cell
            processed_event_data["truth_EM_frac"].append(cell_hits_E_EM[threshold_E] / cell_hits_E[threshold_E])

            # get cluster cell energy
            processed_event_data["cluster_cell_E"].append(event_data["cluster_cell_E"][event_idx][cluster_idx][threshold_E])

            for feature in node_feature_names:
                processed_event_data[feature].append(cell_geo_data[feature][0][cell_ID_map]) # nested in 2 arrays so take first element

            # get cartesian coords
            processed_event_data["x"].append([])
            processed_event_data["y"].append([])
            processed_event_data["z"].append([])
            thetas = [2*np.arctan(np.exp(-eta)) for eta in processed_event_data["cell_geo_eta"][cluster_count]]
            processed_event_data["x"][cluster_count], processed_event_data["y"][cluster_count], processed_event_data["z"][cluster_count] = spherical_to_cartesian(processed_event_data["cell_geo_rPerp"][cluster_count], processed_event_data["cell_geo_phi"][cluster_count], thetas)

            cluster_count += 1
            if len(cell_IDs) > max_cells:
                max_cells = len(cell_IDs)
    
    for file_type in file_features:
        point_data = np.zeros((cluster_count, max_cells, len(file_features[file_type])))
        point_label = np.zeros((cluster_count, max_cells, 1))

        for cluster_idx in range(cluster_count):
            len_cluster = len(processed_event_data["cell_geo_eta"][cluster_idx])
            cluster_point_data = [processed_event_data[file_feature][cluster_idx] for file_feature in file_features[file_type]]

            point_data[cluster_idx, :len_cluster] = np.transpose(cluster_point_data)
            point_label[cluster_idx, :len_cluster] = np.transpose([processed_event_data["truth_EM_frac"][cluster_idx]])

        file_path = save_dir + file_type + file_name.split(".")[0] + "_" + "_".join(file_type.split("_")[2:]) + '.npz'
        np.savez(file_path, X=point_data, Y=point_label)

    max_points_queue.put(max_cells)


if __name__ == "__main__":
    t_start = time.time()
    args = get_args()
    
    config_file = "./preprocess_events_to_pnet_npz_config.yaml"
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    pion_dir = config["pion_dir"]
    save_dir = config["save_dir"]
    pi0_file_nums = config["pi0_file_nums"]
    pipm1_file_nums = config["pipm1_file_nums"]
    pipm2_file_nums = config["pipm2_file_nums"]
    len_file = config["len_file"]
    i_low = config["i_low"]
    i_high = config["i_high"]
    num_procs = config["num_procs"]
    energy_threshold = config["energy_threshold"]

    # load cell geo tree dict
    file = uproot.open("/data/atlas/data/rho_delta/rho_small.root")
    cell_geo_tree = file["CellGeo"]

    node_feature_names = cell_geo_tree.keys()[1:7] # 'cell_geo_sampling', 'cell_geo_eta', 'cell_geo_phi', 'cell_geo_rPerp', 'cell_geo_deta', 'cell_geo_dphi'

    file_features = {
        'pion_pointnet_spherical': ['cluster_cell_E', *(node_feature_names[1:4])],
        'pion_pointnet_cartesian': ['cluster_cell_E', 'x', 'y', 'z'],
        'pion_pointnet_lawrence': ['cluster_cell_E', *node_feature_names]
    }

    cell_geo_data = cell_geo_tree.arrays(library='np')
    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]
    sorter = np.argsort(cell_geo_ID)

    # get list of paths to pion files to convert
    pion_files = []
    for j, pi0_num in enumerate(pi0_file_nums):
        pion_files.extend(list(map(lambda i:  "/pi0_" + str(pi0_num) + "_pipm_" + str(pipm1_file_nums[j]) + "_" + str(pipm2_file_nums[j]) + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1))))
    
    max_points = 0

    manager = multiprocessing.Manager()
    max_points_queue = manager.Queue()
    pool = multiprocessing.Pool(num_procs)

    pool.map(process_file, [(file_name, pion_dir, save_dir, node_feature_names, file_features, cell_geo_data, sorter, energy_threshold,  max_points_queue) for file_name in pion_files])

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
    
        