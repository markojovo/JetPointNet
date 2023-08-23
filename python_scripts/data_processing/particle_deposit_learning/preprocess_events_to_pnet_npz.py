import argparse
import time
import numpy as np
import yaml
import sys
import uproot
import awkward as ak
import multiprocessing
import vector
from utils.track_metadata import *
from PointNet_Segmentation.utils.data_processing_util import *


LOG_ENERGY_MEAN = -1 # unrounded mean is ~ -0.93
LOG_MEAN_TRACK_MOMETUM = 2


def process_file(args):
    file_name, rho_dir, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, weight_cells, include_tracks, left_from_include_tracks, add_3_min_dists_and_layer = args
    print(file_name)
    
    events_arr = np.load(rho_dir + file_name, allow_pickle=True).item()
    
    """ PREPROCESSING """
    # aggregate the data with the cluster it is in and its EM probability (for analysis not training)
    events_arr["clus_idx"] = []
    events_arr["clus_em_prob"] = []
    for event_idx, event_clus_cell_E in enumerate(events_arr["cluster_cell_ID"]):
        events_arr["clus_idx"].append([])
        events_arr["clus_em_prob"].append([])
        for clus_idx, clus_cell_E in enumerate(event_clus_cell_E):
            events_arr["clus_idx"][event_idx].append(np.tile(clus_idx, len(clus_cell_E)))
            events_arr["clus_em_prob"][event_idx].append(np.tile(events_arr["cluster_EM_PROBABILITY"][event_idx][clus_idx], len(clus_cell_E)))
    events_arr["clus_idx"] = ak.Array(events_arr["clus_idx"])
    events_arr["clus_em_prob"] = ak.Array(events_arr["clus_em_prob"])


    # Fix 1 - cut out the negative energy deposits
    events_arr["cluster_cell_hitsTruthE"] = events_arr["cluster_cell_hitsTruthE"][events_arr["cluster_cell_hitsTruthE"] >= 0]
    events_arr["cluster_cell_hitsTruthIndex"] = events_arr["cluster_cell_hitsTruthIndex"][events_arr["cluster_cell_hitsTruthE"] >= 0]

    # Fix 2 - cut out the events w a electron/positron, no clusters, or no cell hits
    # for electron/positron pairs find the occurrences of this array in the list [ 213.  211.  111.  -11.   11.   22.]  (if there is a pdgId == 11)
    events_arr_resized = {}
    elec_pos_events = ak.Array([np.count_nonzero(np.array(event_truthPartPdgId) == 11) > 0 for event_truthPartPdgId in events_arr["truthPartPdgId"]])
    print("num events with electron/positron:", np.count_nonzero(elec_pos_events))

    num_truth_contributions = np.array([[len(cell_truth_con) for clus_truth_con in event_truth_con for cell_truth_con in clus_truth_con] for event_truth_con in events_arr["cluster_cell_hitsTruthIndex"]])
    no_cell_deposits = np.array([np.count_nonzero(np.array(num_truth_contributions_event) == 0) == len(num_truth_contributions_event) for num_truth_contributions_event in num_truth_contributions])
    print("num events no cell deposits:", np.count_nonzero(no_cell_deposits))

    for key in events_arr.keys():
        events_arr_resized[key] = events_arr[key][~elec_pos_events & (ak.Array(events_arr["nCluster"]) != 0) & ~no_cell_deposits]

    events_arr = events_arr_resized

    # Fix 3 - cut out the padding from the end of particle deposits arrays (cluster_cell_hitsTruthE and cluster_cell_hitsTruthIndex)
    events_arr["cluster_cell_hitsTruthE"] = [[cluster_cell_hitsTruthE_clus[:len(events_arr["cluster_cell_hitsE_EM"][event_idx][clus_idx])] for clus_idx, cluster_cell_hitsTruthE_clus in enumerate(cluster_cell_hitsTruthE_event)] for event_idx, cluster_cell_hitsTruthE_event in enumerate(events_arr["cluster_cell_hitsTruthE"])]
    events_arr["cluster_cell_hitsTruthIndex"] = [[cluster_cell_hitsTruthIndex_clus[:len(events_arr["cluster_cell_hitsE_EM"][event_idx][clus_idx])] for clus_idx, cluster_cell_hitsTruthIndex_clus in enumerate(cluster_cell_hitsTruthIndex_event)] for event_idx, cluster_cell_hitsTruthIndex_event in enumerate(events_arr["cluster_cell_hitsTruthIndex"])]

    # Fix 4 - flatten clusters to events & remove cell repeats from when a cell is in more than one topocluster
    cellwise_data_keys = ["cluster_cell_E", "cluster_cell_hitsE_EM", "cluster_cell_hitsE_nonEM", "cluster_cell_hitsTruthE", "cluster_cell_hitsTruthIndex", "cluster_cell_ID", "clus_idx", "clus_em_prob"]

    # flatten cellwise data to only be grouped by events, not clus and events
    for key in cellwise_data_keys:
        events_arr[key] = flatten_clus_in_events(events_arr[key])

    # define unique versions of the cellwise data arrays
    cellwise_data_unique = {}
    for key in cellwise_data_keys:
        cellwise_data_unique[key + "_unique"] = []

    # iterate through the events and remove any repeated cell deposits (if 2+ of same cell ID in one event remove the extra)
    for event_idx in np.arange(len(events_arr["cluster_cell_ID"])):
        # get the idx of the unique cell IDs
        unique_events_cells_IDs, unique_cells_idx = np.unique(events_arr["cluster_cell_ID"][event_idx], return_index=True)

        # cut the particle ID and energy deposits to just have the unique cells per event
        for key in cellwise_data_keys:
            cellwise_data_unique[key + "_unique"].append(ak.Array(events_arr[key][event_idx])[unique_cells_idx])

    for key in cellwise_data_keys:
        events_arr[key] = cellwise_data_unique[key + "_unique"]

    print("num events:", len(events_arr["cluster_cell_hitsTruthIndex"]))

    event_data = events_arr
    """ DONE PREPROCESSING """
    """"""""""""""""""""""""""

    # save the preprocessed data to be used for analysis
    np.save("/fast_scratch_1/jbohm/cell_particle_deposit_learning/rho_processed_test_data/" + file_name, event_data)

    num_events = len(event_data["eventNumber"])

    # cluster data dict to look up data by feature name
    processed_event_data = {}
    processed_event_track_data = {}
    processed_event_track_flags = []

    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]

    samples_count = 0
    max_cells = 0
    num_cells = 0

    for feature in [*node_feature_names, 'x', 'y', 'z', 'cell_hitsTruthIndex', 'cell_hitsTruthE', 'class_pi0_lead_energy', 'cell_E', 'cell_E_weight', 'sampling_layer', 'delta_R', 'clus_idx', 'track_pipm_delta_R', 'track_pi0_delta_R']:
        processed_event_data[feature] = []

    for feature in ['x', 'y', 'z', 'P', 'min_dist', 'min_eta', 'min_phi', 'sampling_layer']:
        processed_event_track_data[feature] = []

    # iterate over events and compute/save necessary features to processed_event_data/processed_event_track_data
    for event_idx in range(num_events):
        print(event_idx)
        num_tracks = event_data["nTrack"][event_idx]

        if ((include_tracks and num_tracks == 1) or (left_from_include_tracks and num_tracks != 1) or (not left_from_include_tracks and not include_tracks)):
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
                cell_IDs = event_data['cluster_cell_ID'][event_idx]

                # get fraction cell energy deposited by pi0 (particle idx != 1 since particle idx1 is charged pion)
                frac_pi0_energy = ak.sum(events_arr["cluster_cell_hitsTruthE"][event_idx][events_arr["cluster_cell_hitsTruthIndex"][event_idx] != 1], axis=1)/ak.sum(events_arr["cluster_cell_hitsTruthE"][event_idx], axis=1)
                # find cells with no particles contributing energy - could be from removing the neg cell deposits
                empty_contr_frac = np.isnan(frac_pi0_energy)

                # filter out cells without truth particle deposits
                threshold_E_deposit = ~empty_contr_frac
                
                cell_IDs = cell_IDs[threshold_E_deposit]
                num_cells = len(cell_IDs)
                cell_ID_map = sorter[np.searchsorted(cell_geo_ID, cell_IDs, sorter=sorter)]

                # classify leading particle contributor
                class_frac_pi0_energy = frac_pi0_energy[threshold_E_deposit] > 0.5 # class 1 if pi0 frac > 0.5, otherwise class 2

                # get cluster cell energy
                cell_E = event_data["cluster_cell_E"][event_idx][threshold_E_deposit]
                cell_E_weight = cell_E / np.sum(cell_E)
                cell_E = np.log10(cell_E) - LOG_ENERGY_MEAN

                # node features
                node_features = {}
                for feature in node_feature_names:
                    node_features[feature] = cell_geo_data[feature][0][cell_ID_map]

                # get cartesian coords
                thetas = [2*np.arctan(np.exp(-eta)) for eta in node_features["cell_geo_eta"]]
                x, y, z = spherical_to_cartesian(node_features["cell_geo_rPerp"], node_features["cell_geo_phi"], thetas)

            
                processed_event_data["cell_E"].append(cell_E)
                processed_event_data["cell_E_weight"].append(cell_E_weight)
                processed_event_data["class_pi0_lead_energy"].append(class_frac_pi0_energy)

                
                # calculate delta R between particle and track
                pipm_idx = 1
                pi0_idx = 2

                pipm_eta = event_data['truthPartEta'][event_idx][pipm_idx]
                pipm_phi = event_data['truthPartPhi'][event_idx][pipm_idx]
                pipm_pT = event_data['truthPartPt'][event_idx][pipm_idx]

                pi0_eta = event_data['truthPartEta'][event_idx][pi0_idx]
                pi0_phi = event_data['truthPartPhi'][event_idx][pi0_idx]
                pi0_pT = event_data['truthPartPt'][event_idx][pi0_idx]

                track_eta = np.squeeze(event_data['trackEta'][event_idx])
                track_phi = np.squeeze(event_data['trackPhi'][event_idx])
                track_pT = np.squeeze(event_data['trackP'][event_idx])


                # angular separation between the vectors in the eta/phi plane - over events
                pipm_vector = vector.obj(eta=pipm_eta, phi=pipm_phi, rho=pipm_pT)
                pi0_vector = vector.obj(eta=pi0_eta, phi=pi0_phi, rho=pi0_pT)
                track_vector = vector.obj(eta=track_eta, phi=track_phi, rho=track_pT)
                track_pipm_delta_R = pipm_vector.deltaR(track_vector)
                track_pi0_delta_R = pi0_vector.deltaR(track_vector)
                pipm_pi0_delta_R = pi0_vector.deltaR(pipm_vector)

                processed_event_data["track_pipm_delta_R"].append(track_pipm_delta_R)
                processed_event_data["track_pi0_delta_R"].append(track_pi0_delta_R)
                processed_event_data["delta_R"].append(pipm_pi0_delta_R)

                
                for feature in node_feature_names:
                    processed_event_data[feature].append(np.array(node_features[feature]))

                processed_event_data["x"].append(x)
                processed_event_data["y"].append(y)
                processed_event_data["z"].append(z)                
                
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
                    
                    #processed_event_track_data["min_dist"].append(np.zeros(NUM_TRACK_POINTS))
                    #processed_event_track_data["min_eta"].append(np.zeros(NUM_TRACK_POINTS))
                    #processed_event_track_data["min_phi"].append(np.zeros(NUM_TRACK_POINTS))


                    processed_event_track_flags.append(np.zeros(num_cells + NUM_TRACK_POINTS))
                    processed_event_track_flags[samples_count][-NUM_TRACK_POINTS:] = 1*(non_null_tracks)

                    # add min dists from cells to closest track point
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
                    
                    #min_dists = np.min(dists, axis=1)
                    #min_eta_dists = np.min(eta_dists, axis=1)
                    #min_phi_dists = np.min(phi_dists, axis=1)
                    #processed_event_data["min_dist"][samples_count][:NUM_TRACK_POINTS] = min_dists
                    #processed_event_data["min_eta"][samples_count][:NUM_TRACK_POINTS] = min_eta_dists
                    #processed_event_data["min_phi"][samples_count][:NUM_TRACK_POINTS] = min_phi_dists

                    if num_cells + NUM_TRACK_POINTS > max_cells:
                            max_cells = num_cells + NUM_TRACK_POINTS
                            
                else:
                    if num_cells > max_cells:
                        max_cells = num_cells

                samples_count += 1

    file_features = ['cell_E', 'x', 'y', 'z']
    if include_tracks and not add_3_min_dists_and_layer:
        point_data = np.zeros((samples_count, max_cells, len(file_features) + 1)) # pad X data with 0's
    elif add_3_min_dists_and_layer:
        point_data = np.zeros((samples_count, max_cells, 9)) # pad X data with 0's
    else:
        point_data = np.zeros((samples_count, max_cells, len(file_features))) # pad X data with 0's

    if weight_cells:
        label_len = 2
    else:
        label_len = 1
    
    point_label = np.negative(np.ones((samples_count, max_cells, label_len))) # pad Y data with -1's, since some EM fracs are 0


    for idx in range(samples_count):
        print("idx:", idx)
        len_cluster = len(processed_event_data["cell_geo_eta"][idx])
        if(len_cluster == 0):
            print("zero len clus!!!")

        if include_tracks and not add_3_min_dists_and_layer:
            len_cluster += NUM_TRACK_POINTS
            track_features = ["P", "x", "y", "z"]
            cluster_point_data = np.concatenate(([np.concatenate((processed_event_data[file_feature][idx], processed_event_track_data[track_features[i]][idx]), -1)  for i, file_feature in enumerate(file_features)], [processed_event_track_flags[idx]]), 0)
        elif add_3_min_dists_and_layer:
            len_cluster += NUM_TRACK_POINTS
            track_features = ["P", "x", "y", "z", 'min_dist', 'min_phi', 'min_eta', 'sampling_layer']
            file_features = ['cell_E', 'x', 'y', 'z', 'min_dist', 'min_phi', 'min_eta', 'cell_geo_sampling']
            cluster_point_data = np.concatenate(([np.concatenate((processed_event_data[file_feature][idx], processed_event_track_data[track_features[i]][idx]), -1)  for i, file_feature in enumerate(file_features)], [processed_event_track_flags[idx]]), 0)
        else:
            cluster_point_data = [processed_event_data[file_feature][idx] for file_feature in file_features]
        

        point_data[idx, :len_cluster] = np.transpose(cluster_point_data)
        if weight_cells:
            point_label[idx, :len_cluster] = np.transpose([processed_event_data["class_pi0_lead_energy"][idx], processed_event_data["cluster_cell_E_weight"][idx]])
        else:
            if include_tracks:
                point_label[idx, :len_cluster] = np.transpose([np.concatenate((processed_event_data["class_pi0_lead_energy"][idx], np.full(NUM_TRACK_POINTS, -1)))]) # label all tracks as -1
            else:
                point_label[idx, :len_cluster] = np.transpose([processed_event_data["class_pi0_lead_energy"][idx]])


    # save X and Y train point data
    file_path = save_dir + "/" + file_name.split(".")[0] + '_1_track.npz'
    np.savez(file_path, X=point_data, Y=point_label)

    # save extra data to make cuts
    np.save(save_dir + "/delta_R/" + file_name.split(".")[0] + "_delta_R.npy", processed_event_data["delta_R"])
    np.save(save_dir + "/delta_R/" + file_name.split(".")[0] + "_track_pipm_delta_R.npy", processed_event_data["track_pipm_delta_R"])
    np.save(save_dir + "/delta_R/" + file_name.split(".")[0] + "_track_pi0_delta_R.npy", processed_event_data["track_pi0_delta_R"])

    max_points_queue.put(max_cells)


if __name__ == "__main__":
    t_start = time.time()

    # get arguments from yaml file
    args = get_args()
    
    config_file = "./preprocess_events_to_pnet_npz_config.yaml"
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    save_dir = config["save_dir"]
    files_name = config["files_name"]
    len_file = config["len_file"]
    i_low = config["i_low"]
    i_high = config["i_high"]
    num_procs = config["num_procs"]
    energy_threshold = config["energy_threshold"]
    weight_cells = config["weight_cells"]
    include_tracks = config["include_tracks"]
    left_from_include_tracks = config["left_from_include_tracks"]
    add_3_min_dists_and_layer = config["add_3_min_dists_and_layer"]

    rho_dir = "/fast_scratch_1/jbohm/cell_particle_deposit_learning/rho_npy_files/" 

    # load cell geo tree to look uo cells location
    file = uproot.open("/data/atlas/data/rho_delta/rho_small.root")
    cell_geo_tree = file["CellGeo"]

    node_feature_names = cell_geo_tree.keys()[1:7] # 'cell_geo_sampling', 'cell_geo_eta', 'cell_geo_phi', 'cell_geo_rPerp', 'cell_geo_deta', 'cell_geo_dphi'

    cell_geo_data = cell_geo_tree.arrays(library='np')
    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]
    sorter = np.argsort(cell_geo_ID)

    # get list of paths to pion files to convert
    rho_files = list(map(lambda i:  files_name + "_len_" + str(len_file) + "_i_" + str(i) + ".npy", np.arange(i_low, i_high + 1)))
    max_points = 0

    manager = multiprocessing.Manager()
    max_points_queue = manager.Queue()
    pool = multiprocessing.Pool(num_procs)

    # start a multiprocessing pool to convert each file on its own process
    pool.map(process_file, [(file_name, rho_dir, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, weight_cells, include_tracks, left_from_include_tracks, add_3_min_dists_and_layer) for file_name in rho_files])

    # as files are processed and the max num points are added to queue on completion, compare with current max 
    while not max_points_queue.empty():
        q_max_points = max_points_queue.get()

        if q_max_points > max_points:
            max_points = q_max_points

    # update max points if a exceeded
    try:
        with open(save_dir + 'max_points_1_track.txt') as f:
            current_max_points = int(f.readline())
    except IOError:
        current_max_points = 0
    
    if max_points > current_max_points:
        with open(save_dir + 'max_points_1_track.txt', 'w') as f:
            f.write('%d' % max_points)

    print("elapsed time:",  time.time() - t_start)
    
        