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

sys.path.append('/home/jbohm/start_tf/LCStudies')

LOG_ENERGY_MEAN = -1 # unrounded mean is ~ -0.93
LOG_MEAN_TRACK_MOMETUM = 2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, type=str)
    return parser.parse_args()


def process_file(args):
    file_name, rho_dir, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, weight_cells, include_tracks, left_from_include_tracks, add_3_min_dists_and_layer = args
    print(file_name)
    
    events_arr = np.load(rho_dir + file_name, allow_pickle=True).item()
    
    """ PREPROCESSING """
    # aggregate the data with the cluster it is in
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
    # [ 213.  211.  111.  -11.   11.   22.]
    # find the occurances of this array in the list 
    events_arr_resized = {}
    elec_pos_events = ak.Array([np.count_nonzero(np.array(event_truthPartPdgId) == 11) > 0 for event_truthPartPdgId in events_arr["truthPartPdgId"]])
    print("num events with electron/positron:", np.count_nonzero(elec_pos_events))

    num_truth_contributions = np.array([[len(cell_truth_con) for clus_truth_con in event_truth_con for cell_truth_con in clus_truth_con] for event_truth_con in events_arr["cluster_cell_hitsTruthIndex"]])
    no_cell_deposits = np.array([np.count_nonzero(np.array(num_truth_contributions_event) == 0) == len(num_truth_contributions_event) for num_truth_contributions_event in num_truth_contributions])
    print("num events no cell deposits:", np.count_nonzero(no_cell_deposits))

    for key in events_arr.keys():
        events_arr_resized[key] = events_arr[key][~elec_pos_events & (ak.Array(events_arr["nCluster"]) != 0) & ~no_cell_deposits]

    events_arr = events_arr_resized

    # Fix 3 - cut out the padding from the end of particle deposits array
    events_arr["cluster_cell_hitsTruthE"] = [[cluster_cell_hitsTruthE_clus[:len(events_arr["cluster_cell_hitsE_EM"][event_idx][clus_idx])] for clus_idx, cluster_cell_hitsTruthE_clus in enumerate(cluster_cell_hitsTruthE_event)] for event_idx, cluster_cell_hitsTruthE_event in enumerate(events_arr["cluster_cell_hitsTruthE"])]
    events_arr["cluster_cell_hitsTruthIndex"] = [[cluster_cell_hitsTruthIndex_clus[:len(events_arr["cluster_cell_hitsE_EM"][event_idx][clus_idx])] for clus_idx, cluster_cell_hitsTruthIndex_clus in enumerate(cluster_cell_hitsTruthIndex_event)] for event_idx, cluster_cell_hitsTruthIndex_event in enumerate(events_arr["cluster_cell_hitsTruthIndex"])]

    # Fix 4 - flatten events to cells & remove cell repeats
    cellwise_data_keys = ["cluster_cell_E", "cluster_cell_hitsE_EM", "cluster_cell_hitsE_nonEM", "cluster_cell_hitsTruthE", "cluster_cell_hitsTruthIndex", "cluster_cell_ID", "clus_idx", "clus_em_prob"]

    # flatten cellwise data to only be grouped by events, not clus and events
    for key in cellwise_data_keys:
        events_arr[key] = flatten_clus_in_events(events_arr[key])

    # iterate through the events and remove any repeated cell deposits (if 2+ of same cell ID in one event remove the extra)

    # define unique versions of the cellwise data arrays
    cellwise_data_unique = {}
    for key in cellwise_data_keys:
        cellwise_data_unique[key + "_unique"] = []

    # for every event remove the repeat 
    for event_idx in np.arange(len(events_arr["cluster_cell_ID"])):
        # get the idx of the unique cell IDs
        unique_events_cells_IDs, unique_cells_idx = np.unique(events_arr["cluster_cell_ID"][event_idx], return_index=True)

        # cut the particle ID and energy deposits to just have the unique cells per event
        for key in cellwise_data_keys:
            cellwise_data_unique[key + "_unique"].append(ak.Array(events_arr[key][event_idx])[unique_cells_idx])

    for key in cellwise_data_keys:
        events_arr[key] = cellwise_data_unique[key + "_unique"]

    print("num events:", len(events_arr["cluster_cell_hitsTruthIndex"]))

    # add feature of decay group to events_arr
    class DecayGroup():
        DELTA_PLUS_PI0 = 1
        DELTA_PLUS_PIPM = 2
        DELTA_0_PI0_N = 3
        DELTA_0_PIPM_P = 4
        DELTA_PLUS_PLUS = 5
        DELTA_MIN = 6
                    
    class PartDepositType():
        TRACK_OF_INTEREST = 1
        OTHER_TRACKED = 2
        PI0 = 3
        OTHER_NEUTRAL = 4                                                                                                                                                                                                                                                                                             

    # label events with their decay group
    truthPartPdgIds = events_arr["truthPartPdgId"]
    decay_group_cuts = {
        1: (truthPartPdgIds[:, 0] == 2214) & (truthPartPdgIds[:, 2] == 111),
        2: (truthPartPdgIds[:, 0] == 2214) & (truthPartPdgIds[:, 2] == 211),
        3: (truthPartPdgIds[:, 0] == 2114) & (truthPartPdgIds[:, 1] == 2112),
        4: (truthPartPdgIds[:, 0] == 2114) & (truthPartPdgIds[:, 1] == 2212),
        5: (truthPartPdgIds[:, 0] == 2224),
        6: (truthPartPdgIds[:, 0] == 1114),
    }

    events_arr["decay_group"] = np.zeros(len(events_arr["nTrack"])) # [decay_group_cuts["delta_plus_pi0"]] = 0
    events_arr["decay_group"][decay_group_cuts[DecayGroup.DELTA_PLUS_PI0]] = DecayGroup.DELTA_PLUS_PI0
    events_arr["decay_group"][decay_group_cuts[DecayGroup.DELTA_PLUS_PIPM]] = DecayGroup.DELTA_PLUS_PIPM
    events_arr["decay_group"][decay_group_cuts[DecayGroup.DELTA_0_PI0_N]] = DecayGroup.DELTA_0_PI0_N
    events_arr["decay_group"][decay_group_cuts[DecayGroup.DELTA_0_PIPM_P]] = DecayGroup.DELTA_0_PIPM_P
    events_arr["decay_group"][decay_group_cuts[DecayGroup.DELTA_PLUS_PLUS]] = DecayGroup.DELTA_PLUS_PLUS
    events_arr["decay_group"][decay_group_cuts[DecayGroup.DELTA_MIN]] = DecayGroup.DELTA_MIN

    event_data = events_arr

    """ DONE PREPROCESSING """
    """"""""""""""""""""""""""

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

    for feature in [*node_feature_names, 'x', 'y', 'z', 'cell_hitsTruthIndex', 'cell_hitsTruthE', 'class_pi0_lead_energy', 'cell_E', 'cell_E_weight', 'sampling_layer', 'delta_R', 'clus_idx', 'track_pipm_delta_R', 'track_pi0_delta_R', 'decay_group', 'cell_labels']:
        processed_event_data[feature] = []

    for feature in ['x', 'y', 'z', 'P', 'min_dist', 'min_eta', 'min_phi', 'sampling_layer', 'non_null_tracks', 'track_classes']:
        processed_event_track_data[feature] = []

    for event_idx in range(num_events):
        print(event_idx)
        num_tracks = event_data["nTrack"][event_idx]

        if len(events_arr["cluster_cell_ID"][event_idx]) and (((events_arr["decay_group"][event_idx] == DecayGroup.DELTA_PLUS_PI0 or events_arr["decay_group"][event_idx] == DecayGroup.DELTA_PLUS_PIPM or events_arr["decay_group"][event_idx] == DecayGroup.DELTA_MIN) and num_tracks == 1) or \
        (events_arr["decay_group"][event_idx] == DecayGroup.DELTA_0_PI0_N and num_tracks == 0) or \
        ((events_arr["decay_group"][event_idx] == DecayGroup.DELTA_PLUS_PLUS or events_arr["decay_group"][event_idx] == DecayGroup.DELTA_0_PIPM_P) and num_tracks == 2)):
            if num_tracks > 0:
                x_tracks = []
                y_tracks = []
                z_tracks = []
                eta_tracks = []
                phi_tracks = []
                rPerp_track = []
                non_null_tracks = []
                for track_idx in range(num_tracks):
                    x_tracks.append([])
                    y_tracks.append([])
                    z_tracks.append([])
                    eta_tracks.append([])
                    phi_tracks.append([])
                    rPerp_track.append([])
                    non_null_tracks.append([])
                    # get the eta, phi, and rPerp of each layers hit
                    for layer_name in calo_layers:
                        if has_fixed_r[layer_name]:
                            eta_tracks[track_idx].append(event_data['trackEta_' + layer_name][event_idx][track_idx])
                            phi_tracks[track_idx].append(event_data['trackPhi_' + layer_name][event_idx][track_idx])
                            rPerp_track[track_idx].append(fixed_r[layer_name])
                        else:
                            eta = event_data['trackEta_' + layer_name][event_idx][track_idx]
                            eta_tracks[track_idx].append(eta)
                            phi_tracks[track_idx].append(event_data['trackPhi_' + layer_name][event_idx][track_idx])
                            
                            z = fixed_z[layer_name]
                            aeta = np.abs(eta)
                            rPerp = z*2*np.exp(aeta)/(np.exp(2*aeta) - 1)
                            rPerp_track[track_idx].append(rPerp)
                    
                    # convert each hit to the cartesian coords
                    thetas = [2*np.arctan(np.exp(-eta)) for eta in eta_tracks[track_idx]]
                    x_tracks_i, y_tracks_i, z_tracks_i = spherical_to_cartesian(rPerp_track[track_idx], phi_tracks[track_idx], thetas)
                    x_tracks[track_idx].append(x_tracks_i)
                    y_tracks[track_idx].append(y_tracks_i)
                    z_tracks[track_idx].append(z_tracks_i)

                    non_null_tracks_i = np.full(NUM_TRACK_POINTS, True)

                    # if eta > 4.9 or phi > pi mark the track as null (why??)
                    non_null_tracks_i[(np.abs(eta_tracks[track_idx]) >= 2.5) | (np.abs(phi_tracks[track_idx]) > np.pi)] = False

                    non_null_tracks[track_idx].append(non_null_tracks_i)
                    #eta_tracks = np.array(eta_tracks)
                    #phi_tracks = np.array(phi_tracks)
            else:
                non_null_tracks = [np.full(NUM_TRACK_POINTS, 0)]
                x_tracks, y_tracks, z_tracks = [np.zeros(NUM_TRACK_POINTS)], [np.zeros(NUM_TRACK_POINTS)], [np.zeros(NUM_TRACK_POINTS)]


            cell_IDs = event_data['cluster_cell_ID'][event_idx]

            # get truth energy deposits
            frac_pi0_energy = ak.sum(events_arr["cluster_cell_hitsTruthE"][event_idx][events_arr["cluster_cell_hitsTruthIndex"][event_idx] != 1], axis=1)/ak.sum(events_arr["cluster_cell_hitsTruthE"][event_idx], axis=1)
            #empty_contr_frac = np.isnan(frac_pi0_energy)

            # filter out cells without truth particle deposits
            threshold_E_deposit = np.full(len(cell_IDs), True)#~empty_contr_frac
            
            cell_IDs = cell_IDs[threshold_E_deposit]
            num_cells = len(cell_IDs)
            cell_ID_map = sorter[np.searchsorted(cell_geo_ID, cell_IDs, sorter=sorter)]

            
            # classify leading particle contributor
            class_frac_pi0_energy = frac_pi0_energy[threshold_E_deposit] > 0.5

            # get cluster cell energy
            cell_E = event_data["cluster_cell_E"][event_idx][threshold_E_deposit]
            cell_E_weight = cell_E / np.sum(cell_E)
            cell_E = np.log10(cell_E) - LOG_ENERGY_MEAN

            # label cells
        if event_data["decay_group"][event_idx] == DecayGroup.DELTA_PLUS_PI0:
            class_part_idx_1 = 0
            class_part_idx_not_1 = 2
        elif event_data["decay_group"][event_idx] == DecayGroup.DELTA_PLUS_PIPM or event_data["decay_group"][event_idx] == DecayGroup.DELTA_MIN:
            class_part_idx_1 = 3
            class_part_idx_not_1 = 0
        elif event_data["decay_group"][event_idx] == DecayGroup.DELTA_0_PI0_N:
            class_part_idx_1 = 3
            class_part_idx_not_1 = 2
        elif event_data["decay_group"][event_idx] == DecayGroup.DELTA_PLUS_PLUS or event_data["decay_group"][event_idx] == DecayGroup.DELTA_0_PIPM_P:
            class_part_idx_1 = 3
            class_part_idx_not_1 = 0

        # get all cells with particle idx 1 (proton/neutron)
        cut_part_idx_1_deposits = (ak.Array(event_data["cluster_cell_hitsTruthIndex"][event_idx]) == 1)
        # get fraction of energy from the proton/neutron (always p/n in a rho event - it deposits some energy and the pion deposits the remaining)
        frac_cell_energy_from_part_idx_1 = ak.sum(ak.Array(event_data["cluster_cell_hitsTruthE"][event_idx])[cut_part_idx_1_deposits], axis=1) / ak.sum(event_data["cluster_cell_hitsTruthE"][event_idx], axis=1)

        cell_labels = [class_part_idx_not_1 if cell_frac_cell_energy_from_part_idx_1 < 0.5 else class_part_idx_1 for cell_frac_cell_energy_from_part_idx_1 in frac_cell_energy_from_part_idx_1]
        
        # if the particle has 2 tracks match the track to the particle closest & threshold that they must be close enough together
        dist_track_part_thresh = 1
        if num_tracks == 2:
            #print("TWO TRACKS !!")

            part1_idx = 1 # proton
            part2_idx = 2 # charged pion

            part1_phi = event_data["truthPartPhi"][event_idx][part1_idx]
            part1_eta = event_data["truthPartEta"][event_idx][part1_idx]
            part1_pt = event_data["truthPartPt"][event_idx][part1_idx]

            part2_phi = event_data["truthPartPhi"][event_idx][part2_idx]
            part2_eta = event_data["truthPartEta"][event_idx][part2_idx]
            part2_pt = event_data["truthPartPt"][event_idx][part2_idx]

            track1_phi = event_data["trackPhi"][event_idx][0]
            track1_eta = event_data["trackEta"][event_idx][0]
            track1_pt = event_data["trackPt"][event_idx][0]

            track2_phi = event_data["trackPhi"][event_idx][1]
            track2_eta = event_data["trackEta"][event_idx][1]
            track2_pt = event_data["trackPt"][event_idx][1]

            part1_track1_dist = measure_track_part_dists(track1_phi, track1_eta, track1_pt, part1_phi, part1_eta, part1_pt)
            part1_track2_dist = measure_track_part_dists(track2_phi, track2_eta, track2_pt, part1_phi, part1_eta, part1_pt)
            part2_track1_dist = measure_track_part_dists(track1_phi, track1_eta, track1_pt, part2_phi, part2_eta, part2_pt)
            part2_track2_dist = measure_track_part_dists(track2_phi, track2_eta, track2_pt, part2_phi, part2_eta, part2_pt)


            #print("part1_track1_dist:", part1_track1_dist)
            #print("part1_track2_dist:", part1_track2_dist)
            #print("part2_track1_dist:", part2_track1_dist)
            #print("part2_track2_dist:", part2_track2_dist)

            # either pair part1 with track1 and part2 with track2 or part1 with track2 and part2 with track1
            # or discard event if no pairing exists with all dists < thresh

            paring_one_sum_dist = part1_track1_dist + part2_track2_dist if part1_track1_dist < dist_track_part_thresh and part2_track2_dist < dist_track_part_thresh else 2*dist_track_part_thresh
            paring_two_sum_dist = part1_track2_dist + part2_track1_dist if part1_track1_dist < dist_track_part_thresh and part2_track2_dist < dist_track_part_thresh else 2*dist_track_part_thresh
            
            #print("min pairing 1:", paring_one_sum_dist)
            #print("min pairing 1:", paring_one_sum_dist)
            if max(paring_one_sum_dist, paring_two_sum_dist) >= 2*dist_track_part_thresh:
                num_tracks = 0
            else:
                if paring_one_sum_dist < paring_two_sum_dist:
                    pairing_one = True
                else:
                    pairing_one = False

        track_idx = 0
        execute_once = True
        # execute once for 0-1 track, and 2 times for 2 tracks
        while execute_once or track_idx < num_tracks:
            #print("TRACK 1")
            # node features
            node_features = {}
            for feature in node_feature_names:
                node_features[feature] = cell_geo_data[feature][0][cell_ID_map]

            # get cartesian coords
            thetas = [2*np.arctan(np.exp(-eta)) for eta in node_features["cell_geo_eta"]]
            x, y, z = spherical_to_cartesian(node_features["cell_geo_rPerp"], node_features["cell_geo_phi"], thetas)


            processed_event_data["cell_hitsTruthIndex"].append(event_data["cluster_cell_hitsTruthIndex"][event_idx][threshold_E_deposit])
            processed_event_data["cell_hitsTruthE"].append(event_data["cluster_cell_hitsTruthE"][event_idx][threshold_E_deposit])
            processed_event_data["cell_E"].append(cell_E)
            processed_event_data["cell_E_weight"].append(cell_E_weight)
            #processed_event_data["class_frac_pi0_energy"].append(class_frac_pi0_energy)
            #processed_event_data["frac_pi0_energy"].append(frac_pi0_energy[threshold_E_deposit])
            processed_event_data["clus_idx"].append(event_data["clus_idx"][event_idx][threshold_E_deposit])
            processed_event_data["clus_em_prob"].append(event_data["clus_em_prob"][event_idx][threshold_E_deposit])
            processed_event_data["sampling_layer"].append(node_features["cell_geo_sampling"])

            processed_event_data["truthPartPt"].append(event_data["truthPartPt"][event_idx])
            processed_event_data["truthPartEta"].append(event_data["truthPartEta"][event_idx])
            processed_event_data["truthPartPhi"].append(event_data["truthPartPhi"][event_idx])  

            processed_event_data["decay_group"].append(event_data["decay_group"][event_idx])            

            
            for feature in node_feature_names:
                processed_event_data[feature].append(np.array(node_features[feature]))

            processed_event_data["x"].append(x)
            processed_event_data["y"].append(y)
            processed_event_data["z"].append(z)     
                      
            track_classes = np.zeros(NUM_TRACK_POINTS)
            if num_tracks == 2:
                #print("track_idx:", track_idx)
                #print("pairing 1:", pairing_one)
                # set track_idx to be the track of interest
                
                
                if track_idx == 0:
                    track_classes[:NUM_TRACK_POINTS/2] = 1
                else:
                    track_classes[NUM_TRACK_POINTS/2:] = 1
                #print("track_classes:", track_classes)

                if (pairing_one and track_idx == 0) or (not pairing_one and track_idx == 1): # pair track 0 and part 1
                    class_part_idx_1 = 0 # track of interest
                    class_part_idx_not_1 = 1 # other tracked charged particle 

                else: # pairing 1 and track idx == 1 or paring 2 and track idx == 0
                    class_part_idx_1 = 1
                    class_part_idx_not_1 = 0


                cell_labels = [class_part_idx_not_1 if cell_frac_cell_energy_from_part_idx_1 < 0.5 else class_part_idx_1 for cell_frac_cell_energy_from_part_idx_1 in frac_cell_energy_from_part_idx_1]
                #print("cell_labels:", cell_labels)
                track_Ps = np.concatenate((np.tile([events_arr["trackPt"][event_idx][0]], NUM_TRACK_POINTS), np.tile([events_arr["trackPt"][event_idx][1]], NUM_TRACK_POINTS)))
            elif num_tracks == 1: # the first NUM_TRACK_POINTS/2 are the track points - the remaining are padding
                track_classes[:NUM_TRACK_POINTS/2] = 1
            
            # else no tracks => Pt = 0

            processed_event_data["cell_labels"].append(cell_labels)

            processed_event_track_data["x"].append(np.array(flatten_one_layer(x_tracks)))
            processed_event_track_data["y"].append(np.array(flatten_one_layer(y_tracks)))
            processed_event_track_data["z"].append(np.array(flatten_one_layer(z_tracks)))
            processed_event_track_data["P"].append(flatten_one_layer(track_Ps)) # don't normalize for analysis
            #processed_event_track_data["sampling_layer"].append(sampling_layer_num)
            #processed_event_track_data["track_eta"].append(flatten_one_layer(event_data["trackEta"][event_idx]))
            #processed_event_track_data["track_phi"].append(flatten_one_layer(event_data["trackPhi"][event_idx]))
            processed_event_track_data["non_null_tracks"].append(np.array(flatten_one_layer(non_null_tracks)))

            # track classes - 0 => point, 1 => track of interest, 2 => other track
            processed_event_track_data["track_classes"].append(np.array(track_classes))
                

            samples_count += 1
            execute_once = False
            track_idx += 1


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


    # TODO: update labels then start training !!!
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


    file_path = save_dir + "/" + file_name.split(".")[0] + '_1_track.npz'
    np.savez(file_path, X=point_data, Y=point_label)

    np.save(save_dir + "/delta_R/" + file_name.split(".")[0] + "_delta_R.npy", processed_event_data["delta_R"])
    np.save(save_dir + "/delta_R/" + file_name.split(".")[0] + "_track_pipm_delta_R.npy", processed_event_data["track_pipm_delta_R"])
    np.save(save_dir + "/delta_R/" + file_name.split(".")[0] + "_track_pi0_delta_R.npy", processed_event_data["track_pi0_delta_R"])

    max_points_queue.put(max_cells)


if __name__ == "__main__":
    t_start = time.time()
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

    # load cell geo tree dict
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

    pool.map(process_file, [(file_name, rho_dir, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, weight_cells, include_tracks, left_from_include_tracks, add_3_min_dists_and_layer) for file_name in rho_files])

    while not max_points_queue.empty():
        q_max_points = max_points_queue.get()

        if q_max_points > max_points:
            max_points = q_max_points

        try:
            with open(save_dir + 'max_points_1_track.txt') as f:
                current_max_points = int(f.readline())
        except IOError:
            current_max_points = 0
    
    if max_points > current_max_points:
        with open(save_dir + 'max_points_1_track.txt', 'w') as f:
            f.write('%d' % max_points)

    print("elapsed time:",  time.time() - t_start)
    
        