import time
import numpy as np
import yaml
import sys
import uproot
import awkward as ak
import multiprocessing
import vector

#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append("/home/mjovanovic/Work/PointNet_Segmentation")
from utils.track_metadata import *
from utils.data_processing_util import *

# computed with preliminary analysis
LOG_ENERGY_MEAN = -1 # unrounded mean is ~ -0.93
LOG_MEAN_TRACK_MOMETUM = 2

def process_events(args):
    event_root_data, preprocessed, preprocessed_file_name, event_start_idx, file_len, features_of_interest, dataset, file_name, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, include_delta_p_pi0, include_delta_n_pi0, include_delta_p_pipm, include_delta_n_pipm, niche_case, regression = args
    print(f"Processing: {preprocessed_file_name}...")
    decay_group = {
        "delta+_p": 0,
        "delta+_n": 1,
        "delta0_n": 2,
        "delta0_p": 3,
        "delta++": 4,
        "delta-": 5,
    }   

    part_deposit_type_class = {
        "track_of_interest": 0,
        "other_tracked": 1,
        "pi0": 2,
        "other_neutral": 3,
    }

    particle_to_pdgid = {
        "proton": 2212,
        "neutron": 2112,
        "delta++": 2224,
        "delta+": 2214,
        "delta0": 2114,
        "delta-": 1114
    }     
    if not preprocessed:
        # load file_len sized chunk of root file, starting at event_start_idx to process into one file

        print("Entering preprocessing...")
        print("len of stuff: ",len(event_root_data))

        for events_arr in event_root_data.iterate(['cluster_cell_E'], step_size=10, entry_start=0, entry_stop=10, library="np"):
            print("Iteration successful.")
        for events_arr in event_root_data.iterate(features_of_interest, step_size=file_len, entry_start=event_start_idx, entry_stop=(event_start_idx + file_len), library="np"):
            print("wadwa")
            print("Going into keys...")
            file_idx = int(event_start_idx/file_len)
            for key in events_arr.keys():
                events_arr[key] = ak.Array(events_arr[key])


            print("START PROCESSING file", file_idx)

            """ PREPROCESSING """
            # aggregate the data with the cluster it is in
            events_arr["clus_idx"] = []
            events_arr["clus_em_prob"] = []
            for event_idx, event_clus_cell_E in enumerate(events_arr["cluster_cell_E"]):
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

            # TODO: look over this step !!
            # Fix 2 - cut out the events w no clusters, or no cell hits
            events_arr_resized = {}

            num_truth_contributions = np.array([[len(cell_truth_con) for clus_truth_con in event_truth_con for cell_truth_con in clus_truth_con] for event_truth_con in events_arr["cluster_cell_hitsTruthIndex"]])
            no_cell_deposits = np.array([np.count_nonzero(np.array(num_truth_contributions_event) == 0) == len(num_truth_contributions_event) for num_truth_contributions_event in num_truth_contributions])

            for key in events_arr.keys():
                events_arr_resized[key] = events_arr[key][(ak.Array(events_arr["nCluster"]) != 0) & ~no_cell_deposits]

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

            num_events = len(events_arr["cluster_cell_ID"])
            # for every event remove the repeat 
            for event_idx in np.arange(num_events):
                # get the idx of the unique cell IDs
                unique_events_cells_IDs, unique_cells_idx = np.unique(events_arr["cluster_cell_ID"][event_idx], return_index=True)

                # cut the particle ID and energy deposits to just have the unique cells per event
                for key in cellwise_data_keys:
                    cellwise_data_unique[key + "_unique"].append(ak.Array(events_arr[key][event_idx])[unique_cells_idx])

            for key in cellwise_data_keys:
                events_arr[key] = cellwise_data_unique[key + "_unique"]

            # if Delta dataset add feature of decay group to events_arr
            if dataset == "delta":
                                                                                                                                                                                                                                                                                      

                # label events with their decay group
                truthPartPdgIds = events_arr["truthPartPdgId"]

                truthPartPdgIds_idx_0 = np.abs(truthPartPdgIds[:, 0]) # delta id
                truthPartPdgIds_idx_1 = np.abs(truthPartPdgIds[:, 1]) # proton/neutron id

                decay_group_cuts = {
                    "delta+_p": (truthPartPdgIds_idx_0 == particle_to_pdgid["delta+"]) & (truthPartPdgIds_idx_1 == particle_to_pdgid["proton"]),
                    "delta+_n": (truthPartPdgIds_idx_0 == particle_to_pdgid["delta+"]) & (truthPartPdgIds_idx_1 == particle_to_pdgid["neutron"]),
                    "delta0_n": (truthPartPdgIds_idx_0 == particle_to_pdgid["delta0"]) & (truthPartPdgIds_idx_1 == particle_to_pdgid["neutron"]),
                    "delta0_p": (truthPartPdgIds_idx_0 == particle_to_pdgid["delta0"]) & (truthPartPdgIds_idx_1 == particle_to_pdgid["proton"]),
                    "delta++": (truthPartPdgIds_idx_0 == particle_to_pdgid["delta++"]),
                    "delta-": (truthPartPdgIds_idx_0 == particle_to_pdgid["delta-"]),
                }
                
                # label each event with its decay group class
                events_arr["decay_group"] = np.zeros(num_events)
                for decay_group_name, decay_cut in decay_group_cuts.items():
                    events_arr["decay_group"][decay_cut] = decay_group[decay_group_name]
                    

            event_data = events_arr

            print("DONE PREPROCESSING file", file_idx)

            # save preprocessed event data flattened to events
            file_details = '_len_' + str(file_len) + '_i_' + str(int(file_idx))
            file_label = "delta_full"
            np.save(save_dir + dataset + "_processed_test_files/" + file_label + file_details, event_data)

    else:
        print("already preprocessed")
        event_data = np.load(save_dir + dataset + "_processed_test_files/" + preprocessed_file_name[:-8] + ".npz", allow_pickle=True).item()
        file_label = "_".join(preprocessed_file_name.split("_")[:1])
        file_details = "_".join(preprocessed_file_name.split("_")[2:6])

    if dataset == "delta":
        if niche_case == "1_track_1_n_3_pi0":
            file_name = "delta_1_track_1_n_3_pi0_" + file_details
        
        elif include_delta_p_pi0 and include_delta_n_pi0 and include_delta_p_pipm and include_delta_n_pipm: # full delta dataset
            file_name = file_label + "_" + file_details
            
        elif include_delta_p_pi0 and not include_delta_n_pi0 and include_delta_p_pipm and include_delta_n_pipm: # all tracked delta events (don't include delta0 -> n + pi0)
            file_name = file_label + "_tracked_" + file_details

        elif include_delta_p_pi0 and not include_delta_n_pi0 and not include_delta_p_pipm and include_delta_n_pipm: # all events w one track
            file_name = file_label + "_1_track_all_neutral_2" + file_details # file_name = file_label + "_1_track_" + file_details # TEMPORARY UPDATE

        elif include_delta_p_pi0 and not include_delta_n_pi0 and not include_delta_p_pipm and not include_delta_n_pipm: # only delta+ -> p + pi0
            file_name = "delta+_p_pi0_" + file_details

        elif not include_delta_p_pi0 and not include_delta_n_pi0 and include_delta_p_pipm and not include_delta_n_pipm: # delta++ or delta0 -> p + pi+/-
            file_name = "delta_p_pipm_" + file_details

        elif not include_delta_p_pi0 and not include_delta_n_pi0 and not include_delta_p_pipm and include_delta_n_pipm: # delta -> n + pi+/-
            file_name = "delta_n_pipm_" + file_details
        
    
    else:
        print("rho")
        file_name = file_label + "_" + file_details


    # cluster data dict to look up data by feature name
    processed_event_data = {}
    processed_event_track_data = {}

    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]

    num_events_saved = 0
    max_cells = 0
    num_cells = 0
    max_num_tracks = 1 if dataset == "rho" else 2 # just delta and rho dataset processing for now

    for feature in [*node_feature_names, 'x', 'y', 'z', 'cell_E', 'cell_part_deposit_labels', 'cell_weights', 'cell_regression_deposit_labels']:
        processed_event_data[feature] = []

    for feature in ['x', 'y', 'z', 'P', 'track_classes']:
        processed_event_track_data[feature] = []

    num_events = len(event_data["eventNumber"])

    delta_n_pipm_count = 0
    tot_n_pipm = 0
    tot_p_pi0 = 0

    for event_idx in range(num_events):
        num_tracks = event_data["nTrack"][event_idx]

        if (len(event_data["cluster_cell_ID"][event_idx]) and \
        (dataset == "delta" and  \
         (((num_tracks == 1) and ((include_delta_p_pi0 and event_data["decay_group"][event_idx] == decay_group["delta+_p"]) or (include_delta_n_pipm and (event_data["decay_group"][event_idx] == decay_group["delta+_n"] or event_data["decay_group"][event_idx] == decay_group["delta-"])))) or \
         (include_delta_n_pi0 and (num_tracks == 0) and event_data["decay_group"][event_idx] == decay_group["delta0_n"]) or \
         (include_delta_p_pipm and (num_tracks == 2) and (event_data["decay_group"][event_idx] == decay_group["delta0_p"] or event_data["decay_group"][event_idx] == decay_group["delta++"])))) \
        or (dataset == "rho" and num_tracks == 1)):
            
            # if case where 1_track_1_n_3_pi0 need to discard 5 delta->neutron+pipm events for every 1 kept
            if niche_case == "1_track_1_n_3_pi0" and (event_data["decay_group"][event_idx] == decay_group["delta+_n"] or event_data["decay_group"][event_idx] == decay_group["delta-"]):
                if delta_n_pipm_count == 0:
                    delta_n_pipm_count += 1

                else:
                    delta_n_pipm_count = (delta_n_pipm_count + 1) % 6
                    continue # don't save event when delta_n_pipm_count = 1, 2, 3, 4, 5
            
            

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
                    x_tracks[track_idx] = x_tracks_i
                    y_tracks[track_idx] = y_tracks_i
                    z_tracks[track_idx] = z_tracks_i

                    non_null_tracks_i = np.full(NUM_TRACK_POINTS, True)

                    # if eta > 4.9 or phi > pi mark the track as null (why??)
                    non_null_tracks_i[(np.abs(eta_tracks[track_idx]) >= 2.5) | (np.abs(phi_tracks[track_idx]) > np.pi)] = False

                    non_null_tracks[track_idx] = non_null_tracks_i

                if dataset == "delta":
                    if len(non_null_tracks) != 2:
                        non_null_tracks = np.concatenate((non_null_tracks, np.full((1, NUM_TRACK_POINTS), False)))
                        x_tracks = np.concatenate((x_tracks, np.full((1, NUM_TRACK_POINTS), False)))
                        y_tracks = np.concatenate((y_tracks, np.full((1, NUM_TRACK_POINTS), False)))
                        z_tracks = np.concatenate((z_tracks, np.full((1, NUM_TRACK_POINTS), False)))
            else:
                non_null_tracks = np.full((max_num_tracks, NUM_TRACK_POINTS), False)
                x_tracks, y_tracks, z_tracks = np.zeros((max_num_tracks, NUM_TRACK_POINTS)), np.zeros((max_num_tracks, NUM_TRACK_POINTS)), np.zeros((max_num_tracks, NUM_TRACK_POINTS))
            

            cell_IDs = event_data['cluster_cell_ID'][event_idx]
            cell_IDs = cell_IDs
            cell_ID_map = sorter[np.searchsorted(cell_geo_ID, cell_IDs, sorter=sorter)]

            # get cluster cell energy
            cell_E = event_data["cluster_cell_E"][event_idx]

            cell_weights = cell_E / np.sum(cell_E)
            cell_E = np.log10(cell_E) - LOG_ENERGY_MEAN


            # node features
            node_features = {}
            for feature in node_feature_names:
                node_features[feature] = cell_geo_data[feature][0][cell_ID_map]

            # get cartesian coords
            thetas = [2*np.arctan(np.exp(-eta)) for eta in node_features["cell_geo_eta"]]
            x, y, z = spherical_to_cartesian(node_features["cell_geo_rPerp"], node_features["cell_geo_phi"], thetas)

            # label cells (4 classes)
            # all decay groups only have 2 types particles depositing energy, a proton/neutron and a pion
            if dataset == "delta":
                class_part_idx_1 = 0
                class_part_idx_not_1 = 0

                if event_data["decay_group"][event_idx] == decay_group["delta+_p"]:
                    class_part_idx_1 = part_deposit_type_class["track_of_interest"] # proton
                    class_part_idx_not_1 = part_deposit_type_class["pi0"] # pi0
                elif event_data["decay_group"][event_idx] == decay_group["delta+_n"] or event_data["decay_group"][event_idx] == decay_group["delta-"]:
                    class_part_idx_1 = part_deposit_type_class["pi0"] # neutron # class_part_idx_1 = part_deposit_type_class["other_neutral"] # neutron # TEMPORARY
                    class_part_idx_not_1 = part_deposit_type_class["track_of_interest"] # pi+/-
                elif event_data["decay_group"][event_idx] == decay_group["delta0_n"]:
                    class_part_idx_1 = part_deposit_type_class["other_neutral"] # neutron
                    class_part_idx_not_1 = part_deposit_type_class["pi0"] # pi0
                # if decay is delta++ or delta0_p then set labels elsewhere (since 2 tracks)

                # get all cells with particle idx 1 (proton/neutron)
                cut_part_idx_1_deposits = (ak.Array(event_data["cluster_cell_hitsTruthIndex"][event_idx]) == 1)
                # get fraction of energy from the proton/neutron (always p/n in a rho event - it deposits some energy and the pion deposits the remaining)
                frac_cell_energy_from_part_idx_1 = ak.sum(ak.Array(event_data["cluster_cell_hitsTruthE"][event_idx])[cut_part_idx_1_deposits], axis=1) / ak.sum(event_data["cluster_cell_hitsTruthE"][event_idx], axis=1)
                # if frac_cell_energy_from_part_idx_1 < 0.5 set label class_part_idx_not_1 else set cell label to class_part_idx_1
                cell_part_deposit_labels = [class_part_idx_not_1 if cell_frac_cell_energy_from_part_idx_1 < 0.5 else class_part_idx_1 for cell_frac_cell_energy_from_part_idx_1 in frac_cell_energy_from_part_idx_1]

            # only pi0 and pi+/- depositing energy -> binary classification
            elif dataset == "rho":
                # if the pi0 deposits the majority of the energy label cell 1 else if pi+/- deposits majority label cell 0
                frac_pi0_energy = ak.sum(event_data["cluster_cell_hitsTruthE"][event_idx][event_data["cluster_cell_hitsTruthIndex"][event_idx] != 1], axis=1)/ak.sum(event_data["cluster_cell_hitsTruthE"][event_idx], axis=1)
                cell_part_deposit_labels = [1 if cell_frac_pi0_energy > 0.5 else 0 for cell_frac_pi0_energy in frac_pi0_energy]
                cell_regression_deposit_labels = [cell_frac_pi0_energy for cell_frac_pi0_energy in frac_pi0_energy]

            # if the particle has 2 tracks match the track to the particle closest & threshold that they must be close enough together
            # for delta dataset either delta++ -> proton + pi+/- or delta0 -> proton + pi+/-
            track_part_dist_thresh = 1
            if num_tracks == 2:

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

                # either pair part1 with track1 and part2 with track2 or part1 with track2 and part2 with track1
                # or discard event if no pairing exists with both track-part dists < thresh
                paring_one_sum_dist = part1_track1_dist + part2_track2_dist if part1_track1_dist < track_part_dist_thresh and part2_track2_dist < track_part_dist_thresh else 2*track_part_dist_thresh
                paring_two_sum_dist = part1_track2_dist + part2_track1_dist if part1_track1_dist < track_part_dist_thresh and part2_track2_dist < track_part_dist_thresh else 2*track_part_dist_thresh
                
                if max(paring_one_sum_dist, paring_two_sum_dist) >= 2*track_part_dist_thresh:
                    num_tracks = 0
                else:
                    if paring_one_sum_dist < paring_two_sum_dist:
                        pairing_one = True
                    else:
                        pairing_one = False

            track_idx = 0
            added_one_sample = False # for each event add one sample to dataset

            non_null_tracks = np.array(flatten_one_layer(non_null_tracks))
            x_tracks = np.array(flatten_one_layer(x_tracks))
            y_tracks = np.array(flatten_one_layer(y_tracks))
            z_tracks = np.array(flatten_one_layer(z_tracks))
            
            x_tracks[~non_null_tracks] = 0
            y_tracks[~non_null_tracks] = 0
            z_tracks[~non_null_tracks] = 0

            cell_has_E_deposit = ak.sum(event_data["cluster_cell_hitsTruthE"][event_idx], axis=1) > 0
            num_cells = len(cell_E[cell_has_E_deposit])
            
            # execute once for 0-1 track, and 2 times for 2 tracks
            while not added_one_sample or track_idx < num_tracks:
                processed_event_data["cell_E"].append(cell_E[cell_has_E_deposit])
                processed_event_data["x"].append(x[cell_has_E_deposit])
                processed_event_data["y"].append(y[cell_has_E_deposit])
                processed_event_data["z"].append(z[cell_has_E_deposit])   
                processed_event_data["cell_weights"].append(cell_weights[cell_has_E_deposit])  

                if dataset == "delta":
                    track_classes = np.zeros((2, NUM_TRACK_POINTS))
                    track_Ps = np.zeros((2, NUM_TRACK_POINTS))
                else:
                    track_classes = np.zeros((1, NUM_TRACK_POINTS))
                    track_Ps = np.zeros((1, NUM_TRACK_POINTS))
                        
                if num_tracks == 2:
                    # set track_idx to be the track of interest (for track idx 0 track 0 is of interest, else track 1)
                    if track_idx == 0:
                        track_classes[0] = np.ones(NUM_TRACK_POINTS)
                        track_classes[1] = np.full(NUM_TRACK_POINTS, 2)
                    else:
                        track_classes[1] = np.ones(NUM_TRACK_POINTS)
                        track_classes[0] = np.full(NUM_TRACK_POINTS, 2)

                    if (pairing_one and track_idx == 0) or (not pairing_one and track_idx == 1): # pair track 0 and part 1
                        class_part_idx_1 = 0 # track of interest
                        class_part_idx_not_1 = 1 # other tracked charged particle 
                        cell_regression_frac = 1 - frac_cell_energy_from_part_idx_1

                    else: # pairing 1 and track idx == 1 or paring 2 and track idx == 0
                        class_part_idx_1 = 1
                        class_part_idx_not_1 = 0
                        cell_regression_frac = frac_cell_energy_from_part_idx_1

                    cell_part_deposit_labels = [class_part_idx_not_1 if cell_frac_cell_energy_from_part_idx_1 < 0.5 else class_part_idx_1 for cell_frac_cell_energy_from_part_idx_1 in frac_cell_energy_from_part_idx_1]
                    cell_regression_deposit_labels = [regress_label for regress_label in cell_regression_frac]
                    #print("cell_labels:", cell_labels)
                    track_1_P =  np.log10((event_data["trackP"][event_idx][0])) - LOG_ENERGY_MEAN
                    track_2_P =  np.log10((event_data["trackP"][event_idx][1])) - LOG_ENERGY_MEAN
                    track_Ps[0] = np.full(NUM_TRACK_POINTS, track_1_P)
                    track_Ps[1] = np.full(NUM_TRACK_POINTS, track_2_P)
                
                elif num_tracks == 1:
                    # get tracks momentum readout
                    track_P = np.log10(event_data['trackP'][event_idx][0]) - LOG_MEAN_TRACK_MOMETUM
                    track_Ps[0] = np.full(NUM_TRACK_POINTS, track_P)
                    track_classes[0] = np.ones(NUM_TRACK_POINTS)

                track_classes = np.array(flatten_one_layer(track_classes))
                track_Ps = np.array(flatten_one_layer(track_Ps))

                track_classes[~non_null_tracks] = 0
                track_Ps[~non_null_tracks] = 0
                
                # else no tracks => Pt = 0

                processed_event_data["cell_part_deposit_labels"].append(np.array(cell_part_deposit_labels)[cell_has_E_deposit])
                processed_event_data["cell_regression_deposit_labels"].append(np.array(cell_regression_deposit_labels)[cell_has_E_deposit])

                processed_event_track_data["x"].append(x_tracks)
                processed_event_track_data["y"].append(y_tracks)
                processed_event_track_data["z"].append(z_tracks)
                processed_event_track_data["P"].append(track_Ps) # don't normalize for analysis

                # track classes - 0 => point, 1 => track of interest, 2 => other track
                processed_event_track_data["track_classes"].append(np.array(track_classes))
                
                if num_cells + NUM_TRACK_POINTS*max_num_tracks > max_cells:
                    max_cells = num_cells + NUM_TRACK_POINTS*max_num_tracks
                if not dataset == "rho":
                    if event_data["decay_group"][event_idx] == decay_group["delta+_p"]:
                        tot_p_pi0 += 1
                    elif event_data["decay_group"][event_idx] == decay_group["delta+_n"] or event_data["decay_group"][event_idx] == decay_group["delta-"]:
                        tot_n_pipm += 1

                num_events_saved += 1
                added_one_sample = True
                track_idx += 1

    # save all event data
    track_features = ["P", "x", "y", "z"]
    file_features = ['cell_E', 'x', 'y', 'z']
    
    point_data = np.zeros((num_events_saved, max_cells, len(file_features) + 1)) # pad X data with 0's

    label_len = 1
    point_label = np.negative(np.ones((num_events_saved, max_cells, label_len))) # pad Y data with -1's, since some labels are 0


    for idx in range(num_events_saved):
        num_points = len(processed_event_data["cell_part_deposit_labels"][idx]) + NUM_TRACK_POINTS*max_num_tracks
        # labels points as point (0), track of interest (1), or other track (2)
        type_of_point_labels = np.zeros(num_points)
        type_of_point_labels[-NUM_TRACK_POINTS*max_num_tracks:] = processed_event_track_data["track_classes"][idx]

        event_point_data = np.concatenate(([np.concatenate((processed_event_data[file_feature][idx], processed_event_track_data[track_features[i]][idx]), -1)  for i, file_feature in enumerate(file_features)], [type_of_point_labels]), 0)


        point_data[idx, :num_points] = np.transpose(event_point_data)
        point_label[idx, :num_points] = np.transpose([np.concatenate((processed_event_data["cell_regression_deposit_labels"][idx], np.full(NUM_TRACK_POINTS*max_num_tracks, -1)))]) # label all tracks as -1

    file_path = save_dir + dataset + "_processed_train_files/" + file_name.split(".")[0]
    if(True): # TODO: update
        np.savez(file_path, X=point_data, Y=point_label)#, cell_weights=processed_event_data["cell_weights"])
    else:    
        np.savez(file_path, X=point_data, Y=point_label)

    max_points_queue.put(max_cells)
    print("SAVED file", file_path)

    print("num events", num_events_saved)


if __name__ == "__main__":
    t_start = time.time() # time the whole data processing process

    # get arguments from yaml file
    args = get_args()
    config_file = args.config
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    events_root_file = config["events_root_file"]
    preprocessed = config["preprocessed"]
    dataset = config["dataset"]
    file_len = config["file_len"]
    starting_event_idx = config["starting_event_idx"]
    num_files_to_process = config["num_files_to_process"]
    save_dir = config["save_dir"]

    preprocessed_file_name = config["preprocessed_file_name"]
    i_low = config["i_low"]
    i_high = config["i_high"]
    num_procs = config["num_procs"]

    include_delta_p_pi0 = config["include_delta_p_pi0"]
    include_delta_n_pi0 = config["include_delta_n_pi0"]
    include_delta_p_pipm = config["include_delta_p_pipm"]
    include_delta_n_pipm = config["include_delta_n_pipm"]

    niche_case = config["niche_case"]

    regression = config["regression"]

    # load cell geo tree to look uo cells location
    cell_geo_tree_file = uproot.open("/data/atlas/data/rho_delta/rho_small.root")
    cell_geo_tree = cell_geo_tree_file["CellGeo"]
    node_feature_names = cell_geo_tree.keys()[1:7] # 'cell_geo_sampling', 'cell_geo_eta', 'cell_geo_phi', 'cell_geo_rPerp', 'cell_geo_deta', 'cell_geo_dphi'
    cell_geo_data = cell_geo_tree.arrays(library='np')
    cell_geo_ID = cell_geo_data['cell_geo_ID'][0]
    sorter = np.argsort(cell_geo_ID)

    max_points = 0
    file_name = events_root_file.split('/')[-1].split('.')[0]

    # set up multiprocessing pool
    manager = multiprocessing.Manager()
    max_points_queue = manager.Queue()
    pool = multiprocessing.Pool(num_procs)

    # start a multiprocessing pool to convert each file on its own process
    starting_event_idxs = [file_len*batch_idx + starting_event_idx for batch_idx in range(num_files_to_process)]
    preprocessed_file_names = list(map(lambda i:  preprocessed_file_name + "_len_" + str(file_len) + "_i_" + str(i) + ".npz", np.arange(i_low, i_high + 1)))
    features_of_interest = ["cluster_cell_E", "cluster_cell_hitsE_EM", "cluster_cell_hitsE_nonEM", "cluster_cell_hitsTruthE", "cluster_cell_hitsTruthIndex", "cluster_cell_ID",  "truthPartPhi", "truthPartEta", "truthPartPt"]
    print("Starting processing...")
    if preprocessed:
        event_root_data = "null"
        event_start_idx = "null"
        pool.map(process_events, [(event_root_data, preprocessed, preprocessed_file_name, event_start_idx, file_len, features_of_interest, dataset, file_name, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, include_delta_p_pi0, include_delta_n_pi0, include_delta_p_pipm, include_delta_n_pipm, niche_case, regression) for preprocessed_file_name in preprocessed_file_names])
    else:
        event_root_data = uproot.open(events_root_file)["EventTree"]
        pool.map(process_events, [(event_root_data, preprocessed, preprocessed_file_name, event_start_idx, file_len, features_of_interest, dataset, file_name, save_dir, node_feature_names, cell_geo_data, sorter, max_points_queue, include_delta_p_pi0, include_delta_n_pi0, include_delta_p_pipm, include_delta_n_pipm, niche_case, regression) for event_start_idx in starting_event_idxs])

    # as files are processed and the max num points are added to queue on completion, compare with current max 
    while not max_points_queue.empty():
        q_max_points = max_points_queue.get()

        if q_max_points > max_points:
            max_points = q_max_points

    # update max points if a exceeded
    try:
        with open(save_dir + 'max_points.txt') as f:
            current_max_points = int(f.readline())
    except IOError:
        current_max_points = 0
    
    if max_points > current_max_points:
        with open(save_dir + 'max_points.txt', 'w') as f:
            f.write('%d' % max_points)

    print("elapsed time:",  time.time() - t_start)
    