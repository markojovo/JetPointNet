import argparse
import numpy as np
import yaml
import uproot
import math
import multiprocessing

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None)
    return parser.parse_args()

def process_some_events(args):
    rho_event_data, file_len, entry_start, batch_size, save_dir = args

    file_idx_start = int(entry_start / file_len)

    for rho_event_data_batch in rho_event_data.iterate(keys, step_size=batch_size, entry_start=entry_start, entry_stop=(entry_start + batch_size)):
            print("loaded rho event data batch", entry_start)
            for file_idx_relative in range(int(batch_size / file_len)):
                rho_data = {}
                for key in keys:
                    rho_data[key] = rho_event_data_batch[key][file_len*file_idx_relative:file_len*(file_idx_relative + 1)]

                np.save(save_dir + "/rho_full_len_" + str(file_len) + "_i_" + str(file_idx_start + file_idx_relative), rho_data, allow_pickle=True)



if __name__ == "__main__":
    args = get_args()

    # load config info
    config_file = args.config
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    rho_files = config["rho_files"]
    save_dir = config["save_dir"]
    num_procs = config["num_procs"]
    file_len = config["file_len"]

    keys = ["eventNumber", "nCluster", "truthPartE", "truthPartPt", "cluster_E", "cluster_E_LCCalib", "cluster_EM_PROBABILITY", 
        "cluster_E", "cluster_HAD_WEIGHT", "truthPartPdgId", "cluster_ENG_CALIB_TOT", "cluster_Eta", "cluster_cell_ID", "cluster_cell_E", "cluster_cell_ID", "cluster_cell_hitsE_EM", "cluster_cell_hitsE_nonEM", 
        "cluster_cell_E", "cluster_Pt", 'trackPhi_EMB1', 'trackPhi_EMB2', 'trackPhi_EMB3',
        'trackPhi_EME1', 'trackPhi_EME2', 'trackPhi_EME3', 'trackPhi_HEC0',
        'trackPhi_HEC1', 'trackPhi_HEC2', 'trackPhi_HEC3', 'trackPhi_TileBar0',
        'trackPhi_TileBar1', 'trackPhi_TileBar2', 'trackPhi_TileGap1',
        'trackPhi_TileGap2', 'trackPhi_TileGap3', 'trackPhi_TileExt0',
        'trackPhi_TileExt1', 'trackPhi_TileExt2', 'trackEta_EMB1', 'trackEta_EMB2', 'trackEta_EMB3',
        'trackEta_EME1', 'trackEta_EME2', 'trackEta_EME3', 'trackEta_HEC0',
        'trackEta_HEC1', 'trackEta_HEC2', 'trackEta_HEC3', 'trackEta_TileBar0',
        'trackEta_TileBar1', 'trackEta_TileBar2', 'trackEta_TileGap1',
        'trackEta_TileGap2', 'trackEta_TileGap3', 'trackEta_TileExt0',
        'trackEta_TileExt1', 'trackEta_TileExt2', "nTrack", 'trackMass', 'trackEta', 'trackPhi', 'trackPt', 'trackP',
        'cluster_cell_hitsTruthIndex', 'cluster_cell_hitsTruthE', 'truthPartE', 'truthPartPt', 'truthPartEta', 'truthPartPhi']
    

    for i, rho_file in enumerate(rho_files):
        # load event tree to numpy arrays
        rho_event_data_tree = uproot.open(rho_files[i])
        rho_event_data = rho_event_data_tree["EventTree"]

        num_events = len(rho_event_data["eventNumber"].array())
        batch_size = file_len*5
        num_jobs = 3*num_procs

        manager = multiprocessing.Manager()
        max_points_queue = manager.Queue()
        pool = multiprocessing.Pool(num_procs)

        result = pool.map(process_some_events, [(rho_event_data, file_len, entry_start_i*batch_size, batch_size, save_dir) for entry_start_i in range(num_jobs)])
        