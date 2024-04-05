import argparse
import numpy as np
import yaml
import uproot

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # load config info
    config_file = args.config
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    file_nums = config["file_nums"]
    save_dir = config["save_dir"]
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
        'trackEta_TileExt1', 'trackEta_TileExt2', "nTrack", 'trackMass', 'trackEta', 'trackPhi', 'trackPt', 'trackP']
    
    len_file = 6000


    for file_num in file_nums:
        file_name = "/data/atlas/data/allCellTruthv1/pipm/user.mswiatlo.27153452.OutputStream._0000" + str(file_num) + ".root"
        # load event tree to numpy arrays
        event_data = uproot.open(file_name + ":EventTree").arrays(library="np")
        print("loaded file")

        num_events = len(event_data["eventNumber"])
        
        file_count = 0
        while (file_count + 1)*len_file <= num_events:
            pion_data = {}
            for key in keys:
                pion_data[key] = np.zeros(len_file, dtype=object)
                pion_data[key] = event_data[key][file_count*len_file:(file_count+1)*len_file]

            np.save(save_dir + "/pipm_" + str(file_num) + "_len_" + str(len_file) + "_i_" + str(file_count), pion_data, allow_pickle=True)
            file_count += 1


