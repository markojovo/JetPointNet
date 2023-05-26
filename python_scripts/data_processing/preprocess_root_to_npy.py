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
            for key in event_data:
                pion_data[key] = np.zeros(len_file, dtype=object)
                pion_data[key] = event_data[key][file_count*len_file:(file_count+1)*len_file]

            np.save(save_dir + "/pipm_" + str(file_num) + "_len_" + str(len_file) + "_i_" + str(file_count), pion_data, allow_pickle=True)
            file_count += 1


