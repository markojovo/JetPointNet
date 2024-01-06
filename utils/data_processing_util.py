import vector
import numpy as np
import argparse



def spherical_to_cartesian(rs, phis, thetas):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.cos(phis[i]))
        y.append(r*np.sin(phis[i]))
        z.append( r/np.tan(thetas[i]))

    return np.array(x), np.array(y), np.array(z)

def flatten_clus_in_events(events):
    return [[cell for clus in event for cell in clus] for event in events]

def measure_track_part_dists(track_phi, track_eta, track_p, part_phi, part_eta, part_p):
    part_vectors = vector.obj(eta=part_eta, phi=part_phi, rho=part_p)
    track_vectors = vector.obj(eta=track_eta, phi=track_phi, rho=track_p)

    return np.array(part_vectors.deltaR(track_vectors))

def flatten_one_layer(data):
    return [data_layer2 for data_layer1 in data for data_layer2 in data_layer1]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, type=str)
    return parser.parse_args()

features_of_interest = ["eventNumber", "nCluster", "truthPartE", "truthPartPt", "cluster_E", "cluster_E_LCCalib", "cluster_EM_PROBABILITY", 
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