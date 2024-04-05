#==========================================================================================
# A collection of useful things that I've been using for creating deep set arrays as well as
# some tools for training the networks
# Author: R. Bate - russellbate@phas.ubc.ca
#===========================================================================================

import numpy as np
import awkward as ak

track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

ak_event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E",
                    "truthPartPdgId", "cluster_Eta", "cluster_Phi", "trackPt", "trackP", "trackEta",
                    "cluster_cell_hitsE_EM", "cluster_cell_hitsE_nonEM", 'truthPartE',
                    'cluster_ENG_CALIB_TOT', "cluster_E"]

np_event_branches = ["nCluster", "eventNumber", "nTrack", "nTruthPart"]

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]

cell_meta = {
    'EMB1': {
        'cell_size_phi': 0.098,
        'cell_size_eta': 0.0031,
        'len_phi': 4,
        'len_eta': 128
    },
    'EMB2': {
        'cell_size_phi': 0.0245,
        'cell_size_eta': 0.025,
        'len_phi': 16,
        'len_eta': 16
    },
    'EMB3': {
        'cell_size_phi': 0.0245,
        'cell_size_eta': 0.05,
        'len_phi': 16,
        'len_eta': 8
    },
    'TileBar0': {
        'cell_size_phi': 0.1,
        'cell_size_eta': 0.1,
        'len_phi': 4,
        'len_eta': 4
    },
    'TileBar1': {
        'cell_size_phi': 0.1,
        'cell_size_eta': 0.1,
        'len_phi': 4,
        'len_eta': 4
    },
    'TileBar2': {
        'cell_size_phi': 0.1,
        'cell_size_eta': 0.2,
        'len_phi': 4,
        'len_eta': 2
    },
}

def tvt_num(data, tvt=(75, 10, 15)):
    ''' Function designed to output appropriate numbers for traning validation and testing given
    a variable length input. TVT expressed as ratios and do not need to add to 100. '''
    tot = len(data)
    train, val, test = tvt
    tvt_sum = train + val + test
    
    train_rtrn = round(train*tot/tvt_sum)
    val_rtrn = round(val*tot/tvt_sum)
    test_rtrn = tot - train_rtrn - val_rtrn
    
    return train_rtrn, val_rtrn, test_rtrn

def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref.
    speed this up to accept 1x2 and 1x2 to bypass the tile function. '''
    
    DeltaCoords = np.subtract(coords, ref)
    
    ## Mirroring ##
    gt_pi_mask = DeltaCoords > np.pi
    lt_pi_mask = DeltaCoords < - np.pi
    DeltaCoords[lt_pi_mask] = DeltaCoords[lt_pi_mask] + 2*np.pi
    DeltaCoords[gt_pi_mask] = DeltaCoords[gt_pi_mask] - 2*np.pi
    
    rank = DeltaCoords.ndim
    retVal = None
    
    if rank == 1:
        retVal = np.sqrt(DeltaCoords[0]**2 + DeltaCoords[1]**2)
    elif rank == 2:
        retVal = np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2)
    else:
        raise ValueError('Too many dimensions for DeltaR')
    return retVal

def dict_from_tree(tree, branches=None, np_branches=None):
    ''' Loads branches as default awkward arrays and np_branches as numpy arrays.
    Note need to modify this to pass either branches or np branches and still work '''
    ak_dict = dict()
    if branches is not None:
        ak_arrays = tree.arrays(filter_name=branches)[branches]
        for branch in branches:
            ak_dict[branch] = ak_arrays[branch]
    
    np_dict = dict()
    if np_branches is not None:
        np_arrays = tree.arrays(filter_name=np_branches)
        for np_key in np_branches:
            np_dict[np_key] = ak.to_numpy( np_arrays[np_key] ).flatten()
    
    if branches is not None and np_branches is not None:
        dictionary = {**np_dict, **ak_dict}
    
    elif branches is not None:
        dictionary = reg_dict
    
    elif np_branches is not None:
        dictionary = np_dict
        
    else:
        raise ValueError("No branches passed to function.")
        
    return dictionary

def find_max_calo_cells(events, event_dict):
    ''' This function is designed to work as the find_max_dim_tuple except we are negating tracks here '''
    return None
    
def find_max_dim_tuple(events, event_dict):
    ''' This function is designed to return the sizes of a numpy array such that we are efficient
    with zero padding. Please feel free to write this faster, it can be done. Notes: we add six
    to the maximum cluster number such that we have room for track info.
    Inputs:
        _events: filtered list of events to choose from in an Nx3 format for event, track, cluster index 
        _event_tree: the event tree dictionary
    Returns:
        3-tuple consisting of (number of events, maximum cluster_size, 5), 5 because of how we have structured
        the X data format in energyFlow to be Energy, Eta, Phi, rPerp, track flag, sampling layer
        _cluster_ENG_CALIB_TOT, turthPartE '''    
    nEvents = len(events)
    max_clust = 0
    
    for i in range(nEvents):
        event = events[i,0]
        track_nums = events[i,1]
        clust_nums = events[i,2]
        
        clust_num_total = 0
        # set this to six for now to handle single track events, change later
        track_num_total = 6
        
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                # add the number in each cluster to the total
                clust_num_total += nInClust

        total_size = clust_num_total + track_num_total
        if total_size > max_clust:
            max_clust = total_size
    
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    return (nEvents, max_clust, 6)

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec

def to_xyz(coords, mask=None):
    ''' Simple geometric conversion to xyz from eta, phi, rperp (READ: in this order)
    Inputs: np array of shape (N, 3) where columns are [eta, phi, rPerp]
    Outputs: np array of shape (N, 3) where columns are [x,y,z] '''
    if len(coords.shape) == 2:
        eta = coords[:,0]
        phi = coords[:,1]
        rperp = coords[:,2]
        theta = 2*np.arctan( np.exp(-eta) )
    
        cell_x = rperp*np.cos(phi)
        cell_y = rperp*np.sin(phi)
        cell_z = rperp/np.tan(theta)
    
        return np.column_stack([cell_x,cell_y,cell_z])

    elif len(coords.shape) == 3:
        
        eta = coords[:,:,0]
        phi = coords[:,:,1]
        rperp = coords[:,:,2]
        shape = eta.shape
        
        if mask is None:
            mask = eta == 0
            mask = np.logical_and(mask, phi == 0)
            mask = np.invert(np.logical_and(mask, rPerp == 0))
        
        theta = 2*np.arctan( np.exp(-eta[mask]) )
        
        cell_x = np.zeros(shape)
        cell_y = np.zeros(shape)
        cell_z = np.zeros(shape)

        cell_x[mask] = rperp[mask]*np.cos(phi[mask])
        cell_y[mask] = rperp[mask]*np.sin(phi[mask])
        cell_z[mask] = rperp[mask]/np.tan(theta)
        
        return np.stack((cell_x,cell_y,cell_z), axis=2)
    
    else:
        raise ValueError('Unsupported array type in to_xyz()')
        
def weighted_sum(variable, weights):
    ''' Discrete first moment. '''
    Ans = np.dot(weights, variable)
    Ans = Ans / np.sum(weights)
    return Ans
    
def find_centroid(coords_3d, targets):
    ''' Designed to find the energy weighted centroid of all the cells in the cluster.
    Inputs:
        coords_3d: raw x, y, z without masking 
        target: energies in order of EM, nonEM'''
    
    xmask = coords_3d[:,0] != 0
    ymask = coords_3d[:,1] != 0
    mask = np.logical_or(xmask, ymask)

    EM_weights = targets[mask,0]
    nonEM_weights = targets[mask,1]
    
    EM_centr = weighted_sum(variable=coords_3d[mask,:3], weights=EM_weights)
    nonEM_centr = weighted_sum(variable=coords_3d[mask,:3], weights=nonEM_weights)

    return EM_centr, nonEM_centr


def create_image(cell_size_eta, cell_size_phi, nEta, nPhi, layer_eta, layer_phi, layer_E, mask,
                 eta_shift=0, phi_shift=0, logscale=True):
    ''' Takes layer and cluster information to produce a pixel image using binning.
    Inputs: 
        cell_size_eta/phi: size of each calo cell for that layer
        nEta/nPhi: discrete number of points which to include
        later_eta/phi/E: the input values of the cluster
        mask: which values to choose from
        eta/phi_shift: artificially move the cluster in the eta phi plane
        logscale: whether or not to take the log of the energy values
    Outputs:
        (n,m) array with energy values filled in the appropriate bins, otherwise np.nan
    '''
    
    eta_span = cell_size_eta * nEta
    phi_span = cell_size_phi * nPhi
    eta_start = - int(nEta/2) * cell_size_eta
    phi_start = - int(nPhi/2) * cell_size_phi
    
    # fill with np.nan so hopefully imshow wont plot this?
    image = np.full((nEta, nPhi), np.nan)
    point_out = 0
    
    for i in range(np.count_nonzero(mask)):
        
        N = int(np.floor( ( (layer_eta[i] - eta_start)/eta_span) * nEta ))
        M = int(np.floor( ( (layer_phi[i] - phi_start)/phi_span) * nPhi ))
        
        inbounds = True
        if N < 0 or M < 0:
            inbounds = False
        if N >= nEta or M >= nPhi:
            inbounds = False
            
        if inbounds == True:
            if logscale:
                image[N,M] = np.log(layer_E[i])
            else:
                image[N,M] = layer_E[i]
        else:
            point_out += 1
            
    print(str(point_out)+' points out of bounds')
    return image
