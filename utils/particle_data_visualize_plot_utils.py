import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append('/home/jbohm/start_tf/LCStudies')
from  util import graph_util as gu


def plot_hist(data, bins, title, x_label, y_label, normalize=True, colour=None, xlim=None, discrete=False):
    counts, bins = np.histogram(data, bins, (min(data), max(data) + (1 if discrete == True else 0)))
    # normalize counts
    if normalize:
        counts = counts / np.sum(counts)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.hist(bins[:-1], bins, weights=counts, color=colour)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlim)
    plt.show()

def plot_overlay_hist(data, bins, title, x_label, y_label, labels, colours=None, xlim=None, legend_loc=None):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    if xlim == None:
        xlim = (np.min(data, axis=None), np.max(data, axis=None))
    for i, data in enumerate(data):
        counts, bins = np.histogram(data, bins, xlim)
        # normalize counts
        counts = counts / np.sum(counts)
        plt.stairs(counts, bins, label=labels[i], color=(colours[i] if colours != None else None))
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)
    plt.xlim()
    plt.show()

# dicts for plotting particle data
particle_id_to_name = {213: "rho+", -213: "rho-", 22: "y", 211: "pi+", -211: "pi-", 111: "pi0"}
quantity_to_label = {"Pt": "Transverse momentum (GeV)", "E": "Energy (GeV)", "Mass": "Mass (GeV)"}
particle_id_to_colour = {213: 'r', -213: 'pink', 22: "yellow", 211: "b", -211: "lightblue", 111: "g"}

# plots histograms of passed quantities for all particles of pdg ids in events
# accepted qunatities: Pt, E, Energy, Mass
def plot_truth_data(events, quantities, pdg_ids, num_bins):
    truth_quantities_key = ["truthPart" + truth_quantity for truth_quantity in quantities]

    # extract arrays of desired truth quantities
    truth_data = events.arrays(truth_quantities_key)

    for particle_id in pdg_ids:
        particle_cut = (events["truthPartPdgId"].array() == particle_id)
        for quant in quantities:
            hist_data = ak.flatten(truth_data["truthPart"+quant, particle_cut])

            # plot particle-quantity data as histogram
            if ak.num(hist_data, axis=0) == 0:
                print("no "+ quant + " data for particle with pdg id " + str(particle_id))
                continue
                
            particle_name = particle_id_to_name[particle_id] if particle_id_to_name.get(particle_id) else particle_id
            quantity_label = quantity_to_label[quant] if quantity_to_label.get(quant) else quant
            plot_hist(hist_data, num_bins, particle_name, quantity_label, 'Frequency')


# plot truth particle trajectories for list of events
def plot_particle_trajectories(truth_traj_data, event_nums, quantity):
    for event in event_nums:  
        event_data = truth_traj_data[truth_traj_data.eventNumber == event][0]
        #particle_colours = [particle_id_to_colour[id] if particle_id_to_colour.get(id) else 'black' for id in event_data["truthPartPdgId"]]
        particle_names = [particle_id_to_name[id] if particle_id_to_name.get(id) else id for id in event_data["truthPartPdgId"]]
        quantity_label = quantity_to_label[quantity] if quantity_to_label.get(quantity) else quantity
        data_x = event_data.truthPartEta
        data_y = event_data.truthPartPhi

        space_x = (np.abs(np.max(data_x) - np.min(data_x)))/60
        space_y = (np.abs(np.max(data_y) - np.min(data_y)))/60

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        scatter = plt.scatter(data_x, data_y, c=event_data["truthPart"+quantity])

        for i, particle_label in enumerate(particle_names):
            #label = particle_label if not ak.any(event_data["truthPartPdgId"][:i] == event_data["truthPartPdgId"][i]) else None
            #scatter = plt.scatter(event_data.truthPartEta[i], event_data.truthPartPhi[i], c=particle_colours[i], label=label)
            plt.annotate(particle_label, (data_x[i] + space_x, data_y[i] + space_y))
        plt.title("Event " + str(event) + " particle trajectories")
        plt.xlabel("Eta")
        plt.ylabel("Phi")
        #plt.legend()
        plt.colorbar(label=quantity_label)
        plt.show()


# by generic definition of spherical to cartesian coord conversion
def spherical_to_cartesian(rs, thetas, phis):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.sin(phis[i])*np.cos(thetas[i]))
        y.append(r*np.sin(phis[i])*np.sin(thetas[i]))
        z.append(r*np.cos(phis[i]))

    return x,y,z

# by generic definition of spherical to cartesian coord conversion
def spherical_to_cartesian_cor(rs, phis, thetas):
    # convert to x, y, z
    x, y, z = [], [], []

    for i, r in enumerate(rs):
        x.append(r*np.cos(phis[i]))
        y.append(r*np.sin(phis[i]))
        z.append( r/np.tan(thetas[i]))

    return x,y,z

# plot cells in 3d space, from ids, with array of coresponding data as 4th dimension
def plot_cells(cell_ids, cell_dict, cell_data, title, xlabel, ylabel, zlabel, colour_theme, bar_label, vmin=None, vmax=None):
    phis = gu.convertIDToGeo(cell_ids, "cell_geo_phi", cell_dict)
    etas = gu.convertIDToGeo(cell_ids, "cell_geo_eta", cell_dict)
    rs = gu.convertIDToGeo(cell_ids, "cell_geo_rPerp", cell_dict)

    thetas = [2*np.arctan(np.exp(-eta)) for eta in etas]
    x, y, z = spherical_to_cartesian_cor(rs, phis, thetas)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')

    """
    # temp zoom to check for photons
    filter_x, filter_y, filter_z, filter_energy = [], [], [], []
    for i, z_i in enumerate(z):
        if z_i < 1050 or z_i > 1100:
            continue
        #if x[i] > 2000:
        #    continue
        filter_x.append(x[i])
        filter_y.append(y[i])
        filter_z.append(z[i])
        filter_energy.append(cell_data[i])
    """

    p = ax.scatter(z, x, y, c=cell_data, vmin=vmin, vmax=vmax, cmap=colour_theme)
    #p = ax.scatter(filter_z, filter_x, filter_y, c=filter_energy, cmap=colour_theme)
    fig.colorbar(p, label=bar_label, fraction=0.025)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.show()

def plot_proccessed_cells(x,y,z,cell_data, title, xlabel, ylabel, zlabel, colour_theme, bar_label, vmin=None, vmax=None):
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')

    p = ax.scatter(z, x, y, c=cell_data, vmin=vmin, vmax=vmax, cmap=colour_theme)
    fig.colorbar(p, label=bar_label, fraction=0.025)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)
    plt.show()

def plot_clusters(events, event_num, cell_dict, cluster_idx):
    event_idx = np.where(events["eventNumber"].array() == event_num)
    if not ak.any(event_idx):
        print("No event of index", str(event_num))
        return
    print(events["cluster_cell_E"].array()[event_idx][0])
    energies = ak.flatten(events["cluster_cell_E"].array()[event_idx][0][cluster_idx], axis=None)
    if not ak.any(energies):
        print("No clusters in event", str(event_num))
        return
    energies = [np.log10(energy) for energy in energies]
    cell_ids = ak.flatten(events["cluster_cell_ID"].array()[event_idx][0][cluster_idx], axis=None)
    plot_cells(cell_ids, cell_dict, energies, "Cluster " + str(event_num) + " energy", "z", "x", "y", plt.cm.magma, "log10(Energy) GeV")