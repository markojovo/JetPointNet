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


pastel_colors = [
    (0.6784, 0.8471, 0.9020),  # Lightblue
    (1.0000, 0.7137, 0.7569),  # Lightpink
    (0.8196, 0.7686, 0.9412),  # Lavender
    
    (0.5647, 0.9333, 0.5647),  # Lightgreen
    (1.0000, 0.9804, 0.8039),  # Lightyellow
    
    (1.0000, 0.6275, 0.4784),  # Lightsalmon
    (0.8471, 0.7490, 0.8471),  # Thistle
    (1.0000, 0.8941, 0.8824),  # Mistyrose
    (1.0000, 0.8549, 0.7255)   # Peachpuff
]


def plot_proccessed_cells_in_line(x, y, z, cell_data, title, xlabel, ylabel, zlabel, colour_theme, bar_label, event_idx=None, vmin=None, vmax=None, same_axis=True, add_track=False, add_topo=False):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(5*len(cell_data), 5)
    for i, cell_data_i in enumerate(cell_data):
        ax = fig.add_subplot(1, len(cell_data), i + 1, projection='3d')
        x_i = x[i] if len(x) > 1 else x[0]
        y_i = y[i] if len(y) > 1 else y[0]
        z_i = z[i] if len(z) > 1 else z[0]
        xlabel_i = xlabel[i] if len(xlabel) > 1 else xlabel[0]
        ylabel_i = ylabel[i] if len(ylabel) > 1 else ylabel[0]
        zlabel_i = zlabel[i] if len(zlabel) > 1 else zlabel[0]
        title_i = title[i] if len(title) > 1 else title[0]

        if title[i] == "Clustered cells":
            for clus_idx in range(2):#range(0, np.max(processed_event_data["clus_idx"][event_idx]) + 1):
                ax.scatter(processed_event_data["z"][event_idx][processed_event_data["clus_idx"][event_idx] == clus_idx], processed_event_data["x"][event_idx][processed_event_data["clus_idx"][event_idx] == clus_idx], processed_event_data["y"][event_idx][processed_event_data["clus_idx"][event_idx] == clus_idx], label=["pi class", "pi/pi class"][clus_idx], color=pastel_colors[clus_idx])
            ax.legend()
            
        elif bar_label[i] == "Pi+/- frac class" or bar_label[i] == "Pred EM frac class":
            cut_em_class = np.array(cell_data_i) == 1
            clus_x = np.array(x_i)
            clus_y = np.array(y_i)
            clus_z = np.array(z_i)
            ax.scatter(clus_z[cut_em_class], clus_x[cut_em_class], clus_y[cut_em_class], color="fuchsia")
            ax.scatter(clus_z[~cut_em_class], clus_x[~cut_em_class], clus_y[~cut_em_class], color="cyan")  
        #elif title[i] == "Cluster EM probability":
        #    print("not loaded atm")
        elif title[i] == "Input point data":
            p = ax.scatter(z_i, x_i, y_i, c=cell_data_i, vmin=vmin[i], vmax=vmax[i], cmap=colour_theme[i], zorder=1, norm=mpl.colors.LogNorm())
            #if bar_label[i] != "EM frac class":
            #fig.colorbar(p, label=bar_label[i], fraction=0.025, pad=0.15)
        elif title[i] == "Cell classes": # 0 - track of interest, 1 - other charged particles, 2 - pions, 3 - other neutral hadrons
            for class_i in range(4):
                cut_class = np.array(cell_data_i) == class_i
                ax.scatter(np.array(z_i)[cut_class], np.array(x_i)[cut_class], np.array(y_i)[cut_class], color=class_colour[class_i], label=class_labels[class_i])
            ax.legend()
        else:
            p = ax.scatter(z_i, x_i, y_i, c=cell_data_i, vmin=vmin[i], vmax=vmax[i], cmap=colour_theme[i], zorder=1)
            #if bar_label[i] != "EM frac class":
            #fig.colorbar(p, label=bar_label[i], fraction=0.025, pad=0.15)
        
        if add_track: #and (title[i] == "Input point data" or title[i] == "Input point data" or title[i] == "Fraction of cell energy from pi" or title[i] == "Cell truth classes"):
            #print(event_idx)
            for track_idx, track_classes in enumerate(processed_event_track_data["track_classes"][event_idx]):
                is_track_point = track_classes != 0
                is_track_point = np.array(is_track_point)
                
                if np.count_nonzero(is_track_point):
                    #print(np.array(track_classes)[is_track_point])
                    colour = "black" if np.array(track_classes)[is_track_point][0] == 1 else "mediumorchid"
                    ax.scatter(np.array(processed_event_track_data["z"][event_idx][track_idx])[is_track_point], np.array(processed_event_track_data["x"][event_idx][track_idx])[is_track_point], np.array(processed_event_track_data["y"][event_idx][track_idx])[is_track_point], color=colour)
                    ax.plot(np.array(processed_event_track_data["z"][event_idx][track_idx])[is_track_point], np.array(processed_event_track_data["x"][event_idx][track_idx])[is_track_point], np.array(processed_event_track_data["y"][event_idx][track_idx])[is_track_point], color=colour)

            ax.set_xlim(np.min(z_i), np.max(z_i))
            ax.set_ylim(np.min(x_i), np.max(x_i))
            ax.set_zlim(np.min(y_i), np.max(y_i))

        if add_topo and title[i] == "Predicted classes (+ clusters)":
            for clus_idx in range(0, np.max(processed_event_data["clus_idx"][event_idx]) + 1):
                ax.scatter(processed_event_data["z"][event_idx][processed_event_data["clus_idx"][event_idx] == clus_idx], processed_event_data["x"][event_idx][processed_event_data["clus_idx"][event_idx] == clus_idx], processed_event_data["y"][event_idx][processed_event_data["clus_idx"][event_idx] == clus_idx], marker=",", s=90, alpha=0.2, zorder=-1, label="cluster " + str(clus_idx + 1), color=pastel_colors[clus_idx])
            ax.legend()
        ax.set_xlabel(xlabel_i)
        ax.set_ylabel(ylabel_i)
        ax.set_zlabel(zlabel_i)
        plt.title(title_i)
    plt.show()


# plot 3d event info
def plot_event_3d_info(event_idx):#, pipm_frac, class_pipm_frac, preds):
    x = processed_event_data["x"][event_idx]
    y = processed_event_data["y"][event_idx] 
    z = processed_event_data["z"][event_idx]
    
    title = ["Sampling layers", "Input point data"]#, "Fraction of cell energy from pi", "Clustered cells", "Cell truth classes", "Cluster EM probability", "Predicted classes", "Predicted classes (+ clusters)"]
    xlabel = ["z"]
    ylabel = ["x"]
    zlabel = ["y"]

    colour_theme = [plt.cm.tab20b, plt.cm.magma]#, plt.cm.viridis_r, None, plt.cm.viridis_r, plt.cm.viridis_r, plt.cm.viridis_r, plt.cm.viridis_r]
    bar_label = ["Sampling layers", "Cell energy (GeV)"]#, "pi energy frac", "", "Truth Pi0 frac class",  "EM probability", "Predicted classes", "Predicted classes"]
    vmin = [0, None , 0, None, 0, 0, 0, 0]
    vmax = [23, None, 1, None, 1, 1, 1, 1]
    cell_data = [processed_event_data["sampling_layer"][event_idx],  10**(np.array(processed_event_data["cell_E"][event_idx]) + LOG_ENERGY_MEAN)]#, pipm_frac, None, class_pipm_frac, processed_event_data["clus_em_prob"][event_idx], preds, preds]
    #cell_data = [processed_event_data["sampling_layer"][event_idx],  processed_event_data["cell_E"][event_idx], pipm_frac, None, class_pipm_frac, None, preds, preds]
    
    #plot_proccessed_cells_in_line([x], [y], [z], cell_data[:4], title[:4], xlabel, ylabel, zlabel, colour_theme[:4], bar_label[:4], event_idx, vmin[:4], vmax[:4], add_track=True, add_topo=True)
    #plot_proccessed_cells_in_line([x], [y], [z], cell_data[4:], title[4:], xlabel, ylabel, zlabel, colour_theme[4:], bar_label[4:], event_idx, vmin[4:], vmax[4:], add_track=True, add_topo=True)
    for i in range(len(title)):
        plot_proccessed_cells_in_line([x], [y], [z], [cell_data[i]], [title[i]], xlabel, ylabel, zlabel, [colour_theme[i]], [bar_label[i]], event_idx, [vmin[i]], [vmax[i]], add_track=True, add_topo=True)
        