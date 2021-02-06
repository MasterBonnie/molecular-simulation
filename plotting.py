import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from io_sim import radial_distribution_function

""" File for plotting stuff """

legend_size = 18
axes_label_size = 17

def plot_csv_phase(file):
    data = np.genfromtxt(file, delimiter=',')

    plt.plot(data[:, 0], data[:,1])
    plt.show()

def plot_debugging(file):
    data = np.genfromtxt(file, delimiter = ",")
    time_step = np.linspace(3.5, 4, num=data.shape[0])

    fig, ax = plt.subplots(figsize = (11, 9))

    ax.plot(time_step, data[:, 0])
    ax.plot(time_step, data[:, 1])
    ax.plot(time_step, data[:, 2])
    plt.legend(["potential", "kinetic", "total"], loc = "center right", prop={'size': legend_size})
    plt.title("Energy of the system over time")
    plt.ylabel(r"$E \; \;(Kj \; mol^{-1}) $")
    plt.xlabel(r"$T \; \;(ns)$")

    ax.tick_params(axis='both', which='major', labelsize=axes_label_size)

    plt.show()

    fig, ax = plt.subplots(figsize = (11, 9))

    ax.plot(time_step, data[:, 3])
    ax.plot(time_step, data[:, 4])
    ax.plot(time_step, data[:, 5])
    plt.legend(["bonds", "angles", "dihedrals", "lj"], loc = (0.75,0.3), prop={'size': legend_size})
    plt.title("Distribution over potential energy")
    plt.ylabel(r"$E \; \;(Kj \; mol^{-1}) $")
    plt.xlabel(r"$T \; \;(ns)$")
    plt.show()


    fig, ax = plt.subplots(figsize = (11, 9))


    ax.plot(time_step, data[:, -1])
    #plt.legend(["Temperature"])
    plt.title("Temperature distribution")
    plt.ylabel(r"$Temp \;\; (K)$")
    plt.ylim(296, 300)
    plt.xlabel(r"$T \;\; (ns)$")

    ax.tick_params(axis='both', which='major', labelsize=axes_label_size)

    plt.show()

    # Instead of plotting the temperature, we compute mean and standard deviation:

    temperature = data[:, -1]
    mean_temperature = np.mean(temperature)
    std_temperature = np.std(temperature)

    print(f"Mean temperature: {mean_temperature}")
    print(f"Std of temperature: {std_temperature}")

def plot_integrator_information(file):
    data = np.genfromtxt(file, delimiter = ",")

    plt.plot(data[:,0], data[:,1])
    plt.title("Potential versus Kinetic energy")
    plt.show()


def setup_matplotlib():
    """
    For the visual style of the matplotlib plots, call this function
    """
    # Remove the frame from the legend
    #color_list = ["#1AE0B1", "#740D0E", "#3F3B24"]

    sns.set(rc={
    'axes.axisbelow': False,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',
    'axes.grid': False,
    'axes.labelcolor': 'dimgrey',
    'axes.spines.right': False,
    'axes.spines.top': False,
    #'axes.prop_cycle' : plt.cycler(color=color_list),
    'figure.facecolor': 'white',
    "font.size": 22,
    "axes.labelsize": 17,
    "axes.titlesize": 26,
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',
    'xtick.bottom': False,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.left': False,
    'ytick.right': False})


def make_rdf_plot(xyz_file, top_file, output_file, first_molecule, second_molecule, dr, box_size):
    
    fig, ax = plt.subplots(figsize = (11, 9))


    second_atom = "H"

    (x,y) = radial_distribution_function(xyz_file, top_file, output_file, dr, box_size, 
                           first_molecule, second_molecule, second_atom)

    ax.plot(x,y, linewidth=2)

    second_atom = "O"
    (x,y) = radial_distribution_function(xyz_file, top_file, output_file, dr, box_size, 
                           first_molecule, second_molecule, second_atom)

    ax.plot(x,y, linewidth=2)
    plt.title(f"{first_molecule}-{second_molecule} radial distribution function")
    plt.xlabel(r"$r \;\; (\mathrm{\AA})$")
    plt.legend(["OH", "OO"], prop={'size': legend_size})

    ax.tick_params(axis='both', which='major', labelsize=axes_label_size)

    plt.show()


# If we plot somewhere else, importing this file sets the correct 
# visual style
setup_matplotlib()


if __name__ == "__main__":
    plot_debugging("output/ethanol_3nm_redo_2.csv")
    #plot_integrator_information("output/result_phase_mix_3nm_2.csv")


    xyz_file = "output/ethanol_3nm_redo_2.xyz"
    top_file = "data/ethanol_3nm.itp"
    output_file = "output/rdf/rdf_mix_water_water_h.csv"

    first_molecule = "ethanol"
    second_molecule = "ethanol"

    box_size = 30
    dr = 0.05

    #make_rdf_plot(xyz_file, top_file, output_file, first_molecule, second_molecule, dr, box_size)
