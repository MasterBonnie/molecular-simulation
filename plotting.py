import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

""" File for plotting stuff """

def plot_csv_phase(file):
    data = np.genfromtxt(file, delimiter=',')

    plt.plot(data[:, 0], data[:,1])
    plt.show()

def plot_debugging(file):
    data = np.genfromtxt(file, delimiter = ",")
    time_step = np.linspace(1.5, 2, num=data.shape[0])

    plt.plot(time_step, data[:, 0])
    plt.plot(time_step, data[:, 1])
    plt.plot(time_step, data[:, 2])
    plt.legend(["potential", "kinetic", "total"], loc = "center right")
    plt.title("Energy of the system over time")
    plt.ylabel("E (Kj mol^-1)")
    plt.xlabel("T (ns)")
    plt.show()

    plt.plot(time_step, data[:, 3])
    plt.plot(time_step, data[:, 4])
    plt.plot(time_step, data[:, 5])
    #plt.plot(time_step, data[:, 6])
    plt.legend(["bonds", "angles", "dihedrals", "lj"], loc = (0.75,0.3))
    plt.title("Distribution over potential energy")
    plt.ylabel("E (Kj mol^-1)")
    plt.xlabel("T (ns)")
    plt.show()

    plt.plot(time_step, data[:, -1])
    #plt.legend(["Temperature"])
    plt.title("Temperature distribution")
    plt.ylabel("Temp (K)")
    plt.ylim(270, 320)
    plt.xlabel("T (ns)")
    plt.show()

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

    mpl.rcParams.update({'font.size': 22})
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
    'font.size' : 22,
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


setup_matplotlib()


if __name__ == "__main__":
    plot_debugging("output/test_r.csv")
    #plot_integrator_information("output/result_phase_mix_3nm_2.csv")