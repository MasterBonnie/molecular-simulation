import matplotlib.pyplot as plt
import numpy as np

""" File for plotting stuff """

def plot_csv_phase(file):
    data = np.genfromtxt(file, delimiter=',')

    plt.plot(data[:, 0], data[:,1])
    plt.show()

def plot_debugging(file):
    data = np.genfromtxt(file, delimiter = ",")
    plt.plot(data[:, 0])
    plt.plot(data[:, 1])
    plt.plot(data[:, 2])
    plt.legend(["potential", "kinetic", "total"])
    plt.title("Energy of the system over time")
    plt.ylabel("E (Kj mol^-1)")
    plt.xlabel("T (10^-13 s)")
    plt.show()

    plt.plot(data[:, 3])
    plt.plot(data[:, 4])
    plt.plot(data[:, 5])
    plt.legend(["bonds", "angles", "dihedrals"])
    plt.title("Distribution over potential energy")
    plt.ylabel("E (Kj mol^-1)")
    plt.xlabel("T (10^-13 s)")
    plt.show()

    plt.plot(data[:, 6])
    plt.legend(["Temperature"])
    plt.title("Temperature distribution")
    plt.ylabel("K (K)")
    plt.xlabel("T (10^-13 s)")
    plt.show()

if __name__ == "__main__":
    plot_debugging("output/result_phase_5nm.csv")
