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
    plt.show()

if __name__ == "__main__":
    plot_debugging("output/result_phase.csv")
