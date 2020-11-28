import matplotlib.pyplot as plt
import numpy as np

""" File for plotting stuff """

def plot_csv_phase(file):
    data = np.genfromtxt(file, delimiter=',')

    plt.plot(data[:, 0], data[:,1])
    plt.show()


if __name__ == "__main__":
    plot_csv_phase("output/result_phase.csv")
