import numpy as np
from scipy.spatial import distance_matrix

import helper

"""File for extracting/calculating information about a state at a particular time step"""

def distance(pos):
    """
    Calculates the distance between each atom, which it returns as an matrix.
    
    Input: 
        A matrix whoses rows are the position of each atom.
    Ouput: 
        A matrix where M[i][j] = dist(atom_i,atom_j)
    """
    # NOTE: OLD
    # size = np.shape(pos)[0]
    # dis = np.zeros([size,size])
    
    # # We loop over the matrix, but since it is symmetric and
    # # the diagonal is 0 we can skip certain parts of the full
    # # loop
    # for i in range(size):
    #     for j in range(i, size):
    #         if i - j:
    #             dis[i][j] = np.linalg.norm(pos[i] - pos[j])
    #             dis[j][i] = np.linalg.norm(pos[i] - pos[j])
    # return dis


    # Very much more quicker than the above method
    # NOTE: diff should probably be calculated somewhere else later on
    diff = pos - pos[:, np.newaxis]
    dis = np.linalg.norm(diff, axis=2)
    return dis
            
def force_bond(diff, dis, k, r_0):
    """
    Calculates the force resulting from bond potential
    
    Input:
        diff: vector pointing in the direction of the force 
        dis: distance between atoms in the bond
        k: constant specific to each bond
        r_0: constant specific to each bond
    Output: 
        force: vector representing the force
    """
    
    magnitude_force = -k*(dis - r_0)
    force = magnitude_force*helper.unit_vector(diff)

    return force


def force_angular(direction, theta, dis, k, theta_0):
    """
    Calculates the force resulting from angular potential

    Input:
        direction: vector pointing in the direction of the force
        theta: angle of the molecules
        dis: length of the arm
        k: constant specific to the angle
        theta_0: constant specific to the angle
    Output:
        force: vector representing the force
    """

    magnitude_force = -k*(theta - theta_0)/dis
    force = magnitude_force* helper.unit_vector(direction)

    return force
