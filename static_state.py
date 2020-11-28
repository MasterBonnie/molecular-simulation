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

def compute_force(pos, v, k, r_0):
    """
    Computes the force on a linear model
    """

    diff = pos - pos[:, np.newaxis]
    dis = np.linalg.norm(diff, axis=2)

    force = force_bond(diff[0][1], dis[0][1], k, r_0)

    return np.array([-force, force])

def compute_force_h2o(pos):
    """
    Computes the force on H2O molecule

    Hard coded for reasons I cannot explain
    """
    # Calculate the force on each molecule
    k_b = 5024.16    # Kj mol^-1 A^-2
    r_0 = 0.9572   # A

    theta_0 = 104.52 *(np.pi/180)   # Radians
    k_ang = 628.02  # Kj mol^-1 rad^-2

    diff = pos - pos[:, np.newaxis]
    dis = np.linalg.norm(diff, axis=2)

    theta = helper.angle_between(diff[0][1], diff[0][2])
    #print(theta*180/np.pi)

    # Create the unit vector along which the angular force acts
    # This is the right molecule, and so we need this -1 here
    angular_force_unit_1 = helper.unit_vector(-np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][1]))

    # Calculate the angular and bond force on Hydrogen atom 1
    angular_force_1 = force_angular(angular_force_unit_1,
                                                    theta,
                                                    dis[0][1],
                                                    k_ang,
                                                    theta_0)
    bond_force_1 = force_bond(diff[0][1], dis[0][1], k_b, r_0)

    # Total force on hydrogen atom 1
    force_hydrogen_1 = (angular_force_1 + bond_force_1)

    # Again create unit vector for angular force
    # This one already points in the right direction
    angular_force_unit_2 = helper.unit_vector(np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][2]))

    # Angular force
    angular_force_2 = force_angular(angular_force_unit_2,
                                                    theta,
                                                    dis[0][2],
                                                    k_ang,
                                                    theta_0)
    bond_force_2 = force_bond(diff[0][2], dis[0][2], k_b, r_0)

    # Total force on Hydrogen atom 2
    force_hydrogen_2 = (angular_force_2 + bond_force_2)
    force_oxygen = -(force_hydrogen_1 + force_hydrogen_2)

    return np.array([force_oxygen, force_hydrogen_1, force_hydrogen_2])
