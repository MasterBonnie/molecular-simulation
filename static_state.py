import numpy as np
from numba import vectorize, float64, jit, guvectorize
from helper import unit_vector, angle_between, distance_PBC


"""File for extracting/calculating information about a state at a particular time step"""

# TODO: rewrite this as numba function, using jit
# This might be difficult as numba does not like lists of lists, such as molecules
# otherwise rewrite as numpy function, without the loop
def centre_of_mass(pos, m, molecules):
    """
    Computes the centre of mass for each molecule

    Input:
        pos: array containing the positions
        m: array containing the mass
        molecules: list of lists specifying which molecule contains
                    which atom
    Output:
        centre_of_mass: array containg the centres of mass
    """
    centre_of_mass = np.zeros((len(molecules), 3))
    for i, molecule in enumerate(molecules):
        M = np.sum(m[molecule])
        Mpos = np.sum(m[molecule, np.newaxis]*pos[molecule], axis = 0)
        centre_of_mass[i] = Mpos/M

    return centre_of_mass

def compute_force(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, molecules, nr_atoms,
                    box_size):
    """
    Computes the force on each atom, given the position and information from a 
    topology file.

    Input:
        pos: np array containing the positions
        bonds: index array of the bonds
        const_bonds: array containing the constant associated with each bond
        angles: index array of the angles
        const_angles: array containing the constant associated with each angle
        lj_atoms: index array of Lennard Jones interaction
        lj_sigma: array containing the constant associated with each lj interaction
        lj_eps: array containing the constant associated with each lj interaction
        molecules: list of which atoms belongs in which molecule
        nr_atoms: number of atoms in the system
        box_size: box size of the PCB
    Output:
        force_total: numpy array containing the force acting on each molecule

    NOTE: See also the implementation of read_topology in io_sim, and the definitions of lj_sigma 
          and lj_eps in simulator.integration
    """
    force_total = np.zeros((nr_atoms, 3))

    # Forces due to bonds between atoms
    #----------------------------------
    # Difference vectors for the bonds, and the
    # distance between these atoms
    diff = pos[bonds[:,0]] - pos[bonds[:,1]]
    dis = np.linalg.norm(diff, axis=1)

    # Calculate the forces between the atoms
    magnitudes = np.multiply(-const_bonds[:,0], dis - const_bonds[:,1])
    force = magnitudes[:, np.newaxis]*unit_vector(diff)
    
    # Add them to the total force
    np.add.at(force_total, bonds[:,0], force)
    np.add.at(force_total, bonds[:,1], -force)

    #----------------------------------
    # Forces due to angles in molecules
    #----------------------------------
    
    # The difference vectors we need for the angles
    diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
    dis_1 = np.linalg.norm(diff_1, axis=1)
    diff_2 = pos[angles[:,1]] - pos[angles[:,2]]
    dis_2 = np.linalg.norm(diff_2, axis=1)
    ang = angle_between(diff_1, diff_2)
    
    # The constant we need for the force calculation
    mag_ang = np.multiply(-const_angles[:,0], ang - const_angles[:,1])

    # Calculate the direction vectors for the forces 
    # TODO: does cross return a unit vector already?
    cross_vector = np.cross(diff_1, diff_2)
    angular_force_unit_1 = unit_vector(np.cross(cross_vector, diff_1))
    angular_force_unit_2 = -unit_vector(np.cross(cross_vector, diff_2))

    # Actually calculate the forces
    force_ang_1 = np.multiply(np.true_divide(mag_ang, dis_1)[:, np.newaxis], angular_force_unit_1)
    force_ang_2 = np.multiply(np.true_divide(mag_ang, dis_2)[:, np.newaxis], angular_force_unit_2)
    
    # Add them to the total force
    np.add.at(force_total, angles[:,0], force_ang_1)
    np.add.at(force_total, angles[:,2], force_ang_2)
    np.add.at(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    #----------------------------------
    # Forces due to Lennard Jones interaction
    #----------------------------------
    # if lj_atoms is not empty
    if lj_atoms.shape[0] != 0:
        diff = np.zeros((lj_atoms.shape[0], 3))
        dis = np.zeros(diff.shape[0])
        distance_PBC(pos[lj_atoms[:,0]], pos[lj_atoms[:,1]],  box_size, dis, diff)

        term = np.true_divide(lj_sigma[lj_atoms[:,0], lj_atoms[:,1]], dis)
        term_1 = 2*np.power(term, 12)
        term_2 = -1*np.power(term, 6)

        magnitudes = 6*np.multiply(np.true_divide(lj_eps[lj_atoms[:,0], lj_atoms[:,1]], dis), term_1 + term_2)
        force = magnitudes[:, np.newaxis]*unit_vector(diff)

        np.add.at(force_total, lj_atoms[:,0], force)
        np.add.at(force_total, lj_atoms[:,1], -force)

    return force_total

if __name__ == "__main__":
    pass

