import numpy as np

import io_sim
import static_state
import helper
import integrators

def integration_verlet(k, r_0, delta_t, m, file_name):
    """
    Numerical integration using the verlet algorithm,
    for linear atoms
    
    Input:
        k: bond constant, in Kj mol^-1 A^-2
        r_0: stationary distance, in A
        delta_t: timesetp, in 0.1 ps or 10^-13 s
        m: masses of atoms, in amu (numpy array)
        file_name: file name of xyz file with initial pos
    """
    T = 1 # in 0.1 ps, or 10^-13s
    t = 0

    # Converts our units to the force unit amu A (0.1ps)^-2
    # We now have Kj/(mol * A) = 1000 kg m mol^-1 A^-1 s^-2
    #cf = 1.6605*6.022e-1 # Is really close to 1

    # Read the xyz file and retrieve initial positions
    pos, atoms, nr_atoms = io_sim.read_xyz(file_name)

    # Generate random initial velocity
    v_init_1 = helper.random_unit_vector(0.1)
    v_init_2 = helper.random_unit_vector(0.1)
    v = np.array([v_init_1, v_init_2])

    # Open the file we write the output to 
    with open("output/result_h2.xyz", "a") as output_file:
        output_file.write("{}".format(nr_atoms) + '\n')
        output_file.write("Comments" + '\n')

        for atom_name, atom in pos:
            output_file.write(helper.atom_string(atoms[atom_name], atom))

        # We set one step using euler
        t += delta_t
        pos_old = pos
        (pos, v) = update_euler(pos, v, k, r_0, delta_t)

        while t < T:
            t += delta_t

            f = compute_force(pos, v, k, r_0)
            pos_old, pos = pos, integrators.integrator_verlet_pos(pos, pos_old, f, m, delta_t)
            
            # TODO:
            # compute the velocity

            output_file.write("{}".format(nr_atoms) + '\n')
            output_file.write("Comments" + '\n')

            for atom_name, atom in enumerate(pos):
                output_file.write(helper.atom_string(atoms[atom_name], atom))

    return

def integration(k, r_0, delta_t, m, file_name, update):
    """
    Numerical integration using either the euler or velocity verlet algorithm,
    for linear atoms
    
    Input:
        k: bond constant, in Kj mol^-1 A^-2
        r_0: stationary distance, in A
        delta_t: timesetp, in 0.1 ps or 10^-13 s
        m: masses of atoms, in amu (numpy array)
        file_name: file name of xyz file with initial pos
    """
    T = 1 # in 0.1 ps, or 10^-13s
    t = 0

    # Converts our units to the force unit amu A (0.1ps)^-2
    # We now have Kj/(mol * A) = 1000 kg m mol^-1 A^-1 s^-2
    #cf = 1.6605*6.022e-1 # Is really close to 1

    # Read the xyz file and retrieve initial positions
    pos, atoms, nr_atoms = io_sim.read_xyz(file_name)

    # Generate random initial velocity
    v_init_1 = helper.random_unit_vector(0.1)
    v_init_2 = helper.random_unit_vector(0.1)
    v = np.array([v_init_1, v_init_2])

    # Open the file we write the output to 
    with open("output/result_h2.xyz", "a") as output_file:
        output_file.write("{}".format(nr_atoms) + '\n')
        output_file.write("Comments" + '\n')

        for atom_name, atom in pos:
            output_file.write(helper.atom_string(atoms[atom_name], atom))

        while t < T:
            t += delta_t

            (pos, v) = update(pos, v, k, r_0, delta_t)

            output_file.write("{}".format(nr_atoms) + '\n')
            output_file.write("Comments" + '\n')

            for atom_name, atom in enumerate(pos):
                output_file.write(helper.atom_string(atoms[atom_name], atom))

    return


def update_euler(pos, v, k, r_0, delta_t):
    """
    Performs an update step using the euler algorithm
    """

    f = compute_force(pos, v, k, r_0)
    (pos_new, v_new) = integrators.integrator_euler(pos, v, f, m, delta_t)

    return pos_new, v_new


def update_vv(pos, v, k, r_0, delta_t):
    """
    Performs an update step using the velocity verlet algorithm
    """

    f = compute_force(pos, v, k, r_0)
    pos_new = integrators.integrator_velocity_verlet_pos(pos, v, f, m, delta_t)
    f_new = compute_force(pos, v, k, r_0)
    v_new = integrators.integrator_velocity_verlet_vel(v, f, f_new, m, delta_t)

    return pos_new, v_new


def compute_force(pos, v, k, r_0):
    """
    Computes the force on a linear model
    """

    diff = pos - pos[:, np.newaxis]
    dis = np.linalg.norm(diff, axis=2)

    force = static_state.force_bond(diff[0][1], dis[0][1], k, r_0)

    return np.array([-force, force])

def compute_force_h2o(pos):
    # Calculate the force on each molecule
    k_b = 5024.16    # Kj mol^-1 A^-2
    r_0 = 0.9572   # nm

    theta_0 = 104.52 *(np.pi/180)   # Radians
    k_ang = 628.02  # Kj mol^-1 rad^-2

    diff = pos - pos[:, np.newaxis]
    dis = np.linalg.norm(diff, axis=2)

    theta = helper.angle_between(diff[0][1], diff[0][2])
    #print(theta*180/np.pi)

    # Converts our units to the force unit amu A/(0.1ps)^2
    # We now have Kj/(mol * nm) = 1000 kg m mol^-1 nm^-1 s^-2
    conversion_factor = 1.6605*6.022e-1 # is really close to 0.1

    # Create the unit vector along which the angular force acts
    # This is the right molecule, and so we need this -1 here
    angular_force_unit_1 = helper.unit_vector(-np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][1]))

    # Calculate the angular and bond force on Hydrogen atom 1
    angular_force_1 = static_state.force_angular(angular_force_unit_1,
                                                    theta,
                                                    dis[0][1],
                                                    k_ang,
                                                    theta_0)
    bond_force_1 = static_state.force_bond(diff[0][1], dis[0][1], k_b, r_0)

    # Total force on hydrogen atom 1
    force_hydrogen_1 = conversion_factor*(angular_force_1 + bond_force_1)

    # Again create unit vector for angular force
    # This one already points in the right direction
    angular_force_unit_2 = helper.unit_vector(np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][2]))

    # Angular force
    angular_force_2 = static_state.force_angular(angular_force_unit_2,
                                                    theta,
                                                    dis[0][2],
                                                    k_ang,
                                                    theta_0)
    bond_force_2 = static_state.force_bond(diff[0][2], dis[0][2], k_b, r_0)

    # Total force on Hydrogen atom 2
    force_hydrogen_2 = conversion_factor*(angular_force_2 + bond_force_2)

    force_oxygen = -(force_hydrogen_1 + force_hydrogen_2)

    return np.array([force_oxygen, force_hydrogen_1, force_hydrogen_2])

# Testing of the functions
if __name__ == "__main__":

    # Hydrogen atoms
    k = 245.31 # Kj mol^-1 A^-2
    r_0 = 0.74 # A
    delta_t = 0.001 # 0.1 ps or 10^-13 s
    m = np.array([1.00784, 1.00784]) # amu 
    file_name = "data/water2.xyz"

    # # Oxygen atoms
    # k = 1
    # r_0 = 1
    # delta_t = 0.001 # 0.1 ps 
    # m = np.array([15.999, 15.999]) # amu
    # file_name = "data/oxygenSmall.xyz"


    integration(k, r_0, delta_t, m, file_name, update_euler)
    #integration_verlet(k, r_0, delta_t, m, file_name)
