import numpy as np

import io_sim
from static_state import compute_force_h2o, compute_force
from helper import atom_string, random_unit_vector
import integrators

def integration_2(m, dt, file_xyz, file_top, file_out, verlet = False):

    T = 0.1 # in 0.1 ps, or 10^-13s
    t = 0

    # Get all external variables
    pos, atoms, nr_atoms = io_sim.read_xyz(file_xyz)
    bonds, const_bonds, angles, const_angles = io_sim.read_topology(file_top)

    # Open the output file
    with open(file_out, "w") as output_file:
        output_file.write("{}".format(nr_atoms) + '\n')
        output_file.write("Comments" + '\n')

        # I/O operations
        for atom_name, atom in enumerate(pos):
            output_file.write(atom_string(atoms[atom_name], atom))

        while t < T:
            

            
            
            t += delta_t

            # I/O operations
            for atom_name, atom in enumerate(pos):
                output_file.write(atom_string(atoms[atom_name], atom))

    return




def integration(k, r_0, delta_t, m, file_name, update, verlet = False):
    """
    Numerical integration using either the euler or velocity verlet algorithm 
    (if verlet is false)
    or use the regular verlet algorithm if verlet is true
    
    Input:
        k: bond constant, in Kj mol^-1 A^-2
        r_0: stationary distance, in A
        delta_t: timesetp, in 0.1 ps or 10^-13 s
        m: masses of atoms, in amu (numpy array)
        file_name: file name of xyz file with initial pos
        update: which update function to use (euler vs vv)
        verlet: Boolean specifying if verlet should be run or not

    outputs a xyz file
    """
    T = 0.1 # in 0.1 ps, or 10^-13s
    t = 0

    # Converts our units to the force unit amu A (0.1ps)^-2
    # We now have Kj/(mol * A) = 1000 kg m mol^-1 A^-1 s^-2
    cf = 1.6605*6.022e-1 # Is really close to 1

    # Read the xyz file and retrieve initial positions
    pos, atoms, nr_atoms = io_sim.read_xyz(file_name)

    # Generate random initial velocity
    v = np.array([random_unit_vector(0.1) for i in range(nr_atoms)])

    # Open the file we write the output to 
    with open("output/result_h2.xyz", "w") as output_file, open("output/result_phase.csv", "w") as result_file:
        output_file.write("{}".format(nr_atoms) + '\n')
        output_file.write("Comments" + '\n')

        # I/O operations
        for atom_name, atom in enumerate(pos):
            output_file.write(atom_string(atoms[atom_name], atom))

        # TODO: pls change thenk
        # Should not recalculate diff and dis here
        diff = pos - pos[:, np.newaxis]
        dis = np.linalg.norm(diff, axis=2)
        r = dis[0][1]
        v_plot = np.linalg.norm(v)
        result_file.write("{}, {} \n".format(r, v_plot))

        if verlet:   
            # We set one step using euler
            t += delta_t
            pos_old = pos
            (pos, v) = update_euler(pos, v, delta_t, cf, k, r_0)

        while t < T:
            t += delta_t

            if verlet:
                f = compute_force(pos, k, r_0)
                pos_old, pos = pos, integrators.integrator_verlet_pos(pos, pos_old, f, m, delta_t)
                
                # This v is the velocity at the previous timestep
                v = integrators.integrator_verlet_vel(pos, pos_old, delta_t)            
            else:
                (pos, v) = update(pos, v, delta_t, cf, k, r_0)
            
            # I/O operations
            output_file.write("{}".format(nr_atoms) + '\n')
            output_file.write("Comments" + '\n')

            # TODO: pls change this it bad
            diff = pos - pos[:, np.newaxis]
            dis = np.linalg.norm(diff, axis=2)
            r = dis[0][1]
            v_plot = np.linalg.norm(v)
            result_file.write("{}, {} \n".format(r, v_plot))

            for atom_name, atom in enumerate(pos):
                output_file.write(atom_string(atoms[atom_name], atom))

    return


def update_euler(pos, v, delta_t, cf, k, r_0):
    """
    Performs an update step using the euler algorithm
    """

    f = cf*compute_force(pos, k, r_0)
    (pos_new, v_new) = integrators.integrator_euler(pos, v, f, m, delta_t)

    return pos_new, v_new

def update_vv(pos, v, delta_t, cf, k, r_0):
    """
    Performs an update step using the velocity verlet algorithm
    """

    f = cf*compute_force(pos, k, r_0)
    pos_new = integrators.integrator_velocity_verlet_pos(pos, v, f, m, delta_t)
    f_new = cf*compute_force(pos_new, k, r_0)
    v_new = integrators.integrator_velocity_verlet_vel(v, f, f_new, m, delta_t)

    return pos_new, v_new

# Testing of the functions
if __name__ == "__main__":

    # Hydrogen atoms
    k = 245.31 # Kj mol^-1 A^-2
    r_0 = 0.74 # A
    delta_t = 0.001 # 0.1 ps or 10^-13 s
    m = np.array([1.00784, 1.00784]) # amu 
    file_name = "data/hydrogen_small.xyz"

    # # Oxygen atoms
    # k = 1
    # r_0 = 1
    # delta_t = 0.001 # 0.1 ps 
    # m = np.array([15.999, 15.999]) # amu
    # file_name = "data/oxygenSmall.xyz"

    # # Water molecule
    # k = 1
    # r_0 = 1
    # delta_t = 0.0005 # 0.1 ps 
    # m = np.array([15.999, 1.00784, 1.00784]) # amu
    # file_name = "data/water_small.xyz"

    integration(k, r_0, delta_t, m, file_name, update_euler)
    #integration_verlet(k, r_0, delta_t, m, file_name)
