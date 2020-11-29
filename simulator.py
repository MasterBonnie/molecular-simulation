import numpy as np

import io_sim
from static_state import compute_force_h2o, compute_force
from helper import atom_string, random_unit_vector, unit_vector
import integrators

def integration_2(m, dt, file_xyz, file_top, file_out, file_observable, 
                integrator="vv", observable_function = None):
    """
    Numerical integration using either the euler algorithm,
    velocity verlet (vv) algorithm, or the verlet (v) algorithm.
    Requires a topology file, i.e. cannot pas constant through the function
    anymore.

    Input:
        m: Array of masses of atoms in amu, in same order as in the xyz file
        dt: time step in 0.1 ps, or 10^-13 s
        file_xyz: relative path to the xyz file
        file_top: relative path to the topology file
        file_out: relative path to the desired output file
        file_observable: relative path to csv file for possible output
                        other then xyz file.
        integrator: string selecting which integrator to use, element of
                    [euler, v, vv]
        observable_function: function reference for possible calculations to
                            write to file_observable
    Output:
        Writes an xyz file to the file located at file_out
    """
    # Check for correct integrator
    if integrator not in ["euler", "v", "vv"]:
        print("No integrator selected")
        return

    T = 10 # in 0.1 ps, or 10^-13s
    t = 0

    # Converts our force units to the force 
    # with unit amu A (0.1ps)^-2
    cf = 1.6605*6.022e-1 

    # Get all external variables
    pos, atoms, nr_atoms = io_sim.read_xyz(file_xyz)
    bonds, const_bonds, angles, const_angles = io_sim.read_topology(file_top)

    # Random initial velocity
    v = np.array([random_unit_vector(0.1) for i in range(nr_atoms)])

    # Open the output file
    # I dont think it matters to much to have these files open during the calculation
    with open(file_out, "w") as output_file, open(file_observable, "w") as obs_file:
        output_file.write("{}".format(nr_atoms) + '\n')
        output_file.write("Comments" + '\n')

        # I/O operations
        for atom_name, atom in enumerate(pos):
            output_file.write(atom_string(atoms[atom_name], atom))

        if integrator == "v":
            pos_old = pos
            f = cf*compute_force_n(pos, bonds, const_bonds, angles,
                            const_angles, nr_atoms)

            # If we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)
            
            (pos, v) = integrators.integrator_euler(pos, v, f, m, dt)
            
            t += dt

        while t < T:
            
            # Compute the force on the entire system
            f = cf*compute_force_n(pos, bonds, const_bonds, angles,
                            const_angles, nr_atoms)

            # if we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)

            if integrator == "euler":
                (pos, v) = integrators.integrator_euler(pos, v, f, m, dt)

            elif integrator == "vv":
                pos = integrators.integrator_velocity_verlet_pos(pos, v, f, m, dt)
                f_new = cf*compute_force_n(pos, bonds, const_bonds, angles,
                                    const_angles, nr_atoms)
                v = integrators.integrator_velocity_verlet_vel(v, f, f_new, m, dt)

            elif integrator == "v":
                pos_old, pos = pos, integrators.integrator_verlet_pos(pos, pos_old, f, m, dt)
                # This v is the velocity at the previous timestep
                v = integrators.integrator_verlet_vel(pos, pos_old, dt) 
            
            t += dt

            # I/O operations
            output_file.write("{}".format(nr_atoms) + '\n')
            output_file.write("Comments" + '\n')

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
    T = 10 # in 0.1 ps, or 10^-13s
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

def compute_force_n(pos, bonds, const_bonds, angles, const_angles, nr_atoms):
    force_total = np.zeros((nr_atoms, 3))

    # Bonds
    # These are all diferences we need for bonds
    diff = pos[bonds[:,0]] - pos[bonds[:,1]]
    
    dis = np.linalg.norm(diff, axis=1)

    magnitudes = np.multiply(-const_bonds[:,0], dis - const_bonds[:,1])
    force = magnitudes[:, np.newaxis]*unit_vector(diff)

    np.add.at(force_total, bonds[:,0], force)
    np.add.at(force_total, bonds[:,1], -force)

    # Angles
    diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
    diff_2 = pos[angles[:,1]] - pos[angles[:,2]]

    dot = np.einsum('ij,ij->i', diff_1, diff_2)

    ang = np.arccos(dot)
    
    mag_ang = np.multiply(-const_angles[:,0], ang - const_angles[:,1])

    angular_force_unit_1 = unit_vector(np.cross(np.cross(diff_1, diff_2), diff_1))
    angular_force_unit_2 = -unit_vector(np.cross(np.cross(diff_1, diff_2), diff_2))

    force_ang_1 = np.multiply(np.multiply(mag_ang, np.linalg.norm(diff_1, axis=1))[:, np.newaxis], angular_force_unit_1)
    force_ang_2 = np.multiply(np.multiply(mag_ang, np.linalg.norm(diff_2, axis=1))[:, np.newaxis], angular_force_unit_2)
    
    np.add.at(force_total, angles[:,0], force_ang_1)
    np.add.at(force_total, angles[:,2], force_ang_2)
    np.add.at(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    return force_total

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
    # k = 245.31 # Kj mol^-1 A^-2
    # r_0 = 0.74 # A
    # delta_t = 0.001 # 0.1 ps or 10^-13 s
    # m = np.array([1.00784, 1.00784]) # amu 
    # file_name = "data/hydrogen_small.xyz"

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

    #integration(k, r_0, delta_t, m, file_name, update_euler)
    #integration_verlet(k, r_0, delta_t, m, file_name)

    m = np.array([15.999, 1.00784, 1.00784, 15.999, 1.00784, 1.00784])
    dt = 0.001
    file_xyz = "data/water_top.xyz"
    file_top = "data/top.itp"
    file_out = "output/result.xyz"
    file_observable = "output/result_phase.csv"

    integration_2(m, dt, file_xyz, file_top, file_out, file_observable)
