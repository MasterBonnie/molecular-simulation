import numpy as np
import math
from collections import deque
from numba import vectorize, float64, jit, guvectorize, double
import cProfile

import io_sim
from helper import atom_string, random_unit_vector, unit_vector, angle_between, atom_name_to_mass, cartesianprod, create_list, neighbor_list, distance_PBC
import integrators
from static_state import centre_of_mass, compute_force, project_pos, kinetic_energy, potential_energy, temperature, compute_force_old

def integration(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, 
                integrator="vv", write_output = True, fill_in_molecules = 3, write_output_threshold = 0):
    """
    Numerical integration using either the euler algorithm,
    velocity verlet (vv) algorithm, or the verlet (v) algorithm.
    Requires a topology file, i.e. cannot pas constant through the function
    anymore.

    Input:
        dt: time step in 0.1 ps, or 10^-13 s
        T: Length of simulation, in 0.1 ps
        r_cut: cutoff length for the LJ potential, in A
        box_size: size of the box for PBC, in A
        file_xyz: relative path to the xyz file
        file_top: relative path to the topology file
        file_out: relative path to the desired output file
        file_observable: relative path to csv file for possible output
                        other then xyz file.
        integrator: string selecting which integrator to use, element of
                    [euler, v, vv]
        observable_function: function reference for possible calculations to
                            write to file_observable
        write_output: boolean, whether the computed pos needs to be written to file_out 
    Output:
        Writes an xyz file to the file located at file_out
    """
    # Check for correct integrator
    if integrator not in ["euler", "v", "vv"]:
        print("No integrator selected")
        return

    print("Initialization of simulation")

    t = 0

    # Progress bar variables
    total_progress = int(T/dt)
    progress = 0
    
    # Desired temperature 
    T_desired = 298.5 #kelvin

    # Converts our force units to the force 
    # with unit amu A (0.1ps)^-2
    cf = 1.6605*6.022e-1 

    # Get all external variables
    print("reading variables...", end="\n")
    pos, atoms, nr_atoms = io_sim.read_xyz(file_xyz)
    bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, dihedrals, const_dihedrals = io_sim.read_topology(file_top, nr_atoms, fill_in_molecules)

    pos = np.append(pos, [[0,0,0]], axis=0)

    # Mass is given in amu
    m = atom_name_to_mass(atoms)

    # Pre-calculate these, because we need them all anyway, probably
    lj_sigma = np.zeros(nr_atoms)
    lj_sigma[lj] = const_lj[:,0]
    lj_sigma = 0.5*(lj_sigma + lj_sigma[:, np.newaxis])

    # NOTE: multiply with 4 here already
    lj_eps = np.zeros(nr_atoms)
    lj_eps[lj] = const_lj[:, 1]
    lj_eps = 4*np.sqrt(lj_eps*lj_eps[:, np.newaxis])

    # Create the conversion from molecules to atoms
    # This is for the Lennard-Jones potential
    molecule_to_atoms = create_list(molecules, fill_in_molecules)

    # Random initial velocity
    v = unit_vector(np.random.uniform(size=[nr_atoms,3]))
    energy_kinetic = kinetic_energy(v,m[:-1])

    temp = temperature(energy_kinetic,nr_atoms)
    Lambda = np.sqrt(T_desired/temp)

    v = Lambda*v

    print("Starting with simulation:                       ")

    # Open the output file
    with open(file_out, "w") as output_file, open(file_observable, "w") as obs_file:

        # If we use the verlet integrator, we take one step using the 
        # Euler algorithm at first. It is easier to do this outside
        # the while loop
        # if integrator == "v":
        #     centres_of_mass = centre_of_mass(pos, m, molecules)
        #     lj_atoms = neighbor_list(pos, molecule_to_atoms, centres_of_mass, r_cut, box_size)

        #     pos_old = pos
        #     f = cf*compute_force(pos, bonds, const_bonds, angles,
        #                     const_angles, lj_atoms, lj_sigma, lj_eps, molecules, nr_atoms, box_size)

        #     # If we want to calculate something
        #     if observable_function:
        #         observable_function(pos, v, f, obs_file)
            
        #     pos, v, _ = integrators.integrator_euler(pos, v, f, m, dt)
            
        #     t += dt
        #     progress += 1

        # Compute the force on the entire system
        centres_of_mass = centre_of_mass(pos, m, molecules)
        project_pos(centres_of_mass, box_size, pos, molecules)

        lj_atoms = neighbor_list(molecule_to_atoms, centres_of_mass, r_cut, box_size, nr_atoms, fill_in_molecules)

        f = cf*compute_force(pos, bonds, const_bonds, angles,
                        const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms, box_size)

        while t < T:
            if progress % 10 == 0:
                io_sim.printProgressBar(progress, total_progress)
                #pass

            # Based on the integrator we update the current pos and v
            # if integrator == "euler":
            #     pos, v, _ = integrators.integrator_euler(pos, v, f, m, dt)

            #     centres_of_mass = centre_of_mass(pos, m, molecules)
            #     project_pos(centres_of_mass, box_size, pos, molecules)
            #     lj_atoms = neighbor_list(pos, molecule_to_atoms, centres_of_mass, r_cut, box_size)
            #     f = cf*compute_force(pos, bonds, const_bonds, angles,
            #                 const_angles, lj_atoms, lj_sigma, lj_eps, molecules, nr_atoms, box_size)

            if integrator == "vv":
                pos[:-1] = integrators.integrator_velocity_verlet_pos(pos[:-1], v, f, m[:-1], dt)

                centres_of_mass = centre_of_mass(pos, m, molecules)
                project_pos(centres_of_mass, box_size, pos, molecules)
                lj_atoms = neighbor_list(molecule_to_atoms, centres_of_mass, r_cut, box_size, nr_atoms, fill_in_molecules)

                f_old = f
                #f = cf*compute_force_old(pos, bonds, const_bonds, angles,
                #                    const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms, box_size)
                f = cf*compute_force(pos, bonds, const_bonds, angles,
                                    const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms, box_size)
                #print(np.allclose(f, f_test))

                v = integrators.integrator_velocity_verlet_vel(v, f_old, f, m[:-1], dt)

            # elif integrator == "v":
            #     pos_old, (pos, _) = pos, integrators.integrator_verlet_pos(pos, pos_old, f, m, dt)
            #     # This v is the velocity at the previous timestep
            #     v = integrators.integrator_verlet_vel(pos, pos_old, dt) 

            #     centres_of_mass = centre_of_mass(pos, m, molecules)
            #     lj_atoms = neighbor_list(pos, molecule_to_atoms, centres_of_mass, r_cut, box_size)
            #     f = cf*compute_force(pos, bonds, const_bonds, angles,
            #             const_angles, lj_atoms, lj_sigma, lj_eps, molecules, nr_atoms, box_size)
            
            t += dt
            progress += 1

            energy_kinetic = kinetic_energy(v,m[:-1])
            temp = temperature(energy_kinetic,nr_atoms)
            Lambda = np.sqrt(T_desired/temp)

            v = Lambda*v

            if write_output and progress/total_progress > write_output_threshold:
                # I/O operations
                # energy_potential, energy_bond, energy_angle, energy_dihedral = potential_energy(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms,
                #         box_size)
                # energy_total = energy_kinetic + energy_potential

                # obs_file.write(f"{energy_potential}, {energy_kinetic}, {energy_total}, {energy_bond}, {energy_angle}, {energy_dihedral}, {temp} \n")

                output_file.write(f"{nr_atoms}" + '\n')
                output_file.write("Comments" + '\n')

                for atom_name, atom in enumerate(pos[:-1]):
                    output_file.write(atom_string(atoms[atom_name], atom))

    return

# Testing of the functions
if __name__ == "__main__":

    # Water file
    dt = 0.02 # 0.1 ps
    T = 10 # 10^-13 s
    r_cut = 8 # A
    box_size = 50 # A
    file_xyz = "data/water.xyz"
    file_top = "data/water.itp"
    file_out = "output/result.xyz"
    file_observable = "output/result_phase.csv"
    integrator = "vv"
    write_output = True
    write_output_threshold = 0

    #NOTE: DO NOT FORGET TO CHANGE THIS 
    fill_in_molecules = 3

    cProfile.run("integration(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, integrator, write_output, fill_in_molecules, write_output_threshold)")
    #integration(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, integrator, write_output, fill_in_molecules, write_output_threshold)
