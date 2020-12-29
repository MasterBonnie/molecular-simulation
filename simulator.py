import numpy as np
import math
from collections import deque
from numba import vectorize, float64, jit, guvectorize, double
import cProfile

import io_sim
from helper import atom_string, random_unit_vector, unit_vector, angle_between, atom_name_to_mass, cartesianprod, create_list, neighbor_list, distance_PBC
import integrators
from static_state import centre_of_mass, compute_force, project_pos, kinetic_energy, potential_energy, temperature

def integration(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, 
                observable_function = None, integrator="vv", write_output = True):
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

    t = 0

    # Progress bar variables
    total_progress = int(T/dt)
    progress = 1
    
    # Desired temperature 
    T_desired = 298.5 #kelvin

    # Converts our force units to the force 
    # with unit amu A (0.1ps)^-2
    cf = 1.6605*6.022e-1 

    # Get all external variables
    pos, atoms, nr_atoms = io_sim.read_xyz(file_xyz)
    bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, dihedrals, const_dihedrals = io_sim.read_topology(file_top)

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
    molecule_to_atoms = create_list(molecules)

    # Random initial velocity
    v = unit_vector(np.random.uniform(size=[nr_atoms,3]))
    energy_kinetic = kinetic_energy(v,m)

    temp = temperature(energy_kinetic,nr_atoms)
    Lambda = np.sqrt(T_desired/temp)

    v = Lambda*v

    # Open the output file
    with open(file_out, "w") as output_file, open(file_observable, "w") as obs_file:
        # I/O operations
        if write_output:
            output_file.write(f"{nr_atoms}" + '\n')
            output_file.write("Comments" + '\n')
 
            for atom_name, atom in enumerate(pos):
                output_file.write(atom_string(atoms[atom_name], atom))

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
        lj_atoms = neighbor_list(pos, molecule_to_atoms, centres_of_mass, r_cut, box_size)
        f = cf*compute_force(pos, bonds, const_bonds, angles,
                        const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms, box_size)

        while t < T:
            io_sim.printProgressBar(progress, total_progress)

            # if we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)

            # Based on the integrator we update the current pos and v
            # if integrator == "euler":
            #     pos, v, _ = integrators.integrator_euler(pos, v, f, m, dt)

            #     centres_of_mass = centre_of_mass(pos, m, molecules)
            #     project_pos(centres_of_mass, box_size, pos, molecules)
            #     lj_atoms = neighbor_list(pos, molecule_to_atoms, centres_of_mass, r_cut, box_size)
            #     f = cf*compute_force(pos, bonds, const_bonds, angles,
            #                 const_angles, lj_atoms, lj_sigma, lj_eps, molecules, nr_atoms, box_size)

            if integrator == "vv":
                pos = integrators.integrator_velocity_verlet_pos(pos, v, f, m, dt)

                centres_of_mass = centre_of_mass(pos, m, molecules)
                project_pos(centres_of_mass, box_size, pos, molecules)
                lj_atoms = neighbor_list(pos, molecule_to_atoms, centres_of_mass, r_cut, box_size)

                f_old = f
                f = cf*compute_force(pos, bonds, const_bonds, angles,
                                    const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms, box_size)
                v = integrators.integrator_velocity_verlet_vel(v, f_old, f, m, dt)

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

            # I/O operations
            energy_kinetic = kinetic_energy(v,m)
            energy_potential, energy_bond, energy_angle, energy_dihedral = potential_energy(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms,
                    box_size)
            energy_total = energy_kinetic + energy_potential

            temp = temperature(energy_kinetic,nr_atoms)
            Lambda = np.sqrt(T_desired/temp)

            v = Lambda*v

            obs_file.write(f"{energy_potential}, {energy_kinetic}, {energy_total}, {energy_bond}, {energy_angle}, {energy_dihedral}, {temp} \n")

            if write_output:
                output_file.write(f"{nr_atoms}" + '\n')
                output_file.write("Comments" + '\n')

                for atom_name, atom in enumerate(pos):
                    output_file.write(atom_string(atoms[atom_name], atom))

    return

def phase_space_h(pos, v, f, obs_file):
    """
    Example of how the observable function can be used in the integrator function
    Calculates phase space data for a single hydrogen molecule, for use in the 
    report
    """
    # NOTE: this is pretty bad if it was not used for 
    # only the toy example of a single hydrogen molecule
    diff = pos - pos[:, np.newaxis]
    dis = np.linalg.norm(diff, axis=2)

    r = dis[0][1]
    v_plot = np.linalg.norm(v)
    
    obs_file.write(f"{r}, {v_plot} \n")

# Testing of the functions
if __name__ == "__main__":

    # Water file
    dt = 0.02 # 0.1 ps
    T = 100  # 10^-13 s
    r_cut = 8 # A
    box_size = 50 # A
    file_xyz = "data/water.xyz"
    file_top = "data/water.itp"
    file_out = "output/result.xyz"
    file_observable = "output/result_phase.csv"
    observable_function = None  
    integrator = "vv"
    write_output = True

    #cProfile.run("integration(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, observable_function, integrator, write_output)")
    integration(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, observable_function, integrator, write_output)
