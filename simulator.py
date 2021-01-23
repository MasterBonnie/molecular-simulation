import numpy as np
import math
from numba import vectorize, float64, jit, guvectorize, double
import cProfile
import time

import io_sim
from helper import atom_string, random_unit_vector, unit_vector,  atom_name_to_mass, cartesianprod, create_list, neighbor_list, distance_PBC, project_pos
import integrators
from static_state import centre_of_mass, compute_force, kinetic_energy, potential_energy_jit, temperature

def simulator(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, 
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

    print_simulation_info(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, 
                integrator, write_output, fill_in_molecules, write_output_threshold)
    
    print("Initialization of simulation")

    t = 0

    # Progress bar variables
    total_progress = int(T/dt)
    progress = 0
    
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

    lj_sigma = None
    lj_eps = None
    molecule_to_atoms = None

    if lj is not None:
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

    if T_desired:
        temp = temperature(energy_kinetic,nr_atoms)
        Lambda = np.sqrt(T_desired/temp)

        v = Lambda*v

    print("Starting with simulation:                       ")

    # Open the output file
    with open(file_out, "w") as output_file, open(file_observable, "w") as obs_file:

        # Compute the force on the entire system
        f, _ = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)
        f_old = f
        pos_old = pos

        while t < T:
            if progress % 1000 == 0:
                io_sim.printProgressBar(progress, total_progress)

            # Select which integrator we use. In general, these follow the same pattern, update the positions and velocity,
            # and then compute the new force on the system. We also compute lj_atoms for use in calculating the potential energy
            if integrator == "vv":
                pos[:-1] = integrators.integrator_velocity_verlet_pos(pos[:-1], v, f, m[:-1], dt)
                
                f_new, lj_atoms = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)

                v = integrators.integrator_velocity_verlet_vel(v, f, f_new, m[:-1], dt)

                f = f_new

            elif integrator == "beeman":
                pos[:-1] = integrators.integrator_beeman_pos(pos[:-1], v, f, f_old, m[:-1], dt)
    
                f_new, lj_atoms = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)

                v = integrators.integrator_beeman_vel(v, f_new, f, f_old, m[:-1], dt)

                f_old = f
                f = f_new

            elif integrator == "euler":
                pos[:-1], v = integrators.integrator_euler(pos[:-1], v, f, m[:-1], dt)

                f, lj_atoms = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)

            elif integrator == "verlet":
                pos[:-1] = integrators.integrator_verlet_pos(pos[:-1], pos_old[:-1], f, m[:-1], dt)
                v = integrators.integrator_verlet_vel(pos[:-1], pos_old[:-1], dt)
                
                pos_old = pos
                f, lj_atoms = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)

            t += dt
            progress += 1

            energy_kinetic = kinetic_energy(v, m[:-1])
            temp = temperature(energy_kinetic, nr_atoms)
            
            if T_desired:
                Lambda = np.sqrt(T_desired/temp)
                v = Lambda*v

            # We only write output in certain cases, to save on some time, as this is pretty slow.
            if write_output and progress/total_progress > write_output_threshold:
                # I/O operations
                energy_potential, energy_bond, energy_angle, energy_dihedral = potential_energy_jit(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms,
                         box_size)
                         
                energy_total = energy_kinetic + energy_potential

                obs_file.write(f"{energy_potential}, {energy_kinetic}, {energy_total}, {energy_bond}, {energy_angle}, {energy_dihedral}, {temp} \n")

                output_file.write(f"{nr_atoms}" + '\n')
                output_file.write("Comments" + '\n')

                for atom_name, atom in enumerate(pos[:-1]):
                    output_file.write(atom_string(atoms[atom_name], atom))

    return

def compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals):
    centres_of_mass = centre_of_mass(pos, m, molecules)
    project_pos(centres_of_mass, box_size, pos, molecules)

    lj_atoms = None
    if molecule_to_atoms is not None:
        lj_atoms = neighbor_list(molecule_to_atoms, centres_of_mass, r_cut, box_size, nr_atoms, fill_in_molecules)

    f = cf*compute_force(pos, bonds, const_bonds, angles,
                    const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms, box_size)

    return f, lj_atoms

def print_simulation_info(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, 
                integrator, write_output, fill_in_molecules, write_output_threshold):

    info_string =  f"running simulation with the following variables: \n \n \
                        Total time: {T} 10^-13 s\n \
                        Time step: {dt} 10^-13 s\n \
                        Number of iterations: {T//dt} \n \
                        \n \
                        Box_size: {box_size} A \n \
                        cut off distance: {r_cut} A \n \
                        \n \
                        Input file: {file_xyz} \n \
                        Input top file: {file_top} \n \
                        Ouput file: {file_out} \n \
                        Output file energy: {file_observable} \n \
                        \n \
                        Integrator: {integrator}\n"
    print(info_string)
    print()

# Testing of the functions
if __name__ == "__main__":

    # Water file
    dt = 0.005 # 0.1 ps
    T = 1500 # 10^-13 s
    r_cut = 7 # A
    box_size = 30 # A
    file_xyz = "data/mix_3nm_2.xyz"
    file_top = "data/mix_3nm_2.itp"
    file_out = "output/result_mix_3nm_2.xyz"
    file_observable = "output/result_phase_mix_3nm_2.csv"
    T_desired =  298.15   #kelvin     if zero we do not use a thermostat
    integrator = "vv"
    write_output = True
    write_output_threshold = 0.75
    
    #NOTE: DO NOT FORGET TO CHANGE THIS 
    fill_in_molecules = 9

    time_1 = time.time()
    #cProfile.run("simulator(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, integrator, write_output, fill_in_molecules, write_output_threshold)")
    simulator(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, integrator, write_output, fill_in_molecules, write_output_threshold)
    time_2 = time.time()
    print(f"Min: {(time_2 - time_1)/60}")
