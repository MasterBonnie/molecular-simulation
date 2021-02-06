import numpy as np
import time

from io_sim import read_topology, read_xyz, printProgressBar
from helper import atom_string, unit_vector,  atom_name_to_mass, create_list, create_list_s, neighbor_list, neighbor_list_s, project_pos, project_pos_s
import integrators
from static_state import centre_of_mass, centre_of_mass_s, compute_force, kinetic_energy, potential_energy, temperature

def simulator(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, 
                integrator="vv", write_output = True, fill_in_molecules = 3, write_output_threshold = 0):
    """
    Numerical integration using either the euler algorithm,
    velocity verlet (vv) algorithm, or the verlet (v) algorithm.
    Requires a topology file, i.e. cannot pas constant through the function
    anymore.

    params:
        dt: time step in 0.1 ps, or 10^-13 s
        T: Length of simulation, in 0.1 ps
        r_cut: cutoff length for the LJ potential, in A
        box_size: size of the box for PBC, in A
        file_xyz: relative path to the xyz file
        file_top: relative path to the topology file
        file_out: relative path to the desired output file
        file_observable: relative path to csv file for possible output
                        other then xyz file.
        T_desired: desired temperature of the simulation. If 0, no thermostat is used.
        integrator: string selecting which integrator to use, element of
                    [euler, v, vv, beeman]
        write_output: boolean, whether to write to the xyz and observable file.
        fill_in_molecule: Amount to fill in molecules, DOES NOT WORK YET FOR MIXED MOLECULES.
                          see also the function read_topology in the io_sim.py file. Uses slower
                          versions of certain functions if 0.
        write_output_threshold: From which percentage onwards to start writing ouput, if write_ouput
                                is true.

    Output:
        Writes to the specified files if write_output is true. Furthermore if the variable debug is set,
        prints the average time each iteration takes.
    """
    debug = False
    print_simulation_info(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, 
                integrator, write_output, fill_in_molecules, write_output_threshold)
    
    print("Initialization of simulation and parameters")

    # If this is equal to zero, we use the "slower" version of several functions
    # There seems to be a bug if we use this on datasets consisting of different molecules

    # The reason we do it in this way is to prevent the need for many if statements that are 
    # executed all the time. We could have done the same thing to prevent the if statements
    # deciding which integrator is selected.
    if fill_in_molecules == 0:
        c_o_m = centre_of_mass_s
        nl = neighbor_list_s
        pp = project_pos_s
        cl = create_list_s

    else:
        c_o_m = centre_of_mass
        nl = neighbor_list
        pp = project_pos
        cl = create_list

    def compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                        const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals):
        centres_of_mass = c_o_m(pos, m, molecules)
        pp(centres_of_mass, box_size, pos, molecules)

        lj_atoms = None
        if molecule_to_atoms is not None:
            lj_atoms = nl(molecule_to_atoms, centres_of_mass, r_cut, box_size, nr_atoms, fill_in_molecules)

        f = cf*compute_force(pos, bonds, const_bonds, angles,
                        const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms, box_size)

        return f, lj_atoms


    t = 0

    # Progress bar variables
    total_progress = int(T/dt)
    progress = 0
    
    # Converts our force units to the force 
    # with unit amu A (0.1ps)^-2
    cf = 1.6605*6.022e-1 

    # Get all external variables
    print("reading variables...", end="\n")
    pos, atoms, nr_atoms = read_xyz(file_xyz)
    bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, dihedrals, const_dihedrals = read_topology(file_top, nr_atoms, fill_in_molecules)

    pos = np.append(pos, [[0,0,0]], axis=0)

    # Mass is given in amu
    m = atom_name_to_mass(atoms)

    # Pre-calculate these.
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
        #molecule_to_atoms = create_list(molecules, fill_in_molecules)
        molecule_to_atoms = cl(molecules, fill_in_molecules)

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
        pos_old = np.copy(pos)

        # If we use the verlet integrator, we first take on euler step
        if integrator == "verlet":
            pos[:-1], v = integrators.integrator_euler(pos[:-1], v, f, m[:-1], dt)

            f, _ = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)

        if debug:
            # We start a timer to measure the total time it takes to run the simulation
            # without initialization and file reading.
            time_1 = time.time()
        while t < T:
            if progress % 1000 == 0:
                printProgressBar(progress, total_progress)

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
                pos_old_temp = np.copy(pos)
                pos[:-1] = integrators.integrator_verlet_pos(pos[:-1], pos_old[:-1], f, m[:-1], dt)
                v = integrators.integrator_verlet_vel(pos[:-1], pos_old[:-1], dt)
                
                pos_old = pos_old_temp

                f, lj_atoms = compute_force_and_project(pos, m, cf, molecules, molecule_to_atoms, nr_atoms, bonds, const_bonds, angles,
                    const_angles, lj_sigma, lj_eps, dihedrals, const_dihedrals)

            t += dt
            progress += 1

            energy_kinetic = kinetic_energy(v, m[:-1])
            temp = temperature(energy_kinetic, nr_atoms)
            
            if T_desired:
                Lambda = np.sqrt(T_desired/temp)
                v = Lambda*v

            # I/O operations
            # We only write output in certain cases, to save on some time, as this is pretty slow.
            if write_output and progress/total_progress > write_output_threshold:
                energy_potential, energy_bond, energy_angle, energy_dihedral, energy_lj = potential_energy(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms,
                         box_size)
                         
                energy_total = energy_kinetic + energy_potential

                obs_file.write(f"{energy_potential}, {energy_kinetic}, {energy_total}, {energy_bond}, {energy_angle}, {energy_dihedral}, {energy_lj}, {temp} \n")

                output_file.write(f"{nr_atoms}" + '\n')
                output_file.write("Comments" + '\n')

                position_string = ""
                
                for atom_name, atom in enumerate(pos[:-1]):
                    position_string += atom_string(atoms[atom_name], atom)
                    
                output_file.write(position_string)
    
    print()
    
    if debug:
        time_2 = time.time()
    
        print(f"Total time for all iterations: {time_2 - time_1}")
        print(f"Time per iteration, on average, in seconds: {(time_2 - time_1) / total_progress }")
    
    return


def print_simulation_info(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, 
                integrator, write_output, fill_in_molecules, write_output_threshold):
    """ prints a small summary of the simulation parameters """

    info_string =  f"running simulation with the following variables: \n \n \
                        Total time: {T} 10^-13 s\n \
                        Time step: {dt} 10^-13 s\n \
                        Number of iterations: {int(T/dt)} \n \
                        \n \
                        Box_size: {box_size} A \n \
                        cut off distance: {r_cut} A \n \
                        \n \
                        Input file: {file_xyz} \n \
                        Input top file: {file_top} \n \
                        Ouput file: {file_out} \n \
                        Output file energy: {file_observable} \n \
                        \n \
                        Integrator: {integrator}\n \
                        \n \
                        Writing output: {write_output}, from {write_output_threshold*100} %\n"
    print(info_string)

    if fill_in_molecules:
        print(f"Using molecule fill in: {fill_in_molecules}")
    if T_desired:
        print(f"Using thermostat with desired temperature: {T_desired}")

    print()


# Testing of the functions
if __name__ == "__main__":

    dt = 0.0075 # 10^-13 s
    T = 2000 # 10^-13 s
    r_cut = 7 # A
    box_size = 30 # A
    file_xyz = "data/ethanol_3nm"
    file_top = "data/ethanol_3nm.itp"
    file_out = "output/result.xyz"
    file_observable = "output/result.csv"
    T_desired = 298.15   #kelvin     if zero we do not use a thermostat
    integrator = "vv"
    write_output = True
    write_output_threshold = 0.75
    
    #NOTE: DO NOT FORGET TO CHANGE THIS 
    fill_in_molecules = 9

    simulator(dt, T, r_cut, box_size, file_xyz, file_top, file_out, file_observable, T_desired, integrator, write_output, fill_in_molecules, write_output_threshold)

