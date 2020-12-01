import numpy as np

import io_sim
from helper import atom_string, random_unit_vector, unit_vector, angle_between
import integrators

def integration(m, dt, T, file_xyz, file_top, file_out, file_observable, 
                observable_function = None, integrator="vv"):
    """
    Numerical integration using either the euler algorithm,
    velocity verlet (vv) algorithm, or the verlet (v) algorithm.
    Requires a topology file, i.e. cannot pas constant through the function
    anymore.

    Input:
        m: Array of masses of atoms in amu, in same order as in the xyz file
        dt: time step in 0.1 ps, or 10^-13 s
        T: Length of simulation, in 0.1 ps
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

    t = 0

    # Converts our force units to the force 
    # with unit amu A (0.1ps)^-2
    cf = 1.6605*6.022e-1 

    # Get all external variables
    pos, atoms, nr_atoms = io_sim.read_xyz(file_xyz)
    bonds, const_bonds, angles, const_angles = io_sim.read_topology(file_top)

    # Random initial velocity
    v = 0.1*unit_vector(np.random.uniform(size=[nr_atoms,3]))

    # Open the output file
    # I dont think it matters too much to have these files open during the calculation
    with open(file_out, "w") as output_file, open(file_observable, "w") as obs_file:
        output_file.write(f"{nr_atoms}" + '\n')
        output_file.write("Comments" + '\n')

        # I/O operations
        for atom_name, atom in enumerate(pos):
            output_file.write(atom_string(atoms[atom_name], atom))

        # If we use the verlet integrator, we take one step using the 
        # Euler algorithm at first. It is easier to do this outside
        # the while loop
        if integrator == "v":
            pos_old = pos
            f = cf*compute_force(pos, bonds, const_bonds, angles,
                            const_angles, nr_atoms)

            # If we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)
            
            (pos, v) = integrators.integrator_euler(pos, v, f, m, dt)
            
            t += dt

        while t < T:
            
            # Compute the force on the entire system
            f = cf*compute_force(pos, bonds, const_bonds, angles,
                            const_angles, nr_atoms)

            # if we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)

            # Based on the integrator we update the current pos and v
            if integrator == "euler":
                (pos, v) = integrators.integrator_euler(pos, v, f, m, dt)

            elif integrator == "vv":
                pos = integrators.integrator_velocity_verlet_pos(pos, v, f, m, dt)
                f_new = cf*compute_force(pos, bonds, const_bonds, angles,
                                    const_angles, nr_atoms)
                v = integrators.integrator_velocity_verlet_vel(v, f, f_new, m, dt)

            elif integrator == "v":
                pos_old, pos = pos, integrators.integrator_verlet_pos(pos, pos_old, f, m, dt)
                # This v is the velocity at the previous timestep
                v = integrators.integrator_verlet_vel(pos, pos_old, dt) 
            
            t += dt

            # I/O operations
            output_file.write(f"{nr_atoms}" + '\n')
            output_file.write("Comments" + '\n')

            for atom_name, atom in enumerate(pos):
                output_file.write(atom_string(atoms[atom_name], atom))

    return


def compute_force(pos, bonds, const_bonds, angles, const_angles, nr_atoms):
    """
    Computes the force on each atom, given the position and information from a 
    topology file.

    Input:
        pos: np array containing the positions
        bonds: index array of the bonds
        const_bonds: array containing the constant associated with each bond
        angles: index array of the angles
        const_angles: array containing the constant associated with each angle
        nr_atoms: number of atoms in the system
    Output:
        force_total: numpy array containing the force acting on each molecule

    NOTE: See also the implementation of read_topology in io_sim
    """
    force_total = np.zeros((nr_atoms, 3))

    # Forces due to bonds between atoms

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

    # Forces due to angles in molecules
    # If there are no angles in the molecule,
    # we just return
    if not angles:
        return force_total
    
    # The difference vectors we need for the angles
    # 
    # TODO: see if there is a way to combine these 
    # with the differences calculated for the bonds,
    # to avoid calculating some twice
    diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
    diff_2 = pos[angles[:,1]] - pos[angles[:,2]]

    ang = angle_between(diff_1, diff_2)
    
    # The constant we need for the force calculation
    mag_ang = np.multiply(-const_angles[:,0], ang - const_angles[:,1])

    # Calculate the direction vectors for the forces 
    # TODO: does cross return a unit vector already?
    angular_force_unit_1 = unit_vector(np.cross(np.cross(diff_1, diff_2), diff_1))
    angular_force_unit_2 = -unit_vector(np.cross(np.cross(diff_1, diff_2), diff_2))

    # Actually calculate the forces
    force_ang_1 = np.multiply(np.multiply(mag_ang, np.linalg.norm(diff_1, axis=1))[:, np.newaxis], angular_force_unit_1)
    force_ang_2 = np.multiply(np.multiply(mag_ang, np.linalg.norm(diff_2, axis=1))[:, np.newaxis], angular_force_unit_2)
    
    # Add them to the total force
    np.add.at(force_total, angles[:,0], force_ang_1)
    np.add.at(force_total, angles[:,2], force_ang_2)
    np.add.at(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    return force_total

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
    m = np.array([15.999, 1.00784, 1.00784, 15.999, 1.00784, 1.00784]) # amu
    dt = 0.001 # 0.1 ps
    T = 1 # 0.1 ps
    file_xyz = "data/water_top.xyz"
    file_top = "data/top.itp"
    file_out = "output/result.xyz"
    file_observable = "output/result_phase.csv"
    observable_function = None

    # Hydrogen file
    # m = np.array([1.00784, 1.00784]) # amu
    # dt = 0.001 # 0.1 ps
    # T = 1 # 0.1 ps
    # file_xyz = "data/hydrogen_top.xyz"
    # file_top = "data/hydrogen_top.itp"
    # file_out = "output/result_h2.xyz"
    # file_observable = "output/result_phase.csv"
    # observable_function = phase_space_h

    integration(m, dt, T, file_xyz, file_top, file_out, file_observable, observable_function)
