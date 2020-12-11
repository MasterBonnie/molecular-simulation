import numpy as np
import math
from collections import deque

import io_sim
from helper import atom_string, random_unit_vector, unit_vector, angle_between, atom_name_to_mass
import integrators

def integration(dt, T, file_xyz, file_top, file_out, file_observable, 
                observable_function = None, integrator="vv", write_output = True):
    """
    Numerical integration using either the euler algorithm,
    velocity verlet (vv) algorithm, or the verlet (v) algorithm.
    Requires a topology file, i.e. cannot pas constant through the function
    anymore.

    Input:
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
        write_output: boolean, whether the computed pos needs to be written to file_out 
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
    bonds, const_bonds, angles, const_angles, lj, const_lj, molecules = io_sim.read_topology(file_top)
    # Mass is given in amu
    m = atom_name_to_mass(atoms)

    lj_sigma = np.zeros(nr_atoms)
    lj_sigma[lj] = const_lj[:,0]
    lj_sigma = 0.5*(lj_sigma + lj_sigma[:, np.newaxis])

    lj_eps = np.zeros(nr_atoms)
    lj_eps[lj] = const_lj[:, 1]
    lj_eps = np.sqrt(lj_eps*lj_eps[:, np.newaxis])

    # Create the conversion from molecules to atoms
    # This is for the Lennard-Jones potential
    molecule_to_atoms = create_list(molecules)

    # Random initial velocity
    v = 1*unit_vector(np.random.uniform(size=[nr_atoms,3]))

    # Open the output file
    # I dont think it matters too much to have these files open during the calculation
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
        if integrator == "v":
            pos_old = pos
            f = cf*compute_force(pos, bonds, const_bonds, angles,
                            const_angles, lj, const_lj, molecules, nr_atoms)

            # If we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)
            
            pos, v, _ = integrators.integrator_euler(pos, v, f, m, dt)
            
            t += dt

        while t < T:

            # Then we compute the neighbor list
            # This list contains arrays of the form
            # [i,j], which are molecules which need a Lennard Jones 
            # interaction
            nl = neighbor_list(pos, m, molecules, r_cut)
            # This contains the pure atom interactions
            lj_atoms = np.concatenate([molecule_to_atoms[i[0]][i[1]] for i in nl])

            # Compute the force on the entire system
            f = cf*compute_force(pos, bonds, const_bonds, angles,
                            const_angles, lj, const_lj, molecules, nr_atoms)

            # if we want to calculate something
            if observable_function:
                observable_function(pos, v, f, obs_file)

            # Based on the integrator we update the current pos and v
            if integrator == "euler":
                pos, v, update = integrators.integrator_euler(pos, v, f, m, dt)

            elif integrator == "vv":
                pos, update = integrators.integrator_velocity_verlet_pos(pos, v, f, m, dt)
                f_new = cf*compute_force(pos, bonds, const_bonds, angles,
                                    const_angles, lj, const_lj, molecules, nr_atoms)
                v = integrators.integrator_velocity_verlet_vel(v, f, f_new, m, dt)

            elif integrator == "v":
                pos_old, (pos, update) = pos, integrators.integrator_verlet_pos(pos, pos_old, f, m, dt)
                # This v is the velocity at the previous timestep
                v = integrators.integrator_verlet_vel(pos, pos_old, dt) 
            
            t += dt

            # Check if we need to update the verlet list
            # if update_displacement(displacement, update) > r_m - r_cut:
            #     displacement = np.zeros(())
            #     create_verlet_list()

            # I/O operations
            if write_output:
                output_file.write(f"{nr_atoms}" + '\n')
                output_file.write("Comments" + '\n')

                for atom_name, atom in enumerate(pos):
                    output_file.write(atom_string(atoms[atom_name], atom))

    return

def centreOfMass(pos,m,molecules):
    M = np.sum(m[molecules], axis = 1)
    Mpos = np.sum(m[molecules,np.newaxis]*pos[molecules], axis = 1)
    Cm = Mpos/M[:,np.newaxis]
    return Cm
 
# TODO: faster version on stackoverflow? Numba using lists in python?
def cartesianprod(x,y):
    Cp = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return Cp    

def update_displacement(displacement, update):
    displacement += update    
    distances = np.linalg.norm(displacement, axis=1)
    distances = np.sort(distances)
    return distances[-1] + distances[-2]

def neighbor_list(pos, m, molecules, r_cut):
    pos_matrix = centreOfMass(pos, m, molecules)
    dis_matrix = np.linalg.norm(pos_matrix - pos_matrix[:, np.newaxis], axis = 2)
    adj = (0 < dis_matrix) & (dis_matrix < r_cut)
    # TODO: maybe this can be done better?
    # Prevents the dubble pairs to appear
    iu1 = np.tril_indices(adj.shape[0])
    adj[iu1] = 0
    return np.transpose(np.nonzero(adj))

# Maybe we can compile this using numba, or even pre-compile this, as we only call it once
def create_list(molecules):
    """
    Creates list of what atoms are connected given molecules that are
    connected
    """
    matrix = [[0 for j in range(molecules.shape[0])] for i in range(molecules.shape[0])]

    for i in range(molecules.shape[0]):
        for j in range(molecules.shape[0]):
            if j > i:
                matrix[i][j] = cartesianprod(molecules[i], molecules[j])

    return matrix

#@jit
def norm(x,y,z):
    return math.sqrt(x*x + y*y + z*z)

#@jit
def compute_distance_PBC(pos_1, pos_2, box_length):
    """
    Function to compute the distance between two positions when considering
    periodic boundary conditions
    """
    res = np.zeros(pos_1.shape[0])

    for i in range(pos_1.shape[0]):
        x = min([pos_1[i][0] - pos_2[i][0], 
                 pos_1[i][0] - pos_2[i][0] + box_length, 
                 pos_1[i][0] - pos_2[i][0] - box_length])   

        y = min([pos_1[i][1] - pos_2[i][1], 
                 pos_1[i][1] - pos_2[i][1] + box_length, 
                 pos_1[i][1] - pos_2[i][1] - box_length])        

        z = min([pos_1[i][2] - pos_2[i][2], 
                 pos_1[i][2] - pos_2[i][2] + box_length, 
                 pos_1[i][2] - pos_2[i][2] - box_length])        
        
        res[i] = norm(x,y,z)

    return res

def create_verlet_list(pos, dis_c, dis_s, nr_atoms, box_length):
    """
    Function to create verlet list given positions
    """
    vl = []
    vl.append(0)
    # For every atom:
    start_list = 0
    for i in range(pos.shape[0]):
        nneighbors = 0
        for j in range(pos.shape[0]):
            if i == j:
                continue
            dis = compute_distance_PBC(pos[i], pos[j], box_length)

            if (dis < dis_s):
                nneighbors += 1
                vl.append(j)
            vl[start_list] = nneighbors
            start_list = nneighbors + 1

    return vl


def compute_force(pos, bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, nr_atoms):
    """
    Computes the force on each atom, given the position and information from a 
    topology file.

    Input:
        pos: np array containing the positions
        bonds: index array of the bonds
        const_bonds: array containing the constant associated with each bond
        angles: index array of the angles
        const_angles: array containing the constant associated with each angle
        lj: index array of Lennard Jones interaction
        const_lj: array containing the constant associated with each lj interaction
        nr_atoms: number of atoms in the system
    Output:
        force_total: numpy array containing the force acting on each molecule

    NOTE: See also the implementation of read_topology in io_sim
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
    # If there are no angles in the molecule,
    # we just return
    if angles is None:
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
    force_ang_1 = np.multiply(np.true_divide(mag_ang, np.linalg.norm(diff_1, axis=1))[:, np.newaxis], angular_force_unit_1)
    force_ang_2 = np.multiply(np.true_divide(mag_ang, np.linalg.norm(diff_2, axis=1))[:, np.newaxis], angular_force_unit_2)
    
    # Add them to the total force
    np.add.at(force_total, angles[:,0], force_ang_1)
    np.add.at(force_total, angles[:,2], force_ang_2)
    np.add.at(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    #----------------------------------
    # Forces due to Lennard Jones interaction
    #----------------------------------
    # Difference vectors
    diff = pos[lj[:,0]] - pos[lj[:,1]]
    dis = np.linalg.norm(diff, axis=1)

    #x1 = np.true_divide(4*const_lj[:,1], dis)
    #x2 = np.true_divide(const_lj[:,0], dis)

    # TODO: Fix this, we need derivative not this one!!!
    #   this is energy not force
    magnitudes = np.multiply(4*const_lj[:,1], np.power(x2, 12) - np.power(x2, 6))
    #print(magnitudes)
    force = magnitudes[:, np.newaxis]*diff
    #print(force)

    np.add.at(force_total, lj[:,0], force)
    np.add.at(force_total, lj[:,1], -force)

    #print(force_total)
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
    dt = 0.001 # 0.1 ps
    T = 10 # 0.1 ps
    rcut = 5 # A
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

    integration(dt, T, file_xyz, file_top, file_out, file_observable, observable_function)
