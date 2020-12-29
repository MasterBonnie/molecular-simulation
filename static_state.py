import numpy as np
from numba import vectorize, float64, jit, guvectorize, double
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
        molecule_np = np.array(molecule, dtype=np.int)
        M = np.sum(m[molecule_np])
        Mpos = np.sum(m[molecule_np, np.newaxis]*pos[molecule_np], axis = 0)
        centre_of_mass[i] = Mpos/M

    return centre_of_mass

def kinetic_energy(v, m):
    """
    Computes the kinetic energy of the system
    """
    dot = np.einsum('ij,ij->i', v, v)
    summands = np.multiply(dot, m)
    cf = 1.6605*6.022e-1 
    return cf*0.5*np.sum(summands)

def potential_energy(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms,
                    box_size):
    """
    Calculates the potential energy of the system
    """
    energy = 0
    # print(energy)
    # Energy due to bonds 
    #----------------------------------
    # Difference vectors for the bonds, and the
    # distance between these atoms
    diff = pos[bonds[:,0]] - pos[bonds[:,1]]
    dis = np.linalg.norm(diff, axis=1)

    # Calculate the energy between the atoms
    energy_bond = np.multiply(0.5*const_bonds[:,0], np.power((dis - const_bonds[:,1]),2))
    energy += np.sum(energy_bond)
    # print(energy_bond)

    #----------------------------------
    # Energy due to angles
    #----------------------------------
    # The difference vectors we need for the angles
    diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
    diff_2 = pos[angles[:,1]] - pos[angles[:,2]]
    ang = angle_between(diff_1, diff_2)
    
    # The constant we need for the force calculation
    energy_angle = np.multiply(0.5*const_angles[:,0], np.power((ang - const_angles[:,1]),2))
    energy += np.sum(energy_angle)
    # print(energy_angle)

    #----------------------------------
    # Energy due to LJ interactions
    #----------------------------------
    if lj_atoms.shape[0] != 0:
        diff = np.zeros((lj_atoms.shape[0], 3))
        dis = np.zeros(diff.shape[0])
        distance_PBC(pos[lj_atoms[:,0]], pos[lj_atoms[:,1]],  box_size, dis, diff)

        term = np.true_divide(lj_sigma[lj_atoms[:,0], lj_atoms[:,1]], dis)
        term_1 = np.power(term, 6)

        energy_lj = np.multiply(lj_eps[lj_atoms[:,0], lj_atoms[:,1]], np.power(term_1, 2) - term_1)
        energy += np.sum(energy_lj)

    if dihedrals is not None:
        i = pos[dihedrals[:,0]]
        j = pos[dihedrals[:,1]]
        k = pos[dihedrals[:,2]]
        l = pos[dihedrals[:,3]]

        # Using https://www.rug.nl/research/portal/files/3251566/c5.pdf
        # Equations (5.3a) for the dihederal angle

        f_l = cross(k - j, k - l)
        sign_angle = np.sign(np.einsum('ij,ij->i', i-j, f_l))

        R = (i - j) - np.einsum('ij,ij->i', i-j, unit_vector(k - j))[:, np.newaxis]*unit_vector(k - j)
        S = (l - k) - np.einsum('ij,ij->i', l-k, unit_vector(k - j))[:, np.newaxis]*unit_vector(k - j)

        psi = sign_angle*angle_between(R,S) - np.pi
        
        C_1 = const_dihedrals[:,0]
        C_2 = const_dihedrals[:,1]
        C_3 = const_dihedrals[:,2]
        C_4 = const_dihedrals[:,3]
        energy_dihedral = 0.5*(C_1*(1.0 + np.cos(psi)) + C_2*(1.0 - np.cos(2*psi)) + C_3*(1.0 + np.cos(3*psi)) + C_4*(1.0 - np.cos(4*psi)))
        energy += np.sum(energy_dihedral)
    else:
        energy_dihedral = np.array([])

    return energy, np.sum(energy_bond), np.sum(energy_angle), np.sum(energy_dihedral)

#https://gist.github.com/ufechner7/98bcd6d9915ff4660a10
@jit(nopython=True, cache=True)
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros((vec1.shape[0],3))
    return cross_(vec1, vec2, result)

@jit(nopython=True, cache=True)
def cross_(vec1, vec2, result):
    """ Calculate the cross product of array of 3d vectors. """
    for i in range(vec1.shape[0]):
        a1, a2, a3 = double(vec1[i][0]), double(vec1[i][1]), double(vec1[i][2])
        b1, b2, b3 = double(vec2[i][0]), double(vec2[i][1]), double(vec2[i][2])
        result[i][0] = a2 * b3 - a3 * b2
        result[i][1] = a3 * b1 - a1 * b3
        result[i][2] = a1 * b2 - a2 * b1
    return result

def compute_force(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, molecules, nr_atoms,
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
    cross_vector = cross(diff_1, diff_2)
    angular_force_unit_1 = unit_vector(cross(cross_vector, diff_1))
    angular_force_unit_2 = -unit_vector(cross(cross_vector, diff_2))

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
    #if lj_atoms is not empty
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

    #----------------------------------
    # Forces due to dihedral angles
    #----------------------------------
    if dihedrals is not None:
        # pass
        # Using https://www.rug.nl/research/portal/files/3251566/c5.pdf
        # Equations (5.11), (5.12), (5.21) and (5.22) and (5.3a) for the
        # dihederal angle

        # Stores the positions of the atoms
        # in the dihderal angle
        i = pos[dihedrals[:,0]]
        j = pos[dihedrals[:,1]]
        k = pos[dihedrals[:,2]]
        l = pos[dihedrals[:,3]]

        # Retrieves the coefficients 
        C_1 = const_dihedrals[:,0]
        C_2 = const_dihedrals[:,1]
        C_3 = const_dihedrals[:,2]
        C_4 = const_dihedrals[:,3]

        f_i = cross(i - j, k - j)
        f_l = cross(k - j, k - l)

        # The einsum takes the row-wise inner product of a matrix
        sign_angle = np.sign(np.einsum('ij,ij->i', i-j, f_l))

        R = (i - j) - np.einsum('ij,ij->i', i-j, unit_vector(k - j))[:, np.newaxis]*unit_vector(k - j)
        S = (l - k) - np.einsum('ij,ij->i', l-k, unit_vector(k - j))[:, np.newaxis]*unit_vector(k - j)

        psi = sign_angle*angle_between(R,S) - np.pi  

        # Derivative of the potential 
        magnitude = -0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.sin(3*psi) - 4*C_4*np.sin(4*psi))
        #print(magnitude)

        force_i = -(magnitude*np.linalg.norm(k - j, axis=1)/(np.linalg.norm(f_i, axis=1)))[:, np.newaxis]*unit_vector(f_i)
        force_l =  (magnitude*np.linalg.norm(k - j, axis=1)/(np.linalg.norm(f_l, axis=1)))[:, np.newaxis]*unit_vector(f_l)

        term = np.reciprocal(np.linalg.norm(j - k, axis=1)**2)[:, np.newaxis]*(np.einsum('ij,ij->i', i-j, k-j)[:, np.newaxis]*force_i - np.einsum('ij,ij->i', k - l, k - j)[:, np.newaxis]*force_l)

        force_j = -force_i + term
        force_k = -force_l - term

        # To check if the calculations are correct
        # This is indeed 0 when running the simulation

        # Middle of bond jk
        #o = (j+k)/2.0
        #torque = cross(i - o, force_i) + cross(j - o, force_j) + cross(k - o, force_k) + cross(l - o, force_l)

        np.add.at(force_total, dihedrals[:,0], force_i)
        np.add.at(force_total, dihedrals[:,1], force_j)
        np.add.at(force_total, dihedrals[:,2], force_k)
        np.add.at(force_total, dihedrals[:,3], force_l)
    #print(force_total)
    return force_total


#@guvectorize([(float64[:,:], float64, float64[:,:])], "(n,p), ()->(n,p)",
#            nopython=True, cache=True)
def calculate_displacement(centres_of_mass, box_size, res):
    for i in range(centres_of_mass.shape[0]):
        for j in range(3):
            if centres_of_mass[i][j] < 0:
                res[i][j] = box_size
            elif centres_of_mass[i][j] > box_size:
                res[i][j] = - box_size

#@jit(nopython=True, cache=True)
def project_pos(centres_of_mass, box_size, pos, molecules):
    
    displacement = np.zeros(centres_of_mass.shape)
    calculate_displacement(centres_of_mass, box_size, displacement)

    for i, molecule in enumerate(molecules):
        molecule_np = np.array(molecule, dtype=np.int)
        pos[molecule_np] += displacement[i]

def temperature(Ekin,N):
    conversion = 3*1.3806*6.022/1000
    T = Ekin/(N*conversion)
    return T

if __name__ == "__main__":
    pos = np.array([[1.,1.,1.],
                     [2.5, 1.9, 1.9],
                     [1.9, 1.9, 1.9]])
    molecules = [[0], [1,2]]
    m = np.array([1.,2.,3.])
    box_size = 2
    # com = centre_of_mass(pos, m, molecules)
    # calculate_displacement(com, box_size, np.zeros(com.shape))
    # project_pos(com, box_size, pos, molecules)
    