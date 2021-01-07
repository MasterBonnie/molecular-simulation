import numpy as np
from numba import vectorize, float64, jit, guvectorize, double, prange, int32
from helper import unit_vector, angle_between, distance_PBC, norm, dot_product, angle_between_jit


"""File for extracting/calculating information about a state at a particular time step"""

# TODO: rewrite this as numba function, using jit
# This might be difficult as numba does not like lists of lists, such as molecules
# otherwise rewrite as numpy function, without the loop
@jit(nopython=True, cache=True)
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
    centre_of_mass = np.zeros((molecules.shape[0], 3))

    for i in range(molecules.shape[0]):
        molecule = molecules[i]
        M = np.sum(m[molecule])

        Mpos = np.zeros((3))
        for j in molecule:
            Mpos += m[j] * pos[j]

        centre_of_mass[i] = Mpos/M

    return centre_of_mass

@jit(nopython=True, cache=True)
def kinetic_energy(v, m):
    """
    Computes the kinetic energy of the system
    """
    dot = dot_product(v, v)
    summands = np.multiply(dot, m)
    cf = 1.6605*6.022e-1 
    return cf*0.5*np.sum(summands)

@jit(nopython=True, cache=True)
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
    dis = r_norm(diff)

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
    ang = angle_between_jit(diff_1, diff_2)
    
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
        energy += lj_energy(dis, lj_atoms, lj_eps, lj_sigma, nr_atoms)

    if dihedrals is not None:
        i = pos[dihedrals[:,0]]
        j = pos[dihedrals[:,1]]
        k = pos[dihedrals[:,2]]
        l = pos[dihedrals[:,3]]

        # Using https://www.rug.nl/research/portal/files/3251566/c5.pdf
        # Equations (5.3a) for the dihederal angle

        ij = i - j
        kj = k - j
        kl = k - l

        f_l = cross(kj, kl)
        sign_angle = np.sign(dot_product(ij, f_l))

        kj_norm = unit_vector(kj)

        R = (ij) - sv_mult(dot_product(ij, kj_norm), kj_norm)
        S = (-kl) - sv_mult(dot_product(-kl, kj_norm), kj_norm)

        psi = sign_angle*angle_between(R,S) - np.pi  
        
        C_1 = const_dihedrals[:,0]
        C_2 = const_dihedrals[:,1]
        C_3 = const_dihedrals[:,2]
        C_4 = const_dihedrals[:,3]
        energy_dihedral = 0.5*(C_1*(1.0 + np.cos(psi)) + C_2*(1.0 - np.cos(2*psi)) + C_3*(1.0 + np.cos(3*psi)) + C_4*(1.0 - np.cos(4*psi)))
        energy += np.sum(energy_dihedral)
    else:
        energy_dihedral = np.zeros((1))

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
 

def compute_force_old(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms,
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
    # np.add.at(force_total, bonds[:,0], force)
    # np.add.at(force_total, bonds[:,1], -force)

    add_jit(force_total, bonds[:,0], force)
    add_jit(force_total, bonds[:,1], -force)

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
    cross_vector = cross(diff_1, diff_2)
    angular_force_unit_1 = unit_vector(cross(cross_vector, diff_1))
    angular_force_unit_2 = -unit_vector(cross(cross_vector, diff_2))

    # Actually calculate the forces
    force_ang_1 = np.true_divide(mag_ang, dis_1)[:, np.newaxis]*angular_force_unit_1
    force_ang_2 = np.true_divide(mag_ang, dis_2)[:, np.newaxis]*angular_force_unit_2

    # Add them to the total force
    # np.add.at(force_total, angles[:,0], force_ang_1)
    # np.add.at(force_total, angles[:,2], force_ang_2)
    # np.add.at(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    add_jit(force_total, angles[:,0], force_ang_1)
    add_jit(force_total, angles[:,2], force_ang_2)
    add_jit(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    #----------------------------------
    # Forces due to Lennard Jones interaction
    #----------------------------------
    #if lj_atoms is not empty
    if lj_atoms.shape[0] != 0:

        diff = np.zeros((lj_atoms.shape[0], 3))
        dis = np.zeros(diff.shape[0])
        distance_PBC(pos[lj_atoms[:,0]], pos[lj_atoms[:,1]], box_size, dis, diff)

        # term = np.true_divide(lj_sigma[lj_atoms[:,0], lj_atoms[:,1]], dis)
        # term_1 = 2*np.power(term, 12)
        # term_2 = -1*np.power(term, 6)

        # magnitudes = 6*np.multiply(np.true_divide(lj_eps[lj_atoms[:,0], lj_atoms[:,1]], dis), term_1 + term_2)
        # force = magnitudes[:, np.newaxis]*unit_vector(diff)

        force_1, force_2 = lj_force(dis, unit_vector(diff), lj_atoms, lj_eps, lj_sigma, nr_atoms)
        
        force_total += force_1
        force_total += force_2

        # add_jit(force_total, lj_atoms[:,0], force)
        # add_jit(force_total, lj_atoms[:,1], -force)

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

        ij = i - j
        kj = k - j
        kl = k - l

        f_i = cross(ij, kj)
        f_l = cross(kj, kl)

        # The einsum takes the row-wise inner product of a matrix
        sign_angle = np.sign(np.einsum('ij,ij->i', i-j, f_l))

        kj_norm = unit_vector(kj)

        R = (ij) - np.einsum('ij,ij->i', ij, kj_norm)[:, np.newaxis]*kj_norm
        S = (-kl) - np.einsum('ij,ij->i', -kl, kj_norm)[:, np.newaxis]*kj_norm

        psi = sign_angle*angle_between(R,S) - np.pi  

        # Derivative of the potential 
        magnitude = -0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.sin(3*psi) - 4*C_4*np.sin(4*psi))

        r_kj = np.linalg.norm(kj, axis=1)

        force_i = -(magnitude*r_kj/(np.linalg.norm(f_i, axis=1)))[:, np.newaxis]*unit_vector(f_i)
        force_l =  (magnitude*r_kj/(np.linalg.norm(f_l, axis=1)))[:, np.newaxis]*unit_vector(f_l)

        term = np.reciprocal(r_kj**2)[:, np.newaxis]*(np.einsum('ij,ij->i', ij, kj)[:, np.newaxis]*force_i - np.einsum('ij,ij->i', kl, kj)[:, np.newaxis]*force_l)

        force_j = -force_i + term
        force_k = -force_l - term

        # np.add.at(force_total, dihedrals[:,0], force_i)
        # np.add.at(force_total, dihedrals[:,1], force_j)
        # np.add.at(force_total, dihedrals[:,2], force_k)
        # np.add.at(force_total, dihedrals[:,3], force_l)

        add_jit(force_total, dihedrals[:,0], force_i)
        add_jit(force_total, dihedrals[:,1], force_j)
        add_jit(force_total, dihedrals[:,2], force_k)
        add_jit(force_total, dihedrals[:,3], force_l)

    return force_total

@jit(nopython=True, cache=True)
def compute_force(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms,
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
    dis = r_norm(diff)

    # Calculate the forces between the atoms
    magnitudes = np.multiply(-const_bonds[:,0], dis - const_bonds[:,1])
    force = sv_mult(magnitudes, unit_vector(diff))
    
    # Add them to the total force
    # np.add.at(force_total, bonds[:,0], force)
    # np.add.at(force_total, bonds[:,1], -force)

    add_jit(force_total, bonds[:,0], force)
    add_jit(force_total, bonds[:,1], -force)

    #----------------------------------
    # Forces due to angles in molecules
    #----------------------------------
    # The difference vectors we need for the angles
    diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
    dis_1 = r_norm(diff_1)
    diff_2 = pos[angles[:,1]] - pos[angles[:,2]]
    dis_2 = r_norm(diff_2)
    ang = angle_between_jit(diff_1, diff_2)
    
    # The constant we need for the force calculation
    mag_ang = np.multiply(-const_angles[:,0], ang - const_angles[:,1])

    # Calculate the direction vectors for the forces 
    cross_vector = cross(diff_1, diff_2)
    angular_force_unit_1 = unit_vector(cross(cross_vector, diff_1))
    angular_force_unit_2 = -unit_vector(cross(cross_vector, diff_2))

    # Actually calculate the forces
    force_ang_1 = sv_mult(mag_ang / dis_1, angular_force_unit_1)
    force_ang_2 = sv_mult(mag_ang / dis_2, angular_force_unit_2)

    # Add them to the total force
    # np.add.at(force_total, angles[:,0], force_ang_1)
    # np.add.at(force_total, angles[:,2], force_ang_2)
    # np.add.at(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    add_jit(force_total, angles[:,0], force_ang_1)
    add_jit(force_total, angles[:,2], force_ang_2)
    add_jit(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    #----------------------------------
    # Forces due to Lennard Jones interaction
    #----------------------------------
    #if lj_atoms is not empty
    if lj_atoms.shape[0] != 0:

        diff = np.zeros((lj_atoms.shape[0], 3))
        dis = np.zeros(diff.shape[0])
        distance_PBC(pos[lj_atoms[:,0]], pos[lj_atoms[:,1]], box_size, dis, diff)

        # term = np.true_divide(lj_sigma[lj_atoms[:,0], lj_atoms[:,1]], dis)
        # term_1 = 2*np.power(term, 12)
        # term_2 = -1*np.power(term, 6)

        # magnitudes = 6*np.multiply(np.true_divide(lj_eps[lj_atoms[:,0], lj_atoms[:,1]], dis), term_1 + term_2)
        # force = magnitudes[:, np.newaxis]*unit_vector(diff)

        force_1, force_2 = lj_force(dis, unit_vector(diff), lj_atoms, lj_eps, lj_sigma, nr_atoms)
        
        force_total += force_1
        force_total += force_2

        # add_jit(force_total, lj_atoms[:,0], force)
        # add_jit(force_total, lj_atoms[:,1], -force)

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

        ij = i - j
        kj = k - j
        kl = k - l

        f_i = cross(ij, kj)
        f_l = cross(kj, kl)

        # The einsum takes the row-wise inner product of a matrix
        sign_angle = np.sign(dot_product(ij, f_l))

        kj_norm = unit_vector(kj)

        R = (ij) - sv_mult(dot_product(ij, kj_norm), kj_norm)
        S = (-kl) - sv_mult(dot_product(-kl, kj_norm), kj_norm)

        psi = sign_angle*angle_between(R,S) - np.pi  

        # Derivative of the potential 
        magnitude = -0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.sin(3*psi) - 4*C_4*np.sin(4*psi))

        r_kj = r_norm(kj)

        force_i = -sv_mult(magnitude*(r_kj/r_norm(f_i)), unit_vector(f_i))
        force_l =  sv_mult(magnitude*(r_kj/r_norm(f_l)), unit_vector(f_l))

        term = sv_mult(np.reciprocal(r_kj**2), sv_mult(dot_product(ij, kj), force_i) - sv_mult(dot_product(kl, kj),force_l))

        force_j = -force_i + term
        force_k = -force_l - term

        # np.add.at(force_total, dihedrals[:,0], force_i)
        # np.add.at(force_total, dihedrals[:,1], force_j)
        # np.add.at(force_total, dihedrals[:,2], force_k)
        # np.add.at(force_total, dihedrals[:,3], force_l)

        add_jit(force_total, dihedrals[:,0], force_i)
        add_jit(force_total, dihedrals[:,1], force_j)
        add_jit(force_total, dihedrals[:,2], force_k)
        add_jit(force_total, dihedrals[:,3], force_l)

    return force_total

@jit(nopython=True, cache=True)
def sv_mult(arg_1, arg_2):
    res = np.zeros(arg_2.shape)
    for i in range(arg_1.shape[0]):
        for j in range(3):
            res[i][j] = arg_2[i][j]*arg_1[i]
    return res
    
@jit(nopython=True, cache=True)
def r_norm(arg_1):
    res = np.zeros((arg_1.shape[0]))
    for i in range(arg_1.shape[0]):
        res[i] = norm(arg_1[i][0], arg_1[i][1], arg_1[i][2])
    return res

# @guvectorize([(float64[:,:], float64[:])], "(n,p)->(n)",
#             nopython=True, cache=True)
@jit(nopython=True, cache=True)
def _r_norm(arg_1, res):
    for i in range(arg_1.shape[0]):
        res[i] = norm(arg_1[i][0], arg_1[i][1], arg_1[i][2])

# @guvectorize([(float64[:], float64[:,:], float64[:,:])], "(n),(n,p)->(n,p)",
#             nopython=True, cache=True)
@jit(nopython=True, cache=True)
def _scalar_vector_mult(arg_1, arg_2, res):
    for i in range(arg_1.shape[0]):
        for j in range(3):
            res[i][j] = arg_2[i][j]*arg_1[i]

@jit(nopython=True, cache=True, fastmath=True)
def lj_force(dis, direction, lj_atoms, lj_eps, lj_sigma, nr_atoms):
    force_1 = np.zeros((nr_atoms, 3))
    force_2 = np.zeros((nr_atoms, 3))

    for i in range(lj_atoms.shape[0]):
        if lj_atoms[i,0] == nr_atoms:
            break
        if lj_atoms[i,1] == nr_atoms:
            break

        term = lj_sigma[lj_atoms[i,0],lj_atoms[i,1]] / dis[i]
        term_1 = 2*np.power(term, 12)
        term_2 = -1*np.power(term, 6)

        magnitudes = 6*(lj_eps[lj_atoms[i,0],lj_atoms[i,1]]/dis[i])*(term_1 + term_2)

        for j in range(3):
            force_1[lj_atoms[i,0],j] += magnitudes*direction[i,j]
            force_2[lj_atoms[i,1],j] += -magnitudes*direction[i,j]
    
    return force_1, force_2

@jit(nopython=True, cache=True, fastmath=True)
def lj_energy(dis, lj_atoms, lj_eps, lj_sigma, nr_atoms):
    energy = 0
    for i in range(lj_atoms.shape[0]):
        if lj_atoms[i,0] == nr_atoms:
            break
        if lj_atoms[i,1] == nr_atoms:
            break

        term = lj_sigma[lj_atoms[i,0],lj_atoms[i,1]] / dis[i]
        term = np.power(term, 6)

        energy += lj_eps[lj_atoms[i,0],lj_atoms[i,1]]*(term**2 - term)

    return energy

@jit(nopython=True, cache=True)
def add_jit(total, index, addition):
    for i in range(index.shape[0]):
        for j in range(3):
            total[index[i]][j] += addition[i][j]

@jit(nopython=True, cache=True)
def calculate_displacement(centres_of_mass, box_size, res):
    for i in range(centres_of_mass.shape[0]):
        for j in range(3):
            if centres_of_mass[i][j] < 0:
                res[i][j] = box_size
            elif centres_of_mass[i][j] > box_size:
                res[i][j] = - box_size

@jit(nopython=True, cache=True)
def project_pos(centres_of_mass, box_size, pos, molecules):
    
    displacement = np.zeros(centres_of_mass.shape)
    calculate_displacement(centres_of_mass, box_size, displacement)

    for i, molecule in enumerate(molecules):
        pos[molecule] += displacement[i]

    pos[-1] = np.array([0,0,0])

@jit(nopython=True, cache=True)
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
    