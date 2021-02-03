import numpy as np
from numba import jit
from helper import unit_vector, distance_PBC, dot_product, angle_between_jit, add_jit, sv_mult, r_norm, cross


"""File for extracting/calculating information about a state at a particular time step"""

@jit(nopython=True, cache=True)
def centre_of_mass(pos, m, molecules):
    """
    Computes the centre of mass for each molecule

    params:
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
    """ Computes the kinetic energy of the system """
    dot = dot_product(v, v)
    summands = np.multiply(dot, m)
    # Conversion factor to get the right units
    cf = 1.6605*6.022e-1 
    return cf*0.5*np.sum(summands)

@jit(nopython=True, cache=True)
def potential_energy(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms,
                    box_size):
    """
        Calculates the potential energy of the system

        params:
            pos: np array of positions
            bonds:  a nr_bonds x 2 np array, whose rows
                    are the pair which have a bond
            const_bonds: a nr_bonds x 2 np array, whose
                    rows contain the constants for the
                    associated bond in the same row as bonds
            angles: a nr_angles x 3 np array, whose rows
                    specify the three atoms in an angle
            const_angles: a nr_angles x 2 np array, whose
                    rows contain the constants for the 
                    associated angle in the same row as in 
                    angles
            lj_atoms: a nr_lj x 2 np array, whose rows are the pairs
                    between a lj interaction
            lj_sigma: a nr_atoms x nr_atoms array, where index i j contains
                    the sigma const of the lj interaction between these atoms
            lj_eps: similar to lj_sigma, containing the epsilon variable
            dihedrals:
                    a nr_dihedrals x 4 np array, containing the indices of atoms in one
                    dihedral angle
            const_dihedrals: a nr_dihedrals x 4 np array, containing the associated constants
            nr_atoms: number of atoms in the simulation
            box_size: size of the simulation box, in A

        Output:
            energy: total energy of the system, in
            energy_bond: energy of the bonds in the system
            energy_angle: energy of the angles in the system
            energy_dihedral: energy of the dihedrals in the system
                            if none are present, equals 0

        TODO: Maybe this can be combined in some way with the compute force function, to save some time
    """
    energy = 0

    energy_bond = np.zeros((1))
    energy_angle = np.zeros((1))
    energy_dihedral = np.zeros((1))
    energy_lj = 0

    # Energy due to bonds 
    #----------------------------------
    if bonds is not None:
        # Difference vectors for the bonds, and the
        # distance between these atoms
        diff = pos[bonds[:,0]] - pos[bonds[:,1]]
        dis = r_norm(diff)

        # Calculate the energy between the atoms
        energy_bond = np.multiply(0.5*const_bonds[:,0], np.power((dis - const_bonds[:,1]),2))
        energy += np.sum(energy_bond)

    #----------------------------------
    # Energy due to angles
    #----------------------------------
    if angles is not None:
        # The difference vectors we need for the angles
        diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
        diff_2 = pos[angles[:,1]] - pos[angles[:,2]]
        ang = angle_between_jit(diff_1, diff_2)
        
        # The constant we need for the force calculation
        energy_angle = np.multiply(0.5*const_angles[:,0], np.power((ang - const_angles[:,1]),2))
        energy += np.sum(energy_angle)

    #----------------------------------
    # Energy due to LJ interactions
    #----------------------------------
    if lj_atoms is not None:
        if lj_atoms.shape[0] != 0:
            diff = np.zeros((lj_atoms.shape[0], 3))
            dis = np.zeros(diff.shape[0])
            distance_PBC(pos[lj_atoms[:,0]], pos[lj_atoms[:,1]],  box_size, dis, diff)
            energy_lj = lj_energy(dis, lj_atoms, lj_eps, lj_sigma, nr_atoms)
            energy += energy_lj

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

        psi = sign_angle*angle_between_jit(R,S) - np.pi  
        
        C_1 = const_dihedrals[:,0]
        C_2 = const_dihedrals[:,1]
        C_3 = const_dihedrals[:,2]
        C_4 = const_dihedrals[:,3]
        energy_dihedral = 0.5*(C_1*(1.0 + np.cos(psi)) + C_2*(1.0 - np.cos(2*psi)) + C_3*(1.0 + np.cos(3*psi)) + C_4*(1.0 - np.cos(4*psi)))
        energy += np.sum(energy_dihedral)

    return energy, np.sum(energy_bond), np.sum(energy_angle), np.sum(energy_dihedral), energy_lj
 
@jit(nopython=True, cache=True)
def compute_force(pos, bonds, const_bonds, angles, const_angles, lj_atoms, lj_sigma, lj_eps, dihedrals, const_dihedrals, nr_atoms,
                    box_size):
    """
        Computes the force on each atom, given the position and information from a 
        topology file.

        params:
            pos: np array of positions
            bonds:  a nr_bonds x 2 np array, whose rows
                    are the pair which have a bond
            const_bonds: a nr_bonds x 2 np array, whose
                    rows contain the constants for the
                    associated bond in the same row as bonds
            angles: a nr_angles x 3 np array, whose rows
                    specify the three atoms in an angle
            const_angles: a nr_angles x 2 np array, whose
                    rows contain the constants for the 
                    associated angle in the same row as in 
                    angles
            lj_atoms: a nr_lj x 2 np array, whose rows are the pairs
                    between a lj interaction
            lj_sigma: a nr_atoms x nr_atoms array, where index i j contains
                    the sigma const of the lj interaction between these atoms
            lj_eps: similar to lj_sigma, containing the epsilon variable
            dihedrals:
                    a nr_dihedrals x 4 np array, containing the indices of atoms in one
                    dihedral angle
            const_dihedrals: a nr_dihedrals x 4 np array, containing the associated constants
            molecules:
                    if fixed_atom_length is not 0:
                        a nr_molecules x fixed_atom_length np array containing
                        the index of atoms in one molecule
                    else:
                        a python list of numpy arrays containing the index of atoms
                        in one molecule
            nr_atoms: number of atoms in the simulation
            box_size: size of the simulation box, in A
        Output:
            force_total: numpy array containing the force acting on each molecule

        NOTE: See also the implementation of read_topology in io_sim, and the definitions of lj_sigma 
            and lj_eps in the simulator function in simulator.py
        NOTE: Some of these variables might be None, which signifies that there are for example no angles
            in the molecules we are simulating
    """
    force_total = np.zeros((nr_atoms, 3))


    # Forces due to bonds between atoms
    #----------------------------------
    if bonds is not None:
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
    if angles is not None:
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
        add_jit(force_total, angles[:,0], force_ang_1)
        add_jit(force_total, angles[:,2], force_ang_2)
        add_jit(force_total, angles[:,1], -(force_ang_1 + force_ang_2))

    #----------------------------------
    # Forces due to Lennard Jones interaction
    #----------------------------------
    #if lj_atoms is not empty
    if lj_atoms is not None:
        if lj_atoms.shape[0] != 0:

            diff = np.zeros((lj_atoms.shape[0], 3))
            dis = np.zeros(diff.shape[0])
            distance_PBC(pos[lj_atoms[:,0]], pos[lj_atoms[:,1]], box_size, dis, diff)

            force_1, force_2 = lj_force(dis, unit_vector(diff), lj_atoms, lj_eps, lj_sigma, nr_atoms)
            
            force_total += force_1
            force_total += force_2

    #----------------------------------
    # Forces due to dihedral angles
    #----------------------------------
    if dihedrals is not None:
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

        sign_angle = np.sign(dot_product(ij, f_l))

        kj_norm = unit_vector(kj)

        R = (ij) - sv_mult(dot_product(ij, kj_norm), kj_norm)
        S = (-kl) - sv_mult(dot_product(-kl, kj_norm), kj_norm)

        psi = sign_angle*angle_between_jit(R,S) - np.pi  

        # Derivative of the potential 
        magnitude = -0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.sin(3*psi) - 4*C_4*np.sin(4*psi))

        r_kj = r_norm(kj)

        force_i = -sv_mult(magnitude*(r_kj/r_norm(f_i)), unit_vector(f_i))
        force_l =  sv_mult(magnitude*(r_kj/r_norm(f_l)), unit_vector(f_l))

        term = sv_mult(np.reciprocal(r_kj**2), sv_mult(dot_product(ij, kj), force_i) - sv_mult(dot_product(kl, kj),force_l))

        force_j = -force_i + term
        force_k = -force_l - term

        add_jit(force_total, dihedrals[:,0], force_i)
        add_jit(force_total, dihedrals[:,1], force_j)
        add_jit(force_total, dihedrals[:,2], force_k)
        add_jit(force_total, dihedrals[:,3], force_l)

    return force_total

@jit(nopython=True, cache=True, fastmath=True)
def lj_force(dis, direction, lj_atoms, lj_eps, lj_sigma, nr_atoms):
    """ Computes force due to the lennard-jones interaction """
    force_1 = np.zeros((nr_atoms, 3))
    force_2 = np.zeros((nr_atoms, 3))

    for i in range(lj_atoms.shape[0]):
        # If this is the case, we are dealing with the "virtual particle"
        if lj_atoms[i,0] == nr_atoms:
            break
        if lj_atoms[i,1] == nr_atoms:
            break

        # first dividing and then raising to the power, to (hopefully)
        # prevent numerical errors.
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
    """ Computes the energy due to the lennard-jones interaction """
    energy = 0
    for i in range(lj_atoms.shape[0]):
        # If this is the case, we are dealing with the "virtual particle"
        if lj_atoms[i,0] == nr_atoms:
            break
        if lj_atoms[i,1] == nr_atoms:
            break

        term = lj_sigma[lj_atoms[i,0],lj_atoms[i,1]] / dis[i]
        term = np.power(term, 6)

        energy += lj_eps[lj_atoms[i,0],lj_atoms[i,1]]*(term**2 - term)

    return energy

@jit(nopython=True, cache=True)
def temperature(Ekin,N):
    """ converts kinetic energy to temperature """
    # This converts it to the correct unit
    conversion = 3*1.3806*6.022/1000
    T = Ekin/(N*conversion)
    return T

""" Slower versions of the functions above with the same name, see also helper.py """

def centre_of_mass_s(pos, m, molecules):
    """
    Computes the centre of mass for each molecule, non fixed-length version
    is slower then the above version
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
        M = np.sum(m[molecule])
        Mpos = np.sum(m[molecule, np.newaxis]*pos[molecule], axis = 0)
        centre_of_mass[i] = Mpos/M

    return centre_of_mass

if __name__ == "__main__":
    pass