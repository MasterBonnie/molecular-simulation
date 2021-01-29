import numpy as np
from numba import vectorize, float64, jit, guvectorize, int16, prange, int8, int32
import math
import perfplot

""" helper/misc. functions and constants """

# Atom constant
atom_mass = {
    "O": 15.999,
    "H": 1.00784,
    "C": 12.011
}

# Template for a water molecule
water_patern = np.array([
                          [1.93617934,      2.31884508,      1.72261570],
                          [1.78931374,      3.24075634,      1.51114298],
                          [2.30448689,      1.98045541,      0.90160232]
                           ])

water_atoms = ["O", "H", "H"]

# Template for an ethanol molecule
ethanol_patern = np.array([
                        [0.826028, -0.40038, -0.826028],
                        [1.42445, -1.03723, -0.171629],
                        [1.49617, 0.1448, -1.49617],
                        [0.171629, -1.03723, -1.42445],
                        [0.0, 0.55946, 0.0],
                        [-0.597, 1.20751, -0.657249],
                        [0.657249, 1.20751, 0.59706],
                        [-0.841514, -0.22767, 0.841514],
                        [-1.37647, 0.38153, 1.37647]
                            ])

ethanol_atoms = ["C","H","H","H","C","H","H","O","H"]

def angle_to_radian(angle):
    """
    converts angles to radians
    """
    return (angle*np.pi)/180.0

""" 
Replaced numpy functions, mostly using numba in order to gain some speed
"""

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
        a1, a2, a3 = vec1[i][0], vec1[i][1], vec1[i][2]
        b1, b2, b3 = vec2[i][0], vec2[i][1], vec2[i][2]
        result[i][0] = a2 * b3 - a3 * b2
        result[i][1] = a3 * b1 - a1 * b3
        result[i][2] = a1 * b2 - a2 * b1
    return result


@jit(nopython=True, cache=True)
def unit_vector(matrix):
    """
    returns unit vector of the input, row-wise
    if the row is small, we keep that vector the same

    this behaviour is used in the LJ force computation, 
    these entries are ignored anyway, but we dont want an error
    """
    res = np.zeros(matrix.shape)
    
    for i in range(matrix.shape[0]):
        div = norm(matrix[i,0], matrix[i,1], matrix[i,2])
        if div > 1e-3:
            for j in range(3):
                res[i,j] = matrix[i,j]/div

    return res

@jit(nopython=True, cache=True) 
def angle_between_jit(arg_1, arg_2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2', rowwise
    """
    # Calculates the row-wise dot product between
    # diff_1 and diff_2
    v1 = unit_vector(arg_1)
    v2 = unit_vector(arg_2)

    # We then get the angle from this
    dot = dot_product(v1,v2)
    dot_clipped = clip_jit(dot)
    ang = np.arccos(dot_clipped)
    return ang

@jit(nopython=True, cache=True)
def dot_product(arg_1, arg_2):
    res = np.zeros((arg_1.shape[0]))
    _dot_product(arg_1, arg_2, res)
    return res

@jit(nopython=True, cache=True)
def _dot_product(arg_1, arg_2, res):
    for i in range(arg_1.shape[0]):
        res[i] = arg_1[i][0]*arg_2[i][0] + arg_1[i][1]*arg_2[i][1] + arg_1[i][2]*arg_2[i][2]

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

@jit(nopython=True, cache=True)
def clip_jit(arg):
    res = np.zeros((arg.shape[0]))
    _clip_jit(arg, res)
    return res

@jit(nopython=True, cache=True)
def _clip_jit(arg, res):
    for i in range(arg.shape[0]):
        if arg[i] < -1:
            res[i] = -1
        elif arg[i] > 1:
            res[i] = 1
        else:
            res[i] = arg[i]

def random_unit_vector(const = 1):
    """
    returns random unit vector scaled with const
    """

    u = np.random.uniform(size=3)
    u /= np.linalg.norm(u) # normalize
    vRand = const*u

    return vRand
    
def atom_string(atom, pos):
    """
    returns string format correct for xyz file
    """
    return f"{atom} {pos[0]} {pos[1]} {pos[2]} \n"

def atom_name_to_mass(atoms):
    """ converts an atom name to its mass"""
    mass = [atom_mass[atom] for atom in atoms]
    return np.array(mass + [0])

@jit(nopython=True, cache=True)
def cartesianprod(x,y):
    res = np.zeros((x.shape[0]*y.shape[0],2), dtype=int16)

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            res[i*x.shape[0] + j] = np.array([x[i],y[j]], dtype=int16)
    return res

#@jit(nopython=True, cache=True)
def neighbor_list(molecule_to_atoms, centre_of_mass, r_cut, box_size, nr_atoms, atom_length):
    """
    Returns which atoms are close, based on centres of mass that are withtin 
    r_cut distance of eachother 
    """

    # print("Old calculations")
    dis_matrix = np.zeros((centre_of_mass.shape[0], centre_of_mass.shape[0]), np.int8)
    difference_matrix = matrix_difference(centre_of_mass)
    distance_PBC_matrix(difference_matrix, box_size, r_cut, dis_matrix)
    indices = check_index(dis_matrix, molecule_to_atoms, nr_atoms, atom_length)

    # nr_of_par = 7
    # max_number_atoms = 30

    # print("New calculations")
    # partition = partition_pos(centre_of_mass, box_size, nr_of_par, max_number_atoms)
    # indices_molecules = neighbor_list_creation(partition, centre_of_mass, nr_of_par, nr_atoms, box_size, r_cut)
    # indices = index_molecule_2_atom(indices_molecules, molecule_to_atoms, nr_atoms, atom_length)

    return indices

@jit(nopython=True, cache=True)
def matrix_difference(arg):
    res = np.zeros((arg.shape[0], arg.shape[0],3))

    for i in range(arg.shape[0]):
        for j in range(arg.shape[0]):
            for k in range(3):
                res[i][j][k] = arg[i][k] - arg[j][k]

    return res

@jit(nopython=True, cache=True)
def check_index(dis_matrix, molecule_to_atoms, nr_atoms, atom_length):
    i_1, i_2 = np.nonzero(dis_matrix)
    n = len(i_1)

    indices = np.zeros((n*atom_length*atom_length,2), dtype=np.int16)

    for i in range(n):
        atom_2_atom = molecule_to_atoms[i_1[i], i_2[i]]

        for j in range(atom_2_atom.shape[0]):
            indices[i*atom_length*atom_length + j] = atom_2_atom[j]

    return indices

@jit(nopython=True, cache=True)
def index_molecule_2_atom(indices, molecule_to_atoms, nr_atoms, atom_length):
    n = indices[0,0]
    indices_atoms = np.zeros((n*atom_length*atom_length,2), dtype=np.int16)

    for i in range(n):
        atom_2_atom = molecule_to_atoms[indices[i+1, 0], indices[i+1, 1]]

        for j in range(atom_2_atom.shape[0]):
            indices_atoms[i*atom_length*atom_length + j] = atom_2_atom[j]
    
    return indices_atoms

@jit(nopython=True, cache=True)
def partition_pos(pos, box_size, nr_of_par, max_number_atoms):
    partition_size = box_size/nr_of_par
    partition = np.zeros((nr_of_par, nr_of_par, nr_of_par, max_number_atoms), dtype=np.int16)

    for i in range(pos.shape[0]):
        x_box = math.floor(pos[i][0]/partition_size) % nr_of_par
        y_box = math.floor(pos[i][1]/partition_size) % nr_of_par
        z_box = math.floor(pos[i][2]/partition_size) % nr_of_par
        
        partition[x_box, y_box, z_box, 0] += 1
        partition[x_box, y_box, z_box][partition[x_box, y_box, z_box, 0]] = i

    return partition

# NOTE: this is slower then the naive method? I dont know why, but it is...
# the "slowness" seems to come from this function, not the other ones
@jit(nopython=True, cache=True)
def neighbor_list_creation(partition, pos, nr_of_par, nr_atoms, box_length, r_cut):
    """
    Creates neighbor list using the cell list computed in previous function
    """

    # Array in which the first element stores the current length of the 
    # array, and the remainder is indices of interactions between molecules
    indices = np.zeros((nr_atoms**2, 2), dtype=int32)

    for x in range(nr_of_par):
        for y in range(nr_of_par):
            for z in range(nr_of_par):
                partition_box = partition[x][y][z]
                for i in range(partition_box[0]):
                    index_p = partition_box[i+1]
                    point = pos[index_p]

                    for x_n in range(x-1, x+2):
                        x_n_p = x_n % nr_of_par
                        for y_n in range(y-1, y+2):
                            y_n_p = y_n % nr_of_par
                            for z_n in range(z-1, z+2):
                                z_n_p = z_n % nr_of_par
                                
                                other_partition_box = partition[x_n_p][y_n_p][z_n_p]

                                for j in range(other_partition_box[0]):
                                    index_o = other_partition_box[j+1]

                                    if index_p > index_o:
                                        other = pos[index_o]

                                        if check_distance(point, other, box_length, r_cut):
                                            indices[0][0] += 1
                                            indices[indices[0][0]] = [index_p, index_o]
    return indices

@jit(nopython=True, cache=True)
def create_list(molecules, fixed_atom_length):
    """
    Creates list of what atoms are connected given molecules that are
    connected, i.e. matrix[i][j] is the cartesian product of the atoms 
    in molecule i and molecule j

    """
    n = molecules.shape[0]
    matrix = np.zeros((n, n, fixed_atom_length**2, 2), dtype=np.int16)

    for i in range(n):
        for j in range(n):
            if i > j:
                matrix[i][j] = cartesianprod(molecules[i],molecules[j])

    return matrix

@jit(nopython=True, cache=True, fastmath=True)
def norm(x,y,z):
    """ 2-norm of a vector"""
    return math.sqrt(x*x + y*y + z*z)

@jit(nopython=True, cache=True)
def abs_min(x1,x2,x3):
    """
    returns minimum of the absolute value of a (x,y,z) triplet
    """
    res = x1
    if abs(res) > abs(x2):
        res = x2
    if abs(res) > abs(x3):
        res = x3

    return res

@jit(nopython=True, cache=True)
def distance_PBC(pos_1, pos_2, box_length, res, diff):
    """
    Function to compute the distance between two positions when considering
    periodic boundary conditions

    Input:
        pos_1: array of positions
        pos_2: array of positions, same length as pos_1
        box_length: length of the PBC box, in A
        res: array which will be filled with the distances
        diff: array which will be filled with the difference vectors

    """

    for i in range(diff.shape[0]):
        x = abs_min(pos_1[i][0] - pos_2[i][0], pos_1[i][0]  - pos_2[i][0] + box_length, pos_1[i][0]  - pos_2[i][0] - box_length)  

        y = abs_min(pos_1[i][1] - pos_2[i][1], 
                pos_1[i][1]  - pos_2[i][1] + box_length, 
                pos_1[i][1]  - pos_2[i][1] - box_length) 
        
        z = abs_min(pos_1[i][2] - pos_2[i][2], 
                pos_1[i][2]  - pos_2[i][2] + box_length, 
                pos_1[i][2]  - pos_2[i][2] - box_length)        
        
        diff[i] = np.array([x,y,z])
        res[i] = norm(x,y,z)

@jit(nopython=True, cache=True)
def distance_PBC_matrix(diff, box_length, r_cut, res):
    """
    Function to compute the distance of a matrix of vectors when considering
    periodic boundary conditions

    Input:
        diff: (n,n,3) numpy array of vectors
        box_length: length of the PBC box, in A
        r_cut: cutoff distance, in A
        res: (n,n) array which will be filled with the distances

    """

    for i in range(diff.shape[0]):
        for j in range(diff.shape[0]):
            if i > j:
                x = abs_min(diff[i][j][0], 
                        diff[i][j][0] + box_length, 
                        diff[i][j][0] - box_length)

                y = abs_min(diff[i][j][1], 
                        diff[i][j][1] + box_length, 
                        diff[i][j][1] - box_length) 
                
                z = abs_min(diff[i][j][2], 
                        diff[i][j][2] + box_length, 
                        diff[i][j][2] - box_length)       

                length = norm(x,y,z)
                if (length < r_cut):
                    res[i][j] = 1

@jit(nopython=True, cache=True)
def check_distance(pos_1, pos_2, box_length, r_cut):
    diff = pos_1 - pos_2

    x = abs_min(diff[0], 
        diff[0] + box_length, 
        diff[0] - box_length)

    y = abs_min(diff[1], 
        diff[1] + box_length, 
        diff[1] - box_length) 

    z = abs_min(diff[2], 
        diff[2] + box_length, 
        diff[2] - box_length)       

    length = norm(x,y,z)
    if (length < r_cut):
        return 1
    else:
        return 0

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

if __name__ == "__main__":
    
    pass