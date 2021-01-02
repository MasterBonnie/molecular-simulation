import numpy as np
from numba import vectorize, float64, jit, guvectorize, int16
import math
#import static_state

""" helper/misc. functions and constants """

atom_mass = {
    "O": 15.999,
    "H": 1.00784,
    "C": 12.011
}

water_patern = np.array([
                          [1.93617934,      2.31884508,      1.72261570],
                          [1.78931374,      3.24075634,      1.51114298],
                          [2.30448689,      1.98045541,      0.90160232]
                           ])

water_atoms = ["O", "H", "H"]

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
    return (angle*np.pi)/180.0

def unit_vector(matrix_):
    """ 
    Returns the unit vector of the vector.  
    if input is matrix does this for each row.
    """
    return matrix_/np.linalg.norm(matrix_, ord=2, axis=1, keepdims=True)

def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    # Calculates the row-wise dot product between
    # diff_1 and diff_2
    v1 = unit_vector(v1)
    v2 = unit_vector(v2)
    dot = np.einsum('ij,ij->i', v1, v2)

    # We then get the angle from this
    ang = np.arccos(np.clip(dot, -1.0,1.0))
    return ang

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


def neighbor_list(pos, molecule_to_atoms, centre_of_mass, r_cut, box_size, nr_atoms):
    """
    Returns which atoms are close, based on centres of mass that are withtin 
    r_cut distance of eachother 
    """
    dis_matrix = np.zeros((centre_of_mass.shape[0], centre_of_mass.shape[0]))
    distance_PBC_matrix(centre_of_mass - centre_of_mass[:, np.newaxis], box_size, r_cut, dis_matrix)

    return check_index(dis_matrix, molecule_to_atoms, nr_atoms)

@jit(nopython=True, cache=True)
def check_index(dis_matrix, molecule_to_atoms, nr_atoms):
    i_1, i_2 = np.nonzero(dis_matrix)

    indices = []

    for i in range(len(i_1)):
        atom_2_atom = molecule_to_atoms[i_1[i], i_2[i]]

        for connection in atom_2_atom:
            if not (connection[0] == nr_atoms or connection[1] == nr_atoms):
                indices.append(connection)

    return indices

@jit(nopython=True, cache=True)
def create_list(molecules, fixed_atom_length):
    """
    Creates list of what atoms are connected given molecules that are
    connected, i.e. matrix[i][j] is the cartesian product of the atoms 
    in molecule i and molecule j

    # NOTE: inefficient double loop, but we only call this once so it is
            not that bad
    """
    n = molecules.shape[0]
    matrix = np.zeros((n, n, fixed_atom_length**2, 2), dtype=np.int16)
    #matrix = [[0 for j in range(len(molecules))] for i in range(len(molecules))]

    for i in range(n):
        #print(f"working on creating list...  {(100*i)//(len(molecules))} %        ", end="\r")
        for j in range(n):
            if j > i:
                matrix[i][j] = cartesianprod(molecules[i],molecules[j])

    return matrix

@jit(nopython=True, cache=True)
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

@guvectorize([(float64[:,:,:], float64, float64, float64[:,:])], "(n,n,p),(),()->(n,n)",
            nopython=True, cache=True)
def distance_PBC_matrix(diff, box_length, r_cut, res):
    """
    Function to compute the distance of a matrix of vectors when considering
    periodic boundary conditions

    Input:
        diff: (n,n,3) numpy array of vectors
        box_length: length of the PBC box, in A
        res: (n,n) array which will be filled with the distances

    # TODO: combine this with the function below, seemed difficult to get working
    """

    for i in range(diff.shape[0]):
        for j in range(diff.shape[0]):
            if j > i:
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
                if (0 < length) and (length < r_cut):
                    res[i][j] = 1
                
@guvectorize([(float64[:,:], float64[:,:], float64, float64[:], float64[:,:])], "(n,p),(n,p),()->(n),(n,p)",
            nopython=True, cache=True)
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

if __name__ == "__main__":
    pass