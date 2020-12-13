import numpy as np
from numba import vectorize, float64, jit, guvectorize
import math

""" helper/misc. functions and constants """

_atom_mass = {
    "O": 15.999,
    "H": 1.00784,
}

def unit_vector(vector):
    """ 
    Returns the unit vector of the vector.  
    if input is matrix does this for each row.
    """
    # OLD return vector / np.linalg.norm(vector)
    return vector/np.linalg.norm(vector, ord=2, axis=1, keepdims=True)


def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    # Calculates the row-wise dot product between
    # diff_1 and diff_2
    dot = np.einsum('ij,ij->i', v1, v2)
    print(dot)

    # We then get the angle from this
    ang = np.arccos(dot)
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
    return atom + " " +  np.array2string(pos, separator=" ")[1:-1] + "\n"   

def atom_name_to_mass(atoms):
    # Not particulary fast, but good enough 
    mass = [_atom_mass[atom] for atom in atoms]
    return np.array(mass)

def centreOfMass(pos,m,molecules):
    M = np.sum(m[molecules], axis = 1)
    Mpos = np.sum(m[molecules,np.newaxis]*pos[molecules], axis = 1)
    Cm = Mpos/M[:,np.newaxis]
    return Cm

# TODO: faster version on stackoverflow? Numba using lists in python?
def cartesianprod(x,y):
    Cp = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return Cp    

def neighbor_list(pos, molecules, centre_of_mass, r_cut):
    dis_matrix = np.zeros((centre_of_mass.shape[0], centre_of_mass.shape[0]))
    distance_PBC_matrix(centre_of_mass - centre_of_mass[:, np.newaxis], box_size, dis_matrix)

    adj = (0 < dis_matrix) & (dis_matrix < r_cut)
    return np.transpose(np.nonzero(adj))

# TODO: Maybe we can compile this using numba, or even pre-compile this, as we only call it once?
# Dont know probably doesnt matter
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

@vectorize([float64(float64, float64)],
            nopython=True, cache=True)
def project_box(pos, box_size):
    if pos < 0:
        pos += box_size
    elif pos > box_size:
        pos -= box_size
    return pos

@jit(nopython=True, cache=True)
def project_back(molecules, centre_of_mass, pos, box_size):
    for molecule, i in enumerate(centre_of_mass):
        translation = np.zeros(3)
        
        if molecule[0] < 0:
            translation[0] = box_size
        elif molecule[0] > box_size:
            translation[0] = -box_size
        if molecule[1] < 0:
            translation[1] = box_size
        elif molecule[1] > box_size:
            translation[1] = -box_size
        if molecule[2] < 0:
            translation[2] = box_size
        elif molecule[2] > box_size:
            translation[2] = -box_size
        
        pos[molecule[i]] += translation


@jit(nopython=True, cache=True)
def norm(x,y,z):
    return math.sqrt(x*x + y*y + z*z)

@vectorize([float64(float64)],
            nopython=True, cache=True)
def abs_vec(number):
    if number < 0:
        return -number
    else:
        return number

@guvectorize([(float64[:,:,:], float64, float64[:,:])], "(n,n,p),()->(n,n)",
            nopython=True, cache=True)
def distance_PBC_matrix(diff, box_length, res):
    """
    Function to compute the distance between two positions when considering
    periodic boundary conditions
    """

    for i in range(diff.shape[0]):
        for j in range(diff.shape[0]):
            if j > i:
                x = min(abs_vec(np.array([diff[i][j][0], 
                        diff[i][j][0] + box_length, 
                        diff[i][j][0] - box_length])))  

                y = min(abs_vec(np.array([diff[i][j][1], 
                        diff[i][j][1] + box_length, 
                        diff[i][j][1] - box_length])))  
                
                z = min(abs_vec(np.array([diff[i][j][2], 
                        diff[i][j][2] + box_length, 
                        diff[i][j][2] - box_length])))       

                res[i][j] = norm(x,y,z)


@guvectorize([(float64[:,:], float64, float64[:])], "(n,p),()->(n)",
            nopython=True, cache=True)
def distance_PBC(diff, box_length, res):
    """
    Function to compute the distance between two positions when considering
    periodic boundary conditions
    """

    for i in range(diff.shape[0]):
        x = min(abs_vec(np.array([diff[i][0], 
                diff[i][0] + box_length, 
                diff[i][0] - box_length])))  

        y = min(abs_vec(np.array([diff[i][1], 
                diff[i][1] + box_length, 
                diff[i][1] - box_length])))  
        
        z = min(abs_vec(np.array([diff[i][2], 
                diff[i][2] + box_length, 
                diff[i][2] - box_length])))       

        res[i] = norm(x,y,z)

if __name__ == "__main__":
    box_size = 2.5
    test_array = np.array([[3.,3.,3.],
                            [1.,1.,1.],
                            [1.,2.,3.],
                            [1., 2.4, 3.]])
    m = np.array([1.,1.,1.,1.])

    molecules = np.asarray([[0],[1], [2,3]], dtype=np.int)
    print(molecules)
    com = centreOfMass(test_array, m, molecules)

    project_back(molecules, com, test_array, box_size)


    print(test_array)


    # print(test_array)

    # diff = test_array[[2,1]] - test_array[[1,1]]
    # print(diff)
    # res = np.zeros(diff.shape[0])
    # distance_PBC(diff, box_size, res)
    # print(res)