import numpy as np
from numba import vectorize, float64, jit, guvectorize
import math
#import static_state

""" helper/misc. functions and constants """

_atom_mass = {
    "O": 15.999,
    "H": 1.00784,
    "C": 12.011
}

def angle_to_radian(angle):
    return (angle*np.pi)/180.0

def unit_vector(matrix_):
    """ 
    Returns the unit vector of the vector.  
    if input is matrix does this for each row.
    """
    # OLD return vector / np.linalg.norm(vector)
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
    #return atom + " " +  np.array2string(pos, separator=" ")[1:-1] + "\n"   

def atom_name_to_mass(atoms):
    """ converts an atom name to its mass"""
    mass = [_atom_mass[atom] for atom in atoms]
    return np.array(mass)

# TODO: faster version on stackoverflow? 
def cartesianprod(x,y):
    """ returns cartesian product of lists x and y"""
    Cp = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return Cp    

def neighbor_list(pos, molecule_to_atoms, centre_of_mass, r_cut, box_size):
    """
    Returns which atoms are close, based on centres of mass that are withtin 
    r_cut distance of eachother 
    """
    dis_matrix = np.zeros((centre_of_mass.shape[0], centre_of_mass.shape[0]))
    distance_PBC_matrix(centre_of_mass - centre_of_mass[:, np.newaxis], box_size, dis_matrix)

    adj = (0 < dis_matrix) & (dis_matrix < r_cut)
    nl =  np.transpose(np.nonzero(adj))

    if nl.size != 0:
        lj_atoms = np.concatenate([molecule_to_atoms[i[0]][i[1]] for i in nl])
    else:
        lj_atoms = np.array([])

    return lj_atoms

def create_list(molecules):
    """
    Creates list of what atoms are connected given molecules that are
    connected, i.e. matrix[i][j] is the cartesian product of the atoms 
    in molecule i and molecule j

    # NOTE: inefficient double loop, but we only call this once so it is
            not that bad
    """
    matrix = [[0 for j in range(molecules.shape[0])] for i in range(molecules.shape[0])]

    for i in range(molecules.shape[0]):
        for j in range(molecules.shape[0]):
            if j > i:
                matrix[i][j] = cartesianprod(molecules[i], molecules[j])

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

@guvectorize([(float64[:,:,:], float64, float64[:,:])], "(n,n,p),()->(n,n)",
            nopython=True, cache=True)
def distance_PBC_matrix(diff, box_length, res):
    """
    Function to compute the distance of a matrix of vectors when considering
    periodic boundary conditions

    Input:
        diff: (n,n,3) numpy array of vectors
        box_length: length of the PBC box, in A
        res: (n,n) array which will be filled with the distances

    # TODO: rewrite this back to a numba function (this is also the reason for no return statement)
    # TODO: combine this with the function below
    """

    #diff = np.mod(diff, box_length)

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

                res[i][j] = norm(x,y,z)


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
    #pos_1 = np.mod(pos_1, box_length)
    #pos_2 = np.mod(pos_2, box_length)

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
    _ethanol_patern = np.array([
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

    dihedrals = np.array([[1,0,3,5],
                 [2,0,3,5]])

    print(dihedrals[:,0])

    dihedral = _ethanol_patern
    i = dihedral[dihedrals[:,0]]
    j = dihedral[dihedrals[:,1]]
    k = dihedral[dihedrals[:,2]]
    l = dihedral[dihedrals[:,3]]

    ji = i - j
    lk = k - l

    dihedral_angle = angle_between(ji, lk)
    psi = dihedral_angle - np.pi

    c = (j+k)/2.0

    ijk = angle_between(i - j, k - j)
    jkl = angle_between(j - k, l - k)

    C_1 = np.array([1.0, 2.0])
    C_2 = np.array([1.0, 2.0])
    C_3 = np.array([1.0, 2.0])
    C_4 = np.array([1.0, 2.0])
    magnitude = 0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.cos(3*psi) - 4*C_4*np.sin(4*psi))


    # Force on atom i in dihedral angle
    f_i_unit = unit_vector(static_state.cross(j - i, k - j))
    f_l_unit = unit_vector(static_state.cross(l - k, k - j))

    f_i = (magnitude/(np.linalg.norm(j - i, axis=1)*np.sin(ijk)))[:, np.newaxis]*f_i_unit     
    f_l = (magnitude/(np.linalg.norm(k - l, axis=1)*np.sin(jkl)))[:, np.newaxis]*f_l_unit

    f_k = -(1/(np.linalg.norm(k - c, axis=1)**2))[:, np.newaxis]*static_state.cross(static_state.cross(k-c, f_l) + 0.5*static_state.cross(l - k, f_l) + 0.5*static_state.cross(i-j, f_i), k - c)

    f_j = -(f_i + f_l + f_k)

    torque = static_state.cross(i - c, f_i) + static_state.cross(j - c, f_j) + static_state.cross(k - c, f_k) + static_state.cross(l - c, f_l)
    print(torque)
    print(f_i + f_j + f_l + f_k)




    # ji = i - j
    # lk = k - l

    # dihedral_angle = angle_between(np.array([ji]), np.array([lk]))
    # psi = dihedral_angle - np.pi

    # c = (j+k)/2.0

    # ijk = angle_between(np.array([i - j]), np.array([k - j]))
    # jkl = angle_between(np.array([j - k]), np.array([l - k]))

    # C_1 = 1
    # C_2 = 1
    # C_3 = 1
    # C_4 = 1
    # magnitude = 0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.cos(3*psi) - 4*C_4*np.sin(4*psi))


    # # Force on atom i in dihedral angle
    # f_i_unit = unit_vector(static_state.cross(np.array([j - i]), np.array([k - j])))
    # f_l_unit = unit_vector(static_state.cross(np.array([l - k]), np.array([k - j])))

    # f_i = (magnitude/(np.linalg.norm(j - i)*np.sin(ijk)))*f_i_unit     
    # f_l = (magnitude/(np.linalg.norm(k - l)*np.sin(jkl)))*f_l_unit

    # f_k = -(1/(np.linalg.norm(k-c)**2))*static_state.cross(static_state.cross(np.array([k-c]), f_l) + 0.5*static_state.cross(np.array([l - k]), f_l) + 0.5*static_state.cross(np.array([i-j]), f_i), np.array([k-c]))

    # f_j = -(f_i + f_l + f_k)

    # torque = static_state.cross(np.array([i - c]), f_i) + static_state.cross(np.array([j - c]), f_j) + static_state.cross(np.array([k - c]), f_k) + static_state.cross(np.array([l - c]), f_l)
    # print(torque)
    # print(f_i + f_j + f_l + f_k)