import numpy as np
from numba import jit, guvectorize, float64

""" helper/misc. functions and constants """

#@jit(nopython=True)
def unit_vector(vector):
    """ 
    Returns the unit vector of the vector.  
    if input is matrix does this for each row.
    """
    # OLD return vector / np.linalg.norm(vector)
    return vector/np.linalg.norm(vector, ord=2, axis=1, keepdims=True)

@guvectorize([(float64[:,:], float64[:,:], float64[:])],"(n,p),(n,p)->(n)", nopython=True)
def dot_product(m1,m2,res):
    for i in range(m1.shape[0]):
        for j in range(m2.shape[0]):
            res[i] += m1[i][j]*m2[i][j]

#@jit(nopython=True)
def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    # Calculates the row-wise dot product between
    # diff_1 and diff_2
    dot = np.zeros(v1.shape[0])
    dot_product(v1,v2,dot)

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