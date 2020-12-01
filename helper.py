import numpy as np

""" helper/misc. functions and constants """

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