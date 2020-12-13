import numpy as np
from numba import vectorize, float64, jit, guvectorize
import helper

"""File for extracting/calculating information about a state at a particular time step"""

# TODO: werkt nog niet als molecules niet homogenous is
# depcrecated because of reason above
def centreOfMass(pos,m,molecules):
    M = np.sum(m[molecules], axis = 1)
    Mpos = np.sum(m[molecules,np.newaxis]*pos[molecules], axis = 1)
    Cm = Mpos/M[:,np.newaxis]
    return Cm


def centre_of_mass(pos, m, molecules):
    centre_of_mass = np.zeros((len(molecules), 3))
    for i, molecule in enumerate(molecules):
        M = np.sum(m[molecule])
        Mpos = np.sum(m[molecule, np.newaxis]*pos[molecule], axis = 0)
        centre_of_mass[i] = Mpos/M

    return centre_of_mass

#@jit(nopython=True, cache=True)
# NOTE: this was written with the intention to use numba on it, however,
# Numba complains, for example because we use lists of lists, molecules.
# However numpy arrays would not work either as the molecules sizes can be
# different. For now leave this, but needs to be done better!
def project_to_box(molecules, centre_of_mass, pos, box_size):
    for i, molecule in enumerate(centre_of_mass):
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
        
        for atom in molecules[i]:
            pos[atom] = pos[atom] + translation

if __name__ == "__main__":
    box_size = 2.5
    pos = np.array([[3.,3.,3.],
                            [1.,1.,1.],
                            [1.,2.,3.],
                            [1., 2.4, 3.]])
    m = np.array([1.,1.,1.,1.])

    molecules = [[0],[1], [2,3]]    
    com = centre_of_mass(pos, m, molecules)

    project_to_box(molecules, com, pos, box_size)
    print(pos)

