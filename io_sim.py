import numpy as np
import json

import static_state
import helper


""" File for all I/O operations """

def read_xyz(input):
    """
    Read input xyz file.
    Input:
        input: relative path to the xyz file
    Output:
        pos: np array of positions
        atom_names: array of string of the atoms 
                    in the xyz file
        nr_atoms: number of atoms

    NOTE: if there are multiple timesteps, we return 
          the last pos, atom_names and nr_atoms
    """

    with open(input, "r") as inputfile:
        while True:
            # First line is number of atoms
            line  = inputfile.readline()

            # If line is empty, we are done with the file
            if not line:
                break 

            try:
                atoms = int(line)
            except:
                print("Number of atoms is not a number?")
                break

            # Second line is comments
            line = inputfile.readline()
            handle_comment(line)

            # Position stores the positions of all atoms
            # in one time step
            pos = np.zeros((atoms, 3))
            atom_names = []

            # We read out all the atoms
            for i in range(atoms):
                line = inputfile.readline()
                splittedLine = line.split()
                pos[i] = np.asarray(splittedLine[1:], np.float)
                atom_names.append(splittedLine[0])

            # Do stuff with the atoms here, for testing
            # ------------------------------------

    return pos, atom_names, atoms

def read_topology(input):
    """
    Read topology file (.itp) 
    Input:
        input: relative path to the itp file
    Output:
        bonds: a nr_bonds x 2 np array, whose rows
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

    NOTE: if there are no angles in the file, angles and
          const_angles are None. We assume there are always 
          bonds.
    """

    with open(input, "r") as inputfile:
        line = inputfile.readline()
        line = line.split()
        nr_bonds = int(line[1])
        
        bonds = np.zeros((nr_bonds, 2), dtype=np.intp)
        const_bonds = np.zeros((nr_bonds, 2))

        for i in range(nr_bonds):
            line = inputfile.readline()
            line = line.split()

            bonds[i] = np.asarray(line[0:2], dtype=np.intp)
            const_bonds[i]= np.asarray(line[2:], dtype=np.float)

        line = inputfile.readline()
        
        # If there are no angles in the molecule
        if not line:
            return bonds, const_bonds, None, None

        line = line.split()
        nr_angles = int(line[1])

        angles = np.zeros((nr_angles, 3), dtype=np.intp)
        const_angles = np.zeros((nr_angles, 2))

        for i in range(nr_angles):
            line = inputfile.readline()
            line = line.split()

            angles[i] = np.asarray(line[0:3], dtype=np.intc)
            const_angles[i] = np.asarray(line[3:], dtype=np.float)

        line - inputfile.readline()

        if not line:
            return bonds, const_bonds, angles, const_angles, None, None

        line - line.split()
        nr_lj = int(line[1])

        lj = np.zeros((nr_lj, 2), dtype=np.intp)
        const_lj = np.zeros((nr_lj, 2), dtype=np.float)

        for i in range(nr_lj):
            line = inputfile.readline()
            line = line.split()

            lj[i] = np.asarray(line[0:2], dtype=np.intp)
            const_lj = np.asarray(line[2:], dtype=np.float)

    return bonds, const_bonds, angles, const_angles, lj, const_lj

def handle_comment(line):
    """ handles comment line of a xyz file"""
    # Dont know yet what is usefull to do here
    return 

def read_json(input):
    """ reads json file and transforms it into a dict """
    # Possible for later, passing arguments to simulator
    with open(input) as file:
        data = json.load(file)

    return data

# Testing of the functions
if __name__ == "__main__":
    pos, atom_names, atoms = read_xyz("data/water_top.xyz")
    bonds, const_bonds, angles, const_angles = read_topology("data/top.itp")
