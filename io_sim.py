import numpy as np
import json

import static_state
import helper


""" File for all I/O operations """

def read_xyz(input, debug = False):
    """
    Read input xyz file.
    Input:
        input: string containing relative file location
        debug: boolean whether to print extra output
    Output:
        pos: last array of positions
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

            print("number of atoms:")
            print(atoms)

            # Second line is comments
            line = inputfile.readline()
            print("comments:")
            print(line)
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

                if debug: 
                    print("x,y,z")
                    print(np.asarray(splittedLine[1:]))

            # Do stuff with the atoms here, for testing
            # ------------------------------------

    return pos, atom_names, atoms


def handle_comment(line):
    """ handles comment line of a xyz file"""
    # Dont know yet what is usefull to do here
    return 

def read_json(input):
    """ reads json file and transforms it into a dict """
    with open(input) as file:
        data = json.load(file)

    return data

# Testing of the functions
if __name__ == "__main__":
    read_xyz("data/Water2.xyz", True)
    