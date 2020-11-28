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

def read_topology(input, debug = False):
    """
    Read topology file (.itp) 
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

            print(line[0:1])
            bonds[i] = np.asarray(line[0:2], dtype=np.intp)
            print(line[2:])
            const_bonds[i]= np.asarray(line[2:], dtype=np.float)

        line = inputfile.readline()
        line = line.split()
        nr_angles = int(line[1])

        angles = np.zeros((nr_angles, 3), dtype=np.intp)
        const_angles = np.zeros((nr_angles, 2))

        for i in range(nr_angles):
            line = inputfile.readline()
            line = line.split()

            print(line[0:2])
            angles[i] = np.asarray(line[0:3], dtype=np.intc)
            const_angles[i] = np.asarray(line[3:], dtype=np.float)

    return bonds, const_bonds, angles, const_angles

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
    pos, atom_names, atoms = read_xyz("data/water_top.xyz")
    bonds, const_bonds, angles, const_angles = read_topology("data/top.itp")

    force_total = np.zeros((atoms, 3))

    print("TESTING")
    # These are all diferences we need for bonds
    diff = pos[bonds[:,0]] - pos[bonds[:,1]]
    #print(diff)
    
    dis = np.linalg.norm(diff, axis=1)
    #print(dis)

    #print(dis - const_bonds[:,1])
    #magnitude_force = -k*(dis - r_0)
    magnitudes = np.multiply(-const_bonds[:,0], dis - const_bonds[:,1])
    #print(magnitudes)
    force = magnitudes[:, np.newaxis]*helper.unit_vector(diff)

    force_total[bonds[:,0]] += force
    force_total[bonds[:,1]] += -force

    # Now force contains the right forces for bonds

    # Angles
    diff_1 = pos[angles[:,1]] - pos[angles[:,0]]
    diff_2 = pos[angles[:,1]] - pos[angles[:,2]]

    print(diff_1)
    print(diff_2)
    dot = np.einsum('ij,ij->i', diff_1, diff_2)
    print(dot)

    ang = np.arccos(dot)
    print(ang)

    #magnitude_force = -k*(theta - theta_0)/dis
    #force = magnitude_force* helper.unit_vector(direction)
    
    mag_ang = np.multiply(-const_angles[:,0], ang - const_angles[:,1])

    angular_force_unit_1 = helper.unit_vector(np.cross(np.cross(diff_1, diff_2), diff_1))
    angular_force_unit_2 = -helper.unit_vector(np.cross(np.cross(diff_1, diff_2), diff_2))

    force_ang_1 = np.multiply(np.multiply(mag_ang, np.linalg.norm(diff_1, axis=1))[:, np.newaxis], angular_force_unit_1)
    force_ang_2 = np.multiply(np.multiply(mag_ang, np.linalg.norm(diff_2, axis=1))[:, np.newaxis], angular_force_unit_2)
    
    force_total[angles[:,0]] += force_ang_1
    force_total[angles[:,2]] += force_ang_2
    force_total[angles[:,1]] += -(force_ang_1 + force_ang_2)

    print(force_total)