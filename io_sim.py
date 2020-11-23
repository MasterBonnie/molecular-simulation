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


def write_xyz(file, pos, atoms):
    """ 
    Writes to an open xyz file the coordinates of the atoms 
    Input:

    
    """

    return

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
    

# # Do stuff with the atoms here, for testing
# # ------------------------------------
# # For now set unit to nm, as in pos they are Amstrong
# pos = 0.1*pos

# if debug:
#     print(pos)
# diff = (pos - pos[:, np.newaxis])
# if debug:
#     print(diff)
# dis = np.linalg.norm(diff, axis=2)
# if debug:
#     print(dis)

# # Calculate the force on each molecule
# k_b = 502416    # Kj mol^-1 nm^-2
# r_0 = 0.09572   # nm

# theta_0 = 104.52 *(np.pi/180)   # Radians
# k_ang = 628.02  # Kj mol^-1 rad^-2

# theta = helper.angle_between(diff[0][1], diff[0][2])
# #print(theta*180/np.pi)

# # Converts our units to the force unit amu A/(0.1ps)^2
# # We now have Kj/(mol * nm) = 1000 kg m mol^-1 nm^-1 s^-2
# conversion_factor = 1.6605*6.022e-2 # is really close to 0.1

# print("atom 1")
# # Create the unit vector along which the angular force acts
# # This is the right molecule, and so we need this -1 here
# angular_force_unit_1 = helper.unit_vector(-np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][1]))

# # Calculate the angular and bond force on Hydrogen atom 1
# angular_force_1 = static_state.force_angular(angular_force_unit_1,
#                                                 theta,
#                                                 dis[0][1],
#                                                 k_ang,
#                                                 theta_0)
# bond_force_1 = static_state.force_bond(diff[0][1], dis[0][1], k_b, r_0)

# #print("angular force 1")
# #print(angular_force_1)
# #print("bond force 1")
# #print(bond_force_1)

# # Total force on hydrogen atom 1
# force_hydrogen_1 = conversion_factor*(angular_force_1 + bond_force_1)

# print("total force 1")
# print(force_hydrogen_1)

# print("atom 2")
# # Again create unit vector for angular force
# # This one already points in the right direction
# angular_force_unit_2 = helper.unit_vector(np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][2]))

# # Angular force
# angular_force_2 = static_state.force_angular(angular_force_unit_2,
#                                                 theta,
#                                                 dis[0][2],
#                                                 k_ang,
#                                                 theta_0)
# bond_force_2 = static_state.force_bond(diff[0][2], dis[0][2], k_b, r_0)

# #print("Angular force 2")
# #print(angular_force_2)
# #print("bond force 2")
# #print(bond_force_2)

# # Total force on Hydrogen atom 2
# force_hydrogen_2 = conversion_factor*(angular_force_2 + bond_force_2)

# print("total force 2")
# print(force_hydrogen_2)

# # Total force on Oxygen is just the vector which causes the total force of the system to be 0


# if debug:
#     print(pos)
# diff = (pos - pos[:, np.newaxis])
# if debug:
#     print(diff)
# dis = np.linalg.norm(diff, axis=2)
# if debug:
#     print(dis)

# # Calculate the force on each molecule
# k_b = 5024.16    # Kj mol^-1 A^-2
# r_0 = 0.9572   # nm

# theta_0 = 104.52 *(np.pi/180)   # Radians
# k_ang = 628.02  # Kj mol^-1 rad^-2

# theta = helper.angle_between(diff[0][1], diff[0][2])
# #print(theta*180/np.pi)

# # Converts our units to the force unit amu A/(0.1ps)^2
# # We now have Kj/(mol * nm) = 1000 kg m mol^-1 nm^-1 s^-2
# conversion_factor = 1#1.6605*6.022e-2 # is really close to 0.1

# print("atom 1")
# # Create the unit vector along which the angular force acts
# # This is the right molecule, and so we need this -1 here
# angular_force_unit_1 = helper.unit_vector(-np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][1]))

# # Calculate the angular and bond force on Hydrogen atom 1
# angular_force_1 = static_state.force_angular(angular_force_unit_1,
#                                                 theta,
#                                                 dis[0][1],
#                                                 k_ang,
#                                                 theta_0)
# bond_force_1 = static_state.force_bond(diff[0][1], dis[0][1], k_b, r_0)

# #print("angular force 1")
# #print(angular_force_1)
# #print("bond force 1")
# #print(bond_force_1)

# # Total force on hydrogen atom 1
# force_hydrogen_1 = conversion_factor*(angular_force_1 + bond_force_1)

# print("total force 1")
# print(force_hydrogen_1)

# print("atom 2")
# # Again create unit vector for angular force
# # This one already points in the right direction
# angular_force_unit_2 = helper.unit_vector(np.cross(np.cross(diff[0][1], diff[0][2]), diff[0][2]))

# # Angular force
# angular_force_2 = static_state.force_angular(angular_force_unit_2,
#                                                 theta,
#                                                 dis[0][2],
#                                                 k_ang,
#                                                 theta_0)
# bond_force_2 = static_state.force_bond(diff[0][2], dis[0][2], k_b, r_0)

# #print("Angular force 2")
# #print(angular_force_2)
# #print("bond force 2")
# #print(bond_force_2)

# # Total force on Hydrogen atom 2
# force_hydrogen_2 = conversion_factor*(angular_force_2 + bond_force_2)

# print("total force 2")
# print(force_hydrogen_2)

# # Total force on Oxygen is just the vector which causes the total force of the system to be 0