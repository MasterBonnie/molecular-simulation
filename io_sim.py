import numpy as np
import json

import static_state
import helper

import matplotlib.pyplot as plt


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

def radial_distribution_function(file, dr, box_size):
    """
    Calculates the radial distribution function for a given xyz file
    """
    with open(file, "r") as input_file:    
        while True:   
            line = input_file.readline()

            if not line:
                break
            nr_atoms = int(line)

            # Comment
            line = input_file.readline()

            # Position stores the positions of all atoms
            # in one time step
            pos = np.zeros((nr_atoms, 3))
            atom_names = []

            # We read out all the atoms
            for i in range(nr_atoms):
                line = input_file.readline()
                splittedLine = line.split()
                pos[i] = np.asarray(splittedLine[1:], np.float)
                atom_names.append(splittedLine[0])
            
            reference_position = np.ones((nr_atoms, 3))*pos[0]
            res = np.zeros((nr_atoms))
            diff = np.zeros((nr_atoms, 3))
            helper.distance_PBC(reference_position, pos, box_size, res, diff)

            histo = np.histogram(res, np.arange(0, box_size + dr, dr))[0]
            histo = histo/(nr_atoms*box_size**3)
            cf = np.array([(4*np.pi/3)*(((i+1)*dr)**3-(i*dr)**3) for i in range(int(box_size/dr))])
            histo = histo*cf

            plt.plot(histo)
            plt.show()


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
        lj: a nr_lj x 2 np array, whose rows are the pairs
                between a lj interaction
        const_lj: a nr_ljx2 np array, whose rows contain the 
                constant for the associated interaction between
                molecules in the same row as in lj

    NOTE: We assume there are always bonds, angles, lj and molecules,
          dihedrals are optional
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
        line = line.split()
        nr_angles = int(line[1])

        angles = np.zeros((nr_angles, 3), dtype=np.intp)
        const_angles = np.zeros((nr_angles, 2))

        for i in range(nr_angles):
            line = inputfile.readline()
            line = line.split()

            angles[i] = np.asarray(line[0:3], dtype=np.intc)
            const_angles[i] = np.asarray(line[3:], dtype=np.float)

        line = inputfile.readline()
        line = line.split()
        nr_lj = int(line[1])

        lj = np.zeros((nr_lj), dtype=np.int)
        const_lj = np.zeros((nr_lj, 2), dtype=np.float)

        for i in range(nr_lj):
            line = inputfile.readline()
            line = line.split()

            lj[i] = np.asarray(line[0], dtype=np.intp)
            const_lj[i] = np.asarray(line[1:], dtype=np.float)

        line = inputfile.readline()
        line = line.split()
        nr_molecules = int(line[1])

        molecules = []

        for i in range(nr_molecules):
            line = inputfile.readline()
            line = line.split()
            molecules.append(line)
        
        molecules = np.asarray(molecules, dtype=np.int)

        line = inputfile.readline()

        if not line:
            return bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, None, None

        
        line = line.split()
        nr_dihedrals = int(line[1])

        dihedrals = np.zeros((nr_dihedrals, 4), dtype=np.int)
        const_dihedrals = np.zeros((nr_dihedrals, 4))

        for i in range(nr_dihedrals):
            line = inputfile.readline()
            line = line.split()
            dihedrals[i] = np.asarray(line[0:4], dtype=np.int)
            const_dihedrals[i] = np.asarray(line[4:])

    return bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, dihedrals, const_dihedrals

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

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

_water_patern = np.array([
                          [1.93617934,      2.31884508,      1.72261570],
                          [1.78931374,      3.24075634,      1.51114298],
                          [2.30448689,      1.98045541,      0.90160232]
                           ])

_water_atoms = ["O", "H", "H"]

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

_ethanol_atoms = ["C","H","H","H","C","H","H","O","H"]

def create_dataset(nr_h20, nr_ethanol, tol_h20, tol_eth, box_size, output_file_xyz, output_file_top):
    """ 
    Creates a dataset with nr_h20 water molecules and nr_ethanol ethanol molecules
    Randomly placed inside a box of size box_size.
    Writes to output_file_xyz and output_file_top
    """
    with open(output_file_xyz, "w") as xyz_file, open(output_file_top, "w") as top_file:
        xyz_file.write(f"{3*nr_h20 + 9*nr_ethanol} \n")
        xyz_file.write(f"{nr_h20} water molecules and {nr_ethanol} ethanol molecules \n")

        bonds = []
        angles = []
        LJ = []
        molecules = []
        dihedrals = []

        pos = []

        for i in range(nr_h20):
            print(f"{i}th water molecule")
            random_displacement = np.random.uniform(0, box_size, (3))
            water = np.asarray([_water_patern[0] + random_displacement,
                               _water_patern[1] + random_displacement,
                               _water_patern[2] + random_displacement]) 

            com = (15.999*water[0] + water[1] + water[2])/16.

            while check_placement(com, pos, box_size, tol_h20):
                random_displacement = np.random.uniform(0, box_size, (3))
                water = np.asarray([_water_patern[0] + random_displacement,
                               _water_patern[1] + random_displacement,
                               _water_patern[2] + random_displacement]) 

                com = (15.999*water[0] + water[1] + water[2])/16.

            pos.append(com)

            for j, atom in enumerate(_water_atoms):
                xyz_file.write(helper.atom_string(atom, water[j]))


            bonds.append(f"{3*i} {3*i+1} 5024.16 0.9572 \n")
            bonds.append(f"{3*i} {3*i+2} 5024.16 0.9572 \n")

            angles.append(f"{3*i+1} {3*i} {3*i+2} 628.02 1.8242181 \n")

            LJ.append(f"{3*i} 3.15061 0.66386 \n")

            molecules.append(f"{3*i} {3*i+1} {3*i+2} \n")

        for i in range(nr_ethanol):
            print(f"{i}th ethanol molecule")
            random_displacement = np.random.uniform(0, box_size, (3))
            ethanol = np.asarray([_ethanol_patern[i] + random_displacement for i in range(9)])

            m = helper.atom_name_to_mass(_ethanol_atoms)

            com = np.sum(np.array([m[i]*ethanol[i] for i in range(9)]), axis=0)/np.sum(m)

            while check_placement(com, pos, box_size, tol_eth):
                random_displacement = np.random.uniform(0, box_size, (3))
                ethanol = np.asarray([_ethanol_patern[i] + random_displacement for i in range(9)])
                com = np.sum(np.array([m[i]*ethanol[i] for i in range(9)]), axis = 0)/np.sum(m)

            pos.append(com)

            for j, atom in enumerate(_ethanol_atoms):
                xyz_file.write(helper.atom_string(atom, ethanol[j]))

            # Correctly offset counter

            bonds.append(f"{9*i + 3*nr_h20} {9*i + 1 + 3*nr_h20} 2845.12 1.09 \n")
            bonds.append(f"{9*i + 3*nr_h20} {9*i + 2 + 3*nr_h20} 2845.12 1.09 \n")
            bonds.append(f"{9*i + 3*nr_h20} {9*i + 3 + 3*nr_h20} 2845.12 1.09 \n")
            bonds.append(f"{9*i + 4 + 3*nr_h20} {9*i + 5 + 3*nr_h20} 2845.12 1.09 \n")
            bonds.append(f"{9*i + 4 + 3*nr_h20} {9*i + 6 + 3*nr_h20} 2845.12 1.09 \n")

            bonds.append(f"{9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} 2242.624 1.529 \n")

            bonds.append(f"{9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 2677.76 1.41 \n")

            bonds.append(f"{9*i + 7 + 3*nr_h20} {9*i + 8 + 3*nr_h20} 4626.50 0.945  \n")

            angles.append(f"{9*i + 1 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} 292.88 {helper.angle_to_radian(108.5)} \n")
            angles.append(f"{9*i + 2 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} 292.88 {helper.angle_to_radian(108.5)} \n")
            angles.append(f"{9*i + 3 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} 292.88 {helper.angle_to_radian(108.5)} \n")

            angles.append(f"{9*i + 3 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 2 + 3*nr_h20} 276.144 {helper.angle_to_radian(107.8)} \n")
            angles.append(f"{9*i + 3 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 1 + 3*nr_h20} 276.144 {helper.angle_to_radian(107.8)} \n")
            angles.append(f"{9*i + 2 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 1 + 3*nr_h20} 276.144 {helper.angle_to_radian(107.8)} \n")
            angles.append(f"{9*i + 5 + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 6 + 3*nr_h20} 276.144 {helper.angle_to_radian(107.8)} \n")

            angles.append(f"{9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 6 + 3*nr_h20} 313.8 {helper.angle_to_radian(110.7)} \n")
            angles.append(f"{9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 5 + 3*nr_h20} 313.8 {helper.angle_to_radian(110.7)} \n")

            angles.append(f"{9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 414.4 {helper.angle_to_radian(109.5)} \n")

            angles.append(f"{9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} {9*i + 8 + 3*nr_h20} 460.24 {helper.angle_to_radian(108.5)} \n")

            angles.append(f"{9*i + 5 + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 292.88 {helper.angle_to_radian(109.5)} \n")
            angles.append(f"{9*i + 6 + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 292.88 {helper.angle_to_radian(109.5)}\n")


            LJ.append(f"{9*i + 3*nr_h20} 3.5 0.276144 \n")
            LJ.append(f"{9*i + 4 + 3*nr_h20} 3.5 0.276144 \n")

            LJ.append(f"{9*i + 1 + 3*nr_h20} 2.5 0.12552 \n")
            LJ.append(f"{9*i + 2 + 3*nr_h20} 2.5 0.12552 \n")
            LJ.append(f"{9*i + 3 + 3*nr_h20} 2.5 0.12552 \n")
            LJ.append(f"{9*i + 5 + 3*nr_h20} 2.5 0.12552 \n")
            LJ.append(f"{9*i + 6 + 3*nr_h20} 2.5 0.12552 \n")

            LJ.append(f"{9*i + 7 + 3*nr_h20} 3.12 0.71128 \n")

            molecules.append(f"{9*i + 3*nr_h20} {9*i + 1 + 3*nr_h20} {9*i + 2 + 3*nr_h20} {9*i + 3 + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 5 + 3*nr_h20} {9*i + 6 + 3*nr_h20} {9*i + 7 + 3*nr_h20} {9*i + 8 + 3*nr_h20} \n")

            dihedrals.append(f"{9*i + 1 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 5 + 3*nr_h20} 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 2 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 5 + 3*nr_h20} 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 3 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 5 + 3*nr_h20} 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 1 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 6 + 3*nr_h20} 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 2 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 6 + 3*nr_h20} 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 3 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 6 + 3*nr_h20} 0.6276 1.8828 0.0 -3.91622 \n")

            dihedrals.append(f"{9*i + 1 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 0.97905 2.93716 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 2 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 0.97905 2.93716 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 3 + 3*nr_h20} {9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} 0.97905 2.93716 0.0 -3.91622 \n")

            dihedrals.append(f"{9*i + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} {9*i + 8 + 3*nr_h20} -0.4431 3.83255 0.72801 -4.11705 \n")

            dihedrals.append(f"{9*i + 5 + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} {9*i + 8 + 3*nr_h20} 0.94140 2.82420 0.0 -3.76560 \n")
            dihedrals.append(f"{9*i + 6 + 3*nr_h20} {9*i + 4 + 3*nr_h20} {9*i + 7 + 3*nr_h20} {9*i + 8 + 3*nr_h20} 0.94140 2.82420 0.0 -3.76560 \n")


        # Write to topology file
        top_file.write(f"bonds {2*nr_h20 + 8*nr_ethanol} \n")
        for string in bonds:
            top_file.write(string)
        top_file.write(f"angles {nr_h20 + 13*nr_ethanol} \n")
        for string in angles:
            top_file.write(string)
        top_file.write(f"LJ {nr_h20 + 8*nr_ethanol} \n")
        for string in LJ:
            top_file.write(string)
        top_file.write(f"molecules {nr_h20 + nr_ethanol} \n")
        for string in molecules:
            top_file.write(string)
        if dihedrals:
            top_file.write(f"dihedrals {12*nr_ethanol} \n")
            for string in dihedrals:
                top_file.write(string)

def distance(pos_1, pos_2, box_length):
    x = helper.abs_min(pos_1[0] - pos_2[0], pos_1[0]  - pos_2[0] + box_length, pos_1[0]  - pos_2[0] - box_length)  

    y = helper.abs_min(pos_1[1] - pos_2[1], 
                pos_1[1]  - pos_2[1] + box_length, 
                pos_1[1]  - pos_2[1] - box_length) 
        
    z = helper.abs_min(pos_1[2] - pos_2[2], 
                pos_1[2]  - pos_2[2] + box_length, 
                pos_1[2]  - pos_2[2] - box_length)        
        
    return helper.norm(x,y,z)


def check_placement(molecule, pos, box, tol):
    for com in pos:
        if distance(molecule, com, box) < tol:
            return True

    return False

# Testing of the functions
if __name__ == "__main__":
    #pos, atom_names, atoms = read_xyz("data/water_top.xyz")
    #bonds, const_bonds, angles, const_angles, lj, const_lj, molecules = read_topology("data/top.itp")  

    nr_h20 = 0
    tol_h20 = 3
    nr_ethanol = 1
    tol_eth = 10  
    box_size = 5 
    output_file_xyz = "data/ethanol.xyz"
    output_file_top = "data/ethanol.itp"

    create_dataset(nr_h20, nr_ethanol, tol_h20, tol_eth, box_size, output_file_xyz, output_file_top)

    #radial_distribution_function("data/rdf.xyz", 0.2, box_size = 50)
