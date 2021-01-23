import numpy as np

import static_state
import helper
from helper import water_patern, water_atoms, ethanol_patern, ethanol_atoms
from numba import vectorize, float64, jit, guvectorize, double
from plotting import setup_matplotlib


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

def radial_distribution_function(xyz_file, top_file, dr, box_size):
    """
    Calculates the radial distribution function for a given xyz file
    Needs an associated topology file to work.

    NOTE: Saves the computed rdf
    """
    _, _, _, _, _, _, molecules, _, _ = read_topology(top_file)

    # Select which molecules and atoms we want to find the rdf of
    first_molecule = "water"
    #first_atom = "O"  first molecule is always O
    second_molecule = "water"
    second_atom = "O"

    # Number of reference atoms, i.e. atoms to average over
    nr_reference_atoms = 1000

    # We create the appropriate lists to select the right reference
    # atoms and relevant positions
    water_o = []
    water_h = []
    ethanol_o = []
    ethanol_h = []

    for molecule in molecules:
        if len(molecule) == 3:
            # If len is three we are in water
            water_o.append(molecule[0])
            water_h.append(molecule[1])
            water_h.append(molecule[2])
        elif len(molecule) == 9:
            # We are in ethanol molecule
            ethanol_o.append(molecule[7])
            
            for i in [1,2,3,5,6,8]:
                ethanol_h.append(molecule[i])

    water_o = np.array(water_o, dtype=np.int)
    water_h = np.array(water_h, dtype=np.int)

    ethanol_o = np.array(ethanol_o, dtype=np.int)
    ethanol_h = np.array(ethanol_h, dtype=np.int)


    if first_molecule == "water":
        reference_atoms = water_o[:nr_reference_atoms]
    if first_molecule == "ethanol":
        reference_atoms = ethanol_o[:nr_reference_atoms]

    # now we define the position list we want to check ( in terms of indices )
    if second_molecule == "water":
        if second_atom == "O":
            pos_index = water_o
            nr_atoms = water_o.shape[0]
        if second_atom == "H":
            pos_index = water_h
            nr_atoms = water_h.shape[0]
    if second_molecule == "ethanol":
        if second_atom == "O":
            pos_index = ethanol_o
            nr_atoms = ethanol_o.shape[0]
        if second_atom == "H":
            pos_index = ethanol_h
            nr_atoms = ethanol_h.shape[0]

    # r is the maximum distance we check
    r = 9

    # Number of bins
    n = 1000    

    # List of distances:
    distances = np.array([[1+(i+1)*r/n, 1+(i+1)*r/n + dr] for i in range(n)])

    # Final rdf values
    total_rdf = np.zeros((n))

    # Some variables to keep track of how far we are
    # important: max_iterations is number of timesteps we average over,
    #            offset is how far we look in the xyz file
    iteration = 0
    rdf_iteration = 0
    max_iterations = 10
    offset = 50000
    sample_points = [offset + 100*i for i in range(max_iterations)]

    with open(xyz_file, "r") as input_file:    
        while True:   
            line = input_file.readline()

            if not line:
                break
            nr_atoms_total = int(line)

            # Comment
            line = input_file.readline()

            if iteration in sample_points:
                # Position stores the positions of all atoms
                # in one time step
                pos = np.zeros((nr_atoms_total, 3))

                for i in range(nr_atoms_total):
                    line = input_file.readline()
                    splittedLine = line.split()
                    pos[i] = np.asarray(splittedLine[1:], np.float)
                
                rdf = np.zeros((n))
                density = nr_atoms/(box_size**3)
                calculate_rdf(distances, pos[reference_atoms], pos[pos_index], box_size, density, rdf)

                print(f"{rdf_iteration} of {max_iterations}", end="\r")
                total_rdf += rdf
                rdf_iteration += 1
                if rdf_iteration == max_iterations:
                    break

            else:
                for i in range(nr_atoms_total):
                    line = input_file.readline()

            iteration += 1
                
        total_rdf = total_rdf/(len(sample_points))
        
        # Save rdf calculation in csv file
        np.savetxt("output/rdf/rdf_water_5nm_O_O.csv", total_rdf, delimiter=",")

        plt.plot([distance[0] for distance in distances], total_rdf)
        plt.show()

@jit(nopython=True, cache=True)
def calculate_rdf(distances, reference_atoms, pos, box_length, density, rdf):
    """
    rdf calculation, give the parameters.
    """
    for i in range(distances.shape[0]):
        lower = distances[i][0]
        upper = distances[i][1]


        for j in range(reference_atoms.shape[0]):
            reference_atom = reference_atoms[j]

            for k in range(pos.shape[0]):
                if k != j:

                    dis = distance(reference_atom, pos[k], box_length)
                    if lower < dis and dis < upper:
                        rdf[i] = rdf[i] + 1

        rdf[i] = rdf[i] /  reference_atoms.shape[0]
        rdf[i] = rdf[i] / ((4*np.pi/3)*(upper**3-lower**3))
        rdf[i] = rdf[i] / density

def read_topology(input, nr_atoms=0, fixed_atom_length=0):
    """
    Read a topology file from the file at input.

    params:
        input: relative path to the itp file
        nr_atoms: number of atoms in the simulation
        fixed_atom_length: if not 0, we fill any molecule to have the 
                        specified length, in order to pre-allocate some
                        more arrays in the simulation, and be able to use
                        numpy and numba more. We fill the molecules up with the
                        value nr_atoms
    Output:
        bonds:  a nr_bonds x 2 np array, whose rows
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
        lj:     a nr_lj x 2 np array, whose rows are the pairs
                between a lj interaction
        const_lj: a nr_lj x 2 np array, whose rows contain the 
                constant for the associated interaction between
                molecules in the same row as in lj
        molecules:
                if fixed_atom_length is not 0:
                    a nr_molecules x fixed_atom_length np array containing
                    the index of atoms in one molecule
                else:
                    a python list of numpy arrays containing the index of atoms
                    in one molecule
        dihedrals:
                a nr_dihedrals x 4 np array, containing the indices of atoms in one
                dihedral angle
        const_dihedrals:
                a nr_dihedrals x 4 np array, containing the associated constants

    NOTE: We assume there are always bonds, angles, lj and molecules,
          dihedrals are optional
    """

    with open(input, "r") as inputfile:

        # We set the variables here all to None, so that we 
        # only need one return statement.
        bonds = None
        const_bonds = None
        angles = None
        const_angles = None
        lj = None
        const_lj = None
        molecules = None
        dihedrals = None
        const_dihedrals = None


        while True:
            line = inputfile.readline()

            # There are no more parameters
            if not line:
                break

            line = line.split()
            parameter_type = line[0]
            nr_parameters = int(line[1])

            if parameter_type == "bonds":
                # create the arrays with correct sizes
                bonds = np.zeros((nr_parameters, 2), dtype=np.intp)
                const_bonds = np.zeros((nr_parameters, 2))

                # we fill them by reading the topology file
                for i in range(nr_parameters):
                    line = inputfile.readline()
                    line = line.split()

                    bonds[i] = np.asarray(line[0:2], dtype=np.intp)
                    const_bonds[i]= np.asarray(line[2:], dtype=np.float)

            elif parameter_type == "angles":
                angles = np.zeros((nr_parameters, 3), dtype=np.intp)
                const_angles = np.zeros((nr_parameters, 2))

                for i in range(nr_parameters):
                    line = inputfile.readline()
                    line = line.split()

                    angles[i] = np.asarray(line[0:3], dtype=np.intc)
                    const_angles[i] = np.asarray(line[3:], dtype=np.float)
            
            elif parameter_type == "LJ":
                lj = np.zeros((nr_parameters), dtype=np.int)
                const_lj = np.zeros((nr_parameters, 2), dtype=np.float)

                for i in range(nr_parameters):
                    line = inputfile.readline()
                    line = line.split()

                    lj[i] = np.asarray(line[0], dtype=np.intp)
                    const_lj[i] = np.asarray(line[1:], dtype=np.float)

            elif parameter_type == "dihedrals":
                dihedrals = np.zeros((nr_parameters, 4), dtype=np.int)
                const_dihedrals = np.zeros((nr_parameters, 4))

                for i in range(nr_parameters):
                    line = inputfile.readline()
                    line = line.split()
                    dihedrals[i] = np.asarray(line[0:4], dtype=np.int)
                    const_dihedrals[i] = np.asarray(line[4:])

            elif parameter_type == "molecules":
                molecules = []

                for i in range(nr_parameters):
                    line = inputfile.readline()
                    line = line.split()

                    if nr_atoms != 0:
                        length = fixed_atom_length - len(line)
                        molecules.append(np.array(line + [nr_atoms for i in range(length)], dtype=np.int16))
                    else:
                        molecules.append(np.array(line, dtype=np.int16))

                if nr_atoms != 0:
                    molecules = np.asarray(molecules, dtype=np.int16)

    return bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, dihedrals, const_dihedrals

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

        # We first place the ethanol molecules, because they are bigger
        for i in range(nr_ethanol):
            print(f"{i}th ethanol molecule", end="\r")
            random_displacement = np.random.uniform(0, box_size, (3))
            ethanol = np.asarray([ethanol_patern[i] + random_displacement for i in range(9)])

            # We check if any of the atoms in the molecule are too close to any other atoms in already placed
            # molecules. Any smarter way would probably cost a lot more effort to implement.
            while check_placement_per_atom(ethanol, pos, box_size, tol_eth):
                random_displacement = np.random.uniform(0, box_size, (3))
                ethanol = np.asarray([ethanol_patern[i] + random_displacement for i in range(9)])

            # We accepted the molecules placement, so now we add the correct constants to the arrays,
            # with the correct index
            for atom in ethanol:
                pos.append(atom)

            for j, atom in enumerate(ethanol_atoms):
                xyz_file.write(helper.atom_string(atom, ethanol[j]))

            bonds.append(f"{9*i } {9*i + 1 } 2845.12 1.09 \n")
            bonds.append(f"{9*i } {9*i + 2 } 2845.12 1.09 \n")
            bonds.append(f"{9*i } {9*i + 3 } 2845.12 1.09 \n")
            bonds.append(f"{9*i + 4 } {9*i + 5 } 2845.12 1.09 \n")
            bonds.append(f"{9*i + 4 } {9*i + 6 } 2845.12 1.09 \n")

            bonds.append(f"{9*i } {9*i + 4 } 2242.624 1.529 \n")

            bonds.append(f"{9*i + 4 } {9*i + 7 } 2677.76 1.41 \n")

            bonds.append(f"{9*i + 7 } {9*i + 8 } 4626.50 0.945  \n")

            angles.append(f"{9*i + 1 } {9*i } {9*i + 4 } 292.88 {helper.angle_to_radian(108.5)} \n")
            angles.append(f"{9*i + 2 } {9*i } {9*i + 4 } 292.88 {helper.angle_to_radian(108.5)} \n")
            angles.append(f"{9*i + 3 } {9*i } {9*i + 4 } 292.88 {helper.angle_to_radian(108.5)} \n")

            angles.append(f"{9*i + 3 } {9*i } {9*i + 2 } 276.144 {helper.angle_to_radian(107.8)} \n")
            angles.append(f"{9*i + 3 } {9*i } {9*i + 1 } 276.144 {helper.angle_to_radian(107.8)} \n")
            angles.append(f"{9*i + 2 } {9*i } {9*i + 1 } 276.144 {helper.angle_to_radian(107.8)} \n")
            angles.append(f"{9*i + 5 } {9*i + 4 } {9*i + 6 } 276.144 {helper.angle_to_radian(107.8)} \n")

            angles.append(f"{9*i } {9*i + 4 } {9*i + 6 } 313.8 {helper.angle_to_radian(110.7)} \n")
            angles.append(f"{9*i } {9*i + 4 } {9*i + 5 } 313.8 {helper.angle_to_radian(110.7)} \n")

            angles.append(f"{9*i } {9*i + 4 } {9*i + 7 } 414.4 {helper.angle_to_radian(109.5)} \n")

            angles.append(f"{9*i + 4 } {9*i + 7 } {9*i + 8 } 460.24 {helper.angle_to_radian(108.5)} \n")

            angles.append(f"{9*i + 5 } {9*i + 4 } {9*i + 7 } 292.88 {helper.angle_to_radian(109.5)} \n")
            angles.append(f"{9*i + 6 } {9*i + 4 } {9*i + 7 } 292.88 {helper.angle_to_radian(109.5)}\n")


            LJ.append(f"{9*i } 3.5 0.276144 \n")
            LJ.append(f"{9*i + 4 } 3.5 0.276144 \n")

            LJ.append(f"{9*i + 1 } 2.5 0.12552 \n")
            LJ.append(f"{9*i + 2 } 2.5 0.12552 \n")
            LJ.append(f"{9*i + 3 } 2.5 0.12552 \n")
            LJ.append(f"{9*i + 5 } 2.5 0.12552 \n")
            LJ.append(f"{9*i + 6 } 2.5 0.12552 \n")

            LJ.append(f"{9*i + 7 } 3.12 0.71128 \n")

            molecules.append(f"{9*i } {9*i + 1 } {9*i + 2 } {9*i + 3 } {9*i + 4 } {9*i + 5 } {9*i + 6 } {9*i + 7 } {9*i + 8 } \n")

            dihedrals.append(f"{9*i + 1 } {9*i } {9*i + 4 } {9*i + 5 } 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 2 } {9*i } {9*i + 4 } {9*i + 5 } 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 3 } {9*i } {9*i + 4 } {9*i + 5 } 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 1 } {9*i } {9*i + 4 } {9*i + 6 } 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 2 } {9*i } {9*i + 4 } {9*i + 6 } 0.6276 1.8828 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 3 } {9*i } {9*i + 4 } {9*i + 6 } 0.6276 1.8828 0.0 -3.91622 \n")

            dihedrals.append(f"{9*i + 1 } {9*i } {9*i + 4 } {9*i + 7 } 0.97905 2.93716 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 2 } {9*i } {9*i + 4 } {9*i + 7 } 0.97905 2.93716 0.0 -3.91622 \n")
            dihedrals.append(f"{9*i + 3 } {9*i } {9*i + 4 } {9*i + 7 } 0.97905 2.93716 0.0 -3.91622 \n")

            dihedrals.append(f"{9*i } {9*i + 4 } {9*i + 7 } {9*i + 8 } -0.4431 3.83255 0.72801 -4.11705 \n")

            dihedrals.append(f"{9*i + 5 } {9*i + 4 } {9*i + 7 } {9*i + 8 } 0.94140 2.82420 0.0 -3.76560 \n")
            dihedrals.append(f"{9*i + 6 } {9*i + 4 } {9*i + 7 } {9*i + 8 } 0.94140 2.82420 0.0 -3.76560 \n")

        # And now we do the same for water
        for i in range(nr_h20):
            print(f"{i}th water molecule      ", end="\r")
            random_displacement = np.random.uniform(0, box_size, (3))
            water = np.asarray([water_patern[0] + random_displacement,
                               water_patern[1] + random_displacement,
                               water_patern[2] + random_displacement]) 

            

            while check_placement_per_atom(water, pos, box_size, tol_h20):
                random_displacement = np.random.uniform(0, box_size, (3))
                water = np.asarray([water_patern[0] + random_displacement,
                               water_patern[1] + random_displacement,
                               water_patern[2] + random_displacement]) 

            for atom in water:
                pos.append(atom)
            

            for j, atom in enumerate(water_atoms):
                xyz_file.write(helper.atom_string(atom, water[j]))
            
            index = i + 3*nr_ethanol

            bonds.append(f"{3*index} {3*index+1} 5024.16 0.9572 \n")
            bonds.append(f"{3*index} {3*index+2} 5024.16 0.9572 \n")

            angles.append(f"{3*index+1} {3*index} {3*index+2} 628.02 1.8242181 \n")

            LJ.append(f"{3*index} 3.15061 0.66386 \n")

            molecules.append(f"{3*index} {3*index+1} {3*index+2} \n")

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

@jit(nopython=True, cache=True)
def distance(pos_1, pos_2, box_length):
    x = helper.abs_min(pos_1[0] - pos_2[0], pos_1[0]  - pos_2[0] + box_length, pos_1[0]  - pos_2[0] - box_length)  

    y = helper.abs_min(pos_1[1] - pos_2[1], 
                pos_1[1]  - pos_2[1] + box_length, 
                pos_1[1]  - pos_2[1] - box_length) 
        
    z = helper.abs_min(pos_1[2] - pos_2[2], 
                pos_1[2]  - pos_2[2] + box_length, 
                pos_1[2]  - pos_2[2] - box_length)        
        
    return helper.norm(x,y,z)

def check_placement_per_atom(molecule, pos, box, tol):
    """
    Checks if molecule can be placed in the box
    """
    for com in pos:
        for atom in molecule:
            if distance(atom, com, box) < tol:
                return True
    return False

# Testing of the functions
if __name__ == "__main__":
    #pos, atom_names, atoms = read_xyz("data/water_small.xyz")
    #bonds, const_bonds, angles, const_angles, lj, const_lj, molecules, dihedrals, const_dihedrals = read_topology("data/water_small.itp", atoms, 5)  

    nr_h20 = 601
    tol_h20 = 1.82
    nr_ethanol = 100
    tol_eth = 1.82
    box_size = 30
    output_file_xyz = "data/mix_3nm_2.xyz"
    output_file_top = "data/mix_3nm_2.itp"

    #create_dataset(nr_h20, nr_ethanol, tol_h20, tol_eth, box_size, output_file_xyz, output_file_top)

    radial_distribution_function("output/result_water_5nm.xyz", "data/water.itp", 0.05, box_size = 50)
    