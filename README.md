# molecular dynamics simulation
This repository contains a small Molecular Dynamics (MD) simulation, written with the intent to be able to simulate small ethanol-water mixtures. It is probably not written optimal, and so any suggestions are welcome. What follows is a short explanation of how the code is structured and can be used, as well as some points where
it can be improved.

## Overview
The primary functionality, running the simulations, needs to be done through the simulator function inside of `simulator.py`. Example usage can be found at the bottom of this file, but should be self-explanatory. Note that the output files can become quite large if the simulation is run for a long time.

The basic working of the `simulator` function is as follows. It starts by initializing several variables needed during the simulation. This relies partly on certain I/O operation, reading in initial positions and force field constants. It also creates a fairly large numpy array, called `molecules_to_atoms`, whose purpose is to transform interactions that are calculated at molecule level to actions on the atomic level. What this boils down to is that it contains the cartesian product of the index set of molecule i and j in position (i,j).

Next we enter the main loop of the function, that will repeatedly update the positions and velocities of our particles according to some numerical integration scheme. The bulk of time is spend in calculating the pairs of molecules that need an Lennard-Jones interaction between them, and calculating the total force on the system. We will discuss these in more detail.

In general, inter molecular force fields use a cut-off distance as to not calculate interactions that contribute nothing to the overall force. We do exactly that, by calculating the distances between the centres of mass of each molecule, and based on that decide if there should be such a force between these molecules. The code now works with a naive approach, simply calculating the distance between each molecule and looking if this is less than the cut off distance. We tried an approach where the simulation domain is partitioned in smaller boxes so that only the adjacent boxes need to be considered, however this approach seemed to slow down the simulation. We are not quite sure why. The code for this is still present in the repository.

The total force on the system is calculated in `static_state.py`, specifically in `compute_force`. It heavily uses both numpy and numba in order to as fast as possible.

After this, we can optionally calculate the energy and temperature of the system and write these to ouput files. This behaviour can be controlled by several input variables to the function `simulator` function.

Other important functionality can be found in `io_sim.py`. Next to containing some basic I/O functions for reading data, it can be used to compute the radial distribution function from output trajectory (.xyz) files. It also contains a function which creates datasets of water-ethanol mixtures, by generating appropriate .xyz and .itp files.

The file `plotting.py` is an auxilary file we used to generate the plots used in the report about this project.

Lastly, a note about the variable `fill_in_molecules` in the `simulator` function. Its purpose was specify to what length molecules needed to be "filled up". When running a dataset with molecules of different lengths, we cannot use numpy arrays anymore, as they do not support variable lengths. If `fill_in_molecules` is specified, it will fill up all molecules to have the same length, which allows us to use a numpy array to represent the molecules, and we can use Numba to speed certain actions up. If you want to run the simulation faster, this should be a non-zero number. However, it does NOT work yet for datasets with molecules of different sizes. So if we have a dataset consisting of only ethanol molecules, we want to set this variable to 9, the length of an ethanol molecule. If we have a dataset of a mixture of ethanol and water, we need to set it to 0, disabling the behaviour of filling up molecules, and using the slower, but correct, functions. 

## Folder explanation
In the data folder, several datasets can be found that were used during the project. An `.xyz` file and `.itp` file with the same name belong together.

## File formats
The simulator expects two files as input, a initial position file, denoted with the `.xyz` extension, and a topology file, denoted with `.itp`. The topology file specifies which atoms are inside of the same molecule, between which atom there are bonds, etc. The syntax for the `.xyz` files is as follows: The first line denotes the total number of atoms in the file. The next line serves as a comment, and needs to be present. The lines after that will denote the atoms and their positions. The first symbol should be the element of the atom, followed by three number seperated by spaces. There should be as many of these lines as was specified in the first one. An `.xyz` may contain many blocks of this type underneath each other.

The syntax for an topology, or `.itp` file is as follows: the first line should be one of the following keywords: bonds, angles, dihedrals, molecules, LJ, followed by a number specifiyng the number of the keyword present. The next lines should contain the indices of the molecules part of the interaction, followed by possibly the parameters of the interaction. Ater this, another keyword and the same structure can follow. There are several examples in the `data/` folder.