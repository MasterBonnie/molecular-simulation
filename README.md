# molecular-simulation
This repository is a small Molecular Dynamics (MD) simulation, written with the intent to be able to simulate
small ethanol-water mixtures. It is probably not written optimal, and so any suggestions are welcome. What 
follows is a short explanation of how the code is structured and can be used, as well as some points where
it can be improved.

The primary functionality, namely running the simulations, needs to be done trough the simulator function inside
of simulator.py. An example can be found on the bottom of this file. The function will not print any output to standard
output, but will write results to the specified files. Note that these file can become quite large if the simulation takes
a long time.

The next important functionality can be found in io_sim.py. It contains a function to generate datasets, consisting of a variable
amount of ethanol and water molecules, randomly placed inside a box of specified size. It also contains a function which will compute
the Radial Distribution Function given a xyz file. The other functions inside of it are used to read in initial positions and parameters.

The force computations happen in static_state.py
