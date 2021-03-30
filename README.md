# Optimization of a distributed finite difference code for simulation of acoustic wave propagation

Requires PETSc. For instructions on how to install PETSc see https://www.mcs.anl.gov/petsc/documentation/installation.html

In order to build you must first set the environment variables `PETSC_DIR` and `PETSC_ARCH`. See https://www.mcs.anl.gov/petsc/documentation/installation.html#envvars

To build the solver, from the code directory do
  `make ORDER=N` where `N` is one of `2,4,6`. If `ORDER` is not specified, the default order used is 4.

To run the solver, from the code directory do 
  `mpirun -n Nprocs bin/acoustic_wave_sim Nx Ny Tend CFL`
where `Nprocs` is the number of MPI processes, `Nx`, `Ny` are the number of points in the x- and y-directions, `Tend` is the final simulation time and `CFL` is the ration between the grid spacing and the time step. Too high `CFL` number will result in an unstable solution, while too small a `CFL` number will increase run time.
