# Hyperbolic-system-SBP-FD-solver

The repo was started to organize a PhD reading course on HPC for PDE:s given at the division of scientific computing, Uppsala University in the fall 2020 - spring 2021. The main goal of the project is to develop an distributed matrix free finite difference code for solving systems of hyperbolic PDE:s based on the SBP framework. 

Parallelization is carried out using the PETSc library. For instructions on how to install PETSc see https://www.mcs.anl.gov/petsc/documentation/installation.html
In order to build you must first set the environment variables PETSC_DIR and PETSC_ARCH. See https://www.mcs.anl.gov/petsc/documentation/installation.html#envvars

The following demos are available: 
- Acoustic wave equation on first order form in 2D (`acowave_2D`)
- Advection equation in 1D and 2D (`adv_1D`, `adv_2D`)
- The reflection problem (`ref_1D`)

To build a demo, from the code directory do `make target order=N` where target is one of `acowave_2D`, `adv_1D`, `adv_2D`, `ref_1D`)
`N` is one of 2,4,6, specifying the order of accuracy of the SBP operators used in the simulation. If not specified, the default order used is 4.

- To build with optimization flags do `make opt app=target order=N`
- To build with debug flags do `make debug app=target order=N`
- To build with optimization and debug flags do `make opt-debug app=target order=N`

To run the solver, from the code directory do `mpirun -n Nprocs bin/target` to get more information on inputs for the demo.

Authors:
Vidar Stiernström
Gustav Eriksson
