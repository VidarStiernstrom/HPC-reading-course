# Optimization of a distributed finite difference code for simulation of acoustic wave propagation

Requires PETSc. For instructions on how to install PETSc see https://www.mcs.anl.gov/petsc/documentation/installation.html
In order to build you must first set the environment variables `PETSC_DIR` and `PETSC_ARCH`. See https://www.mcs.anl.gov/petsc/documentation/installation.html#envvars
Then build to solver by
  `make OPERATORFLAGS=-DSBP_OPERATOR_ORDER=N` where `N` is one of `(2,4,6)`. If `OPERATORFLAGS` is not specified, the default order used is 4.
