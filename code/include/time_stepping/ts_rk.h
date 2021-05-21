#pragma once

#include <petscts.h>
#include <petscdmda.h>

/**
* Time steps system of ODEs with adaptive Runge-Kutta Fehlberg 45 using the built-in PETSc routines TS.
* Inputs: da        - DMDA context
*         t_end     - Final time
*         dt        - Time step
*         v         - Working vector. Should contain initial data.
*         rhs       - RHS function. Inputs (required by petsc): (TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) 
*         ctx       - User defined context
**/
PetscErrorCode ts_rk45(const DM da, const PetscScalar t_end, const PetscScalar dt, Vec v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* ctx);

/**
* Time steps system of ODEs with standard non-adaptive RK4 using the built-in PETSc routines TS.
* Inputs: da        - DMDA context
*         t_end     - Final time
*         dt        - Time step
*         v         - Working vector. Should contain initial data.
*         rhs       - RHS function. Inputs (required by petsc): (TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) 
*         ctx       - User defined context
**/
PetscErrorCode ts_rk4(const DM da, const PetscScalar t_end, const PetscScalar dt, Vec v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* ctx);