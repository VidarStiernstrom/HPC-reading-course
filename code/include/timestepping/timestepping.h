#pragma once

#include <array>
#include <petscdmda.h>
#include <petscts.h>

/**
* Time steps system of ODEs with RK4 using the built-in PETSc routines TS.
* Inputs: da        - DMDA object
*         Tend      - Final time
*         dt        - Time step
*         v         - Working vector. Should contain initial data.
*         rhs       - RHS function. Inputs (required by petsc): (TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) 
*         appctx    - Application context
**/
PetscErrorCode time_integrate_rk4(const DM& da, const PetscScalar Tend, const PetscScalar dt, Vec& v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* appctx);
