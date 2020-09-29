#pragma once

#include <array>
#include <petscdmda.h>
#include "appctx.h"

/**
* Time steps system of ODEs with RK4 using the built-in PETSc routines TS.
* Inputs: da        - DMDA object
*         appctx    - Application context
a         Tend      - Final time
*         dt        - Time step
*         v         - Working vector. Should contain initial data.
*         rhs       - RHS function. Inputs (required by petsc): (TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) 
**/
PetscErrorCode RK4_petsc(const DM da, AppCtx appctx, const PetscScalar Tend, const PetscScalar dt, Vec v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *));


/**
* Time steps system of ODEs with RK4 using standard looping.
* Inputs: da        - DMDA object
*         appctx    - Application context
a         Tend      - Final time
*         dt        - Time step
*         v         - Working vector. Should contain initial data.
*         rhs       - RHS function. Inputs: (DM da, PetscReal t, Vec v_src, Vec v_dst, void *ctx) 
**/
PetscErrorCode RK4_custom(const DM da, AppCtx appctx, const PetscScalar Tend, PetscScalar dt, Vec v, PetscErrorCode (*rhs)(DM, PetscReal, Vec, Vec, AppCtx *)); 


/**
* Finds local indices of inner points (no ghost points).
* Inputs: da          - 1D, 2D or 3D DMDA object.
*         appctx      - Application context.
a         ln_tot      - Pointer to integer representing number of inner points.
*         linner_ids  - Pointer to array containing inner indices.
**/
PetscErrorCode get_local_inner_ids(DM da, AppCtx appctx, PetscInt *ln_tot, PetscInt **linner_ids);


/**
* Computes v[i] = y[i] + alpha*x[i] for specified indices i.
* Inputs: v, k1, k2, k3, k4   - PETSc vectors used for computation.
*         alpha               - PETSc scalar used for computation.
*         ln                  - Number of indices to update.
*         linner_ids          - Vector of length ln containing the indices to update.
**/
PetscErrorCode apply_WAXPY(Vec v, const PetscScalar alpha, const Vec x, const Vec y, const PetscInt ln, const PetscInt *linner_ids);


/**
* Computes v[i] = v[i] + alpha*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for specified indices i.
* Inputs: v, k1, k2, k3, k4   - PETSc vectors used for computation.
*         alpha               - PETSc scalar used for computation, alpha = dt/6 for RK4.
*         ln                  - Number of indices to update.
*         linner_ids          - Vector of length ln containing the indices to update.
**/
PetscErrorCode rk4_util(Vec v, const Vec k1, const Vec k2, const Vec k3, const Vec k4, const PetscScalar alpha, const PetscInt ln, const PetscInt *linner_ids);


/////////// IMPLEMENTATIONS ///////////

PetscErrorCode RK4_petsc(const DM da, AppCtx appctx, const PetscScalar Tend, const PetscScalar dt, Vec v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *))
{
  TS             ts;
  TSAdapt        adapt;

  TSCreate(PETSC_COMM_WORLD, &ts);
  
  // Problem type and RHS function
  TSSetProblemType(ts, TS_LINEAR);
  TSSetRHSFunction(ts, NULL, rhs, &appctx);
  
  // Integrator
  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK4);
  TSGetAdapt(ts, &adapt);
  TSAdaptSetType(adapt,TSADAPTNONE);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  // DM context
  TSSetDM(ts,da);

  TSSetSolution(ts, v);
  TSSetTime(ts,0);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,Tend);

  // Set all options
  TSSetFromOptions(ts);

  // Simulate
  TSSolve(ts,v);

  TSDestroy(&ts);

  return 0;
}

PetscErrorCode RK4_custom(const DM da, AppCtx appctx, const PetscScalar Tend, PetscScalar dt, Vec v, PetscErrorCode (*rhs)(DM, PetscReal, Vec, Vec, AppCtx *)) 
{
  Vec k1, k2, k3, k4, tmp;
  PetscScalar t = 0.0, dtDIV2, dtDIV6;
  PetscInt tidx, ln_tot, *linner_ids, tlen;

 	tlen = round(Tend/dt);
  if (abs(tlen*dt - Tend) > 1e-14)
  {
    PetscPrintf(PETSC_COMM_WORLD,"Warning: Non-matching time step. dt*tlen = %.12f.\nChanging timestep from %e to %e.\n",dt*tlen,dt,Tend/tlen);
    dt = Tend/tlen;
  } 
  PetscPrintf(PETSC_COMM_WORLD,"tlen: %d, dt: %e\n",tlen,dt);

  dtDIV2 = 0.5*dt;
  dtDIV6 = dt/6;

  // Find which local indices to update (non-ghost points)
  get_local_inner_ids(da, appctx, &ln_tot, &linner_ids);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Apply RK4
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMGetLocalVector(da, &k1);
  DMGetLocalVector(da, &k2);
  DMGetLocalVector(da, &k3);
  DMGetLocalVector(da, &k4);
  DMGetLocalVector(da, &tmp);

  for (tidx = 0; tidx < tlen; tidx++) {
    rhs(da, t, v, k1, &appctx); // k1 = D*v
    apply_WAXPY(tmp, dtDIV2, k1, v, ln_tot, linner_ids); // tmp = v + 0.5*dt*k1

    rhs(da, t + dtDIV2, tmp, k2, &appctx); // k2 = D*(v + 0.5*dt*k1)
    apply_WAXPY(tmp, dtDIV2, k2, v, ln_tot, linner_ids); // tmp = v + 0.5*dt*k2

    rhs(da, t + dtDIV2, tmp, k3, &appctx); // k3 = D*(v + 0.5*dt*k2)
    apply_WAXPY(tmp, dt, k3, v, ln_tot, linner_ids); // tmp = v + dt*k3

    rhs(da, t + dt, tmp, k4, &appctx); // k4 = D*(v + dt*k3)

    rk4_util(v, k1, k2, k3,k4, dtDIV6, ln_tot, linner_ids); // v = v + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    t = t + dt;
  }
  PetscPrintf(PETSC_COMM_WORLD,"Final t: %.8e\n",t);

  DMRestoreLocalVector(da, &k1);
  DMRestoreLocalVector(da, &k2);
  DMRestoreLocalVector(da, &k3);
  DMRestoreLocalVector(da, &k4);
  DMRestoreLocalVector(da, &tmp);

  return 0;
}

PetscErrorCode get_local_inner_ids(DM da, AppCtx appctx, PetscInt *ln_tot, PetscInt **linner_ids)
{
  PetscInt i, j, k, l, idx, nx, ny, nz, lnx, lny, sw, dof;
  int dim;

  DMDAGetStencilWidth(da, &sw);
  DMDAGetCorners(da,NULL,NULL,NULL,&nx,&ny,&nz);
  DMDAGetGhostCorners(da,NULL,NULL,NULL,&lnx,&lny,NULL);
  DMDAGetInfo(da,&dim,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);

  PetscInt li_start[3] = {0,0,0}, li_end[3] = {1,1,1};

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Find starting and stopping indices in each dimension
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (dim > 0) 
  {
    if (appctx.i_start[0] == 0) {                 // Left x
      li_start[0] = 0;
      li_end[0] = nx*dof;
    } else if (appctx.i_end[0] == appctx.N[0]) {  // Right x
      li_start[0] = sw*dof;
      li_end[0] = (sw + nx)*dof;
    } else {                                      // Center x
      li_start[0] = sw*dof;
      li_end[0] = (sw + nx)*dof;
    }
  }

  if (dim > 1)
  {
    if (appctx.i_start[1] == 0) {                 // Left y
      li_start[1] = 0;
      li_end[1] = ny*dof;
    } else if (appctx.i_end[1] == appctx.N[1]) {  // Right y
      li_start[1] = sw*dof;
      li_end[1] = (sw + ny)*dof;
    } else {                                      // Center y
      li_start[1] = sw*dof;
      li_end[1] = (sw + ny)*dof;
    }      
  }

  if (dim > 2) // NOT BUG TESTED!
  {
    if (appctx.i_start[2] == 0) {                 // Left z
      li_start[2] = 0;
      li_end[2] = nz*dof;
    } else if (appctx.i_end[2] == appctx.N[2]) {  // Right z
      li_start[2] = sw*dof;
      li_end[2] = (sw + nz)*dof;
    } else {                                      // Center z
      li_start[2] = sw*dof;
      li_end[2] = (sw + nz)*dof;
    }  
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Loop through inner points and save array indices.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  *ln_tot = nx*ny*nz*dof;
  *linner_ids = (PetscInt *) malloc(*ln_tot*sizeof(PetscInt));
  PetscInt count = 0;
  for (l = 0; l < dof; l++) {
    for (k = li_start[2]; k < li_end[2]; k++) {
      for (j = li_start[1]; j < li_end[1]; j++) {
        for (i = li_start[0]; i < li_end[0]; i++) {
          idx = l + i + lnx*(j + lny*k);
          (*linner_ids)[count] = idx;
          count++;
        }  
      }
    }  
  }


  return 0;
}

PetscErrorCode apply_WAXPY(Vec v, const PetscScalar alpha, const Vec x, const Vec y, const PetscInt ln, const PetscInt *linner_ids) 
{
  PetscErrorCode     ierr;
  PetscInt           i;
  PetscScalar        *v_arr;
  const PetscScalar  *x_arr, *y_arr;

  ierr = VecGetArrayRead(x,&x_arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y,&y_arr);CHKERRQ(ierr);
  ierr = VecGetArray(v,&v_arr);CHKERRQ(ierr);

  for (i = 0; i < ln; i++) {
    v_arr[linner_ids[i]] = y_arr[linner_ids[i]] + alpha*x_arr[linner_ids[i]];
  }

  ierr = VecRestoreArrayRead(x,&x_arr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(y,&y_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&v_arr);CHKERRQ(ierr);

  return 0;
}

PetscErrorCode rk4_util(Vec v, const Vec k1, const Vec k2, const Vec k3, const Vec k4, const PetscScalar alpha, const PetscInt ln, const PetscInt *linner_ids) 
{
  PetscErrorCode     ierr;
  PetscInt           i;
  PetscScalar        *v_arr;
  const PetscScalar  *k1_arr,*k2_arr,*k3_arr,*k4_arr;

  ierr = VecGetArrayRead(k1,&k1_arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(k2,&k2_arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(k3,&k3_arr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(k4,&k4_arr);CHKERRQ(ierr);
  ierr = VecGetArray(v,&v_arr);CHKERRQ(ierr);

  for (i = 0; i < ln; i++) {
    v_arr[linner_ids[i]] += alpha*(k1_arr[linner_ids[i]] + 2*(k2_arr[linner_ids[i]] + k3_arr[linner_ids[i]]) + k4_arr[linner_ids[i]]);
  }

  ierr = VecRestoreArrayRead(k1,&k1_arr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(k2,&k2_arr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(k3,&k3_arr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(k4,&k4_arr);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&v_arr);CHKERRQ(ierr);

  return 0;
}