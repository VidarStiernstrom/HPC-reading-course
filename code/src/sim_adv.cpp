static char help[] = "Solves advection problem u_t + u_x = 0.\n";

#include <petscdmda.h>
#include <petsctime.h>
#include <math.h>
#include "sbpops/make_diff_op.h"
#include "diffops/advection.h"

typedef struct {
  DM da;                 
  PetscInt M, istart, iend;
  PetscScalar h, hi, xl, xr, Tend;
  const sbp::D1_central<3,1,2> D1;
  std::function<PetscScalar(PetscInt)> velocity_field;
} AppCtx;

PetscErrorCode apply_RHS_mat(Mat, Vec, Vec);
PetscErrorCode RK4(Mat, PetscScalar, PetscScalar, Vec);
PetscErrorCode set_initial_condition(Vec, AppCtx *);
PetscScalar gauss(PetscScalar, PetscScalar);
PetscScalar get_l2_err(Vec, PetscScalar, AppCtx *);

int main(int argc,char **argv)  {
  PetscErrorCode ierr;
  PetscInt M, DOF, sw, width;
  Vec v;
  int myrank, size;
  PetscScalar Tend, dt, xl, xr, err, CFL;
  PetscLogDouble v1,v2,elapsed_time = 0;
  Mat D;
  AppCtx appctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set problem parameters
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  M = 2001;
  xl = -1.0;
  xr = 1.0;
  Tend = 1.8;
  CFL = 0.05;
  DOF = 1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  appctx.xl = xl;
  appctx.xr = xr;
  appctx.M = M;
  appctx.h = (xr-xl)/(M-1);
  appctx.hi = 1.0/appctx.h;
  appctx.Tend = Tend;
  dt = CFL*appctx.h;
  appctx.velocity_field = [](PetscInt i){return 1.;};

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  const auto [iw, n_closures, closure_width] = appctx.D1.get_ranges();
  sw = (iw - 1)/2.0; // DMDA stencil width
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, M, DOF, sw, NULL, &appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  DMDAGetCorners(appctx.da,&appctx.istart,NULL,NULL,&width,NULL,NULL);
  appctx.iend = appctx.istart + width;
  PetscPrintf(PETSC_COMM_WORLD, "Global info. M: %d, xl: %f, xr: %f, h: %f, Tend: %f\n", appctx.M, appctx.xl, appctx.xr, appctx.h, appctx.Tend);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create global vector and set to initial data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(appctx.da,&v);CHKERRQ(ierr);
  set_initial_condition(v, &appctx);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define matrix shell
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMSetMatType(appctx.da, MATSHELL);
  DMCreateMatrix(appctx.da, &D);
  MatShellSetContext(D, &appctx);
  MatShellSetOperation(D, MATOP_MULT,(void(*)(void)) apply_RHS_mat);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run simulation, measure elapsed time and compute error
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier(NULL);
  if (myrank == 0) {
    PetscTime(&v1);
  }

  RK4(D, Tend, dt, v);
  
  PetscBarrier(NULL);
  if (myrank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  err = get_l2_err(v, Tend, &appctx);

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f s, Error: %e\n",elapsed_time,err);

  // Writing result
  // PetscViewerBinaryOpen(PETSC_COMM_WORLD, "data/vend.dat", FILE_MODE_WRITE, &viewer);
  // VecView(v, viewer);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Destroy PetSc objects and finalize
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMDestroy(&appctx.da);
  VecDestroy(&v);
  ierr = PetscFinalize();

  return 0;
  }

/**
* Compute l2-error of solution to the reflection problem.
* Inputs: v      - Solution vector of which to compute error
*         t      - time
*         appctx - application context, contains necessary information
**/
PetscScalar get_l2_err(const Vec v, PetscScalar t, AppCtx *appctx) {
  Vec e;
  PetscScalar **varr, **earr, x, uan, err;
  PetscInt i;

  DMGetGlobalVector(appctx->da, &e);

  DMDAVecGetArrayDOF(appctx->da,e,&earr); 
  DMDAVecGetArrayDOF(appctx->da,v,&varr); 

  for (i = appctx->istart; i < appctx->iend; i++) {
    x = appctx->xl + i*appctx->h;
    uan = gauss(x,appctx->Tend);
    earr[i][0] = varr[i][0] - uan;
  }
  DMDAVecRestoreArrayDOF(appctx->da,e,&earr);

  VecNorm(e,NORM_2,&err);
  err = appctx->h*err;
  DMRestoreGlobalVector(appctx->da, &e);
  
  return err;
}

/**
* Set initial condition.
* Inputs: v      - vector to place initial data
*         appctx - application context, contains necessary information
**/
PetscErrorCode set_initial_condition(Vec v, AppCtx *appctx) {
  PetscInt i; 
  PetscScalar **varr, x;

  DMDAVecGetArrayDOF(appctx->da,v,&varr);  

  for (i = appctx->istart; i < appctx->iend; i++) {
    x = appctx->xl + i*appctx->h;
    varr[i][0] = gauss(x,0);
  }
  DMDAVecRestoreArrayDOF(appctx->da, v, &varr);
  return 0;
}

/**
* Required function for initial and analytical solution
**/
PetscScalar gauss(PetscScalar x, PetscScalar t) {
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar));
}

/**
* Define matrix multiplication for matrix shell, y = D*x. Here x and y are global vectors.
**/
PetscErrorCode apply_RHS_mat(Mat D, Vec x, Vec y) {
  AppCtx            *appctx;
  Vec               xlocal;
  PetscScalar **xarr, **yarr;

  MatShellGetContext(D,&appctx);
  DMGetLocalVector(appctx->da, &xlocal);

  DMGlobalToLocalBegin(appctx->da,x,INSERT_VALUES,xlocal);
  DMGlobalToLocalEnd(appctx->da,x,INSERT_VALUES,xlocal);

  DMDAVecGetArrayDOF(appctx->da,xlocal,&xarr);
  DMDAVecGetArrayDOF(appctx->da,y,&yarr);

  sbp::advection_apply(appctx->D1, appctx->velocity_field, xarr, yarr, appctx->istart, appctx->iend, appctx->M, appctx->hi);

  DMDAVecRestoreArrayDOF(appctx->da,xlocal, &xarr);
  DMRestoreLocalVector(appctx->da,&xlocal);

  // Apply BC
  if (appctx->istart == 0) {
    yarr[0][0] = 0.0;  
  }

  DMDAVecRestoreArrayDOF(appctx->da,y, &yarr);
  return 0;
}

/**
* Apply RK4 time stepping.
* Inputs: D - RHS matrix shell
*         Tend - final time
*         dt - time step
*         v - vector with initial and final solution
**/
PetscErrorCode RK4(Mat D, PetscScalar Tend, PetscScalar dt, Vec v) {
  Vec k1, k2, k3, k4, tmp;
  PetscScalar t = 0.0, dt05 = 0.5*dt;
  AppCtx            *appctx;

  MatShellGetContext(D,&appctx);

  DMGetGlobalVector(appctx->da, &k1);
  DMGetGlobalVector(appctx->da, &k2);
  DMGetGlobalVector(appctx->da, &k3);
  DMGetGlobalVector(appctx->da, &k4);
  DMGetGlobalVector(appctx->da, &tmp);

  while (t < Tend) {
    MatMult(D, v, k1); // k1 = D*v
    VecWAXPY(tmp, dt05, k1, v); // tmp = v + 0.5*dt*k1
    MatMult(D, tmp, k2); // k2 = D*(v + 0.5*dt*k1)
    VecWAXPY(tmp, dt05, k2, v); // tmp = v + 0.5*dt*k2
    MatMult(D, tmp, k3); // k3 = D*(v + 0.5*dt*k2)
    VecWAXPY(tmp, dt, k3, v); // tmp = v + dt*k3
    MatMult(D, tmp, k4); // k3 = D*(v + dt*k3)

    VecAXPBYPCZ(tmp,1.0,1.0,0.0,k2,k3); // tmp = k2 + k3
    VecScale(tmp,2.0); // tmp = 2*k2 + 2*k3
    VecAXPBYPCZ(tmp,1.0,1.0,1.0,k1,k4); // tmp = tmp + k1 + k4 = k1 + 2*k2 + 2*k3 + k4
    VecAXPY(v,dt/6.0,tmp); // v = v + dt/6*tmp = v + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    t = t + dt;
  }

  DMRestoreGlobalVector(appctx->da, &k1);
  DMRestoreGlobalVector(appctx->da, &k2);
  DMRestoreGlobalVector(appctx->da, &k3);
  DMRestoreGlobalVector(appctx->da, &k4);
  DMRestoreGlobalVector(appctx->da, &tmp);

  return 0;
}

