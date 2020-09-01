static char help[] = "help text\n";

#include <petscdmda.h>
#include <petsctime.h>
#include <math.h>
#include "sbpops/make_diff_op.h"
#include "diffops/reflection.h"

typedef struct {
  DM da;                 
  PetscInt M, istart, iend;
  PetscScalar h, hi, xl, xr, Tend;
  const sbp::D1_central<7,6,9> D1;
} AppCtx;

PetscErrorCode apply_RHS_mat(Mat, Vec, Vec);
PetscErrorCode RK4(Mat, PetscScalar, PetscScalar, Vec);
PetscErrorCode set_initial_condition(AppCtx *, Vec);
PetscScalar theta1(PetscScalar x, PetscScalar t);
PetscScalar theta2(PetscScalar x, PetscScalar t);
PetscScalar get_l2_err(Vec, PetscScalar, AppCtx *);

int main(int argc,char **argv)  {
  PetscErrorCode ierr;
  PetscInt M = 1001, dof = 2, s, width;
  Vec v;
  int myrank, size;
  PetscScalar Tend = 1.8, dt, xl = -1.0, xr = 1.0, err;
  PetscLogDouble v1,v2,elapsed_time = 0;
  Mat D;
  AppCtx appctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  appctx.xl = xl;
  appctx.xr = xr;
  appctx.M = M;
  appctx.h = (xr-xl)/(M-1);
  appctx.hi = 1.0/appctx.h;
  appctx.Tend = Tend;
  dt = 0.05*appctx.h;

  // DMDA
  const auto [iw, n_closures, closure_width] = appctx.D1.get_ranges();
  s = (iw - 1)/2.0;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, M,dof,s,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  DMDAGetCorners(appctx.da,&appctx.istart,NULL,NULL,&width,NULL,NULL);
  appctx.iend = appctx.istart + width;

  PetscPrintf(PETSC_COMM_WORLD, "Global info. M: %d, xl: %f, xr: %f, h: %f, Tend: %f\n", appctx.M, appctx.xl, appctx.xr, appctx.h, appctx.Tend);

  // Initial vector
  ierr = DMCreateGlobalVector(appctx.da,&v);CHKERRQ(ierr);
  set_initial_condition(&appctx, v);

  // Matrix shell
  DMSetMatType(appctx.da, MATSHELL);
  DMCreateMatrix(appctx.da, &D);
  MatShellSetContext(D, &appctx);
  MatShellSetOperation(D, MATOP_MULT,(void(*)(void)) apply_RHS_mat);

  // Simulation
  PetscBarrier(NULL);
  if (myrank == 0) {
    PetscTime(&v1);
  }

  RK4(D, Tend, dt, v);
  // VecView(v, PETSC_VIEWER_STDOUT_WORLD);
  
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

  // Finalizing
  DMDestroy(&appctx.da);
  VecDestroy(&v);
  ierr = PetscFinalize();

  return 0;
  }

PetscScalar get_l2_err(const Vec v, PetscScalar t, AppCtx *appctx) {
  Vec e;
  PetscScalar **varr, **earr, x, W, uan1, uan2, err;
  PetscInt i;

  DMGetGlobalVector(appctx->da, &e);
  W = appctx->xr - appctx->xl;

  DMDAVecGetArrayDOF(appctx->da,e,&earr); 
  DMDAVecGetArrayDOF(appctx->da,v,&varr); 

  for (i = appctx->istart; i < appctx->iend; i++) {
    x = appctx->xl + i*appctx->h;
    uan1 = theta1(x,W - appctx->Tend) - theta2(x, -W + appctx->Tend);
    uan2 = theta1(x,W - appctx->Tend) + theta2(x, -W + appctx->Tend);
    earr[i][0] = varr[i][0] - uan1;
    earr[i][1] = varr[i][1] - uan2;
  }
  DMDAVecRestoreArrayDOF(appctx->da,e,&earr);

  VecNorm(e,NORM_2,&err);
  err = appctx->h*err;
  DMRestoreGlobalVector(appctx->da, &e);
  
  return err;
}

PetscErrorCode set_initial_condition(AppCtx *appctx, Vec v) {
  PetscInt i; 
  PetscScalar **varr, x;

  DMDAVecGetArrayDOF(appctx->da,v,&varr);    

  for (i = appctx->istart; i < appctx->iend; i++) {
    x = appctx->xl + i*appctx->h;
    varr[i][0] = theta2(x,0) - theta1(x,0);
    varr[i][1] = theta2(x,0) + theta1(x,0);
  }
  DMDAVecRestoreArrayDOF(appctx->da,v,&varr);  
  return 0;
}

PetscScalar theta1(PetscScalar x, PetscScalar t) {
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar));
}

PetscScalar theta2(PetscScalar x, PetscScalar t) {
  return -theta1(x,t);
}

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

  sbp::reflection_apply(appctx->D1, xarr, yarr, appctx->istart, appctx->iend, appctx->M, appctx->hi);

  DMDAVecRestoreArrayDOF(appctx->da,xlocal, &xarr);
  DMRestoreLocalVector(appctx->da,&xlocal);

  // Apply BC
  if (appctx->istart == 0) {
    yarr[0][0] = 0.0;  
  }
  if (appctx->iend == appctx->M) {
    yarr[appctx->M-1][0] = 0.0;
  }

  DMDAVecRestoreArrayDOF(appctx->da,y, &yarr);
  return 0;
}

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

