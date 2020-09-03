
static char help[] ="Solves the 1D advection equation u_t + au_x = 0 using the PETSc time stepping contexts.";

#include "petscsys.h" 
#include <petscdmda.h>
#include "petscvec.h"
#include "petscts.h"
#include "sbpops/make_diff_op.h"
#include "diffops/advection.h"

struct AppCtx{
  Mat A;
  DM da;                 
  PetscInt N, i_start, i_end;
  PetscScalar hi;
  std::function<double(int)> velocity_field;
  sbp::D1_central<5,4,6> D1;
};

extern PetscErrorCode initial_condition(const AppCtx&, Vec&);
extern PetscErrorCode analytic_solution(const AppCtx&, const PetscScalar, Vec&);
extern PetscErrorCode apply_rhs_mat(Mat A, Vec x, Vec y);
extern PetscErrorCode rhs(TS,PetscReal,Vec,Vec,void *);
extern PetscScalar theta(PetscScalar, PetscScalar);

int main(int argc,char **argv)
{
  Vec            v, v_tmp;
  PetscInt       n, stencil_radius;
  PetscScalar    xl, xr, dt, t0, Tend;
  TS             ts;
  TSAdapt        adapt;
  AppCtx         appctx;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto velocity_field = [](const int i){ return 1.; };
  xl = -1;
  xr = 1;
  appctx.N = 1001;
  appctx.hi = (appctx.N-1)/(xr-xl);
  Tend = 0.01;
  t0 = 0;
  dt = 0.05/appctx.hi;
  appctx.velocity_field = velocity_field;
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;

  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,appctx.N,1,stencil_radius,0,&appctx.da);
  DMSetFromOptions(appctx.da);
  DMSetUp(appctx.da);
  DMDASetUniformCoordinates(appctx.da, xl, xr, 0, 0, 0, 0);
  DMDAGetCorners(appctx.da,&appctx.i_start,0,0,&n,0,0);
  appctx.i_end = appctx.i_start+n;

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Create matrix Shell defining the RHS operation
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMSetMatType(appctx.da, MATSHELL);
  DMCreateMatrix(appctx.da, &appctx.A);
  MatShellSetContext(appctx.A, &appctx);
  MatShellSetOperation(appctx.A, MATOP_MULT,(void(*)(void)) apply_rhs_mat);


  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(appctx.da,&v);
  VecDuplicate(v,&v_tmp);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Setup time stepping context and set initial condition
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  initial_condition(appctx, v);
  TSCreate(PETSC_COMM_WORLD, &ts);
  TSSetProblemType(ts, TS_LINEAR);
  TSSetRHSFunction(ts, NULL, rhs, &appctx);
  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK4);
  TSGetAdapt(ts, &adapt);
  TSAdaptSetType(adapt,TSADAPTNONE);
  PetscPrintf(PETSC_COMM_WORLD,"dt: %g \n", dt);
  TSSetSolution(ts, v);
  TSSetTime(ts,t0);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,Tend);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,v); CHKERRQ(ierr);
  analytic_solution(appctx, Tend, v_tmp);
  VecAYPX(v_tmp,-1.,v);
  PetscReal l2_error;
  VecNorm(v_tmp,NORM_2,&l2_error);
  l2_error = sqrt(appctx.hi)*l2_error;
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error error is: %g\n",l2_error);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_tmp);
  DMDestroy(&appctx.da);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode initial_condition(const AppCtx& appctx, Vec& v_init)
{ 
  PetscErrorCode ierr;
  Vec coords;
  PetscScalar x, v_i;
  DMGetCoordinates(appctx.da, &coords);
  for (PetscInt i = appctx.i_start; i < appctx.i_end; i++)
  {
    VecGetValues(coords,1,&i,&x);
    v_i = theta(x,0);
    ierr = VecSetValue(v_init, i, v_i, INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(v_init);
  ierr = VecAssemblyEnd(v_init);
  CHKERRQ(ierr);
  return 0;
}

PetscErrorCode analytic_solution(const AppCtx& appctx, const PetscScalar Tend, Vec& v_analytic)
{ 
  PetscErrorCode ierr;
  Vec coords;
  PetscScalar x, v_i;
  DMGetCoordinates(appctx.da, &coords);
  for (PetscInt i = appctx.i_start; i < appctx.i_end; i++)
  {
    VecGetValues(coords,1,&i,&x);
    v_i = theta(x,Tend);
    ierr = VecSetValue(v_analytic, i, v_i, INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(v_analytic);
  ierr = VecAssemblyEnd(v_analytic);
  CHKERRQ(ierr);
  return 0;
};

PetscScalar theta(PetscScalar x, PetscScalar t) {
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar));
}


PetscErrorCode apply_rhs_mat(Mat A, Vec v_src, Vec v_dst) 
{
  AppCtx            *appctx;
  Vec               v_local;
  PetscScalar       *array_src, *array_dst;
  PetscInt          i, i_start, i_end;
  PetscErrorCode    ierr;

  
  MatShellGetContext(A,&appctx);
  DMGetLocalVector(appctx->da, &v_local);
  DMGlobalToLocalBegin(appctx->da,v_src,INSERT_VALUES,v_local);
  DMGlobalToLocalEnd(appctx->da,v_src,INSERT_VALUES,v_local);
  
  ierr = DMDAVecGetArrayRead(appctx->da, v_src, &array_src);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, v_dst, &array_dst);CHKERRQ(ierr);
  i_start = appctx->i_start;
  i_end = appctx->i_end;
  const auto [iw, n_closures, cw] = appctx->D1.get_ranges();
  if (appctx->i_start == 0){
    // Injection BC
    array_dst[0] = 0.0;
    for (i = i_start+1; i < n_closures; i++) 
    { 
      array_dst[i] = -1*sbp::advection_variable_left(appctx->D1, appctx->velocity_field, array_src, appctx->hi, i);
      i_start = n_closures;
    }
  }
  if (i_end == appctx->N)
  {
    for (i = appctx->N-n_closures; i < i_end; i++)
    {
      array_dst[i] = -1*sbp::advection_variable_right(appctx->D1, appctx->velocity_field, array_src, appctx->hi, appctx->N, i);
    }
    i_end = appctx->N-n_closures;
  }
  for (i = i_start; i < i_end; i++)
  {
    array_dst[i] = -1*sbp::advection_variable_interior(appctx->D1, appctx->velocity_field, array_src, appctx->hi, i);
  }

  DMDAVecRestoreArrayRead(appctx->da, v_src, &array_src);
  DMRestoreLocalVector(appctx->da,&v_local);
  DMDAVecRestoreArray(appctx->da, v_dst, &array_dst);
  return 0;
}

PetscErrorCode rhs(TS ts,PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;     /* user-defined application context */
  MatMult(appctx->A, v_src, v_dst);
  return 0;
}

