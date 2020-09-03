
static char help[] ="Solves the 1D advection equation u_t + au_x = 0 using the PETSc time stepping contexts.";

#include "petscsys.h" 
#include <petscdmda.h>
#include "petscvec.h"
#include "petscts.h"
#include "sbpops/make_diff_op.h"
#include "diffops/advection.h"

struct AppCtx{         
  PetscInt N;
  PetscScalar hi;
  std::function<double(int)> velocity_field;
  const sbp::D1_central<5,4,6> D1;
};

extern PetscErrorCode analytic_solution(const DM&, const PetscScalar, const std::function<double(int)>&, Vec&);
extern PetscErrorCode rhs(TS,PetscReal,Vec,Vec,void *);
extern PetscScalar gaussian(PetscScalar, PetscScalar);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error;
  PetscInt       stencil_radius;
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
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  // Space
  xl = -1;
  xr = 1;
  appctx.N = 1001;
  appctx.hi = (appctx.N-1)/(xr-xl);
  
  // Time
  t0 = 0;
  Tend = 0.2;
  dt = 0.01/appctx.hi;

  // Velocity field
  bool constant_speed = true;
  std::function<double(int)> velocity_field;
  if (constant_speed)
  {
    velocity_field = [](const int i){ return 2;};
  }
  else
  {
    velocity_field = [&](const int i){ return 1.+cos(M_PI/2*(xl+i/appctx.hi));};
  }
  appctx.velocity_field = velocity_field;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,appctx.N,1,stencil_radius,0,&da);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDASetUniformCoordinates(da, xl, xr, 0, 0, 0, 0);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Setup time stepping context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  analytic_solution(da, 0, velocity_field, v);
  TSCreate(PETSC_COMM_WORLD, &ts);
  
  // Problem type and RHS function
  TSSetProblemType(ts, TS_LINEAR);
  TSSetRHSFunction(ts, NULL, rhs, &appctx);
  
  // Integrator
  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK5F);
  TSGetAdapt(ts, &adapt);
  TSAdaptSetType(adapt,TSADAPTBASIC);
  TSAdaptSetStepLimits(adapt, dt, 1./appctx.hi);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  // DM context
  TSSetDM(ts,da);

  // Initial solution, starting time and end time.
  TSSetSolution(ts, v);
  TSSetTime(ts,t0);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,Tend);

  // Set all options
  TSSetFromOptions(ts);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error (if analytic solutiona available)
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,v); CHKERRQ(ierr);
  if (constant_speed)
  {
    analytic_solution(da, Tend, velocity_field, v_analytic);
    VecSet(v_error,0);
    VecWAXPY(v_error,-1,v,v_analytic);
    PetscReal l2_error;
    VecNorm(v_error,NORM_2,&l2_error);
    l2_error = sqrt(appctx.hi)*l2_error;
    PetscPrintf(PETSC_COMM_WORLD,"The l2-error error is: %g\n",l2_error);
  }
  else
  {
    VecView(v,PETSC_VIEWER_STDOUT_WORLD);
  }
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_analytic);
  VecDestroy(&v_error);
  TSDestroy(&ts);
  DMDestroy(&da);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode analytic_solution(const DM& da, const PetscScalar t, const std::function<double(int)>& velocity_field, Vec& v_analytic)
{ 
  PetscErrorCode ierr;
  Vec coords;
  PetscScalar x, v_i;
  PetscInt    i_start, n;
  DMGetCoordinates(da, &coords);
  DMDAGetCorners(da,&i_start,NULL,NULL,&n,NULL,NULL);
  for (PetscInt i = i_start; i < i_start + n; i++)
  {
    VecGetValues(coords,1,&i,&x);
    v_i = gaussian(x,velocity_field(i)*t);
    ierr = VecSetValue(v_analytic, i, v_i, INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(v_analytic);
  ierr = VecAssemblyEnd(v_analytic);
  CHKERRQ(ierr);
  return 0;
};

PetscScalar gaussian(PetscScalar x, PetscScalar t) {
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar));
}


PetscErrorCode rhs(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;     /* user-defined application context */
  DM                da;
  Vec               v_local;
  PetscScalar       **array_src, **array_dst;
  PetscInt          i_start, n;

  TSGetDM(ts,&da);
  DMGetLocalVector(da, &v_local);
  // Scatter global vector to local vectors, communicating ghost points
  DMGlobalToLocalBegin(da,v_src,INSERT_VALUES,v_local);
  DMGlobalToLocalEnd(da,v_src,INSERT_VALUES,v_local);
  
  // Extract arrays
  DMDAVecGetArrayDOFRead(da,v_local,&array_src);
  DMDAVecGetArrayDOF(da,v_dst,&array_dst);
  
  // Get ranges and apply the advection operator
  DMDAGetCorners(da,&i_start,NULL,NULL,&n,NULL,NULL);
  sbp::advection_apply(appctx->D1, appctx->velocity_field, array_src, array_dst, i_start, i_start+n, appctx->N, appctx->hi);

  // Apply BC via injection
  if (i_start == 0) {
    array_dst[0][0] = 0.0;  
  }

  // Restore arrays
  DMDAVecRestoreArrayDOFRead(da, v_local, &array_src);
  DMRestoreLocalVector(da,&v_local);
  DMDAVecRestoreArrayDOF(da, v_dst, &array_dst);
  return 0;
}

