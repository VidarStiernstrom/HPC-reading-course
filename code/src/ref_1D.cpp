static char help[] = "Solves 1D reflection problem.\n";

#define PROBLEM_TYPE_1D_O6

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/reflection.h"
#include "timestepping.h"
#include "appctx.h"
#include "grids/create_layout.h"
#include "grids/grid_function.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

extern PetscScalar theta1(PetscScalar x, PetscScalar t);
extern PetscScalar theta2(PetscScalar x, PetscScalar t);
extern PetscErrorCode analytic_solution(const DM& da, const PetscScalar t, const AppCtx& appctx, const Vec& v_analytic, const PetscScalar W);
extern PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx);
extern PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx *appctx);
extern PetscErrorCode rhs_TS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode rhs(DM, PetscReal, Vec, Vec, AppCtx *);
extern PetscScalar gaussian(PetscScalar);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, N, n, dofs;
  PetscScalar    xl, xr, h, hi, dt, t0, Tend, CFL;
  PetscReal      l2_error, max_error, H_error;

  AppCtx         appctx;
  PetscBool      write_data, use_custom_ts, use_custom_sc;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (get_inputs_1d(argc, argv, &N, &Tend, &CFL, &use_custom_ts, &use_custom_sc) == -1) {
    PetscFinalize();
    return -1;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // Space
  xl = -1;
  xr = 1;
  hi = (N-1)/(xr-xl);
  h = 1.0/hi;

  // Time
  t0 = 0;
  dt = CFL*h;

  dofs = 2;

  // Velocity field a(i,j) = 1
  auto a = [](const PetscInt i){ return 1.5;};

  // Set if data should be written.
  write_data = PETSC_FALSE;

  PetscScalar L = xr - xl; // domain width
  if ((Tend < L - L/4) || (Tend > L + L/4)) {
    PetscPrintf(PETSC_COMM_WORLD,"--- Warning ---\nTend = %f is outside range of known solution T = [%f,%f]\n",Tend,L - L/4,L + L/4);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, dofs, stencil_radius, NULL, &da);CHKERRQ(ierr);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDAGetCorners(da,&i_xstart,NULL,NULL,&n,NULL,NULL);
  i_xend = i_xstart + n;

  // Populate application context.
  appctx.N = {N};
  appctx.hi = {hi};
  appctx.h = {h};
  appctx.xl = xl;
  appctx.i_start = {i_xstart};
  appctx.i_end = {i_xend};
  appctx.dofs = dofs;
  appctx.a = a;
  appctx.sw = stencil_radius;
  appctx.layout = grid::create_layout_1d(da);

  // Extract local to local scatter context
  if (use_custom_sc) {
    build_ltol_1D(da, &appctx.scatctx);
  } else {
    DMDAGetScatter(da, NULL, &appctx.scatctx);
  }

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);
  
  // Initial solution, starting time and end time.
  set_initial_condition(da, v, &appctx);

  if (write_data) write_vector_to_binary(v,"data/ref_1D","v_init");

  ierr = DMCreateLocalVector(da,&vlocal);CHKERRQ(ierr);
  DMGlobalToLocalBegin(da,v,INSERT_VALUES,vlocal);  
  DMGlobalToLocalEnd(da,v,INSERT_VALUES,vlocal);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  if (use_custom_ts) {
    RK4_custom(da, appctx, Tend, dt, vlocal, rhs);  
  } else {
    RK4_petsc(da, appctx, Tend, dt, vlocal, rhs_TS);  
  }
  
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  DMLocalToGlobalBegin(da,vlocal,INSERT_VALUES,v);
  DMLocalToGlobalEnd(da,vlocal,INSERT_VALUES,v);

  analytic_solution(da, Tend, appctx, v_analytic, xr-xl);
  get_error(da, v, v_analytic, &v_error, &H_error, &l2_error, &max_error, appctx);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, the H-error is: %g and the maximum error is %g\n",l2_error,H_error,max_error);

  if (write_data)
  {
    write_vector_to_binary(v,"data/ref_1D","v");
    write_vector_to_binary(v_error,"data/ref_1D","v_error");

    char tmp_str[200];
    std::string data_string;
    sprintf(tmp_str,"%d\t%d\t%d\t%e\t%f\t%f\t%e\t%e\t%e\n",size,N,-1,dt,Tend,elapsed_time,l2_error,H_error,max_error);
    data_string.assign(tmp_str);
    write_data_to_file(data_string, "data/ref_1D", "data.tsv");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_analytic);
  VecDestroy(&v_error);
  DMDestroy(&da);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode rhs_TS(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) // Function to utilize PETSc TS.
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;

  TSGetDM(ts,&da);
  rhs(da, t, v_src, v_dst, appctx);
  return 0;
}

PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx) {
  PetscScalar **arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);
  *l2_error = sqrt(appctx.h[0])*(*l2_error);

  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(da, *v_error, &arr);
  *H_error = appctx.H.get_norm_1D(arr, appctx.h[0], appctx.N[0], appctx.i_start[0], appctx.i_end[0], appctx.dofs);
  DMDAVecRestoreArrayDOF(da, *v_error, &arr);

  return 0;
}

PetscErrorCode analytic_solution(const DM& da, const PetscScalar t, const AppCtx& appctx, const Vec& v_analytic, const PetscScalar W)
{
  PetscScalar x, **array_analytic;
  DMDAVecGetArrayDOF(da,v_analytic,&array_analytic);
  for (PetscInt i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
  {
    x = appctx.xl + i/appctx.hi[0];
    array_analytic[i][0] = theta1(x,W - t) - theta2(x, -W + t);
    array_analytic[i][1] = theta1(x,W - t) + theta2(x, -W + t);
  }
  DMDAVecRestoreArrayDOF(da,v_analytic,&array_analytic);  

  return 0;
}

/**
* Set initial condition.
* Inputs: v      - vector to place initial data
*         appctx - application context, contains necessary information
**/
PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx *appctx) 
{
  PetscInt i; 
  PetscScalar **varr, x;

  DMDAVecGetArrayDOF(da,v,&varr);    

  for (i = appctx->i_start[0]; i < appctx->i_end[0]; i++) {
    x = appctx->xl + i/appctx->hi[0];
    varr[i][0] = theta2(x,0) - theta1(x,0);
    varr[i][1] = theta2(x,0) + theta1(x,0);
  }
  DMDAVecRestoreArrayDOF(da,v,&varr);  
  return 0;
}

/**
* Required function for initial and analytical solution
**/
PetscScalar theta1(PetscScalar x, PetscScalar t) 
{
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar));
}

/**
* Required function for initial and analytical solution
**/
PetscScalar theta2(PetscScalar x, PetscScalar t) 
{
  return -theta1(x,t);
}

PetscErrorCode rhs(DM da, PetscReal t, Vec v_src, Vec v_dst, AppCtx *appctx)
{
  PetscScalar       *array_src, *array_dst;

  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);

  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  auto gf_src = grid::grid_function_1d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_1d<PetscScalar>(array_dst, appctx->layout);\

  sbp::reflection_apply1(appctx->D1, gf_src, gf_dst, appctx->i_start[0], appctx->i_end[0], appctx->N[0], appctx->hi[0]);

// Apply BC
  if (appctx->i_start[0] == 0) {
    gf_dst(0,0) = 0.0;  
  }
  if (appctx->i_end[0] == appctx->N[0]) {
    gf_dst(appctx->N[0]-1,0) = 0.0;
  }

  // Restore arrays
  VecRestoreArray(v_src, &array_src);
  VecRestoreArray(v_dst, &array_dst);
  return 0;
}


