static char help[] = "Solves 1D reflection problem.\n";

#include <petsc.h>
#include <array>
#include "reflection_rhs.h"
#include "sbpops/op_defs.h"
#include "time_stepping/ts_rk.h"
#include "grids/create_layout.h"
#include "grids/grid_function.h"
#include "util/io_util.h"
#include "util/vec_util.h"
#include "scatter_ctx/scatter_ctx.h"

struct AppCtx{
    std::array<PetscInt,2> ind_i;
    PetscScalar hi, h, xl, sw;;
    PetscInt N, dofs;
    const FirstDerivativeOp D1;
    VecScatter scatctx;
    grid::partitioned_layout_1d layout;
};

PetscScalar theta1(PetscScalar x, PetscScalar t);
PetscScalar theta2(PetscScalar x, PetscScalar t);
PetscErrorCode initial_condition(const DM da, Vec v, const AppCtx& appctx);
PetscErrorCode analytic_solution(const DM da, const PetscScalar t, const AppCtx& appctx, const Vec v_analytic, const PetscScalar W);
PetscErrorCode rhs(TS, PetscReal, Vec, Vec, void *);
PetscErrorCode rhs_serial(TS, PetscReal, Vec, Vec, void *);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, N, n, dofs;
  PetscScalar    xl, xr, h, hi, dt, t0, Tend, CFL;
  PetscReal      l2_error, max_error;

  AppCtx         appctx;
  PetscBool      write_data, use_custom_sc;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (get_inputs(argc, argv, N, Tend, CFL, use_custom_sc) == -1) {
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
  dofs = 2;

  // Time
  t0 = 0;
  dt = CFL*h;

  // Set if data should be written.
  write_data = PETSC_FALSE;

  PetscScalar L = xr - xl; // domain width
  if ((Tend < L - L/4) || (Tend > L + L/4)) {
    PetscPrintf(PETSC_COMM_WORLD,"--- Warning ---\nTend = %f is outside range of known solution T = [%f,%f]\n",Tend,L - L/4,L + L/4);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  stencil_radius = (appctx.D1.interior_stencil_width()-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, dofs, stencil_radius, NULL, &da);CHKERRQ(ierr);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDAGetCorners(da,&i_xstart,NULL,NULL,&n,NULL,NULL);
  i_xend = i_xstart + n;

  // Populate application context.
  appctx.N = N;
  appctx.hi = hi;
  appctx.h = h;
  appctx.xl = xl;
  appctx.ind_i = {i_xstart, i_xend};
  appctx.dofs = dofs;
  appctx.sw = stencil_radius;
  appctx.layout = grid::create_layout_1d(da);

  // Extract local to local scatter context
  if (use_custom_sc) {
    scatter_ctx_ltol(da, appctx.scatctx);
  } else {
    DMDAGetScatter(da, NULL, &appctx.scatctx);
  }

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);
  
  // Initial solution, starting time and end time.
  initial_condition(da, v, appctx);

  if (write_data) write_vector_to_binary(v,"data/reflection","v_init");

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
  
  if (size == 1) {
    ts_rk4(da, Tend, dt, vlocal, rhs_serial, &appctx);
  }
  else {
    ts_rk4(da, Tend, dt, vlocal, rhs, &appctx);  
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
  l2_error = error_l2(v,v_analytic, appctx.h);
  max_error = error_max(v,v_analytic);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, and the maximum error is %g\n",l2_error,max_error);

  if (write_data) {
    write_vector_to_binary(v,"data/reflection","v");
    Vec v_error = compute_error(v,v_analytic);
    write_vector_to_binary(v_error,"data/reflection","v_error");
    VecDestroy(&v_error);
    char tmp_str[200];
    std::string data_string;
    sprintf(tmp_str,"%d\t%d\t%d\t%e\t%f\t%f\t%e\t%e\n",size,N,-1,dt,Tend,elapsed_time,l2_error,max_error);
    data_string.assign(tmp_str);
    write_data_to_file(data_string, "data/reflection", "data.tsv");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_analytic);
  DMDestroy(&da);
  
  ierr = PetscFinalize();
  return ierr;
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

PetscErrorCode analytic_solution(const DM da, const PetscScalar t, const AppCtx& appctx, const Vec v_analytic, const PetscScalar W)
{
  PetscScalar x, **array_analytic;
  DMDAVecGetArrayDOF(da,v_analytic,&array_analytic);
  for (PetscInt i = appctx.ind_i[0]; i < appctx.ind_i[1]; i++)
  {
    x = appctx.xl + i*appctx.h;
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
PetscErrorCode initial_condition(const DM da, Vec v, const AppCtx& appctx) 
{
  PetscInt i; 
  PetscScalar **varr, x;

  DMDAVecGetArrayDOF(da,v,&varr);    

  for (i = appctx.ind_i[0]; i < appctx.ind_i[1]; i++) {
    x = appctx.xl + i*appctx.h;
    varr[i][0] = theta2(x,0) - theta1(x,0);
    varr[i][1] = theta2(x,0) + theta1(x,0);
  }
  DMDAVecRestoreArrayDOF(da,v,&varr);  
  return 0;
}


PetscErrorCode rhs(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  PetscScalar       *array_src, *array_dst;
  AppCtx *appctx = (AppCtx*) ctx;
  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);  
  auto gf_src = grid::grid_function_1d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_1d<PetscScalar>(array_dst, appctx->layout);
  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  reflection_local(gf_dst, gf_src, appctx->ind_i, appctx->sw, appctx->D1, appctx->hi);
  reflection_bc(gf_dst, gf_src, appctx->ind_i);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  reflection_overlap(gf_dst, gf_src, appctx->ind_i, appctx->sw, appctx->D1, appctx->hi);
  

  // Restore arrays
  VecRestoreArray(v_src, &array_src);
  VecRestoreArray(v_dst, &array_dst);
  return 0;
}

PetscErrorCode rhs_serial(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  PetscScalar       *array_src, *array_dst;
  AppCtx *appctx = (AppCtx*) ctx;
  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);  
  auto gf_src = grid::grid_function_1d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_1d<PetscScalar>(array_dst, appctx->layout);
  reflection_serial(gf_dst, gf_src, appctx->D1, appctx->hi);
  reflection_bc_serial(gf_dst, gf_src);  
  

  // Restore arrays
  VecRestoreArray(v_src, &array_src);
  VecRestoreArray(v_dst, &array_dst);
  return 0;
}



