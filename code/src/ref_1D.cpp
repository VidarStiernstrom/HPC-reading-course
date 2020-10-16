static char help[] = "Solves 1D reflection problem.\n";

#define PROBLEM_TYPE_1D_O6

#include <algorithm>
#include <cmath>
#include <string>
// #include <filesystem>
#include <petscsys.h>
#include <petscdmda.h>
#include <petscvec.h>
#include <functional>
#include <petscts.h>
#include "sbpops/D1_central.h"
#include "diffops/reflection.h"
#include "timestepping.h"
#include <petsc/private/dmdaimpl.h> 
#include "appctx.h"
#include "grids/create_layout.h"
#include "grids/grid_function.h"
#include "IO_utils.h"

extern PetscScalar theta1(PetscScalar x, PetscScalar t);
extern PetscScalar theta2(PetscScalar x, PetscScalar t);
extern PetscScalar get_l2_err(DM da, const Vec v, PetscScalar t, AppCtx *appctx, PetscScalar W); 
extern PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx *appctx);
extern PetscErrorCode rhs_TS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode rhs(DM, PetscReal, Vec, Vec, AppCtx *);
extern PetscScalar gaussian(PetscScalar);
extern PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);
extern PetscErrorCode build_ltol_1D(DM da, VecScatter *ltol);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, N, n;
  PetscScalar    xl, xr, hi, dt, t0, Tend, l2_error, CFL;

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
  
  // Time
  t0 = 0;
  dt = CFL/hi;

  // Velocity field a(i,j) = 1
  auto a = [](const PetscInt i){ return 1.5;};

  // Set if data should be written.
  write_data = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, 2, stencil_radius, NULL, &da);CHKERRQ(ierr);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDAGetCorners(da,&i_xstart,NULL,NULL,&n,NULL,NULL);
  i_xend = i_xstart + n;

  // Populate application context.
  appctx.N = {N};
  appctx.hi = {hi};
  appctx.xl = xl;
  appctx.i_start = {i_xstart};
  appctx.i_end = {i_xend};
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

  // if (write_data) write_vector_to_binary(v,"data/sim_adv_ts","v_init");

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

  l2_error = get_l2_err(da, v, Tend, &appctx, xr - xl);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error error is: %g\n",l2_error);

  // Write solution to file
  // if (write_data)
  // {
  //   write_vector_to_binary(v,"data/sim_adv_ts","v");
  //   write_vector_to_binary(v_error,"data/sim_adv_ts","v_error");
  // }

  if (rank == 0) {
    FILE *f;
    if (use_custom_ts) {
      if (use_custom_sc) {
        f = fopen("data/timings_tsC_scC.txt", "a");
      } else {
        f = fopen("data/timings_tsC_scP.txt", "a");
      }
    } else {
      if (use_custom_sc) {
        f = fopen("data/timings_tsP_scC.txt", "a");
      } else {
        f = fopen("data/timings_tsP_scP.txt", "a");
      }
    }
    fprintf(f,"Size: %d, N: %d, dt: %e, Tend: %f, elapsed time: %f, l2-error: %e, max-error: %e\n",size,N,dt,Tend,elapsed_time,l2_error,max_error);
    fclose(f);
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

/**
* Compute l2-error of solution to the reflection problem.
* Inputs: v      - Solution vector of which to compute error
*         t      - time
*         appctx - application context, contains necessary information
**/
PetscScalar get_l2_err(DM da, const Vec v, PetscScalar t, AppCtx *appctx, PetscScalar W) 
{
  Vec e;
  PetscScalar **varr, **earr, x, uan1, uan2, err;
  PetscInt i;

  DMGetGlobalVector(da, &e);

  DMDAVecGetArrayDOF(da,e,&earr); 
  DMDAVecGetArrayDOF(da,v,&varr); 

  for (i = appctx->i_start[0]; i < appctx->i_end[0]; i++) {
    x = appctx->xl + i/appctx->hi[0];
    uan1 = theta1(x,W - t) - theta2(x, -W + t);
    uan2 = theta1(x,W - t) + theta2(x, -W + t);
    earr[i][0] = varr[i][0] - uan1;
    earr[i][1] = varr[i][1] - uan2;
  }
  DMDAVecRestoreArrayDOF(da,e,&earr);

  VecNorm(e,NORM_2,&err);
  err = err/appctx->hi[0];
  DMRestoreGlobalVector(da, &e);
  
  return err;
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

PetscErrorCode rhs_TS(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) // Function to utilize PETSc TS.
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;

  TSGetDM(ts,&da);
  rhs(da, t, v_src, v_dst, appctx);
  return 0;
}

/**
* Build local to local scatter context containing only ghost point communications
* Inputs: da        - DMDA object
*         ltol      - pointer to local to local scatter context
**/
PetscErrorCode build_ltol_1D(DM da, VecScatter *ltol)
{
  PetscInt    stencil_radius, i_xstart, i_xend, ig_xstart, ig_xend, n, i, j, ln, no_com_vals, count, N, dof;
  IS          ix, iy;
  Vec         vglobal, vlocal;
  VecScatter  gtol;

  DMDAGetStencilWidth(da, &stencil_radius);
  DMDAGetCorners(da,&i_xstart,NULL,NULL,&n,NULL,NULL);
  DMDAGetGhostCorners(da,&ig_xstart,NULL,NULL,&ln,NULL,NULL);
  DMDAGetInfo(da, NULL, &N, NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);

  i_xend = i_xstart + n;
  ig_xend = ig_xstart + ln;

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute how many elements to receive
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  no_com_vals = 0;
  if (i_xstart != 0)  // NOT LEFT BOUNDARY, RECEIVE LEFT
  {
    no_com_vals += 1;
  }
  if (i_xend != N) // NOT RIGHT BOUNDARY, RECEIVE RIGHT
  {
    no_com_vals += 1;
  }
  no_com_vals *= stencil_radius*dof;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define communication pattern, from global index ixx[i] to local index iyy[i]
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt ixx[no_com_vals], iyy[no_com_vals];
  count = 0;

  for (i = ig_xstart; i < i_xstart; i++) { // LEFT
    for (j = 0; j < dof; j++) {
      ixx[count] = dof*i + j;
      iyy[count] = dof*(i - ig_xstart) + j;
      count++;
    }
  }

  for (i = i_xend; i < ig_xend; i++) { // RIGHT
    for (j = 0; j < dof; j++) {
      ixx[count] = dof*i + j;
      iyy[count] = dof*(i - ig_xstart) + j;
      count++;
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Build global to local scatter context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,ixx,PETSC_COPY_VALUES,&ix);  
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,iyy,PETSC_COPY_VALUES,&iy);  

  DMGetGlobalVector(da, &vglobal);
  DMGetLocalVector(da, &vlocal);

  VecScatterCreate(vglobal,ix,vlocal,iy, &gtol);  
  VecScatterSetUp(gtol);

  DMRestoreGlobalVector(da, &vglobal);
  DMRestoreLocalVector(da, &vlocal);

  ISDestroy(&ix);
  ISDestroy(&iy);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Map 2D global to local scatter context to local to local (petsc source code)
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecScatterCopy(gtol,ltol);
  VecScatterDestroy(&gtol);

  PetscInt *idx,left;
  DM_DA *dd = (DM_DA*) da->data; 
  left = dd->xs - dd->Xs;
  PetscMalloc1(dd->xe-dd->xs,&idx);
  for (j=0; j<dd->xe-dd->xs; j++) 
  {
    idx[j] = left + j;
  }
  VecScatterRemap(*ltol,idx,NULL);

  return 0;
}

// PetscErrorCode write_vector_to_binary(const Vec& v, const std::string folder, const std::string file)
// { 
//   std::filesystem::create_directories(folder);
//   PetscErrorCode ierr;
//   PetscViewer viewer;
//   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(folder+"/"+file).c_str(),FILE_MODE_WRITE,&viewer);
//   ierr = VecView(v,viewer);
//   ierr = PetscViewerDestroy(&viewer);
//   CHKERRQ(ierr);
//   return 0;
// }
