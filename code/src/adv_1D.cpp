static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

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
#include "diffops/advection.h"
#include "timestepping.h"
#include <petsc/private/dmdaimpl.h> 
#include "appctx.h"

extern PetscErrorCode analytic_solution(const DM&, const PetscScalar, const AppCtx&, Vec&);
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
  PetscScalar    xl, xr, hi, dt, t0, Tend;

  AppCtx         appctx;
  PetscBool      write_data, use_custom_ts;
  PetscLogDouble v1,v2,elapsed_time = 0;

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
  N = 401;
  hi = (N-1)/(xr-xl);
  
  // Time
  t0 = 0;
  dt = 0.1/hi;
  Tend = 0.4;

  // Velocity field a(i,j) = 1
  auto a = [](const PetscInt i){ return 1.5;};

  // Set if data should be written.
  write_data = PETSC_FALSE;

  // Set which time stepping to use
  use_custom_ts = PETSC_TRUE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, 1, stencil_radius, NULL, &da);CHKERRQ(ierr);
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

  // Extract local to local scatter context
  build_ltol_1D(da, &appctx.scatctx);
  // DMDAGetScatter(da, NULL, &appctx.scatctx);
  
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);
  
  // Initial solution, starting time and end time.
  analytic_solution(da, 0, appctx, v);
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

  analytic_solution(da, Tend, appctx, v_analytic);
  VecSet(v_error,0);
  VecWAXPY(v_error,-1,v,v_analytic);
  PetscReal l2_error, max_error;
  VecNorm(v_error,NORM_2,&l2_error);
  VecNorm(v_error,NORM_INFINITY,&max_error);
  l2_error = sqrt(1./(hi))*l2_error;
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error error is: %g and the maximum error is %g\n",l2_error,max_error);

  // Write solution to file
  // if (write_data)
  // {
  //   write_vector_to_binary(v,"data/sim_adv_ts","v");
  //   write_vector_to_binary(v_error,"data/sim_adv_ts","v_error");
  // }

  if (rank == 0) {
    FILE *f = fopen("data/timings_ts_SCM.txt", "a");
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

PetscErrorCode analytic_solution(const DM& da, const PetscScalar t, const AppCtx& appctx, Vec& v_analytic)
{ 
  PetscScalar x, *array_analytic;
  DMDAVecGetArray(da,v_analytic,&array_analytic);
  for (PetscInt i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
  {
    x = appctx.xl + i/appctx.hi[0];
    array_analytic[i] = gaussian(x-appctx.a(i)*t);
  }
  DMDAVecRestoreArray(da,v_analytic,&array_analytic);  

  return 0;
};

PetscScalar gaussian(PetscScalar x) {
  PetscScalar rstar = 0.1;
  return exp(-x*x/(rstar*rstar));
}

PetscErrorCode rhs(DM da, PetscReal t, Vec v_src, Vec v_dst, AppCtx *appctx)
{
  PetscScalar       **array_src, **array_dst;

  DMDAVecGetArrayDOFRead(da,v_src,&array_src);
  DMDAVecGetArrayDOF(da,v_dst,&array_dst);

  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);

  sbp::advection_apply1(appctx->D1, appctx->a, array_src, array_dst, appctx->i_start[0], appctx->i_end[0], appctx->N[0], appctx->hi[0]);

  //Apply homogeneous Dirichlet BC via injection on west and north boundary
  if (appctx->i_start[0] == 0)
  {
    array_dst[0][0] = 0.0;   
  }

  // Restore arrays
  DMDAVecRestoreArrayDOFRead(da, v_src, &array_src);
  DMDAVecRestoreArrayDOF(da, v_dst, &array_dst);
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
