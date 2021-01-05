static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#define PROBLEM_TYPE_1D_O2

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/ICF_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/first_order_ode.h"
#include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

struct GridCtxs{
  AppCtx F_appctx, C_appctx;
};

extern PetscErrorCode analytic_solution(const AppCtx& appctx, Vec& v_analytic);
extern PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout);
extern PetscErrorCode LHS(Mat, Vec, Vec);
extern PetscErrorCode RHS(const DM& da, const AppCtx& appctx, Vec& b);
extern PetscScalar gaussian(PetscScalar);
extern PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);
extern PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx);
extern PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, GridCtxs *gridctxs);
extern PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, GridCtxs *gridctxs);

int main(int argc,char **argv)
{ 
  Vec            v, v_error, v_analytic;
  Mat            D, D_pc;
  PC             pc;
  PetscInt       stencil_radius, i_xstart, i_xend, N, n, dofs, stencil_radius_pc, i_xstart_pc, i_xend_pc, N_pc, n_pc;
  PetscScalar    xl, xr, hi, h, hi_pc, h_pc;
  PetscReal      l2_error, max_error, H_error;

  GridCtxs       gridctxs;
  PetscBool      write_data, use_custom_ts, use_custom_sc;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  // if (get_inputs_1d(argc, argv, &N, &Tend, &CFL, &use_custom_ts, &use_custom_sc) == -1) {
  //   PetscFinalize();
  //   return -1;
  // }

  use_custom_ts = PETSC_FALSE;
  use_custom_sc = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // Space
  N = 1201;
  xl = -1;
  xr = 1;
  hi = (N-1)/(xr-xl);
  h = 1.0/hi;
  dofs = 1;

  auto a = [](const PetscInt i){ return 1.5;};

  // Set if data should be written.
  write_data = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = gridctxs.F_appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, dofs, stencil_radius, NULL, &gridctxs.F_appctx.da);CHKERRQ(ierr);
  DMSetFromOptions(gridctxs.F_appctx.da);
  DMSetUp(gridctxs.F_appctx.da);
  DMDAGetCorners(gridctxs.F_appctx.da,&i_xstart,NULL,NULL,&n,NULL,NULL);
  i_xend = i_xstart + n;

  // Populate application context.
  gridctxs.F_appctx.N = {N};
  gridctxs.F_appctx.hi = {hi};
  gridctxs.F_appctx.h = {h};
  gridctxs.F_appctx.xl = xl;
  gridctxs.F_appctx.i_start = {i_xstart};
  gridctxs.F_appctx.i_end = {i_xend};
  gridctxs.F_appctx.dofs = dofs;
  gridctxs.F_appctx.a = a;
  gridctxs.F_appctx.sw = stencil_radius;
  gridctxs.F_appctx.layout = grid::create_layout_1d(gridctxs.F_appctx.da);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors for preconditioner
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  N_pc = 0.5*(N + 1);
  hi_pc = (N_pc-1)/(xr-xl);
  h_pc = 1.0/hi_pc;

  auto [stencil_width_pc, nc_pc, cw_pc] = gridctxs.C_appctx.D1.get_ranges();
  stencil_radius_pc = (stencil_width_pc-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N_pc, dofs, stencil_radius_pc, NULL, &gridctxs.C_appctx.da);CHKERRQ(ierr);
  DMSetFromOptions(gridctxs.C_appctx.da);
  DMSetUp(gridctxs.C_appctx.da);
  DMDAGetCorners(gridctxs.C_appctx.da,&i_xstart_pc,NULL,NULL,&n_pc,NULL,NULL);
  i_xend_pc = i_xstart_pc + n_pc;

  PetscReal xend = xl + i_xend*h;
  PetscReal xend_pc = xl + i_xend_pc*h_pc;

  assert(abs(xend - xend_pc) < 2*h);

  // Populate application context.
  gridctxs.C_appctx.N = {N_pc};
  gridctxs.C_appctx.hi = {hi_pc};
  gridctxs.C_appctx.h = {h_pc};
  gridctxs.C_appctx.xl = xl;
  gridctxs.C_appctx.i_start = {i_xstart_pc};
  gridctxs.C_appctx.i_end = {i_xend_pc};
  gridctxs.C_appctx.dofs = dofs;
  gridctxs.C_appctx.a = a;
  gridctxs.C_appctx.sw = stencil_radius_pc;
  gridctxs.C_appctx.layout = grid::create_layout_1d(gridctxs.C_appctx.da);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define matrix object
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatCreateShell(PETSC_COMM_WORLD,n,n,N,N,&gridctxs.F_appctx,&D);
  MatShellSetOperation(D,MATOP_MULT,(void(*)(void))LHS);

  MatCreateShell(PETSC_COMM_WORLD,n_pc,n_pc,N_pc,N_pc,&gridctxs.C_appctx,&D_pc);
  MatShellSetOperation(D_pc,MATOP_MULT,(void(*)(void))LHS);

  DMCreateGlobalVector(gridctxs.F_appctx.da,&v);
  DMCreateGlobalVector(gridctxs.F_appctx.da,&gridctxs.F_appctx.b);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);

  DMCreateGlobalVector(gridctxs.C_appctx.da,&gridctxs.C_appctx.b);

  RHS(gridctxs.F_appctx.da, gridctxs.F_appctx, gridctxs.F_appctx.b);
  RHS(gridctxs.C_appctx.da, gridctxs.C_appctx, gridctxs.C_appctx.b);

  KSPCreate(PETSC_COMM_WORLD, &gridctxs.F_appctx.ksp);
  KSPSetOperators(gridctxs.F_appctx.ksp, D, D);
  KSPSetTolerances(gridctxs.F_appctx.ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetPCSide(gridctxs.F_appctx.ksp, PC_RIGHT);
  KSPSetType(gridctxs.F_appctx.ksp, KSPPIPEFGMRES);
  KSPGMRESSetRestart(gridctxs.F_appctx.ksp, 10);
  // KSPSetInitialGuessNonzero(appctx.ksp, PETSC_TRUE);
  KSPSetFromOptions(gridctxs.F_appctx.ksp);

  KSPGetPC(gridctxs.F_appctx.ksp,&pc);
  PCSetType(pc,PCSHELL);
  // PCSetType(pc,PCNONE);
  PCShellSetContext(pc, &gridctxs);
  PCShellSetApply(pc, apply_pc);
  PCSetUp(pc);

  KSPCreate(PETSC_COMM_WORLD, &gridctxs.C_appctx.ksp);
  KSPSetOperators(gridctxs.C_appctx.ksp, D_pc, D_pc);
  KSPSetTolerances(gridctxs.C_appctx.ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetType(gridctxs.C_appctx.ksp, KSPGMRES);
  KSPGMRESSetRestart(gridctxs.C_appctx.ksp, 800);
  KSPSetInitialGuessNonzero(gridctxs.C_appctx.ksp, PETSC_TRUE);
  // KSPSetFromOptions(gridctxs.C_appctx.ksp);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }
  
  KSPSolve(gridctxs.F_appctx.ksp, gridctxs.F_appctx.b, v);

  // VecView(v,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  analytic_solution(gridctxs.F_appctx, v_analytic);
  get_error(gridctxs.F_appctx.da, v, v_analytic, &v_error, &H_error, &l2_error, &max_error, gridctxs.F_appctx);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, the H-error is: %g and the maximum error is %g\n",l2_error,H_error,max_error);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatDestroy(&D);
  MatDestroy(&D_pc);
  PCDestroy(&pc);
  VecDestroy(&v);
  VecDestroy(&v_error);
  VecDestroy(&v_analytic);
  DMDestroy(&gridctxs.C_appctx.da);
  DMDestroy(&gridctxs.F_appctx.da);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) {
  GridCtxs          *gridctxs;
  Vec               xcoarse;

  PCShellGetContext(pc, (void**) &gridctxs);

  DMGetGlobalVector(gridctxs->C_appctx.da, &xcoarse);

  apply_F2C(xin, xcoarse, gridctxs);
  KSPSolve(gridctxs->C_appctx.ksp, gridctxs->C_appctx.b, xcoarse);
  apply_C2F(xout, xcoarse, gridctxs);

  DMRestoreGlobalVector(gridctxs->C_appctx.da, &xcoarse);

  return 0;
}

PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, GridCtxs *gridctxs) {
  PetscScalar       *array_src, **array_dst;
  Vec               xfine_local;

  DMGetLocalVector(gridctxs->F_appctx.da, &xfine_local);

  DMGlobalToLocalBegin(gridctxs->F_appctx.da,xfine,INSERT_VALUES,xfine_local);
  DMGlobalToLocalEnd(gridctxs->F_appctx.da,xfine,INSERT_VALUES,xfine_local);

  VecGetArray(xfine_local, &array_src);
  auto gf_src = grid::grid_function_1d<PetscScalar>(array_src, gridctxs->F_appctx.layout);

  DMDAVecGetArrayDOF(gridctxs->C_appctx.da,xcoarse,&array_dst);

  sbp::apply_F2C(gridctxs->F_appctx.ICF, gf_src, array_dst, gridctxs->C_appctx.i_start[0], gridctxs->C_appctx.i_end[0], gridctxs->C_appctx.N[0]);

  DMDAVecRestoreArrayDOF(gridctxs->C_appctx.da,xcoarse,&array_dst); 
  VecRestoreArray(xfine_local,&array_src);

  DMRestoreLocalVector(gridctxs->F_appctx.da, &xfine_local);

  return 0;
}

PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, GridCtxs *gridctxs) {
  PetscScalar       *array_src, **array_dst;
  Vec               xcoarse_local;

  DMGetLocalVector(gridctxs->C_appctx.da, &xcoarse_local);

  DMGlobalToLocalBegin(gridctxs->C_appctx.da,xcoarse,INSERT_VALUES,xcoarse_local);
  DMGlobalToLocalEnd(gridctxs->C_appctx.da,xcoarse,INSERT_VALUES,xcoarse_local);

  VecGetArray(xcoarse_local, &array_src);
  auto gf_src = grid::grid_function_1d<PetscScalar>(array_src, gridctxs->C_appctx.layout);

  DMDAVecGetArrayDOF(gridctxs->F_appctx.da,xfine,&array_dst); 

  sbp::apply_C2F(gridctxs->F_appctx.ICF, gf_src, array_dst, gridctxs->F_appctx.i_start[0], gridctxs->F_appctx.i_end[0], gridctxs->F_appctx.N[0]);

  VecRestoreArray(xcoarse_local, &array_src);
  DMDAVecRestoreArrayDOF(gridctxs->F_appctx.da,xfine,&array_dst);

  DMRestoreLocalVector(gridctxs->C_appctx.da, &xcoarse_local);
  return 0;
}


PetscErrorCode analytic_solution(const AppCtx& appctx, Vec& v_analytic)
{ 
  PetscScalar x, *array_analytic;
  DMDAVecGetArray(appctx.da,v_analytic,&array_analytic);
  for (PetscInt i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
  {
    x = appctx.xl + i/appctx.hi[0];
    array_analytic[i] = x + 1;
  }
  DMDAVecRestoreArray(appctx.da,v_analytic,&array_analytic);  

  return 0;
};

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

PetscErrorCode RHS(const DM& da, const AppCtx& appctx, Vec& b)
{ 
  PetscScalar x, **b_arr;
  DMDAVecGetArrayDOF(da,b,&b_arr);
  for (PetscInt i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
  {
    x = appctx.xl + i/appctx.hi[0];
    b_arr[i][0] = 1;
  }
  DMDAVecRestoreArray(da,b,&b_arr);  

  return 0;
};

PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst)
{
  PetscScalar       *array_src, **array_dst;
  Vec               v_src_local;
  AppCtx            *appctx;

  MatShellGetContext(D, &appctx);

  DMGetLocalVector(appctx->da, &v_src_local);

  DMGlobalToLocalBegin(appctx->da,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(appctx->da,v_src,INSERT_VALUES,v_src_local);

  VecGetArray(v_src_local,&array_src);
  auto gf_src = grid::grid_function_1d<PetscScalar>(array_src, appctx->layout);

  DMDAVecGetArrayDOF(appctx->da,v_dst,&array_dst); 

  sbp::ode_apply_all(appctx->D1, appctx->HI, appctx->a, gf_src, array_dst, appctx->i_start[0], appctx->i_end[0], appctx->N[0], appctx->hi[0]);

  VecRestoreArray(v_src_local,&array_src);

  DMDAVecRestoreArray(appctx->da,v_dst,&array_dst); 
  DMRestoreLocalVector(appctx->da,&v_src_local);

  return 0;
}
