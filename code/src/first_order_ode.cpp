static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#define PROBLEM_TYPE_1D_O2

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/ICF_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/first_order_ode.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

struct MatCtx {
  GridCtx gridctx;
};

struct PCCtx {
  MatCtx F_matctx, C_matctx;
};

extern PetscErrorCode analytic_solution(const GridCtx& gridctx, Vec& v_analytic);
extern PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout);
extern PetscErrorCode LHS(Mat, Vec, Vec);
extern PetscErrorCode RHS(const DM& da_x, const GridCtx& gridctx, Vec& b);
extern PetscScalar gaussian(PetscScalar);
extern PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);
extern PetscErrorCode get_error(const DM& da_x, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const GridCtx& gridctx);
extern PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, PCCtx pcctx) ;
extern PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, PCCtx pcctx) ;

int main(int argc,char **argv)
{ 
  Vec            v, v_error, v_analytic;
  PC             pc;
  PetscInt       stencil_radius, i_xstart, i_xend, N, n, dofs, stencil_radius_pc, i_xstart_pc, i_xend_pc, N_pc, n_pc;
  PetscScalar    xl, xr, hi, h, hi_pc, h_pc;
  PetscReal      l2_error, max_error, H_error;
  PCCtx          pcctx;

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
  N = 10001;
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
  auto [stencil_width, nc, cw] = pcctx.F_matctx.gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, dofs, stencil_radius, NULL, &pcctx.F_matctx.gridctx.da_x);CHKERRQ(ierr);
  DMSetFromOptions(pcctx.F_matctx.gridctx.da_x);
  DMSetUp(pcctx.F_matctx.gridctx.da_x);
  DMDAGetCorners(pcctx.F_matctx.gridctx.da_x,&i_xstart,NULL,NULL,&n,NULL,NULL);
  i_xend = i_xstart + n;

  // Populate application context.
  pcctx.F_matctx.gridctx.N = {N};
  pcctx.F_matctx.gridctx.hi = {hi};
  pcctx.F_matctx.gridctx.h = {h};
  pcctx.F_matctx.gridctx.xl = {xl};
  pcctx.F_matctx.gridctx.i_start = {i_xstart};
  pcctx.F_matctx.gridctx.i_end = {i_xend};
  pcctx.F_matctx.gridctx.dofs = dofs;
  pcctx.F_matctx.gridctx.a = a;
  pcctx.F_matctx.gridctx.sw = stencil_radius;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors for preconditioner
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  N_pc = 0.5*(N + 1);
  hi_pc = (N_pc-1)/(xr-xl);
  h_pc = 1.0/hi_pc;

  auto [stencil_width_pc, nc_pc, cw_pc] = pcctx.C_matctx.gridctx.D1.get_ranges();
  stencil_radius_pc = (stencil_width_pc-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N_pc, dofs, stencil_radius_pc, NULL, &pcctx.C_matctx.gridctx.da_x);CHKERRQ(ierr);
  DMSetFromOptions(pcctx.C_matctx.gridctx.da_x);
  DMSetUp(pcctx.C_matctx.gridctx.da_x);
  DMDAGetCorners(pcctx.C_matctx.gridctx.da_x,&i_xstart_pc,NULL,NULL,&n_pc,NULL,NULL);
  i_xend_pc = i_xstart_pc + n_pc;

  PetscReal xend = xl + i_xend*h;
  PetscReal xend_pc = xl + i_xend_pc*h_pc;

  assert(abs(xend - xend_pc) < 2*h);

  // Populate application context.
  pcctx.C_matctx.gridctx.N = {N_pc};
  pcctx.C_matctx.gridctx.hi = {hi_pc};
  pcctx.C_matctx.gridctx.h = {h_pc};
  pcctx.C_matctx.gridctx.xl = {xl};
  pcctx.C_matctx.gridctx.i_start = {i_xstart_pc};
  pcctx.C_matctx.gridctx.i_end = {i_xend_pc};
  pcctx.C_matctx.gridctx.dofs = dofs;
  pcctx.C_matctx.gridctx.a = a;
  pcctx.C_matctx.gridctx.sw = stencil_radius_pc;
  pcctx.C_matctx.gridctx.layout = grid::create_layout_1d(pcctx.C_matctx.gridctx.da_x);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define matrix object
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatCreateShell(PETSC_COMM_WORLD,n,n,N,N,&pcctx.F_matctx.gridctx,&pcctx.F_matctx.gridctx.D);
  MatShellSetOperation(pcctx.F_matctx.gridctx.D,MATOP_MULT,(void(*)(void))LHS);

  MatCreateShell(PETSC_COMM_WORLD,n_pc,n_pc,N_pc,N_pc,&pcctx.C_matctx.gridctx,&pcctx.C_matctx.gridctx.D);
  MatShellSetOperation(pcctx.C_matctx.gridctx.D,MATOP_MULT,(void(*)(void))LHS);

  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x,&v);
  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x,&pcctx.F_matctx.gridctx.b);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);

  DMCreateGlobalVector(pcctx.C_matctx.gridctx.da_x,&pcctx.C_matctx.gridctx.b);

  RHS(pcctx.F_matctx.gridctx.da_x, pcctx.F_matctx.gridctx, pcctx.F_matctx.gridctx.b);
  RHS(pcctx.C_matctx.gridctx.da_x, pcctx.C_matctx.gridctx, pcctx.C_matctx.gridctx.b);

  KSPCreate(PETSC_COMM_WORLD, &pcctx.F_matctx.gridctx.ksp);
  KSPSetOperators(pcctx.F_matctx.gridctx.ksp, pcctx.F_matctx.gridctx.D, pcctx.F_matctx.gridctx.D);
  KSPSetTolerances(pcctx.F_matctx.gridctx.ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetPCSide(pcctx.F_matctx.gridctx.ksp, PC_RIGHT);
  KSPSetType(pcctx.F_matctx.gridctx.ksp, KSPPIPEFGMRES);
  KSPGMRESSetRestart(pcctx.F_matctx.gridctx.ksp, 100);
  // KSPSetInitialGuessNonzero(gridctx.ksp, PETSC_TRUE);
  KSPSetFromOptions(pcctx.F_matctx.gridctx.ksp);

  KSPGetPC(pcctx.F_matctx.gridctx.ksp,&pc);
  // PCSetType(pc,PCSHELL);
  PCSetType(pc,PCNONE);
  PCShellSetContext(pc, &pcctx);
  PCShellSetApply(pc, apply_pc);
  PCSetUp(pc);

  KSPCreate(PETSC_COMM_WORLD, &pcctx.C_matctx.gridctx.ksp);
  KSPSetOperators(pcctx.C_matctx.gridctx.ksp, pcctx.C_matctx.gridctx.D, pcctx.C_matctx.gridctx.D);
  KSPSetTolerances(pcctx.C_matctx.gridctx.ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetType(pcctx.C_matctx.gridctx.ksp, KSPGMRES);
  KSPGMRESSetRestart(pcctx.C_matctx.gridctx.ksp, 100);
  KSPSetInitialGuessNonzero(pcctx.C_matctx.gridctx.ksp, PETSC_TRUE);
  // KSPSetFromOptions(pcctx.C_matctx.gridctx.ksp);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }
  
  KSPSolve(pcctx.F_matctx.gridctx.ksp, pcctx.F_matctx.gridctx.b, v);

  // VecView(v,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  analytic_solution(pcctx.F_matctx.gridctx, v_analytic);
  get_error(pcctx.F_matctx.gridctx.da_x, v, v_analytic, &v_error, &H_error, &l2_error, &max_error, pcctx.F_matctx.gridctx);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, the H-error is: %g and the maximum error is %g\n",l2_error,H_error,max_error);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatDestroy(&pcctx.F_matctx.gridctx.D);
  MatDestroy(&pcctx.C_matctx.gridctx.D);
  PCDestroy(&pc);
  VecDestroy(&v);
  VecDestroy(&v_error);
  VecDestroy(&v_analytic);
  DMDestroy(&pcctx.C_matctx.gridctx.da_x);
  DMDestroy(&pcctx.F_matctx.gridctx.da_x);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) 
{
  Vec               xcoarse;
  PCCtx             *pcctx;
  Vec               F_resi, F_err, C_resi, C_err;

  PCShellGetContext(pc, (void**) &pcctx);

  DMGetGlobalVector(pcctx->F_matctx.gridctx.da_x, &F_resi);
  DMGetGlobalVector(pcctx->F_matctx.gridctx.da_x, &F_err);
  DMGetGlobalVector(pcctx->C_matctx.gridctx.da_x, &C_resi);
  DMGetGlobalVector(pcctx->C_matctx.gridctx.da_x, &C_err);

  // Presmoother
  // KSPSolve(pcctx->F_matctx.gridctx.ksp_smo, pcctx->F_matctx.gridctx.b_smo, xin);

  // Compute fine residual
  MatMult(pcctx->F_matctx.gridctx.D, xin, F_resi); // F_resi = Df*xin
  VecAXPY(F_resi, -1.0, pcctx->F_matctx.gridctx.b); // F_resi = F_resi - b = Df*xin - b

  // // Restrict residual to coarse grid
  apply_F2C(F_resi, C_resi, *pcctx); // C_resi = R*F_resi

  // // Solve error equation 
  VecSet(C_err,0.0);
  KSPSolve(pcctx->C_matctx.gridctx.ksp, C_resi, C_err); // C_err = Dc^-1*C_resi

  // // Prolong error to fine grid
  apply_C2F(F_err, C_err, *pcctx); // F_err = P*C_err

  // // Correct fine grid solution
  VecWAXPY(xout, -1.0, F_err, xin);

  // PetscPrintf(PETSC_COMM_WORLD,"Postsmoothing\n");
  // KSPSolve(pcctx->F_matctx.gridctx.ksp_smo, pcctx->F_matctx.gridctx.b_smo, xout);

  DMRestoreGlobalVector(pcctx->F_matctx.gridctx.da_x, &F_resi);
  DMRestoreGlobalVector(pcctx->F_matctx.gridctx.da_x, &F_err);
  DMRestoreGlobalVector(pcctx->C_matctx.gridctx.da_x, &C_resi);
  DMRestoreGlobalVector(pcctx->C_matctx.gridctx.da_x, &C_err);

  return 0;
}

PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, PCCtx pcctx) 
{
  PetscScalar       **array_src, **array_dst;
  Vec               xfine_local;

  DMGetLocalVector(pcctx.F_matctx.gridctx.da_x, &xfine_local);

  DMGlobalToLocalBegin(pcctx.F_matctx.gridctx.da_x,xfine,INSERT_VALUES,xfine_local);
  DMGlobalToLocalEnd(pcctx.F_matctx.gridctx.da_x,xfine,INSERT_VALUES,xfine_local);

  DMDAVecGetArrayDOF(pcctx.C_matctx.gridctx.da_x,xcoarse,&array_dst);
  DMDAVecGetArrayDOF(pcctx.F_matctx.gridctx.da_x,xfine_local,&array_src);

  sbp::apply_F2C(pcctx.F_matctx.gridctx.ICF, array_src, array_dst, pcctx.C_matctx.gridctx.i_start[0], pcctx.C_matctx.gridctx.i_end[0], pcctx.C_matctx.gridctx.N[0]);

  DMDAVecRestoreArrayDOF(pcctx.C_matctx.gridctx.da_x,xcoarse,&array_dst);
  DMDAVecRestoreArrayDOF(pcctx.F_matctx.gridctx.da_x,xfine_local,&array_src);

  DMRestoreLocalVector(pcctx.F_matctx.gridctx.da_x, &xfine_local);

  return 0;
}

PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, PCCtx pcctx) 
{
  PetscScalar       **array_src, **array_dst;
  Vec               xcoarse_local;

  DMGetLocalVector(pcctx.C_matctx.gridctx.da_x, &xcoarse_local);

  DMGlobalToLocalBegin(pcctx.C_matctx.gridctx.da_x,xcoarse,INSERT_VALUES,xcoarse_local);
  DMGlobalToLocalEnd(pcctx.C_matctx.gridctx.da_x,xcoarse,INSERT_VALUES,xcoarse_local);

  DMDAVecGetArrayDOF(pcctx.F_matctx.gridctx.da_x,xfine,&array_dst); 
  DMDAVecGetArrayDOF(pcctx.C_matctx.gridctx.da_x,xcoarse_local,&array_src); 

  sbp::apply_C2F(pcctx.F_matctx.gridctx.ICF, array_src, array_dst, pcctx.F_matctx.gridctx.i_start[0], pcctx.F_matctx.gridctx.i_end[0], pcctx.F_matctx.gridctx.N[0]);

  DMDAVecRestoreArrayDOF(pcctx.F_matctx.gridctx.da_x,xfine,&array_dst);
  DMDAVecRestoreArrayDOF(pcctx.C_matctx.gridctx.da_x,xcoarse_local,&array_src); 

  DMRestoreLocalVector(pcctx.C_matctx.gridctx.da_x, &xcoarse_local);

  return 0;
}


PetscErrorCode analytic_solution(const GridCtx& gridctx, Vec& v_analytic)
{ 
  PetscScalar x, *array_analytic;
  DMDAVecGetArray(gridctx.da_x,v_analytic,&array_analytic);
  for (PetscInt i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++)
  {
    x = gridctx.xl[0] + i/gridctx.hi[0];
    array_analytic[i] = x + 1;
  }
  DMDAVecRestoreArray(gridctx.da_x,v_analytic,&array_analytic);  

  return 0;
};

PetscErrorCode get_error(const DM& da_x, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const GridCtx& gridctx) {
  PetscScalar **arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);
  *l2_error = sqrt(gridctx.h[0])*(*l2_error);

  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(da_x, *v_error, &arr);
  *H_error = gridctx.H.get_norm_1D(arr, gridctx.h[0], gridctx.N[0], gridctx.i_start[0], gridctx.i_end[0], gridctx.dofs);
  DMDAVecRestoreArrayDOF(da_x, *v_error, &arr);

  return 0;
}

PetscErrorCode RHS(const DM& da_x, const GridCtx& gridctx, Vec& b)
{ 
  PetscScalar x, **b_arr;
  DMDAVecGetArrayDOF(da_x,b,&b_arr);
  for (PetscInt i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++)
  {
    x = gridctx.xl[0] + i/gridctx.hi[0];
    b_arr[i][0] = 1;
  }
  DMDAVecRestoreArray(da_x,b,&b_arr);  

  return 0;
};

PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst)
{
  PetscScalar       **array_src, **array_dst;
  Vec               v_src_local;
  GridCtx            *gridctx;

  MatShellGetContext(D, &gridctx);

  DMGetLocalVector(gridctx->da_x, &v_src_local);

  DMGlobalToLocalBegin(gridctx->da_x,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(gridctx->da_x,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(gridctx->da_x,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(gridctx->da_x,v_src_local,&array_src); 

  sbp::ode_apply_all(gridctx->D1, gridctx->HI, gridctx->a, array_src, array_dst, gridctx->i_start[0], gridctx->i_end[0], gridctx->N[0], gridctx->hi[0]);

  DMDAVecRestoreArrayDOF(gridctx->da_x,v_src_local,&array_src); 
  DMDAVecRestoreArray(gridctx->da_x,v_dst,&array_dst); 

  DMRestoreLocalVector(gridctx->da_x,&v_src_local);

  return 0;
}
