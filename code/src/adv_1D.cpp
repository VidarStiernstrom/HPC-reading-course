static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#define PROBLEM_TYPE_2D_O6

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/advection.h"
#include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

struct GridCtxs{
  AppCtx F_appctx, C_appctx;
};

extern PetscErrorCode analytic_solution(const PetscScalar t, const AppCtx& appctx, Vec& v_analytic);
extern PetscScalar gaussian(PetscScalar);
extern PetscErrorCode LHS(Mat, Vec, Vec);
extern PetscErrorCode RHS(const AppCtx& appctx, Vec& b, const PetscScalar Tend, Vec v0);
extern PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);
extern PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx);
extern PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout);
extern PetscErrorCode get_solution(Vec v_final, Vec v, const AppCtx& appctx);

int main(int argc,char **argv)
{ 
  Mat D;
  Vec            v, v_analytic, v_error, v_final;
  PC             pc;
  PetscInt       stencil_radius, i_xstart, i_xend, i_tstart, i_tend, Nx, nx, Nt, nt, dofs, tblocks;
  PetscInt       ig_xstart, ig_xend, ig_tstart, ig_tend, ngx, ngt, blockidx;
  PetscScalar    xl, xr, dx, dxi, dt, dti, t0, Tend, CFL, Tpb;
  PetscReal      l2_error, max_error, H_error;
  GridCtxs       gridctxs;

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
  Nx = 2001;
  xl = -1;
  xr = 1;
  dx = (xr - xl)/(Nx-1);
  dxi = 1./dx;
  
  // Time
  tblocks = 100;
  t0 = 0;
  Tend = 1;
  Tpb = Tend/tblocks;
  Nt = 4;
  dt = Tpb/(Nt-1);
  dti = 1./dt;

  dofs = 1;

  auto a = [](const PetscInt i){ return 4;};

  PetscPrintf(PETSC_COMM_WORLD,"dx: %f, dt: %f\n",dx,dt);


  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Create distributed array (DMDA) to manage parallel grid and vectors
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = gridctxs.F_appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nt,Nx,1,PETSC_DECIDE,dofs,stencil_radius,NULL,NULL,&gridctxs.F_appctx.da);
  DMSetFromOptions(gridctxs.F_appctx.da);
  DMSetUp(gridctxs.F_appctx.da);
  DMDAGetCorners(gridctxs.F_appctx.da,&i_tstart,&i_xstart,NULL,&nt,&nx,NULL);
  i_xend = i_xstart + nx;
  i_tend = i_tstart + nt;
  DMDAGetGhostCorners(gridctxs.F_appctx.da,&ig_tstart,&ig_xstart,NULL,&ngt,&ngx,NULL);
  ig_xend = ig_xstart + ngx;
  ig_tend = ig_tstart + ngt;

  // printf("Rank: %d, xstart: %d, xend: %d, tstart: %d, tend: %d, nx: %d, nt: %d\n",rank,i_xstart,i_xend,i_tstart,i_tend, nx, nt);
  // printf("Rank: %d, xgstart: %d, xgend: %d, tgstart: %d, tgend: %d, ngx: %d, ngt: %d\n",rank,ig_xstart,ig_xend,ig_tstart,ig_tend, ngx, ngt);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Fill application context
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  gridctxs.F_appctx.N = {Nx,Nt};
  gridctxs.F_appctx.hi = {dxi,dti};
  gridctxs.F_appctx.h = {dx,dt};
  gridctxs.F_appctx.xl = {xl,0};
  gridctxs.F_appctx.Tpb = Tpb;
  gridctxs.F_appctx.i_start = {i_xstart,i_tstart};
  gridctxs.F_appctx.i_end = {i_xend,i_tend};
  gridctxs.F_appctx.dofs = dofs;
  gridctxs.F_appctx.a = a;
  gridctxs.F_appctx.sw = stencil_radius;

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Setup solver
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  DMCreateGlobalVector(gridctxs.F_appctx.da,&v);
  DMCreateGlobalVector(gridctxs.F_appctx.da,&gridctxs.F_appctx.b);

  MatCreateShell(PETSC_COMM_WORLD,nx*nt,nx*nt,Nx*Nt,Nx*Nt,&gridctxs.F_appctx,&D);
  MatShellSetOperation(D,MATOP_MULT,(void(*)(void))LHS);

  KSPCreate(PETSC_COMM_WORLD, &gridctxs.F_appctx.ksp);
  KSPSetOperators(gridctxs.F_appctx.ksp, D, D);
  KSPSetTolerances(gridctxs.F_appctx.ksp, 1e-14, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetPCSide(gridctxs.F_appctx.ksp, PC_RIGHT);
  KSPSetType(gridctxs.F_appctx.ksp, KSPPIPEFGMRES);
  KSPGMRESSetRestart(gridctxs.F_appctx.ksp, 10);
  KSPSetInitialGuessNonzero(gridctxs.F_appctx.ksp, PETSC_TRUE);
  KSPSetFromOptions(gridctxs.F_appctx.ksp);

  KSPGetPC(gridctxs.F_appctx.ksp,&pc);
  PCSetType(pc,PCSHELL);
  // PCSetType(pc,PCNONE);
  PCShellSetContext(pc, &gridctxs);
  PCShellSetApply(pc, apply_pc);
  PCSetUp(pc);

  VecCreate(PETSC_COMM_WORLD, &v_final);
  VecSetSizes(v_final, nx, Nx);
  VecSetType(v_final, VECSTANDARD);
  VecSetUp(v_final);


  analytic_solution(0, gridctxs.F_appctx, v_final);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  for (blockidx = 0; blockidx < tblocks; blockidx++) {
    RHS(gridctxs.F_appctx, gridctxs.F_appctx.b, Tpb, v_final);
    KSPSolve(gridctxs.F_appctx.ksp, gridctxs.F_appctx.b, v);
    get_solution(v_final, v, gridctxs.F_appctx); 
  }

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);



  VecDuplicate(v_final,&v_error);
  VecDuplicate(v_final,&v_analytic);

  write_vector_to_binary(v,"data/adv_1D_try","v");
  write_vector_to_binary(v_final,"data/adv_1D_try","v_final");

  analytic_solution(Tend, gridctxs.F_appctx, v_analytic);

  get_error(gridctxs.F_appctx.da, v_final, v_analytic, &v_error, &H_error, &l2_error, &max_error, gridctxs.F_appctx);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %.8e, the H-error is: %.9e and the maximum error is %.9e\n",l2_error,H_error,max_error);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_final);
  VecDestroy(&v_error);
  VecDestroy(&v_final);
  PCDestroy(&pc);
  MatDestroy(&D);
  DMDestroy(&gridctxs.F_appctx.da);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) 
{
  GridCtxs          *gridctxs;
  Vec               xcoarse;

  // PCShellGetContext(pc, (void**) &gridctxs);

  // DMGetGlobalVector(gridctxs->C_appctx.da, &xcoarse);

  // apply_F2C(xin, xcoarse, gridctxs);
  // KSPSolve(gridctxs->C_appctx.ksp, gridctxs->C_appctx.b, xcoarse);
  // apply_C2F(xout, xcoarse, gridctxs);

  // DMRestoreGlobalVector(gridctxs->C_appctx.da, &xcoarse);
  VecCopy(xin,xout);

  return 0;
}

PetscErrorCode get_solution(Vec v_final, Vec v, const AppCtx& appctx) 
{
  PetscInt i, n, idx, istart;
  PetscScalar *vfinal_arr, ***v_arr;

  PetscScalar e_r[4] = {-0.113917196281990,0.400761520311650,-0.813632449486927,1.526788125457267};

  DMDAVecGetArrayDOF(appctx.da,v,&v_arr); 
  VecGetArray(v_final,&vfinal_arr); 

  VecGetLocalSize(v_final, &n);
  VecGetOwnershipRange(v_final, &istart, NULL);

  for (i = 0; i < n; i++) {
    idx = i + istart;
    vfinal_arr[i] = e_r[0]*v_arr[idx][0][0] + e_r[1]*v_arr[idx][1][0] + e_r[2]*v_arr[idx][2][0] + e_r[3]*v_arr[idx][3][0];
  }
  VecRestoreArray(v_final,&vfinal_arr); 
  DMDAVecRestoreArrayDOF(appctx.da,v,&v_arr); 

  return 0;
}

PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx) {
  // PetscScalar **arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);

  *l2_error = sqrt(appctx.h[0])*(*l2_error);
  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  // DMDAVecGetArrayDOF(da, *v_error, &arr);
  // *H_error = appctx.H.get_norm_1D(arr, appctx.h[0], appctx.N[0], appctx.i_start[0], appctx.i_end[0], appctx.dofs);
  // DMDAVecRestoreArrayDOF(da, *v_error, &arr);
  *H_error = -1;

  return 0;
}

PetscErrorCode analytic_solution(const PetscScalar t, const AppCtx& appctx, Vec& v_analytic)
{ 
  PetscScalar x, *array_analytic;
  PetscInt istart, n;
  VecGetOwnershipRange(v_analytic, &istart, NULL);
  VecGetLocalSize(v_analytic, &n);

  VecGetArray(v_analytic, &array_analytic);

  for (PetscInt i = 0; i < n; i++)
  {
    x = appctx.xl[0] + (i + istart)*appctx.h[0];
    array_analytic[i] = gaussian(x-appctx.a(i)*t);
  }
  VecRestoreArray(v_analytic, &array_analytic);

  return 0;
};

PetscScalar gaussian(PetscScalar x) 
{
  PetscScalar rstar = 0.1;
  return exp(-x*x/(rstar*rstar));
}

PetscErrorCode RHS(const AppCtx& appctx, Vec& b, const PetscScalar Tend, Vec v0)
{ 
  PetscScalar       ***b_arr, val, x, tau, *v0_arr;
  PetscInt          i, istart;

  DMDAVecGetArrayDOF(appctx.da,b,&b_arr); 

  VecGetOwnershipRange(v0, &istart, NULL);
  VecGetArray(v0, &v0_arr);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  tau = 1.0;
  PetscScalar HI_B[4] = {tau*4.389152966531085*2/Tend,tau*-1.247624770988935*2/Tend,tau*0.614528095966794*2/Tend,tau*-0.327484862937516*2/Tend};

  for (i = appctx.i_start[0]; i < appctx.i_end[0]; i++) {
    x = appctx.xl[0] + i*appctx.h[0];
    val = gaussian(x);
    b_arr[i][0][0] = HI_B[0]*v0_arr[i-istart];
    b_arr[i][1][0] = HI_B[1]*v0_arr[i-istart];
    b_arr[i][2][0] = HI_B[2]*v0_arr[i-istart];
    b_arr[i][3][0] = HI_B[3]*v0_arr[i-istart];
  }

  DMDAVecRestoreArrayDOF(appctx.da,b,&b_arr); 

  return 0;
};

PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst)
{
  PetscScalar       ***array_src, ***array_dst;
  Vec               v_src_local;
  AppCtx            *appctx;

  MatShellGetContext(D, &appctx);

  DMGetLocalVector(appctx->da, &v_src_local);

  DMGlobalToLocalBegin(appctx->da,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(appctx->da,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(appctx->da,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(appctx->da,v_src_local,&array_src); 

  sbp::adv_imp_apply_all(appctx->D1, appctx->HI, appctx->a, array_src, array_dst, appctx->i_start[0], appctx->i_end[0], appctx->N[0], appctx->hi[0], appctx->Tpb);

  DMDAVecRestoreArrayDOF(appctx->da,v_dst,&array_dst); 
  DMDAVecRestoreArrayDOF(appctx->da,v_src_local,&array_src); 

  DMRestoreLocalVector(appctx->da,&v_src_local);
  
  return 0;
}
