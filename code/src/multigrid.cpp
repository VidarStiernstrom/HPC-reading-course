#include<petsc.h>
#include "appctx.h"
#include "diffops/advection.h"
#include "imp_timestepping.h"

struct InterpCtx {
  GridCtx *F_gridctx, *C_gridctx;
};

static PetscErrorCode setup_mgsolver(KSP& ksp_fine, PetscInt nlevels, Mat& Afine, Mat Acoarses[], Mat P[], Mat R[]);
static PetscErrorCode setup_operators(MatCtx *C_matctx, MatCtx *F_matctx, InterpCtx *prolctx, InterpCtx *restctx);
static PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse);
static PetscErrorCode apply_P(Mat P, Vec xcoarse, Vec xfine);
static PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) ;
static PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0);
static PetscScalar gaussian(PetscScalar x) ;
static PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic);
static PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) ;

PetscErrorCode mgsolver(Mat& Afine, Vec& v, PetscInt nlevels)
{
  PetscInt level, Nt, blockidx, rank;
  PetscLogDouble v1,v2,elapsed_time = 0;
  Vec            v_analytic, v_error, b, v_curr;
  MatCtx *F_matctx;
  void (*LHS) (void);
  PetscReal      l2_error, max_error, H_error;
  KSP ksp;
  MatCtx C_matctxs[nlevels-1];
  InterpCtx restctx[nlevels-1], prolctx[nlevels-1];
  Mat    Dcoarses[nlevels-1];
  Mat    P[nlevels-1], R[nlevels-1];

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // Extract fine grid context and matmult function
  MatShellGetContext(Afine, &F_matctx);
  MatShellGetOperation(Afine, MATOP_MULT, &LHS);

  // Number of timesteps (gauss = 4)
  Nt = F_matctx->timectx.N;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup contexts for system-/restriction-/prolongation-matrices
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  setup_operators(&C_matctxs[nlevels-2], F_matctx, &prolctx[nlevels-2], &restctx[nlevels-2]);
  for (level = nlevels-2; level > 0; level--) {
    setup_operators(&C_matctxs[level-1], &C_matctxs[level], &prolctx[level-1], &restctx[level-1]);    
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create system-/restriction-/prolongation-matrices
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  // Restriction matrix needed for level = nlevels-1, nlevels-2,...,1
  MatCreateShell(PETSC_COMM_WORLD,C_matctxs[nlevels-2].gridctx.n[0]*Nt,F_matctx->gridctx.n[0]*Nt,C_matctxs[nlevels-2].gridctx.N[0]*Nt,F_matctx->gridctx.N[0]*Nt,&restctx[nlevels-2],&R[nlevels-2]);
  MatShellSetOperation(R[nlevels-2],MATOP_MULT,(void(*)(void))apply_R);
  for (level = 1; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level-1].gridctx.n[0]*Nt,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level-1].gridctx.N[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,&restctx[level-1],&R[level-1]);
    MatShellSetOperation(R[level-1],MATOP_MULT,(void(*)(void))apply_R);    
  }

  // Prolongation matrix needed for level = nlevels-1, nlevels-2,...,1
  MatCreateShell(PETSC_COMM_WORLD,F_matctx->gridctx.n[0]*Nt,C_matctxs[nlevels-2].gridctx.n[0]*Nt,F_matctx->gridctx.N[0]*Nt,C_matctxs[nlevels-2].gridctx.N[0]*Nt,&prolctx[nlevels-2],&P[nlevels-2]);
  MatShellSetOperation(P[nlevels-2],MATOP_MULT,(void(*)(void))apply_P);  
  for (level = 1; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level-1].gridctx.n[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,C_matctxs[level-1].gridctx.N[0]*Nt,&prolctx[level-1],&P[level-1]);
    MatShellSetOperation(P[level-1],MATOP_MULT,(void(*)(void))apply_P);    
  }
  // System matrix needed for level = nlevels-2, nlevels-3,...,0
  for (level = 0; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,&C_matctxs[level],&Dcoarses[level]);
    MatShellSetOperation(Dcoarses[level],MATOP_MULT,(void(*)(void))LHS);
    MatSetDM(Dcoarses[level], C_matctxs[level].gridctx.da_xt);
  }
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup solvers
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  setup_mgsolver(ksp, nlevels, Afine, Dcoarses, P, R);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Print MG configuration
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  for (level = 0; level < nlevels-1; level++) {
    PetscPrintf(PETSC_COMM_WORLD, "Level: %d, Nx: %d, nx: %d\n",level,C_matctxs[level].gridctx.N[0],C_matctxs[level].gridctx.n[0]);
  }
  PetscPrintf(PETSC_COMM_WORLD, "Level: %d, Nx: %d, nx: %d\n",nlevels-1,F_matctx->gridctx.N[0],F_matctx->gridctx.n[0]);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  DMCreateGlobalVector(F_matctx->gridctx.da_x, &v_curr);
  DMCreateGlobalVector(F_matctx->gridctx.da_xt,&b);

  analytic_solution(F_matctx->gridctx, 0, v_curr);

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  for (blockidx = 0; blockidx < F_matctx->timectx.tblocks; blockidx++) 
  {
    RHS(F_matctx->gridctx, F_matctx->timectx, b, v_curr);
    KSPSolve(ksp, b, v);
    get_solution(v_curr, v, F_matctx->gridctx, F_matctx->timectx);
  }

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }
  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute and print error
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(F_matctx->gridctx.da_x, &v_error);
  DMCreateGlobalVector(F_matctx->gridctx.da_x, &v_analytic);

  analytic_solution(F_matctx->gridctx, F_matctx->timectx.Tend, v_analytic);

  get_solution(v_curr, v, F_matctx->gridctx, F_matctx->timectx); 
  get_error(F_matctx->gridctx, v_curr, v_analytic, &v_error, &H_error, &l2_error, &max_error);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %.8e, the H-error is: %.9e and the maximum error is %.9e\n",l2_error,H_error,max_error);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Destroy petsc objects
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v_analytic);
  VecDestroy(&v_error);
  VecDestroy(&b);
  VecDestroy(&v_curr);
  KSPDestroy(&ksp);
  for (level = 0; level < nlevels-1; level++) {
    MatDestroy(&Dcoarses[level]);
    MatDestroy(&P[level]);
    MatDestroy(&R[level]);
  }

  return 0;
}

static PetscErrorCode setup_mgsolver(KSP& ksp_fine, PetscInt nlevels, Mat& Afine, Mat Acoarses[], Mat P[], Mat R[])
{
  KSP ksp_coarse, ksp_fine_smoth, ksp_mid_smoth[nlevels-2];
  PC pcfine, pccoarse, pcfine_smoot, pcmid_smoth[nlevels-2];
  PetscInt level;

  // Finest outer solver
  KSPCreate(PETSC_COMM_WORLD, &ksp_fine);
  KSPGetPC(ksp_fine, &pcfine);
  KSPSetOperators(ksp_fine, Afine, Afine);
  KSPSetTolerances(ksp_fine, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetType(ksp_fine, KSPPIPEFGMRES);
  KSPGMRESSetRestart(ksp_fine, 10);
  KSPSetInitialGuessNonzero(ksp_fine, PETSC_TRUE);
  KSPSetFromOptions(ksp_fine);
  KSPSetPCSide(ksp_fine, PC_RIGHT);
  KSPSetUp(ksp_fine);
  KSPGetPC(ksp_fine, &pcfine);
  PCSetType(pcfine, PCMG);
  PCMGSetLevels(pcfine, nlevels, NULL);
  PCMGSetType(pcfine, PC_MG_MULTIPLICATIVE);
  PCMGSetCycleType(pcfine, PC_MG_CYCLE_W);

  // Finest smoother
  PCMGGetSmoother(pcfine, nlevels-1, &ksp_fine_smoth);
  KSPSetOperators(ksp_fine_smoth, Afine, Afine);
  KSPSetTolerances(ksp_fine_smoth, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 10);
  KSPSetType(ksp_fine_smoth, KSPGMRES);
  // KSPGMRESSetRestart(ksp_fine_smoth, 1);
  KSPGetPC(ksp_fine_smoth, &pcfine_smoot);
  PCSetType(pcfine_smoot, PCNONE);
  KSPSetFromOptions(ksp_fine_smoth);
  PCSetUp(pcfine_smoot);
  KSPSetUp(ksp_fine_smoth);

  // Coarsest inner solver
  PCMGGetCoarseSolve(pcfine, &ksp_coarse);
  KSPSetOperators(ksp_coarse, Acoarses[0], Acoarses[0]);
  KSPSetTolerances(ksp_coarse, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetType(ksp_coarse, KSPGMRES);
  KSPGMRESSetRestart(ksp_coarse, 10);
  KSPSetPCSide(ksp_coarse, PC_RIGHT);
  KSPGetPC(ksp_coarse, &pccoarse);
  KSPSetFromOptions(ksp_coarse);
  PCSetType(pccoarse, PCNONE);
  PCSetUp(pccoarse);
  KSPSetUp(ksp_coarse);

  // Mid smoothers
  for (level = 1; level < nlevels-1; level++) {
      PCMGGetSmoother(pcfine, level, &ksp_mid_smoth[level-1]);
      KSPSetOperators(ksp_mid_smoth[level-1], Acoarses[level], Acoarses[level]);
      KSPSetTolerances(ksp_mid_smoth[level-1], 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 10);
      KSPSetType(ksp_mid_smoth[level-1], KSPGMRES);
      KSPGetPC(ksp_mid_smoth[level-1], &pcmid_smoth[level-1]);
      PCSetType(pcmid_smoth[level-1], PCNONE);
      KSPSetFromOptions(ksp_mid_smoth[level-1]);
      PCSetUp(pcmid_smoth[level-1]);
      KSPSetUp(ksp_mid_smoth[level-1]);
  }

  // Set interpolations
  for (level = 1; level < nlevels; level++) {
    PCMGSetRestriction(pcfine, level, R[level-1]);
    PCMGSetInterpolation(pcfine, level, P[level-1]);    
  }

  PCSetFromOptions(pcfine);
  PCSetUp(pcfine);

  return 0;
}

PetscErrorCode apply_P(Mat P, Vec xcoarse, Vec xfine)
{
  PetscScalar       **array_src, **array_dst;
  Vec               xcoarse_local;
  InterpCtx         *prolctx;
  DM                dmda_fine, dmda_coarse;

  MatShellGetContext(P, &prolctx);

  dmda_fine = prolctx->F_gridctx->da_xt;
  dmda_coarse = prolctx->C_gridctx->da_xt;

  DMGetLocalVector(dmda_coarse, &xcoarse_local);

  DMGlobalToLocalBegin(dmda_coarse, xcoarse, INSERT_VALUES, xcoarse_local);
  DMGlobalToLocalEnd(dmda_coarse, xcoarse, INSERT_VALUES, xcoarse_local);

  DMDAVecGetArrayDOF(dmda_fine, xfine, &array_dst);
  DMDAVecGetArrayDOF(dmda_coarse, xcoarse_local, &array_src);

  sbp::apply_C2F(prolctx->F_gridctx->ICF, array_src, array_dst, prolctx->F_gridctx->i_start[0], prolctx->F_gridctx->i_end[0], prolctx->F_gridctx->N[0]);

  DMDAVecRestoreArrayDOF(dmda_fine, xfine, &array_dst);
  DMDAVecRestoreArrayDOF(dmda_coarse, xcoarse_local, &array_src);

  DMRestoreLocalVector(dmda_coarse, &xcoarse_local);

  return 0;
}

PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse)
{
  PetscScalar       **array_src, **array_dst;
  Vec               xfine_local;
  InterpCtx         *restctx;
  DM                dmda_fine, dmda_coarse;

  MatShellGetContext(R, &restctx);

  dmda_fine = restctx->F_gridctx->da_xt;
  dmda_coarse = restctx->C_gridctx->da_xt;

  DMGetLocalVector(dmda_fine, &xfine_local);

  DMGlobalToLocalBegin(dmda_fine, xfine, INSERT_VALUES, xfine_local);
  DMGlobalToLocalEnd(dmda_fine, xfine, INSERT_VALUES, xfine_local);

  DMDAVecGetArrayDOF(dmda_coarse, xcoarse, &array_dst);
  DMDAVecGetArrayDOF(dmda_fine, xfine_local, &array_src);

  sbp::apply_F2C(restctx->F_gridctx->ICF, array_src, array_dst, restctx->C_gridctx->i_start[0], restctx->C_gridctx->i_end[0], restctx->C_gridctx->N[0]);

  DMDAVecRestoreArrayDOF(dmda_coarse, xcoarse, &array_dst);
  DMDAVecRestoreArrayDOF(dmda_fine, xfine_local, &array_src);

  DMRestoreLocalVector(dmda_fine, &xfine_local);

  return 0;
}

PetscErrorCode setup_operators(MatCtx *C_matctx, MatCtx *F_matctx, InterpCtx *prolctx, InterpCtx *restctx)
{
  PetscInt       stencil_radius, i_xstart, i_xend, Nx, Nt, Nx_pc, nx_pc, dofs;
  PetscScalar    xl, xr, dx, dxi, dx_pc, dxi_pc, Tend, Tpb, tau;

  // System matrices
  Nx = F_matctx->gridctx.N[0];
  dxi = F_matctx->gridctx.hi[0];
  dx = F_matctx->gridctx.h[0];
  xl = F_matctx->gridctx.xl[0];
  xr = F_matctx->gridctx.xr[0];
  dofs = F_matctx->gridctx.dofs;
  auto a = F_matctx->gridctx.a;
  Nt = F_matctx->timectx.N;
  Tend = F_matctx->timectx.Tend;
  Tpb = F_matctx->timectx.Tpb;

  Nx_pc = (Nx - 1)/2 + 1;
  dx_pc = (xr - xl)/(Nx_pc-1);
  dxi_pc = 1./dx_pc;
  Nt = 4;
  tau = 1;

  C_matctx->gridctx.N = {Nx_pc};
  C_matctx->gridctx.hi = {dxi_pc};
  C_matctx->gridctx.h = {dx_pc};
  C_matctx->gridctx.xl = {xl};
  C_matctx->gridctx.dofs = dofs;
  C_matctx->gridctx.a = a;

  C_matctx->timectx.N = Nt;
  C_matctx->timectx.Tend = Tend;
  C_matctx->timectx.Tpb = Tpb;

  auto [stencil_width_pc, nc_pc, cw_pc] = C_matctx->gridctx.D1.get_ranges();
  stencil_radius = (stencil_width_pc-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx_pc, Nt*dofs, stencil_radius, NULL, &C_matctx->gridctx.da_xt);
  
  DMSetFromOptions(C_matctx->gridctx.da_xt);
  DMSetUp(C_matctx->gridctx.da_xt);
  DMDAGetCorners(C_matctx->gridctx.da_xt,&i_xstart,NULL,NULL,&nx_pc,NULL,NULL);
  i_xend = i_xstart + nx_pc;

  C_matctx->gridctx.n = {nx_pc};
  C_matctx->gridctx.i_start = {i_xstart};
  C_matctx->gridctx.i_end = {i_xend};
  C_matctx->gridctx.sw = stencil_radius;

  setup_timestepper(C_matctx->timectx, tau);

  // Interpolation matrices
  restctx->F_gridctx = &F_matctx->gridctx;
  restctx->C_gridctx = &C_matctx->gridctx;

  prolctx->F_gridctx = &F_matctx->gridctx;
  prolctx->C_gridctx = &C_matctx->gridctx;

  return 0;
}

static PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) 
{
  PetscInt i;
  PetscScalar **vfinal_arr, **v_arr;

  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    vfinal_arr[i][0] = timectx.er[0]*v_arr[i][0] + timectx.er[1]*v_arr[i][1] + timectx.er[2]*v_arr[i][2] + timectx.er[3]*v_arr[i][3];
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  return 0;
}

static PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic)
{ 
  PetscScalar x, **array_analytic;
  PetscInt i;

  DMDAVecGetArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    x = gridctx.xl[0] + i*gridctx.h[0];
    array_analytic[i][0] = gaussian(x-gridctx.a(i)*t);
  }

  DMDAVecRestoreArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  return 0;
};

static PetscScalar gaussian(PetscScalar x) 
{
  PetscScalar rstar = 0.1;
  return exp(-x*x/(rstar*rstar));
}

static PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0)
{ 
  PetscScalar       **b_arr, **v0_arr;
  PetscInt          i;

  DMDAVecGetArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x,v0,&v0_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    b_arr[i][0] = timectx.HI_el[0]*v0_arr[i][0];
    b_arr[i][1] = timectx.HI_el[1]*v0_arr[i][0];
    b_arr[i][2] = timectx.HI_el[2]*v0_arr[i][0];
    b_arr[i][3] = timectx.HI_el[3]*v0_arr[i][0];
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x,v0,&v0_arr); 

  return 0;
};

static PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) 
{
  PetscScalar **arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);

  *l2_error = sqrt(gridctx.h[0])*(*l2_error);
  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(gridctx.da_x, *v_error, &arr);
  *H_error = gridctx.H.get_norm_1D(arr, gridctx.h[0], gridctx.N[0], gridctx.i_start[0], gridctx.i_end[0], gridctx.dofs);
  DMDAVecRestoreArrayDOF(gridctx.da_x, *v_error, &arr);

  return 0;
}
