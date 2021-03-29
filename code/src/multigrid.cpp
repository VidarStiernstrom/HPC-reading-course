#include<petsc.h>
#include "appctx.h"
// #include "diffops/advection.h"
// #include "diffops/reflection.h"
#include "diffops/acowave.h"
#include "imp_timestepping.h"
#include "aco_2D.h"
#include "IO_utils.h"
// #include "ref_1D.h"
// #include "adv_1D.h"
#include<unistd.h>

struct InterpCtx {
  GridCtx *F_gridctx, *C_gridctx;
};

static PetscErrorCode setup_mgsolver(KSP& ksp_fine, PetscInt nlevels, Mat& Afine, Mat Acoarses[], Mat P[], Mat R[]);
static PetscErrorCode setup_operators(MatCtx *C_matctx, MatCtx *F_matctx, InterpCtx *prolctx, InterpCtx *restctx);
static PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse);
static PetscErrorCode apply_P(Mat P, Vec xcoarse, Vec xfine);

PetscErrorCode mgsolver(Mat& Afine, Vec& v, std::string filename_reshist)
{
  PetscInt level, Nt, blockidx, rank, dofs, nlevels;
  PetscLogDouble v1,v2,elapsed_time = 0;
  Vec            v_analytic, v_error, b, v_curr;
  MatCtx *F_matctx;
  void (*LHS) (void);
  PetscReal      l2_error, max_error, H_error;
  KSP ksp;

  nlevels = 4;

  filename_reshist.insert(0,"MG_");
  filename_reshist.append("_nlevels"); filename_reshist.append(std::to_string(nlevels));
  PetscPrintf(PETSC_COMM_WORLD,"Output file: %s\n",filename_reshist.c_str());

  MatCtx C_matctxs[nlevels-1];
  InterpCtx restctx[nlevels-1], prolctx[nlevels-1];
  Mat    Dcoarses[nlevels-1];
  Mat    P[nlevels-1], R[nlevels-1];

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  // Extract fine grid context and matmult function
  MatShellGetContext(Afine, &F_matctx);
  MatShellGetOperation(Afine, MATOP_MULT, &LHS);

  // Check if number of grid points / number of multigrid levels / number of cores are compatible
  PetscScalar nxcoarsest, nycoarsest;
  nxcoarsest = ((double) F_matctx->gridctx.n[0])/(1 << (nlevels-1));
  nycoarsest = ((double) F_matctx->gridctx.n[1])/(1 << (nlevels-1));
  assert((nxcoarsest ==  std::floor(nxcoarsest)));
  assert((nycoarsest ==  std::floor(nycoarsest)));

  // Number of timesteps (gauss = 4)
  Nt = F_matctx->timectx.N;
  dofs = F_matctx->gridctx.dofs;

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
  MatCreateShell(PETSC_COMM_WORLD,C_matctxs[nlevels-2].gridctx.n[0]*C_matctxs[nlevels-2].gridctx.n[1]*Nt*dofs,F_matctx->gridctx.n[0]*F_matctx->gridctx.n[1]*Nt*dofs,C_matctxs[nlevels-2].gridctx.N[0]*C_matctxs[nlevels-2].gridctx.N[1]*Nt*dofs,F_matctx->gridctx.N[0]*F_matctx->gridctx.N[1]*Nt*dofs,&restctx[nlevels-2],&R[nlevels-2]);
  MatShellSetOperation(R[nlevels-2],MATOP_MULT,(void(*)(void))apply_R);
  for (level = 1; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level-1].gridctx.n[0]*C_matctxs[level-1].gridctx.n[1]*Nt*dofs,C_matctxs[level].gridctx.n[0]*C_matctxs[level].gridctx.n[1]*Nt*dofs,C_matctxs[level-1].gridctx.N[0]*C_matctxs[level-1].gridctx.N[1]*Nt*dofs,C_matctxs[level].gridctx.N[0]*C_matctxs[level].gridctx.N[1]*Nt*dofs,&restctx[level-1],&R[level-1]);
    MatShellSetOperation(R[level-1],MATOP_MULT,(void(*)(void))apply_R);    
  }

  // Prolongation matrix needed for level = nlevels-1, nlevels-2,...,1
  MatCreateShell(PETSC_COMM_WORLD,F_matctx->gridctx.n[0]*F_matctx->gridctx.n[1]*Nt*dofs,C_matctxs[nlevels-2].gridctx.n[0]*C_matctxs[nlevels-2].gridctx.n[1]*Nt*dofs,F_matctx->gridctx.N[0]*F_matctx->gridctx.N[1]*Nt*dofs,C_matctxs[nlevels-2].gridctx.N[0]*C_matctxs[nlevels-2].gridctx.N[1]*Nt*dofs,&prolctx[nlevels-2],&P[nlevels-2]);
  MatShellSetOperation(P[nlevels-2],MATOP_MULT,(void(*)(void))apply_P);  
  for (level = 1; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level].gridctx.n[0]*C_matctxs[level].gridctx.n[1]*Nt*dofs,C_matctxs[level-1].gridctx.n[0]*C_matctxs[level-1].gridctx.n[1]*Nt*dofs,C_matctxs[level].gridctx.N[0]*C_matctxs[level].gridctx.N[1]*Nt*dofs,C_matctxs[level-1].gridctx.N[0]*C_matctxs[level-1].gridctx.N[1]*Nt*dofs,&prolctx[level-1],&P[level-1]);
    MatShellSetOperation(P[level-1],MATOP_MULT,(void(*)(void))apply_P);    
  }
  // System matrix needed for level = nlevels-2, nlevels-3,...,0
  for (level = 0; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level].gridctx.n[0]*C_matctxs[level].gridctx.n[1]*Nt*dofs,C_matctxs[level].gridctx.n[0]*C_matctxs[level].gridctx.n[1]*Nt*dofs,C_matctxs[level].gridctx.N[0]*C_matctxs[level].gridctx.N[1]*Nt*dofs,C_matctxs[level].gridctx.N[0]*C_matctxs[level].gridctx.N[1]*Nt*dofs,&C_matctxs[level],&Dcoarses[level]);
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
    printf("Rank: %d, level: %d, Nx: %d, Ny: %d, N: %d, nx: %d, ny: %d, n: %d\n",rank,level,C_matctxs[level].gridctx.N[0],C_matctxs[level].gridctx.N[1],
      C_matctxs[level].gridctx.N[0]*C_matctxs[level].gridctx.N[1],C_matctxs[level].gridctx.n[0],C_matctxs[level].gridctx.n[1],C_matctxs[level].gridctx.n[0]*C_matctxs[level].gridctx.n[1]);
  }
  printf("Rank: %d, level: %d, Nx: %d, Ny: %d, N: %d, nx: %d, ny: %d, n: %d\n",rank,level,F_matctx->gridctx.N[0],F_matctx->gridctx.N[1],
      F_matctx->gridctx.N[0]*F_matctx->gridctx.N[1],F_matctx->gridctx.n[0],F_matctx->gridctx.n[1],F_matctx->gridctx.n[0]*F_matctx->gridctx.n[1]);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  DMCreateGlobalVector(F_matctx->gridctx.da_x, &v_curr);
  DMCreateGlobalVector(F_matctx->gridctx.da_xt,&b);

  set_initial_condition(F_matctx->gridctx, v_curr);

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  for (blockidx = 0; blockidx < F_matctx->timectx.tblocks; blockidx++) 
  {
    PetscPrintf(PETSC_COMM_WORLD,"---------------------- Time iteration: %d, t = %f ----------------------\n",blockidx,blockidx*F_matctx->timectx.Tpb);
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

  if (rank == 0) {
    PetscReal *reshistarr;
    PetscInt nits;
    Vec reshist;
    KSPGetResidualHistory(ksp, &reshistarr, &nits);
    reshistarr[nits] = nits;
    reshistarr[nits+1] = elapsed_time;
    VecCreateSeqWithArray(PETSC_COMM_SELF,1,nits+2,reshistarr, &reshist);
    write_vector_to_binary(reshist, "data/aco2D", filename_reshist.c_str(), PETSC_COMM_SELF);
  }


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
    DMDestroy(&C_matctxs[level].gridctx.da_xt);
    DMDestroy(&C_matctxs[level].gridctx.da_x);
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
  PetscInt level, ksp_smooth_iters, ksp_inner_iters, ksp_outer_maxit;

  ksp_smooth_iters = 8;
  ksp_inner_iters = 8;
  ksp_outer_maxit = 1e5;

  // Finest outer solver
  KSPCreate(PETSC_COMM_WORLD, &ksp_fine);
  KSPGetPC(ksp_fine, &pcfine);
  KSPSetOperators(ksp_fine, Afine, Afine);
  KSPSetTolerances(ksp_fine, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, ksp_outer_maxit);
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

  // Set interpolations
  for (level = 1; level < nlevels; level++) {
    PCMGSetRestriction(pcfine, level, R[level-1]);
    PCMGSetInterpolation(pcfine, level, P[level-1]);    
  }


  KSPSetResidualHistory(ksp_fine, NULL, ksp_outer_maxit+10, PETSC_FALSE);



  // Finest smoother
  PCMGGetSmoother(pcfine, nlevels-1, &ksp_fine_smoth);
  KSPSetOperators(ksp_fine_smoth, Afine, Afine);
  KSPSetTolerances(ksp_fine_smoth, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, ksp_smooth_iters);
  KSPSetType(ksp_fine_smoth, KSPGMRES);
  KSPSetInitialGuessNonzero(ksp_fine, PETSC_TRUE);
  KSPGMRESSetRestart(ksp_fine_smoth, ksp_smooth_iters);
  KSPGetPC(ksp_fine_smoth, &pcfine_smoot);
  // KSPSetNormType(ksp_fine_smoth, KSP_NORM_UNPRECONDITIONED);
  PCSetType(pcfine_smoot, PCNONE);
  PCSetUp(pcfine_smoot);
  KSPSetUp(ksp_fine_smoth);

  // Coarsest inner solver
  PCMGGetCoarseSolve(pcfine, &ksp_coarse);
  KSPSetOperators(ksp_coarse, Acoarses[0], Acoarses[0]);
  KSPSetTolerances(ksp_coarse, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, ksp_inner_iters);
  // KSPSetNormType(ksp_coarse, KSP_NORM_UNPRECONDITIONED);
  KSPSetType(ksp_coarse, KSPGMRES);
  KSPSetInitialGuessNonzero(ksp_coarse, PETSC_FALSE);
  KSPGMRESSetRestart(ksp_coarse, ksp_inner_iters);
  KSPSetPCSide(ksp_coarse, PC_RIGHT);
  KSPGetPC(ksp_coarse, &pccoarse);
  PCSetType(pccoarse, PCNONE);
  PCSetUp(pccoarse);
  KSPSetUp(ksp_coarse);

  // Mid smoothers
  for (level = 1; level < nlevels-1; level++) {
      PCMGGetSmoother(pcfine, level, &ksp_mid_smoth[level-1]);
      KSPSetOperators(ksp_mid_smoth[level-1], Acoarses[level], Acoarses[level]);
      KSPSetTolerances(ksp_mid_smoth[level-1], 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, ksp_smooth_iters);
      // KSPSetNormType(ksp_mid_smoth[level-1], KSP_NORM_UNPRECONDITIONED);
      KSPSetType(ksp_mid_smoth[level-1], KSPGMRES);
      KSPSetInitialGuessNonzero(ksp_mid_smoth[level-1], PETSC_TRUE);
      KSPGMRESSetRestart(ksp_mid_smoth[level-1], ksp_smooth_iters);
      KSPGetPC(ksp_mid_smoth[level-1], &pcmid_smoth[level-1]);
      PCSetType(pcmid_smoth[level-1], PCNONE);
      PCSetUp(pcmid_smoth[level-1]);
      KSPSetUp(ksp_mid_smoth[level-1]);
  }



  PCSetFromOptions(pcfine);
  PCSetUp(pcfine);


  return 0;
}

PetscErrorCode apply_P(Mat P, Vec xcoarse, Vec xfine)
{
  PetscScalar       ***array_src, ***array_dst;
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

  sbp::apply_C2F(prolctx->F_gridctx->ICF, array_src, array_dst, prolctx->F_gridctx->i_start, prolctx->F_gridctx->i_end, prolctx->F_gridctx->N);
  // sbp::apply_C2F_1p(prolctx->F_gridctx->ICF, array_src, array_dst, prolctx->F_gridctx->i_start, prolctx->F_gridctx->i_end, prolctx->F_gridctx->N);

  DMDAVecRestoreArrayDOF(dmda_fine, xfine, &array_dst);
  DMDAVecRestoreArrayDOF(dmda_coarse, xcoarse_local, &array_src);

  DMRestoreLocalVector(dmda_coarse, &xcoarse_local);

  return 0;
}

PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse)
{
  PetscScalar       ***array_src, ***array_dst;
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

  sbp::apply_F2C(restctx->F_gridctx->ICF, array_src, array_dst, restctx->C_gridctx->i_start, restctx->C_gridctx->i_end, restctx->C_gridctx->N);
  // sbp::apply_F2C_1p(restctx->F_gridctx->ICF, array_src, array_dst, restctx->C_gridctx->i_start, restctx->C_gridctx->i_end, restctx->C_gridctx->N);

  DMDAVecRestoreArrayDOF(dmda_coarse, xcoarse, &array_dst);
  DMDAVecRestoreArrayDOF(dmda_fine, xfine_local, &array_src);

  DMRestoreLocalVector(dmda_fine, &xfine_local);

  return 0;
}

PetscErrorCode setup_operators(MatCtx *C_matctx, MatCtx *F_matctx, InterpCtx *prolctx, InterpCtx *restctx)
{
  PetscInt dim;

  DMDAGetInfo(F_matctx->gridctx.da_xt, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

  if (dim == 1) {
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
  } else if (dim == 2) {
    PetscInt       stencil_radius, i_xstart, i_ystart, i_xend, i_yend, Nx, Ny, Nt, Nx_pc, Ny_pc, nx_pc, ny_pc, dofs;
    PetscScalar    xl, xr, yl, yr, dx, dy, dxi, dyi, dx_pc, dy_pc, dxi_pc, dyi_pc, Tend, Tpb, tau;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    // System matrices
    Nx = F_matctx->gridctx.N[0];
    Ny = F_matctx->gridctx.N[1];
    dxi = F_matctx->gridctx.hi[0];
    dyi = F_matctx->gridctx.hi[1];
    dx = F_matctx->gridctx.h[0];
    dy = F_matctx->gridctx.h[1];
    xl = F_matctx->gridctx.xl[0];
    yl = F_matctx->gridctx.xl[1];
    xr = F_matctx->gridctx.xr[0];
    yr = F_matctx->gridctx.xr[1];
    dofs = F_matctx->gridctx.dofs;
    auto a = F_matctx->gridctx.a;
    auto b = F_matctx->gridctx.b;
    Nt = F_matctx->timectx.N;
    Tend = F_matctx->timectx.Tend;
    Tpb = F_matctx->timectx.Tpb;

    DMCoarsen(F_matctx->gridctx.da_xt, PETSC_COMM_WORLD, &C_matctx->gridctx.da_xt);
    DMCoarsen(F_matctx->gridctx.da_x, PETSC_COMM_WORLD, &C_matctx->gridctx.da_x);

    DMDAGetCorners(C_matctx->gridctx.da_xt,&i_xstart,&i_ystart,NULL,&nx_pc,&ny_pc,NULL);
    i_xend = i_xstart + nx_pc;
    i_yend = i_ystart + ny_pc;

    DMDAGetInfo(C_matctx->gridctx.da_xt, NULL, &Nx_pc, &Ny_pc, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

    dx_pc = (xr - xl)/(Nx_pc-1);
    dy_pc = (yr - yl)/(Ny_pc-1);
    dxi_pc = 1./dx_pc;
    dyi_pc = 1./dy_pc;
    Nt = 4;
    tau = 1;

    auto [stencil_width_pc, nc_pc, cw_pc] = C_matctx->gridctx.D1.get_ranges();
    stencil_radius = (stencil_width_pc-1)/2;

    C_matctx->gridctx.N = {Nx_pc,Ny_pc};
    C_matctx->gridctx.hi = {dxi_pc,dyi_pc};
    C_matctx->gridctx.h = {dx_pc,dy_pc};
    C_matctx->gridctx.xl = {xl,yl};
    C_matctx->gridctx.xr = {xr,yr};
    C_matctx->gridctx.dofs = dofs;
    C_matctx->gridctx.a = a;
    C_matctx->gridctx.b = b;
    C_matctx->gridctx.n = {nx_pc,ny_pc};
    C_matctx->gridctx.i_start = {i_xstart,i_ystart};
    C_matctx->gridctx.i_end = {i_xend,i_yend};
    C_matctx->gridctx.sw = stencil_radius;

    C_matctx->timectx.N = Nt;
    C_matctx->timectx.Tend = Tend;
    C_matctx->timectx.Tpb = Tpb;

    setup_timestepper(C_matctx->timectx, tau);

    PetscReal Fxend = F_matctx->gridctx.xl[0] + F_matctx->gridctx.i_end[0]*dx;
    PetscReal Cxend = C_matctx->gridctx.xl[0] + C_matctx->gridctx.i_end[0]*dx_pc;

    PetscReal Fyend = F_matctx->gridctx.xl[1] + F_matctx->gridctx.i_end[1]*dy;
    PetscReal Cyend = C_matctx->gridctx.xl[1] + C_matctx->gridctx.i_end[1]*dy_pc;


    if (abs(Fxend - Cxend) >= 2*dx) {
      printf("ERROR! Rank: %d, Fxend: %f, Cxend: %f, dx_f: %f, dx_c: %f, xl_F: %f, xl_C: %f\n",rank,Fxend,Cxend,dx,dx_pc,F_matctx->gridctx.xl[0],C_matctx->gridctx.xl[0]);
      assert(abs(Fxend - Cxend) < 2*dx);
    }

    if (abs(Fyend - Cyend) >= 2*dy) {
      printf("ERROR! Rank: %d, Fyend: %f, Cyend: %f, dy_f: %f, dy_c: %f, yl_F: %f, yl_C: %f\n",rank,Fyend,Cyend,dy,dy_pc,F_matctx->gridctx.xl[1],C_matctx->gridctx.xl[1]);
      assert(abs(Fyend - Cyend) < 2*dy);
    }

    // Interpolation matrices
    restctx->F_gridctx = &F_matctx->gridctx;
    restctx->C_gridctx = &C_matctx->gridctx;

    prolctx->F_gridctx = &F_matctx->gridctx;
    prolctx->C_gridctx = &C_matctx->gridctx;
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "ERROR\n");
    return -1;
  }



  return 0;
}