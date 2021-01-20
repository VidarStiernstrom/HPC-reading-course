#define PROBLEM_TYPE_1D_O6

#include<petsc.h>
#include "appctx.h"
#include "diffops/advection.h"

struct TimeCtx {
  PetscInt N;
  PetscScalar Tend, Tpb;
  PetscScalar er[4], HI_el[4], D[4][4];
};

struct MatCtx {
  GridCtx gridctx;
  TimeCtx timectx;
};

struct InterpCtx {
  GridCtx *F_gridctx, *C_gridctx;
};

// PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse);
// PetscErrorCode apply_P(Mat R, Vec xfine, Vec xcoarse);

/**
* Setup geometric multigrid solver
*
*
**/
PetscErrorCode setup_mgsolver(KSP& ksp_fine, PetscInt nlevels, Mat& Afine, Mat Acoarses[], Mat P[], Mat R[])
{

  ///////////
  KSP ksp_coarse, ksp_fine_smoth, ksp_mid_smoth[nlevels-2];
  PC pcfine, pccoarse, pcfine_smoot, pcmid_smoth[nlevels-2];
  InterpCtx *restctx[nlevels-1], *prolctx[nlevels-1];
  MatCtx *matctx_fine, *matctx_coarses[nlevels-1];
  PetscInt Nxfine, Nxcoarse, nxfine, nxcoarse, dofs, sw, i_xstart, i_xend, Nt, level;
  PetscScalar hx_fine, hx_coarse;
  DM dmda_coarse;
  void (*LHS) (void);

  // Set solvers

  // Finest outer
  KSPCreate(PETSC_COMM_WORLD, &ksp_fine);
  KSPGetPC(ksp_fine, &pcfine);
  KSPSetOperators(ksp_fine, Afine, Afine);
  KSPSetTolerances(ksp_fine, 1e-16, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetType(ksp_fine, KSPPIPEFGMRES);
  KSPGMRESSetRestart(ksp_fine, 4);
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
  KSPSetTolerances(ksp_fine_smoth, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 20);
  KSPSetType(ksp_fine_smoth, KSPGMRES);
  // KSPGMRESSetRestart(ksp_fine_smoth, 1);
  KSPGetPC(ksp_fine_smoth, &pcfine_smoot);
  PCSetType(pcfine_smoot, PCNONE);
  KSPSetFromOptions(ksp_fine_smoth);
  PCSetUp(pcfine_smoot);
  KSPSetUp(ksp_fine_smoth);

  // Inner coarsest
  PCMGGetCoarseSolve(pcfine, &ksp_coarse);
  KSPSetOperators(ksp_coarse, Acoarses[0], Acoarses[0]);
  KSPSetTolerances(ksp_coarse, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetType(ksp_coarse, KSPGMRES);
  // KSPGMRESSetRestart(ksp_coarse, 1);
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
      KSPSetTolerances(ksp_mid_smoth[level-1], 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 20);
      KSPSetType(ksp_mid_smoth[level-1], KSPGMRES);
      KSPGetPC(ksp_mid_smoth[level-1], &pcmid_smoth[level-1]);
      PCSetType(pcmid_smoth[level-1], PCNONE);
      KSPSetFromOptions(ksp_mid_smoth[level-1]);
      PCSetUp(pcmid_smoth[level-1]);
      KSPSetUp(ksp_mid_smoth[level-1]);
  }

  // Set interpolations
  MatShellGetContext(Afine, &matctx_fine);
  for (level = 0; level < nlevels-1; level++) {
    MatShellGetContext(Acoarses[level], &matctx_coarses[level]);    
  }

  for (level = 1; level < nlevels; level++) {
    PCMGSetRestriction(pcfine, level, R[level-1]);
    PCMGSetInterpolation(pcfine, level, P[level-1]);    
  }

  PCSetFromOptions(pcfine);
  PCSetUp(pcfine);

  return 0;
}
