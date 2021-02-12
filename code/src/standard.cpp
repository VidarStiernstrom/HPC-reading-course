#include<petsc.h>
#include "appctx.h"
#include "imp_timestepping.h"
#include "IO_utils.h"
// #include "ref_1D.h"
#include "aco_2D.h"
// #include "adv_1D.h"

static PetscErrorCode shell2diag(Mat& D_shell, Vec& diag);
static PetscErrorCode jacsetup(PC pc);
static PetscErrorCode jacapply(PC pc, Vec vin, Vec vout);

struct JACCtx {
  Vec invdiag;
  Mat *A;
};


PetscErrorCode standard_solver(Mat &A, Vec& v, std::string filename_reshist) 
{
	KSP ksp;
	PC pc;
	PetscLogDouble v1,v2,elapsed_time = 0;
  	Vec            v_analytic, v_error, b, v_curr;
  	PetscReal      l2_error, max_error, H_error;
	MatCtx *matctx;
	JACCtx 		   	jacctx;
	PetscInt rank, blockidx, ksp_maxit;

  	ksp_maxit = 1e5;

  	filename_reshist.insert(0,"stand_");
  	PetscPrintf(PETSC_COMM_WORLD,"Output file: %s\n",filename_reshist.c_str());

	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	MatShellGetContext(A, &matctx);

	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetOperators(ksp, A, A);
	KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, ksp_maxit);
	KSPSetPCSide(ksp, PC_RIGHT);
	KSPGMRESSetRestart(ksp, 10);
	KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPSetFromOptions(ksp);
	KSPSetUp(ksp);
	KSPGetPC(ksp,&pc);

	KSPSetResidualHistory(ksp, NULL, ksp_maxit+10, PETSC_FALSE);

	// none
	PCSetType(pc,PCNONE);
	
	// jacobi
	// PCSetType(pc, PCSHELL);
	// jacctx.A = &A;
	// PCShellSetContext(pc, &jacctx);
	// PCShellSetSetUp(pc, jacsetup);
	// PCShellSetApply(pc, jacapply);
	PCSetUp(pc);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	Solve system
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
	DMCreateGlobalVector(matctx->gridctx.da_x, &v_curr);
	DMCreateGlobalVector(matctx->gridctx.da_xt,&b);

	set_initial_condition(matctx->gridctx, v_curr);

	PetscBarrier((PetscObject) v);
	if (rank == 0) {
		PetscTime(&v1);
	}

	for (blockidx = 0; blockidx < matctx->timectx.tblocks; blockidx++) 
	{
		PetscPrintf(PETSC_COMM_WORLD,"---------------------- Time iteration: %d, t = %f ----------------------\n",blockidx,blockidx*matctx->timectx.Tpb);
		RHS(matctx->gridctx, matctx->timectx, b, v_curr);
		KSPSolve(ksp, b, v);
		get_solution(v_curr, v, matctx->gridctx, matctx->timectx);
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

	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	// Compute and print error
	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	DMCreateGlobalVector(matctx->gridctx.da_x, &v_error);
	DMCreateGlobalVector(matctx->gridctx.da_x, &v_analytic);

	analytic_solution(matctx->gridctx, matctx->timectx.Tend, v_analytic);

	get_solution(v_curr, v, matctx->gridctx, matctx->timectx); 
	get_error(matctx->gridctx, v_curr, v_analytic, &v_error, &H_error, &l2_error, &max_error);
	PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %.8e, the H-error is: %.9e and the maximum error is %.9e\n",l2_error,H_error,max_error);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	Destroy petsc objects
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	VecDestroy(&v_analytic);
	VecDestroy(&v_error);
	VecDestroy(&b);
	VecDestroy(&v_curr);
	KSPDestroy(&ksp);

	return 0;
}

PetscErrorCode jacapply(PC pc, Vec vin, Vec vout) {
	JACCtx 		   	*jacctx;
	PCShellGetContext(pc, (void**) &jacctx);

	VecPointwiseMult(vout, vin, jacctx->invdiag);
	return 0;
}

PetscErrorCode jacsetup(PC pc) {
	JACCtx 		   	*jacctx;
	PCShellGetContext(pc, (void**) &jacctx);

	PetscPrintf(PETSC_COMM_WORLD,"Setting up Jacobi preconditioner... ");

	shell2diag(*jacctx->A, jacctx->invdiag);
	VecReciprocal(jacctx->invdiag);

	PetscPrintf(PETSC_COMM_WORLD,"Done!\n");

	return 0;
}

PetscErrorCode shell2diag(Mat& D_shell, Vec& diag) {
  Vec v, b;
  MatCtx *matctx;
  int j = 0, i, k;

  MatShellGetContext(D_shell, &matctx);


  DMCreateGlobalVector(matctx->gridctx.da_xt, &diag);
  DMGetGlobalVector(matctx->gridctx.da_xt,&v);
  DMGetGlobalVector(matctx->gridctx.da_xt,&b);

  VecSet(v,0.0);

  PetscScalar **arr;
  PetscScalar **diag_arr;

  DMDAVecGetArrayDOF(matctx->gridctx.da_xt, diag, &diag_arr);

  // Loop over all columns
  for (i = 0; i < matctx->gridctx.N[0]; i++) {
    for (k = 0; k < matctx->timectx.N*matctx->gridctx.dofs; k++) {
      j = k + matctx->timectx.N*i;

      // Set v[j] = 1
      VecSetValue(v,j,1,INSERT_VALUES);

      VecAssemblyBegin(v);
      VecAssemblyEnd(v);

      // Compute D*v
      MatMult(D_shell,v,b);
      VecSetValue(v,j,0,INSERT_VALUES);

      DMDAVecGetArrayDOF(matctx->gridctx.da_xt, b, &arr);
      if ((i >= matctx->gridctx.i_start[0]) && (i < matctx->gridctx.i_end[0])) {
        diag_arr[i][k] = arr[i][k];
      }
      DMDAVecRestoreArrayDOF(matctx->gridctx.da_xt, b, &arr); 
    }
  }
  DMDAVecRestoreArrayDOF(matctx->gridctx.da_xt, diag, &diag_arr);

  DMRestoreGlobalVector(matctx->gridctx.da_xt,&v);
  DMRestoreGlobalVector(matctx->gridctx.da_xt,&b);

  VecAssemblyBegin(diag);
  VecAssemblyEnd(diag);

  return 0;
}