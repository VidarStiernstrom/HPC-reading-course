#include<petsc.h>
#include "appctx.h"
#include "diffops/advection.h"
#include "imp_timestepping.h"

static PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) ;
static PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0);
static PetscScalar gaussian(PetscScalar x) ;
static PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic);
static PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) ;

PetscErrorCode standard_solver(Mat &A, Vec& v) 
{
	KSP ksp;
	PC pc;
	PetscLogDouble v1,v2,elapsed_time = 0;
  	Vec            v_analytic, v_error, b, v_curr;
  	PetscReal      l2_error, max_error, H_error;
	MatCtx *matctx;
	PetscInt rank, blockidx;

	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	MatShellGetContext(A, &matctx);

	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetOperators(ksp, A, A);
	KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
	KSPSetPCSide(ksp, PC_RIGHT);
	KSPSetType(ksp, KSPPIPEFGMRES);
	KSPGMRESSetRestart(ksp, 10);
	KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
	KSPSetFromOptions(ksp);
	KSPSetUp(ksp);
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCNONE);
	PCSetUp(pc);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	Solve system
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
	DMCreateGlobalVector(matctx->gridctx.da_x, &v_curr);
	DMCreateGlobalVector(matctx->gridctx.da_xt,&b);

	analytic_solution(matctx->gridctx, 0, v_curr);

	PetscBarrier((PetscObject) v);
	if (rank == 0) {
		PetscTime(&v1);
	}

	for (blockidx = 0; blockidx < matctx->timectx.tblocks; blockidx++) 
	{
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

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	Compute and print error
	- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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