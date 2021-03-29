#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/acowave.h"
#include "appctx.h"
#include "aco_2D.h"

PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst)
{
  PetscScalar       ***array_src, ***array_dst;
  Vec               v_src_local;
  MatCtx            *matctx;
  DM                da;

  MatShellGetContext(D, &matctx);
  MatGetDM(D, &da);

  DMGetLocalVector(da, &v_src_local);

  DMGlobalToLocalBegin(da,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(da,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(da,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(da,v_src_local,&array_src); 

  sbp::aco_imp_apply_all(matctx->timectx.D, matctx->gridctx.D1, matctx->gridctx.HI, matctx->gridctx.a, matctx->gridctx.a, 
    array_src, array_dst, matctx->gridctx.i_start, matctx->gridctx.i_end, matctx->gridctx.N, matctx->gridctx.xl, matctx->gridctx.hi, matctx->gridctx.sw);

  // sbp::aco_imp_apply_1p(matctx->timectx.D, matctx->gridctx.D1, matctx->gridctx.HI, matctx->gridctx.a, matctx->gridctx.a, 
    // array_src, array_dst, matctx->gridctx.i_start, matctx->gridctx.i_end, matctx->gridctx.N, matctx->gridctx.xl, matctx->gridctx.hi, matctx->gridctx.sw);

  DMDAVecRestoreArrayDOF(da,v_dst,&array_dst); 
  DMDAVecRestoreArrayDOF(da,v_src_local,&array_src); 

  DMRestoreLocalVector(da,&v_src_local);
  
  return 0;
}

PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) 
{
  PetscInt i, j;
  PetscScalar ***vfinal_arr, ***v_arr;

  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    for (j = gridctx.i_start[1]; j < gridctx.i_end[1]; j++) {
      vfinal_arr[j][i][0] = timectx.er[0]*v_arr[j][i][0] + timectx.er[1]*v_arr[j][i][3] + timectx.er[2]*v_arr[j][i][6] + timectx.er[3]*v_arr[j][i][9];
      vfinal_arr[j][i][1] = timectx.er[0]*v_arr[j][i][1] + timectx.er[1]*v_arr[j][i][4] + timectx.er[2]*v_arr[j][i][7] + timectx.er[3]*v_arr[j][i][10];
      vfinal_arr[j][i][2] = timectx.er[0]*v_arr[j][i][2] + timectx.er[1]*v_arr[j][i][5] + timectx.er[2]*v_arr[j][i][8] + timectx.er[3]*v_arr[j][i][11];
    }
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  return 0;
}

PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0)
{ 
  PetscScalar       ***b_arr, ***v0_arr;
  PetscInt          i, j;

  DMDAVecGetArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x,v0,&v0_arr); 

  for (j = gridctx.i_start[1]; j < gridctx.i_end[1]; j++) {
    for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
      b_arr[j][i][0] = timectx.HI_el[0]*v0_arr[j][i][0];
      b_arr[j][i][1] = timectx.HI_el[0]*v0_arr[j][i][1];
      b_arr[j][i][2] = timectx.HI_el[0]*v0_arr[j][i][2];
      b_arr[j][i][3] = timectx.HI_el[1]*v0_arr[j][i][0];
      b_arr[j][i][4] = timectx.HI_el[1]*v0_arr[j][i][1];
      b_arr[j][i][5] = timectx.HI_el[1]*v0_arr[j][i][2];
      b_arr[j][i][6] = timectx.HI_el[2]*v0_arr[j][i][0];
      b_arr[j][i][7] = timectx.HI_el[2]*v0_arr[j][i][1];
      b_arr[j][i][8] = timectx.HI_el[2]*v0_arr[j][i][2];
      b_arr[j][i][9] = timectx.HI_el[3]*v0_arr[j][i][0];
      b_arr[j][i][10] = timectx.HI_el[3]*v0_arr[j][i][1];
      b_arr[j][i][11] = timectx.HI_el[3]*v0_arr[j][i][2];
    }
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x,v0,&v0_arr); 

  return 0;
};

PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) 
{
  PetscScalar ***arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);

  *l2_error = sqrt(gridctx.h[0]*gridctx.h[1])*(*l2_error);
  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(gridctx.da_x, *v_error, &arr);
  *H_error = gridctx.H.get_norm_2D(arr, gridctx.h, gridctx.N, gridctx.i_start, gridctx.i_end, gridctx.dofs);
  DMDAVecRestoreArrayDOF(gridctx.da_x, *v_error, &arr);

  return 0;
}

PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic)
{ 
  PetscScalar x, y, ***array_analytic;
  PetscInt i, j, m, n;
  PetscScalar W;

  W = 2;
  n = 3; // n = 1,2,3,4,...
  m = 4; // m = 1,2,3,4,...

  DMDAVecGetArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  for (j = gridctx.i_start[1]; j < gridctx.i_end[1]; j++)
  {
    y = gridctx.xl[1] + j/gridctx.hi[1];
    for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++)
    {
      x = gridctx.xl[0] + i/gridctx.hi[0];
      array_analytic[j][i][0] = -n*cos(n*PETSC_PI*x)*sin(m*PETSC_PI*y)*sin(PETSC_PI*sqrt(n*n + m*m)*t)/sqrt(n*n + m*m);
      array_analytic[j][i][1] = -m*sin(n*PETSC_PI*x)*cos(m*PETSC_PI*y)*sin(PETSC_PI*sqrt(n*n + m*m)*t)/sqrt(n*n + m*m);
      array_analytic[j][i][2] = sin(PETSC_PI*n*x)*sin(PETSC_PI*m*y)*cos(PETSC_PI*sqrt(n*n + m*m)*t);
    }
  }

  DMDAVecRestoreArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  return 0;
};


PetscErrorCode set_initial_condition(const GridCtx& gridctx, Vec& v0)
{
  analytic_solution(gridctx, 0, v0);
  return 0;
}
