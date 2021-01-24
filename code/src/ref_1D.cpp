#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/reflection.h"
#include "appctx.h"
#include "ref_1D.h"

PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst)
{
  PetscScalar       **array_src, **array_dst;
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

  sbp::ref_imp_apply_all(matctx->timectx.D, matctx->gridctx.D1, matctx->gridctx.HI, matctx->gridctx.a, array_src, array_dst, matctx->gridctx.i_start[0], matctx->gridctx.i_end[0], matctx->gridctx.N[0], matctx->gridctx.hi[0], matctx->timectx.Tpb);

  DMDAVecRestoreArrayDOF(da,v_dst,&array_dst); 
  DMDAVecRestoreArrayDOF(da,v_src_local,&array_src); 

  DMRestoreLocalVector(da,&v_src_local);
  
  return 0;
}

PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) 
{
  PetscInt i;
  PetscScalar **vfinal_arr, **v_arr;

  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    vfinal_arr[i][0] = timectx.er[0]*v_arr[i][0] + timectx.er[1]*v_arr[i][2] + timectx.er[2]*v_arr[i][4] + timectx.er[3]*v_arr[i][6];
    vfinal_arr[i][1] = timectx.er[0]*v_arr[i][1] + timectx.er[1]*v_arr[i][3] + timectx.er[2]*v_arr[i][5] + timectx.er[3]*v_arr[i][7];
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  return 0;
}

PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0)
{ 
  PetscScalar       **b_arr, **v0_arr;
  PetscInt          i;

  DMDAVecGetArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x,v0,&v0_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    b_arr[i][0] = timectx.HI_el[0]*v0_arr[i][0];
    b_arr[i][1] = timectx.HI_el[0]*v0_arr[i][1];
    b_arr[i][2] = timectx.HI_el[1]*v0_arr[i][0];
    b_arr[i][3] = timectx.HI_el[1]*v0_arr[i][1];
    b_arr[i][4] = timectx.HI_el[2]*v0_arr[i][0];
    b_arr[i][5] = timectx.HI_el[2]*v0_arr[i][1];
    b_arr[i][6] = timectx.HI_el[3]*v0_arr[i][0];
    b_arr[i][7] = timectx.HI_el[3]*v0_arr[i][1];
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x,v0,&v0_arr); 

  return 0;
};

PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) 
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

PetscScalar theta1(PetscScalar x, PetscScalar t) 
{
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar));
}

PetscScalar theta2(PetscScalar x, PetscScalar t) 
{
  return -theta1(x,t);
}

PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic)
{ 
  PetscScalar x, **array_analytic;
  PetscInt i;
  PetscScalar W;

  W = 2;

  DMDAVecGetArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
  x = gridctx.xl[0] + i/gridctx.hi[0];
    array_analytic[i][0] = theta1(x,W - t) - theta2(x, -W + t);
    array_analytic[i][1] = theta1(x,W - t) + theta2(x, -W + t);
  }

  DMDAVecRestoreArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  return 0;
};

PetscErrorCode set_initial_condition(const GridCtx& gridctx, Vec& v0)
{
  PetscInt i; 
  PetscScalar **varr, x;

  DMDAVecGetArrayDOF(gridctx.da_x,v0,&varr);    

    for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    x = gridctx.xl[0] + i/gridctx.hi[0];
    varr[i][0] = theta2(x,0) - theta1(x,0);
    varr[i][1] = theta2(x,0) + theta1(x,0);
  }
  DMDAVecRestoreArrayDOF(gridctx.da_x,v0,&varr);    
  return 0;
}