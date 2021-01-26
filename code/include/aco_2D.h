#pragma once

PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx);
PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst);
PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0);
PetscScalar gaussian(PetscScalar x) ;
PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic);
PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) ;
PetscScalar theta1(PetscScalar x, PetscScalar t);
PetscScalar theta2(PetscScalar x, PetscScalar t);
PetscErrorCode set_initial_condition(const GridCtx& gridctx, Vec& v_analytic);