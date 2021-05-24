#pragma once

#include <petscvec.h>
#include <numeric>
#include <array>

/**
* Computes the error between the vectors v1 and v2 in the l2-norm
* v1, v2 - vectors being compared.
* h - array storing grid spacings
**/
template <size_t dim>
PetscScalar error_l2(const Vec v1, const Vec v2, const std::array<PetscScalar,dim>& h);

/**
* Computes the error between the vectors v1 and v2 in the l2-norm.
* Special case used for 1D grids.
* v1, v2 - vectors being compared.
* h - grid spacing
**/
PetscScalar error_l2(const Vec v1, const Vec v2, const PetscScalar h);

/**
* Computes the error between the vectors v1 and v2 in the max (infintiy)-norm.
* v1, v2 - vectors being compared.
**/
PetscScalar error_max(const Vec v1, const Vec v2);

/**
* Utility function returning the vector holding the elementwise absolute error between vectors v1, v2
* The returned vectors options is duplicated by v1 and should be destroyed by the caller.
* v1, v2 - vectors being compared.
**/
Vec compute_error(const Vec v1, const Vec v2);

/**
* Utility function computing the error between vectors v1, v2 in the norm n
* v1, v2 - vectors being compared.
* n - norm used to measure error.
**/
PetscScalar compute_error_norm(const Vec v1, const Vec v2, const NormType n);


//=============================================================================
// Implementations
//=============================================================================
template <size_t dim>
PetscScalar error_l2(const Vec v1, const Vec v2, const std::array<PetscScalar,dim>& h) {
  PetscScalar err_l2 = compute_error_norm(v1, v2, NORM_2);
  PetscScalar h_prod = std::accumulate(h.begin(),h.end(), 1.0, std::multiplies<PetscScalar>());
  return err_l2 = sqrt(h_prod)*(err_l2);
}

PetscScalar error_l2(const Vec v1, const Vec v2, const PetscScalar h) {
  std::array<PetscScalar,1> h_arr = {h};
  return error_l2(v1, v2, h_arr);
}

PetscScalar error_max(const Vec v1, const Vec v2) {
  return compute_error_norm(v1, v2, NORM_INFINITY);
}

Vec compute_error(const Vec v1, const Vec v2)
{
  Vec err;
  VecDuplicate(v1,&err);
  VecWAXPY(err,-1,v1,v2);
  VecAbs(err);
  return err;
}

PetscScalar compute_error_norm(const Vec v1, const Vec v2, const NormType n)
{
  PetscScalar err_n;
  Vec err = compute_error(v1, v2);
  VecNorm(err,n,&err_n);
  VecDestroy(&err);
  return err_n;
}



