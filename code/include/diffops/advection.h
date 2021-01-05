#pragma once

#include <petscsystypes.h>
#include <array>
#include "grids/grid_function.h"


namespace sbp{
  //=============================================================================
  // 1D functions
  //=============================================================================

  inline PetscScalar adv_imp_apply_D_time(const PetscScalar D_time[4][4], PetscScalar **src, PetscInt i, PetscInt tcomp)
  {
    return D_time[tcomp][0]*src[i][0] + D_time[tcomp][1]*src[i][1] + D_time[tcomp][2]*src[i][2] + D_time[tcomp][3]*src[i][3];
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_left(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i, tcomp;

    i = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[i][tcomp] = adv_imp_apply_D_time(D_time, src, i, tcomp);
    }

    for (tcomp = 0; tcomp < 4; tcomp++) {
      src[i][tcomp] = 0;
    }
    
    for (i = 1; i < n_closures; i++)
    {
      dst[i][0] = adv_imp_apply_D_time(D_time, src, i, 0) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0);
      dst[i][1] = adv_imp_apply_D_time(D_time, src, i, 1) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,1);
      dst[i][2] = adv_imp_apply_D_time(D_time, src, i, 2) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,2);
      dst[i][3] = adv_imp_apply_D_time(D_time, src, i, 3) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,3);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_right(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i;

    for (i = N-n_closures; i < N; i++)
    {
      dst[i][0] = adv_imp_apply_D_time(D_time, src, i, 0) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,0);
      dst[i][1] = adv_imp_apply_D_time(D_time, src, i, 1) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,1);
      dst[i][2] = adv_imp_apply_D_time(D_time, src, i, 2) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,2);
      dst[i][3] = adv_imp_apply_D_time(D_time, src, i, 3) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,3);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_interior(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i;

    for (i = i_start; i < i_end; i++)
    {
      dst[i][0] = adv_imp_apply_D_time(D_time, src, i, 0) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,0);
      dst[i][1] = adv_imp_apply_D_time(D_time, src, i, 1) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,1);
      dst[i][2] = adv_imp_apply_D_time(D_time, src, i, 2) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,2);
      dst[i][3] = adv_imp_apply_D_time(D_time, src, i, 3) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,3); 
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_all(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi, const PetscScalar Tend)
  {
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      adv_imp_apply_left(D_time, D1, HI, a, src, dst, N, hi, n_closures);
      adv_imp_apply_interior(D_time, D1, HI, a, src, dst, n_closures, i_end, N, hi, n_closures);
    } else if (i_end == N) 
    {
      adv_imp_apply_right(D_time, D1, HI, a, src, dst, N, hi, n_closures);
      adv_imp_apply_interior(D_time, D1, HI, a, src, dst, i_start, N-n_closures, N, hi, n_closures);
    } else 
    {
      adv_imp_apply_interior(D_time, D1, HI, a, src, dst, i_start, i_end, N, hi, n_closures);
    }
    return 0;
  }  

} //namespace sbp