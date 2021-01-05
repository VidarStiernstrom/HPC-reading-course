#pragma once

#include <petscsystypes.h>
#include <array>
#include "grids/grid_function.h"


namespace sbp{
  //=============================================================================
  // 1D functions
  //=============================================================================

  inline PetscScalar adv_imp_apply_D_time(PetscScalar D_time[4][4], PetscScalar ***src, PetscInt i, PetscInt tcomp)
  {
    return D_time[tcomp][0]*src[i][0][0] + D_time[tcomp][1]*src[i][1][0] + D_time[tcomp][2]*src[i][2][0] + D_time[tcomp][3]*src[i][3][0];
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_left(PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar ***src, PetscScalar ***dst, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i, tcomp;

    i = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[i][tcomp][0] = adv_imp_apply_D_time(D_time, src, i, tcomp);
    }

    for (tcomp = 0; tcomp < 4; tcomp++) {
      src[i][tcomp][0] = 0;
    }
    
    for (i = 1; i < n_closures; i++)
    {
      dst[i][0][0] = adv_imp_apply_D_time(D_time, src, i, 0) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0,0);
      dst[i][1][0] = adv_imp_apply_D_time(D_time, src, i, 1) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0,1);
      dst[i][2][0] = adv_imp_apply_D_time(D_time, src, i, 2) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0,2);
      dst[i][3][0] = adv_imp_apply_D_time(D_time, src, i, 3) + std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0,3);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_right(PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar ***src, PetscScalar ***dst, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i;

    for (i = N-n_closures; i < N; i++)
    {
      dst[i][0][0] = adv_imp_apply_D_time(D_time, src, i, 0) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,0,0);
      dst[i][1][0] = adv_imp_apply_D_time(D_time, src, i, 1) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,0,1);
      dst[i][2][0] = adv_imp_apply_D_time(D_time, src, i, 2) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,0,2);
      dst[i][3][0] = adv_imp_apply_D_time(D_time, src, i, 3) + std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,N,i,0,3);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_interior(PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar ***src, PetscScalar ***dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i;

    for (i = i_start; i < i_end; i++)
    {
      dst[i][0][0] = adv_imp_apply_D_time(D_time, src, i, 0) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,0,0);
      dst[i][1][0] = adv_imp_apply_D_time(D_time, src, i, 1) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,0,1);
      dst[i][2][0] = adv_imp_apply_D_time(D_time, src, i, 2) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,0,2);
      dst[i][3][0] = adv_imp_apply_D_time(D_time, src, i, 3) + std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,0,3); 
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode adv_imp_apply_all(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar ***src, PetscScalar ***dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi, const PetscScalar Tend)
  {
    int i, j;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    PetscScalar tau = 1.0;;
    PetscScalar D1_time[4][4] = {
      {-3.3320002363522817*2./Tend,4.8601544156851962*2./Tend,-2.1087823484951789*2./Tend,0.5806281691622644*2./Tend},
      {-0.7575576147992339*2./Tend,-0.3844143922232086*2./Tend,1.4706702312807167*2./Tend,-0.3286982242582743*2./Tend},
      {0.3286982242582743*2./Tend,-1.4706702312807167*2./Tend,0.3844143922232086*2./Tend,0.7575576147992339*2./Tend},
      {-0.5806281691622644*2./Tend,2.1087823484951789*2./Tend,-4.8601544156851962*2./Tend,3.3320002363522817*2./Tend}
    };

    PetscScalar HI_BL_time[4][4] = {
      {6.701306630115196*2./Tend,-3.571157279331500*2./Tend,1.759003615747388*2./Tend,-0.500000000000000*2./Tend},
      {-1.904858685372247*2./Tend,1.015107998460294*2./Tend,-0.500000000000000*2./Tend,0.142125915923019*2./Tend},
      {0.938254199681965*2./Tend,-0.500000000000000*2./Tend,0.246279214013876*2./Tend,-0.070005317729047*2./Tend},
      {-0.500000000000000*2./Tend,0.266452311201742*2./Tend,-0.131243331549891*2./Tend,0.037306157410634*2./Tend}
    };

    PetscScalar D_time[4][4];

    for (i = 0; i < 4; i++) {
      for (j = 0; j < 4; j++) {
        D_time[j][i] = D1_time[j][i] + tau*HI_BL_time[j][i];
      }
    }

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