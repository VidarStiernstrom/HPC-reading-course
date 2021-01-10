#pragma once

#include <petscsystypes.h>
#include <array>
#include "grids/grid_function.h"


namespace sbp{

  //=============================================================================
  // 1D functions
  //=============================================================================


  template <class SbpInterpolator>
  inline PetscErrorCode apply_F2C(const SbpInterpolator& ICF, PetscScalar **src, PetscScalar **dst, PetscInt i_start, PetscInt i_end, const PetscInt N)
  {
    int i;
    const auto [F2C_nc, C2F_nc] = ICF.get_ranges();
    if (i_start == 0) {
      for (i = 0; i < F2C_nc; i++) 
      { 
        dst[i][0] = ICF.F2C_apply_left(src, i, 0);
      }

      for (i = F2C_nc; i < i_end; i++) 
      { 
        dst[i][0] = ICF.F2C_apply_interior(src, i, 0);
      }
    } else if (i_end == N) {

      for (i = i_start; i < i_end-F2C_nc; i++) 
      { 
        dst[i][0] = ICF.F2C_apply_interior(src, i, 0);
      }

      for (i = i_end - F2C_nc; i < i_end; i++) 
      { 
        dst[i][0] = ICF.F2C_apply_right(src, i_end, i, 0);
      }
    } else {
      for (i = i_start; i < i_end; i++) 
      { 
        dst[i][0] = ICF.F2C_apply_interior(src, i, 0);
      }
    }
    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode apply_C2F(const SbpInterpolator& ICF, PetscScalar **src, PetscScalar **dst, PetscInt i_start, PetscInt i_end, const PetscInt N)
  {
    int i;
    const auto [F2C_nc, C2F_nc] = ICF.get_ranges();
    if (i_start == 0) {
      for (i = 0; i < C2F_nc; i++) 
      { 
        dst[i][0] = ICF.C2F_apply_left(src, i, 0);
      }

      for (i = C2F_nc; i < i_end; i++) 
      { 
        if (i % 2 == 0) {
          dst[i][0] = ICF.C2F_even_apply_interior(src, i, 0);
        } else {
          dst[i][0] = ICF.C2F_odd_apply_interior(src, i, 0);  
        }
      }
    } else if (i_end == N) {
      for (i = i_start; i < i_end-C2F_nc; i++) 
      { 
        if (i % 2 == 0) {
          dst[i][0] = ICF.C2F_even_apply_interior(src, i, 0);
        } else {
          dst[i][0] = ICF.C2F_odd_apply_interior(src, i, 0);  
        }
      }

      for (i = i_end - C2F_nc; i < i_end; i++) 
      { 
        // printf("hejhej %f\n",dst[i][0]);
        dst[i][0] = ICF.C2F_apply_right(src, N, i, 0);
      }
    } else {
      for (i = i_start; i < i_end; i++) 
      { 
        if (i % 2 == 0) {
          dst[i][0] = ICF.C2F_even_apply_interior(src, i, 0);
        } else {
          dst[i][0] = ICF.C2F_odd_apply_interior(src, i, 0);  
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode ode_apply_l(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i;
    PetscScalar tmp, sigma;
   sigma = hi;
    // BC using projection, vt = P*D*P*v.

    i = 0;
    tmp = sigma*src[0][0];
    src[0][0] = 0.0;

    dst[i][0] = D1.apply_left(src,hi,i,0) + tmp;

    for (i = 1; i < n_closures; i++) 
    { 
      dst[i][0] = D1.apply_left(src,hi,i,0);
    }

    // BC using SAT, vt = -a*D*v + tau*HI*e_1*e_1'*v
    // PetscScalar tau = -std::forward<VelocityFunction>(a)(0)/2; // assume a(0) good enough
    // dst(0,0) = -std::forward<VelocityFunction>(a)(0)*D1.apply_left(src,hi,0,0) + tau*HI.apply_left(src, hi, 0, 0);
    // for (i = 1; i < n_closures; i++) 
    // { 
    //   dst(i,0) = -std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0);
    // }   

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode ode_apply_r(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, const PetscInt N, const PetscScalar hi, const PetscInt n_closures)
  {
    int i;
    for (i = N-n_closures; i < N; i++)
    {
        dst[i][0] = D1.apply_right(src,hi,N,i,0);
    }
    return 0;
  }

  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode ode_apply_c(const SbpDerivative& D1, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi)
  {
    int i;
    for (i = i_start; i < i_end; i++)
    {
      dst[i][0] = D1.apply_interior(src,hi,i,0);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode ode_apply_1p(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi)
  {
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    ode_apply_l(D1, HI, a, src, dst,  N, hi, n_closures);
    ode_apply_c(D1, a, src, dst, n_closures, i_end-n_closures,  N, hi);
    ode_apply_r(D1, HI, a, src, dst,  N, hi, n_closures);

    return 0;
  };

  /**
  * Approximate RHS of ode problem, u_t = -au_x, between indices i_start <= i < i_end. Direct looping.
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function parametrized on indices, i.e a(x(i)),
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode ode_apply_all(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, PetscScalar **dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi)
  {
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      ode_apply_l(D1, HI, a, src, dst,  N, hi, n_closures);
      ode_apply_c(D1, a, src, dst, n_closures, i_end,  N, hi);
    } else if (i_end == N) 
    {
      ode_apply_r(D1, HI, a, src, dst,  N, hi, n_closures);
      ode_apply_c(D1, a, src, dst, i_start, N - n_closures,  N, hi);
    } else 
    {
      ode_apply_c(D1, a, src, dst, i_start, i_end,  N, hi);
    }
    return 0;
  };

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode ode_apply_inner(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi, const PetscInt sw)
  {
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      ode_apply_l(D1, HI, a, src, dst, N, hi, n_closures);
      ode_apply_c(D1, a, src, dst, n_closures, i_end-sw,  N, hi);
    } else if (i_end == N) 
    {
      ode_apply_r(D1, HI, a, src, dst, N, hi, n_closures);
      ode_apply_c(D1, a, src, dst, i_start+sw, N - n_closures,  N, hi);
    } else 
    {
      ode_apply_c(D1, a, src, dst, i_start, i_end,  N, hi);
    }
    return 0;
  };

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode ode_apply_outer(const SbpDerivative& D1, const SbpInvQuad& HI, VelocityFunction&& a, PetscScalar **src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi, const PetscInt sw)
  {
    if (i_start == 0) 
    {
      ode_apply_c(D1, a, src, dst, i_end-sw, i_end,  N, hi);
    } else if (i_end == N) 
    {
      ode_apply_c(D1, a, src, dst, i_start, i_start+sw,  N, hi);
    } else 
    {
      ode_apply_c(D1, a, src, dst, i_start, i_start+sw,  N, hi);
      ode_apply_c(D1, a, src, dst, i_end-sw, i_end,  N, hi);
    }
    return 0;
  };

} //namespace sbp