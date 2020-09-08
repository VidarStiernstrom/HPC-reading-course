#pragma once

#include<petscsystypes.h>

namespace sbp{

  /**
  * Approximate RHS of advection problem, u_t = -au_x, between indices i_start <= i < i_end.
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function a(x(i)),
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode advection_apply(const SbpDerivative& D1, VelocityFunction&& a, const PetscScalar *const *const array_src, PetscScalar **array_dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      for (i = 0; i < n_closures; i++) 
      { 
        array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_left(array_src,hi,i,0);
      }
      i_start = n_closures;
    }

    if (i_end == N) 
    {
      for (i = N-n_closures; i < N; i++)
      {
          array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_right(array_src,hi,N,i,0);
      }
      i_end = N-n_closures;
    }

    for (i = i_start; i < i_end; i++)
    {
      array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_interior(array_src,hi,i,0);
    }

    return 0;
  };


  /**
  * Approximate RHS of advection problem, u_t = -au_x, between indices i_start <= i < i_end.
  * Inputs: D1        - SBP D1 operator
  *         a         -  
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         Nx         - Global number of points excluding ghost points
  *         hix        - Inverse step length
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode advection_apply_2D_x(const SbpDerivative& D1, VelocityFunction&& a,
                                             const PetscScalar *const *const *const array_src,
                                             PetscScalar *const *const *const array_dst,
                                             PetscInt i_start, const PetscInt j_start,
                                             PetscInt i_end, const PetscInt j_end,
                                             const PetscInt Nx, const PetscScalar hix)
  {
    PetscInt i,j;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    if (i_start == 0) 
    {
      for (j = j_start; j < j_end; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
          array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hix,i,j,0);
        }
      }
      i_start = n_closures;
    }

    if (i_end == Nx) 
    {
      for (j = j_start; j < j_end; j++)
      { 
        for (i = Nx-n_closures; i < Nx; i++)
        {
            array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hix,Nx,i,j,0);
        }
      }
      i_end = Nx-n_closures;
    }
    for (j = j_start; j < j_end; j++)
      { 
      for (i = i_start; i < i_end; i++)
      {
        array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hix,i,j,0);
      }
    }
    return 0;
  };
} //namespace sbp