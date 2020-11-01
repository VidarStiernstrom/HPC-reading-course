#pragma once

#include<petscsystypes.h>
#include <array>
#include "grids/grid_function.h"

namespace sbp{

  /**
  * Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x], between indices i_start <= i < i_end. "Smart" looping.
  * Inputs: D1        - SBP D1 operator
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative>
  inline PetscErrorCode reflection_apply1(const SbpDerivative& D1, const grid::grid_function_1d<PetscScalar> src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, PetscInt N, PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) {
      for (i = 0; i < n_closures; i++) 
      { 
        dst(i,1) = D1.apply_left(src,hi,i,0);
        dst(i,0) = D1.apply_left(src,hi,i,1);
      }
      i_start = n_closures;
    }

    if (i_end == N) {
      for (i = N-n_closures; i < N; i++)
      {
          dst(i,1) = D1.apply_right(src,hi,N,i,0);
          dst(i,0) = D1.apply_right(src,hi,N,i,1);
      }
      i_end = N-n_closures;
    }

    for (i = i_start; i < i_end; i++)
    {
      dst(i,1) = D1.apply_interior(src,hi,i,0);
      dst(i,0) = D1.apply_interior(src,hi,i,1);
    }

    return 0;
  };

  /**
  * Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x], between indices i_start <= i < i_end. Direct looping.
  * Inputs: D1        - SBP D1 operator
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative>
  inline PetscErrorCode reflection_apply2(const SbpDerivative& D1, const grid::grid_function_1d<PetscScalar> src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, PetscInt N, PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      for (i = 0; i < n_closures; i++) 
      { 
        dst(i,1) = D1.apply_left(src,hi,i,0);
        dst(i,0) = D1.apply_left(src,hi,i,1);
      }
      
      for (i = n_closures; i < i_end; i++)
      {
        dst(i,1) = D1.apply_interior(src,hi,i,0);
        dst(i,0) = D1.apply_interior(src,hi,i,1);
      }
    } else if (i_end == N) 
    {
      for (i = N-n_closures; i < N; i++)
      {
        dst(i,1) = D1.apply_right(src,hi,N,i,0);
        dst(i,0) = D1.apply_right(src,hi,N,i,1);
      }

      for (i = i_start; i < N-n_closures; i++)
      {
        dst(i,1) = D1.apply_interior(src,hi,i,0);
        dst(i,0) = D1.apply_interior(src,hi,i,1);
      }
    } else 
    {
      for (i = i_start; i < i_end; i++)
      {
        dst(i,1) = D1.apply_interior(src,hi,i,0);
        dst(i,0) = D1.apply_interior(src,hi,i,1);
      }
    }
    return 0;
  };
} //namespace sbp