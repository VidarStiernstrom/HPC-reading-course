#pragma once

#include<petscsystypes.h>
#include "grids/grid_function.h"
#include "diffops/partitioned_apply.h"

// Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x]
namespace sbp{

template <class SbpDerivative>
inline void reflection_lb(grid::grid_function_1d<PetscScalar> dst,
                                        const grid::grid_function_1d<PetscScalar> src, 
                                        const PetscInt N,
                                        const SbpDerivative& D1, 
                                        const PetscScalar hi)
{
  const PetscInt i = 0;
  dst(i,1) = D1.apply_left(src,hi,i,0);
  dst(i,0) = 0.0;
};

template <class SbpDerivative>
inline void reflection_lc(grid::grid_function_1d<PetscScalar> dst,
                                        const grid::grid_function_1d<PetscScalar> src, 
                                        const PetscInt N,
                                        const PetscInt i,
                                        const SbpDerivative& D1, 
                                        const PetscScalar hi)
{
  dst(i,1) = D1.apply_left(src,hi,i,0);
  dst(i,0) = D1.apply_left(src,hi,i,1);
};


template <class SbpDerivative>
inline void reflection_int(grid::grid_function_1d<PetscScalar> dst,
                                        const grid::grid_function_1d<PetscScalar> src, 
                                        const PetscInt N,
                                        const PetscInt i,
                                        const SbpDerivative& D1, 
                                        const PetscScalar hi)
{
  dst(i,1) = D1.apply_interior(src,hi,i,0);
  dst(i,0) = D1.apply_interior(src,hi,i,1);
};

template <class SbpDerivative>
inline void reflection_rc(grid::grid_function_1d<PetscScalar> dst,
                                        const grid::grid_function_1d<PetscScalar> src, 
                                        const PetscInt N,
                                        const PetscInt i,
                                        const SbpDerivative& D1, 
                                        const PetscScalar hi)
{
  dst(i,1) = D1.apply_right(src,hi,N,i,0);
  dst(i,0) = D1.apply_right(src,hi,N,i,1);
};

template <class SbpDerivative>
inline void reflection_rb(grid::grid_function_1d<PetscScalar> dst,
                          const grid::grid_function_1d<PetscScalar> src, 
                          const PetscInt N,
                          const SbpDerivative& D1, 
                          const PetscScalar hi)
{
  const PetscInt i = N-1;
  dst(i,1) = D1.apply_right(src,hi,N,i,0);
  dst(i,0) = 0.0;
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
inline PetscErrorCode reflection(grid::grid_function_1d<PetscScalar> dst,
                                 const grid::grid_function_1d<PetscScalar> src,
                                 const PetscInt N,
                                 const PetscInt i_start,
                                 const PetscInt i_end,
                                 const SbpDerivative& D1,
                                 const PetscScalar hi)
{
  return rhs_1D(reflection_lb<decltype(D1)>, reflection_lc<decltype(D1)>, reflection_int<decltype(D1)>, reflection_rc<decltype(D1)>, reflection_rb<decltype(D1)>,
                dst, src, N, D1.closure_size(), i_start, i_end, D1, hi);
};

/**
* Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x], for a single core, looping over the entire grid. Usefull for reference timings.
* Inputs: D1        - SBP D1 operator
**/
template <class SbpDerivative>
inline PetscErrorCode reflection_single_core(grid::grid_function_1d<PetscScalar> dst,
                                             const grid::grid_function_1d<PetscScalar> src,
                                             const PetscInt N,
                                             const SbpDerivative& D1,
                                             const PetscScalar hi)
{
  return rhs_1D(reflection_lb, reflection_lc, reflection_int,reflection_rc, reflection_rb,
                dst, src, N, D1.closure_size(), D1, hi);
};

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
  inline PetscErrorCode reflection_old(const SbpDerivative& D1, const grid::grid_function_1d<PetscScalar> src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, PetscInt N, PetscScalar hi)
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

} //namespace sbp