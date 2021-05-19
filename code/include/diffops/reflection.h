#pragma once

#include<petscsystypes.h>
#include "grids/grid_function.h"
#include "partitioned_rhs/rhs.h"
#include "partitioned_rhs/boundary_conditions.h"

// Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x]
namespace sbp{

template <class SbpDerivative>
inline void reflection_l(grid::grid_function_1d<PetscScalar> dst,
                          const grid::grid_function_1d<PetscScalar> src, 
                          const PetscInt cls_sz,
                          const SbpDerivative& D1, 
                          const PetscScalar hi)
{
  for (PetscInt i = 0; i < cls_sz; i++){
    dst(i,1) = D1.apply_left(src,hi,i,0);
    dst(i,0) = D1.apply_left(src,hi,i,1);  
  }
};

template <class SbpDerivative>
inline void reflection_i(grid::grid_function_1d<PetscScalar> dst,
                          const grid::grid_function_1d<PetscScalar> src, 
                          const std::array<PetscInt,2>& ind_i,
                          const SbpDerivative& D1, 
                          const PetscScalar hi)
{
  for (PetscInt i = ind_i[0]; i<ind_i[1]; i++) {
    dst(i,1) = D1.apply_interior(src,hi,i,0);
    dst(i,0) = D1.apply_interior(src,hi,i,1);
  }
};

template <class SbpDerivative>
inline void reflection_r(grid::grid_function_1d<PetscScalar> dst,
                          const grid::grid_function_1d<PetscScalar> src, 
                          const PetscInt cls_sz,
                          const SbpDerivative& D1, 
                          const PetscScalar hi)
{
  const PetscInt nx = src.mapping().nx();
  for (PetscInt i = nx-cls_sz; i < nx; i++) {
    dst(i,1) = D1.apply_right(src,hi,i,0);
    dst(i,0) = D1.apply_right(src,hi,i,1);
  }
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
inline void reflection_local(grid::grid_function_1d<PetscScalar> dst,
                             const grid::grid_function_1d<PetscScalar> src,
                             const std::array<PetscInt,2>& ind_i,
                             const PetscInt halo_sz,
                             const SbpDerivative& D1,
                             const PetscScalar hi)
{
  const PetscInt cls_sz = D1.closure_size();
  rhs_local(reflection_l<decltype(D1)>, reflection_i<decltype(D1)>, reflection_r<decltype(D1)>,
            dst, src, ind_i, cls_sz, halo_sz, D1, hi);
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
inline void reflection_overlap(grid::grid_function_1d<PetscScalar> dst,
                               const grid::grid_function_1d<PetscScalar> src,
                               const std::array<PetscInt,2>& ind_i,
                               const PetscInt halo_sz,
                               const SbpDerivative& D1,
                               const PetscScalar hi)
{
  rhs_overlap(reflection_i<decltype(D1)>, dst, src,  ind_i, halo_sz, D1, hi);
};

/**
* Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x], for a single core, looping over the entire grid. Usefull for reference timings.
* Inputs: D1        - SBP D1 operator
**/
template <class SbpDerivative>
inline void reflection_serial(grid::grid_function_1d<PetscScalar> dst,
                              const grid::grid_function_1d<PetscScalar> src,
                              const SbpDerivative& D1,
                              const PetscScalar hi)
{
  const PetscInt cls_sz = D1.closure_size();
  rhs_serial(reflection_l<decltype(D1)>, reflection_i<decltype(D1)>, reflection_r<decltype(D1)>,
                    dst, src, cls_sz, D1, hi);
};

inline void proj_dirichlet_bc_l(grid::grid_function_1d<PetscScalar> dst,
                                const grid::grid_function_1d<PetscScalar> src)
{
  const PetscInt i = 0;
  dst(i,0) = 0.0;
};

inline void proj_dirichlet_bc_r(grid::grid_function_1d<PetscScalar> dst,
                                const grid::grid_function_1d<PetscScalar> src)
{
  const PetscInt i = src.mapping().nx()-1;
  dst(i,0) = 0.0;
};

inline void reflection_bc(grid::grid_function_1d<PetscScalar> dst,
                          const grid::grid_function_1d<PetscScalar> src,
                          const std::array<PetscInt,2>& ind_i)
{
  bc(proj_dirichlet_bc_l,proj_dirichlet_bc_r,dst,src,ind_i);
};

inline void reflection_bc_serial(grid::grid_function_1d<PetscScalar> dst,
                                 const grid::grid_function_1d<PetscScalar> src)
{
  bc_serial(proj_dirichlet_bc_l,proj_dirichlet_bc_r,dst,src);
};



} //namespace sbp