#pragma once
#include<petscsystypes.h>
#include "grids/grid_function.h"

//=============================================================================
// 1D functions
//=============================================================================

template <typename ApplyLBoundary, 
          typename ApplyLInterior, 
          typename ApplyCInterior, 
          typename ApplyRBoundary, 
          typename ApplyRInterior, 
          typename... Args, 
          class SbpDerivative>
inline PetscErrorCode partitioned_apply_1D_non_overlapping(const ApplyLBoundary& apply_L_boundary,
                                                           const ApplyLInterior& apply_L_interior,
                                                           const ApplyCInterior& apply_C_interior,
                                                           const ApplyRInterior& apply_R_interior,
                                                           const ApplyRBoundary& apply_R_boundary,                                                          
                                                                 grid::grid_function_1d<PetscScalar> dst, 
                                                           const grid::grid_function_1d<PetscScalar> src,
                                                           const PetscInt N,
                                                           const SbpDerivative& D1,
                                                           const PetscScalar hi,
                                                           const PetscInt i_start,
                                                           const PetscInt i_end,
                                                           Args... args)
{
  PetscInt i;
  const auto n_closures = D1.closure_size();
  if (i_start == 0) 
  {
    apply_L_boundary(dst,src,N,D1,hi,args...);
    for (i = 1; i < n_closures; i++) 
    { 
      apply_L_interior(dst,src,N,D1,hi,i,args...);
    }
    for (i = n_closures; i < i_end; i++)
    {
      apply_C_interior(dst,src,N,D1,hi,i,args...);
    }
    
  } else if (i_end == N) 
  {
    for (i = i_start; i < N-n_closures; i++)
    {
      apply_C_interior(dst,src,N,D1,hi,i,args...);
    }
    for (i = N-n_closures; i < N-1; i++)
    {
      apply_R_interior(dst,src,N,D1,hi,i,args...);
    }
    apply_R_boundary(dst,src,N,D1,hi,args...);
  } else 
  {
    for (i = i_start; i < i_end; i++)
    {
      apply_C_interior(dst,src,N,D1,hi,i,args...);
    }
  }
  return 0;
};


template <typename ApplyLBoundary, 
          typename ApplyLInterior, 
          typename ApplyCInterior, 
          typename ApplyRBoundary, 
          typename ApplyRInterior, 
          typename... Args, 
          class SbpDerivative>
inline PetscErrorCode partitioned_apply_1D_single_core(const ApplyLBoundary& apply_L_boundary,
                                                       const ApplyLInterior& apply_L_interior,
                                                       const ApplyCInterior& apply_C_interior,
                                                       const ApplyRInterior& apply_R_interior,
                                                       const ApplyRBoundary& apply_R_boundary,                                                          
                                                             grid::grid_function_1d<PetscScalar> dst, 
                                                       const grid::grid_function_1d<PetscScalar> src,
                                                       const PetscInt N,
                                                       const SbpDerivative& D1,
                                                       const PetscScalar hi,
                                                       Args... args)
{
  PetscInt i;
  const auto n_closures = D1.closure_size();
  apply_L_boundary(dst,src,N,D1,hi,args...);
  for (i = 1; i < n_closures; i++) 
  { 
    apply_L_interior(dst,src,N,D1,hi,i,args...);
  }
  for (i = n_closures; i < N-n_closures; i++)
  {
    apply_C_interior(dst,src,N,D1,hi,i,args...);
  }
  for (i = N-n_closures; i < N-1; i++)
  {
    apply_R_interior(dst,src,N,D1,hi,i,args...);
  }
  apply_R_boundary(dst,src,N,D1,hi,args...);
  return 0;
};

//=============================================================================
// 2D functions
//=============================================================================

template <typename ApplyLLCorner,
          typename ApplyLLWest,
          typename ApplyLLSouth,
          typename ApplyLLInterior,
          typename... Args,
          class SbpDerivative>
inline PetscErrorCode partitioned_apply_LL(ApplyLLCorner apply_LL_corner,
                                           ApplyLLWest apply_LL_west,
                                           ApplyLLSouth apply_LL_south,
                                           ApplyLLInterior apply_LL_interior,
                                           grid::grid_function_2d<PetscScalar> dst,
                                           const grid::grid_function_2d<PetscScalar> src,
                                           const std::array<PetscInt,2>& N,
                                           const SbpDerivative& D1,
                                           const std::array<PetscScalar,2>& hi,
                                           Args args...)
{
    int i,j;
    const auto n_closures = D1.closure_size();
    // Apply corner point
    apply_LL_corner(dst,src,N,D1,hi,args...);

    // Apply points on west boundary (exluding corner)
    for (j = 1; j < n_closures; j++)
    { 
      apply_LL_west(dst,src,N,D1,hi,j,args...);
    }
    // Apply points on south boundary (exluding corner)
    for (i = 1; i < n_closures; i++) 
    { 
      apply_LL_south(dst,src,N,D1,hi,i,args...);
    }

    // Apply remaining interior closure points
    for (j = 1; j < n_closures; j++)
    { 
      for (i = 1; i < n_closures; i++) 
      { 
        apply_LL_south(dst,src,N,D1,hi,i,j,args...);
      }
    }
    return 0;
  }

