#pragma once
#include<petscsystypes.h>
#include "grids/grid_function.h"

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


