#pragma once
#include<petscsystypes.h>
#include "grids/grid_function.h"

//=============================================================================
// 1D functions
//=============================================================================

template <typename RhsLeftBoundary, 
          typename RhsLeftClosure, 
          typename RhsInterior, 
          typename RhsRightClosure,
          typename RhsRightBoundary,
          typename... Args>
inline PetscErrorCode rhs_1D_local(const RhsLeftBoundary& rhs_lb,
                                     const RhsLeftClosure& rhs_lc,
                                     const RhsInterior& rhs_int,
                                     const RhsRightClosure& rhs_rc,
                                     const RhsRightBoundary& rhs_rb,                                                          
                                           grid::grid_function_1d<PetscScalar> dst, 
                                     const grid::grid_function_1d<PetscScalar> src,
                                     const PetscInt N,                                                           
                                     const PetscInt cw,
                                     const PetscInt sw,
                                     const PetscInt i_start,
                                     const PetscInt i_end,
                                     Args... args)
{
  PetscInt i;
  if (i_start == 0) { // Left region
    rhs_lb(dst,src,N,args...); // Boundary point
    for (i = 1; i < cw; i++) { 
      rhs_lc(dst,src,N,i,args...); // Closure points
    } 
    for (i = cw; i < i_end-sw; i++) { 
      rhs_int(dst,src,N,i,args...); // Local interior points
    } 
  }
  else if (i_end == N) { // Right region
    for (i = i_start+sw; i < N-cw; i++) { 
      rhs_int(dst,src,N,i,args...); // Local interior points
    } 
    for (i = N-cw; i < N-1; i++) { 
      rhs_rc(dst,src,N,i,args...); // Closure points
    }
    rhs_rb(dst,src,N,args...); // Boundary point
  }
  else { // Interior region
    for (i = i_start+sw; i < i_end-sw; i++){  
      rhs_int(dst,src,N,i,args...); // local interior points
    }
  }
  return 0;
};

template <typename RhsLeftBoundary, 
          typename RhsLeftClosure, 
          typename RhsInterior, 
          typename RhsRightClosure,
          typename RhsRightBoundary,
          typename... Args>
inline PetscErrorCode rhs_1D_overlap(const RhsLeftBoundary& rhs_lb,
                                     const RhsLeftClosure& rhs_lc,
                                     const RhsInterior& rhs_int,
                                     const RhsRightClosure& rhs_rc,
                                     const RhsRightBoundary& rhs_rb,                                                          
                                           grid::grid_function_1d<PetscScalar> dst, 
                                     const grid::grid_function_1d<PetscScalar> src,
                                     const PetscInt N,                                                           
                                     const PetscInt cw,
                                     const PetscInt sw,
                                     const PetscInt i_start,
                                     const PetscInt i_end,
                                     Args... args)
{
  PetscInt i;
  if (i_start == 0) { // Left region
    for (i = i_end-sw; i < i_end; i++) { 
      rhs_int(dst,src,N,i,args...); // Overlapping interior points
    } 
  }
  else if (i_end == N) { // Right region
    for (i = i_start; i < i_start+sw; i++) { 
      rhs_int(dst,src,N,i,args...); // Overlapping interior points
    } 
  }
  else { // Interior region
    for (i = i_start; i < i_start+sw; i++){  
      rhs_int(dst,src,N,i,args...); // Left overlapping interior points
    }
    for (i = i_end-sw; i < i_end; i++){  
      rhs_int(dst,src,N,i,args...); // Right overlapping interior points
    }
  }
  return 0;
};

//=============================================================================
// 2D functions
//=============================================================================

// template <typename RhsCorner,
//           typename RhsWestBoundary,
//           typename RhsSouthBoundary,
//           typename RhsClosure,
//           typename... Args>
// inline PetscErrorCode rhs_LL(RhsCorner rhs_corner,
//                              RhsWestBoundary rhs_wb,
//                              RhsSouthBoundary rhs_sb,
//                              RhsClosure rhs_c,
//                              grid::grid_function_2d<PetscScalar> dst,
//                              const grid::grid_function_2d<PetscScalar> src,
//                              const std::array<PetscInt,2>& N,
//                              const PetscInt cw,
//                              Args args...)
// {
//   PetscInt i,j;
  
//   rhs_corner(dst,src,N,args...); // Corner point
//   for (i = 1; i < cw; i++) 
//   { 
//     rhs_sb(dst,src,N,i,args...); // south boundary (exluding corner)
//   }
//   for (j = 1; j < cw; j++)
//   { 
//     rhs_wb(dst,src,N,j,args...); // west boundary (exluding corner)
//   }
//   for (j = 1; j < cw; j++) { 
//     for (i = 1; i < cw; i++) { 
//       rhs_c(dst,src,N,i,j,args...); // remaining closure region
//     }
//   }
//   return 0;
// }

// template <typename RhsCorner,
//           typename RhsWestBoundary,
//           typename RhsSouthBoundary,
//           typename RhsClosure,
//           typename... Args>
// inline PetscErrorCode rhs_LI(RhsLLCorner rhs_corner,
//                                            RhsCorner rhs_wb,
//                                            RhsWestBoundary rhs_sb,
//                                            RhsClosure rhs_c,
//                                            grid::grid_function_2d<PetscScalar> dst,
//                                            const grid::grid_function_2d<PetscScalar> src,
//                                            const std::array<PetscInt,2>& N,
//                                            const PetscInt cw,
//                                            Args args...)
// {
//   PetscInt i,j;
  
//   rhs_corner(dst,src,N,args...); // Corner point
//   for (i = 1; i < cw; i++) 
//   { 
//     rhs_sb(dst,src,N,i,args...); // south boundary (exluding corner)
//   }
//   for (j = 1; j < cw; j++)
//   { 
//     rhs_wb(dst,src,N,j,args...); // west boundary (exluding corner)
//   }
//   for (j = 1; j < cw; j++) { 
//     for (i = 1; i < cw; i++) { 
//       rhs_c(dst,src,N,i,j,args...); // remaining closure region
//     }
//   }
//   return 0;
// }


//=============================================================================
// Temporary functions
//=============================================================================


//=============================================================================
// 1D functions
//=============================================================================

template <typename RhsLeftBoundary, 
          typename RhsLeftClosure, 
          typename RhsInterior, 
          typename RhsRightClosure,
          typename RhsRightBoundary,
          typename... Args>
inline PetscErrorCode rhs_1D(const RhsLeftBoundary& rhs_lb,
                             const RhsLeftClosure& rhs_lc,
                             const RhsInterior& rhs_int,
                             const RhsRightClosure& rhs_rc,
                             const RhsRightBoundary& rhs_rb,                                                          
                                   grid::grid_function_1d<PetscScalar> dst, 
                             const grid::grid_function_1d<PetscScalar> src,
                             const PetscInt N,                                                           
                             const PetscInt cw,
                             const PetscInt i_start,
                             const PetscInt i_end,
                             Args... args)
{
  PetscInt i;
  if (i_start == 0) { // Left region
    rhs_lb(dst,src,N,args...); // Boundary point
    for (i = 1; i < cw; i++) { 
      rhs_lc(dst,src,N,i,args...); // Closure points
    }
    for (i = cw; i < i_end; i++) {
      rhs_int(dst,src,N,i,args...); // Interior points
    }
  }
  else if (i_end == N) { // Right region
    for (i = i_start; i < N-cw; i++) {
      rhs_int(dst,src,N,i,args...); // Interior points
    }
    for (i = N-cw; i < N-1; i++) {
      rhs_rc(dst,src,N,i,args...); // Closure points
    }
    rhs_rb(dst,src,N,args...); // Boundary point
  }
  else { // Interior region
    for (i = i_start; i < i_end; i++) {
      rhs_int(dst,src,N,i,args...); // Interior points
    }
  }
  return 0;
};


template <typename RhsLeftBoundary, 
          typename RhsLeftClosure, 
          typename RhsInterior, 
          typename RhsRightClosure,
          typename RhsRightBoundary, 
          typename... Args>
inline PetscErrorCode rhs_1D_single_core(const RhsLeftBoundary& rhs_lb,
                                                       const RhsLeftClosure& rhs_lc,
                                                       const RhsInterior& rhs_int,
                                                       const RhsRightClosure& rhs_rc,
                                                       const RhsRightBoundary& rhs_rb,                                                          
                                                             grid::grid_function_1d<PetscScalar> dst, 
                                                       const grid::grid_function_1d<PetscScalar> src,
                                                       const PetscInt N,
                                                       const PetscInt cw,
                                                       Args... args)
{
  PetscInt i;
  rhs_lb(dst,src,N,args...); // Left boundary point
  for (i = 1; i < cw; i++) { 
    rhs_lc(dst,src,N,i,args...); // Left closure points
  }
  for (i = cw; i < N-cw; i++) {
    rhs_int(dst,src,N,i,args...); // Interior points
  }
  for (i = N-cw; i < N-1; i++) {
    rhs_rc(dst,src,N,i,args...); // Right closure points
  }
  rhs_rb(dst,src,N,args...); // Right boundary point
  return 0;
};

//=============================================================================
// 2D functions
//=============================================================================
