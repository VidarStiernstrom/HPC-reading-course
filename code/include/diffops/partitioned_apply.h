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
                                   const PetscInt cl_sz,
                                   const PetscInt i_start,
                                   const PetscInt i_end,
                                   Args... args)
{
  PetscInt i;
  const PetscInt N = src.mapping().nx();
  const PetscInt sw = src.mapping().stencil_width();
  if (i_start == 0) { // Left region
    rhs_lb(dst,src,args...); // Boundary point
    for (i = 1; i < cl_sz; i++) { 
      rhs_lc(dst,src,i,args...); // Closure points
    } 
    for (i = cl_sz; i < i_end-sw; i++) { 
      rhs_int(dst,src,i,args...); // Local interior points
    } 
  }
  else if (i_end == N) { // Right region
    for (i = i_start+sw; i < N-cl_sz; i++) { 
      rhs_int(dst,src,i,args...); // Local interior points
    } 
    for (i = N-cl_sz; i < N-1; i++) { 
      rhs_rc(dst,src,i,args...); // Closure points
    }
    rhs_rb(dst,src,args...); // Boundary point
  }
  else { // Interior region
    for (i = i_start+sw; i < i_end-sw; i++){  
      rhs_int(dst,src,i,args...); // local interior points
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
                                     const PetscInt cl_sz,
                                     const PetscInt i_start,
                                     const PetscInt i_end,
                                     Args... args)
{
  PetscInt i;
  const PetscInt N = src.mapping().nx();
  const PetscInt sw = src.mapping().stencil_width();
  if (i_start == 0) { // Left region
    for (i = i_end-sw; i < i_end; i++) { 
      rhs_int(dst,src,i,args...); // Overlapping interior points
    } 
  }
  else if (i_end == N) { // Right region
    for (i = i_start; i < i_start+sw; i++) { 
      rhs_int(dst,src,i,args...); // Overlapping interior points
    } 
  }
  else { // Interior region
    for (i = i_start; i < i_start+sw; i++){  
      rhs_int(dst,src,i,args...); // Left overlapping interior points
    }
    for (i = i_end-sw; i < i_end; i++){  
      rhs_int(dst,src,i,args...); // Right overlapping interior points
    }
  }
  return 0;
};

// =============================================================================
// 2D functions
// =============================================================================

// template <typename RhsCorner,
//           typename RhsWestBoundary,
//           typename RhsSouthBoundary,
//           typename RhsClosure,
//           typename... Args>
// inline PetscErrorCode rhs_LL(const RhsCorner& rhs_corner,
//                              const RhsWestBoundary& rhs_wb,
//                              const RhsSouthBoundary& rhs_sb,
//                              const RhsClosure& rhs_c,
//                              grid::grid_function_2d<PetscScalar> dst,
//                              const grid::grid_function_2d<PetscScalar> src,
//                              const std::array<PetscInt,2>& N,
//                              const PetscInt cl_sz,
//                              Args args...)
// {
//   PetscInt i,j;
  
//   rhs_corner(dst,src,args...); // Corner point
//   for (i = 1; i < cl_sz; i++) 
//   { 
//     rhs_sb(dst,src,i,args...); // south boundary (exluding corner)
//   }
//   for (j = 1; j < cl_sz; j++)
//   { 
//     rhs_wb(dst,src,j,args...); // west boundary (exluding corner)
//   }
//   for (j = 1; j < cl_sz; j++) { 
//     for (i = 1; i < cl_sz; i++) { 
//       rhs_c(dst,src,i,j,args...); // remaining closure region
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
//                                            const PetscInt cl_sz,
//                                            Args args...)
// {
//   PetscInt i,j;
  
//   rhs_corner(dst,src,args...); // Corner point
//   for (i = 1; i < cl_sz; i++) 
//   { 
//     rhs_sb(dst,src,i,args...); // south boundary (exluding corner)
//   }
//   for (j = 1; j < cl_sz; j++)
//   { 
//     rhs_wb(dst,src,j,args...); // west boundary (exluding corner)
//   }
//   for (j = 1; j < cl_sz; j++) { 
//     for (i = 1; i < cl_sz; i++) { 
//       rhs_c(dst,src,i,j,args...); // remaining closure region
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
                             const PetscInt cl_sz,
                             const PetscInt i_start,
                             const PetscInt i_end,
                             Args... args)
{
  PetscInt i;
  const PetscInt N = src.mapping().nx();
  if (i_start == 0) { // Left region
    rhs_lb(dst,src,args...); // Boundary point
    for (i = 1; i < cl_sz; i++) { 
      rhs_lc(dst,src,i,args...); // Closure points
    }
    for (i = cl_sz; i < i_end; i++) {
      rhs_int(dst,src,i,args...); // Interior points
    }
  }
  else if (i_end == N) { // Right region
    for (i = i_start; i < N-cl_sz; i++) {
      rhs_int(dst,src,i,args...); // Interior points
    }
    for (i = N-cl_sz; i < N-1; i++) {
      rhs_rc(dst,src,i,args...); // Closure points
    }
    rhs_rb(dst,src,args...); // Boundary point
  }
  else { // Interior region
    for (i = i_start; i < i_end; i++) {
      rhs_int(dst,src,i,args...); // Interior points
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
                                         const PetscInt cl_sz,
                                         Args... args)
{
  PetscInt i;
  const PetscInt N = src.mapping().nx();
  rhs_lb(dst,src,args...); // Left boundary point
  for (i = 1; i < cl_sz; i++) { 
    rhs_lc(dst,src,i,args...); // Left closure points
  }
  for (i = cl_sz; i < N-cl_sz; i++) {
    rhs_int(dst,src,i,args...); // Interior points
  }
  for (i = N-cl_sz; i < N-1; i++) {
    rhs_rc(dst,src,i,args...); // Right closure points
  }
  rhs_rb(dst,src,args...); // Right boundary point
  return 0;
};

//=============================================================================
// 2D functions
//=============================================================================
