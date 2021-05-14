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
PetscErrorCode rhs_1D_local(const RhsLeftBoundary& rhs_lb,
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
PetscErrorCode rhs_1D_overlap(const RhsLeftBoundary& rhs_lb,
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
template <typename RhsLL,
          typename RhsLI,
          typename RhsLR,
          typename RhsIL,
          typename RhsII,
          typename RhsIR,
          typename RhsRL,
          typename RhsRI,
          typename RhsRR,
          typename... Args>
PetscErrorCode rhs_local(const RhsLL& rhs_LL,
                         const RhsLI& rhs_LI,
                         const RhsLR& rhs_LR,
                         const RhsIL& rhs_IL,
                         const RhsII& rhs_II,
                         const RhsIR& rhs_IR,
                         const RhsRL& rhs_RL,
                         const RhsRI& rhs_RI,
                         const RhsRR& rhs_RR,
                         grid::grid_function_2d<PetscScalar> dst,
                         const grid::grid_function_2d<PetscScalar> src,
                         const std::array<PetscInt,2> idx_start,
                         const std::array<PetscInt,2> idx_end,
                         const PetscInt halo_sz,
                         const PetscInt cl_sz,
                         Args... args)
{
  const PetscInt i_start = idx_start[0] + halo_sz; 
  const PetscInt j_start = idx_start[1] + halo_sz;
  const PetscInt i_end = idx_end[0] - halo_sz;
  const PetscInt j_end = idx_end[1] - halo_sz;
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();

  if (idx_start[1] == 0)  // BOTTOM
  {
    if (idx_start[0] == 0) // BOTTOM LEFT
    {
      rhs_LL(dst, src, cl_sz, args);
      rhs_IL(dst, src, {cl_sz, i_end}, cl_sz, args);
      rhs_LI(dst, src, {cl_sz, j_end}, cl_sz, args);
      rhs_II(dst, src, {cl_sz, i_end}, {cl_sz, j_end}, args); 
    } else if (idx_end[0] == nx) // BOTTOM RIGHT
    { 
      rhs_RL(dst, src, cl_sz, args);
      rhs_IL(dst, src, {i_start, nx-cl_sz}, cl_sz, args);
      rhs_RI(dst, src, {cl_sz, j_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, nx-cl_sz}, {cl_sz, j_end}, args); 
    } else // BOTTOM CENTER
    { 
      rhs_IL(dst, src, {i_start, i_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_end}, {cl_sz, j_end}, args); 
    }
  } else if (idx_end[1] == ny) // TOP
  {
    if (idx_start[0] == 0) // TOP LEFT
    {
      rhs_LR(dst, src, cl_sz, args);
      rhs_IR(dst, src, {cl_sz, i_end}, cl_sz, args);
      rhs_LI(dst, src, {j_start, ny - cl_sz}, cl_sz, args);
      rhs_II(dst, src, {cl_sz, i_end}, {j_start, ny-cl_sz}, args);  
    } else if (idx_end[0] == nx) // TOP RIGHT
    { 
      rhs_RR(dst, src, cl_sz, args);
      rhs_IR(dst, src, {i_start, nx-cl_sz} cl_sz, args);
      rhs_RI(dst, src, {j_start, ny - cl_sz}, cl_sz, args);
      rhs_II(dst, src, {i_start, nx-cl_sz}, {j_start, ny - cl_sz}, args);
    } else // TOP CENTER
    { 
      rhs_IR(dst, src, {i_start, i_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_end}, {j_start,  ny - cl_sz}, args);
    }
  } else if (idx_start[0] == 0) // LEFT NOT BOTTOM OR TOP
  { 
    rhs_LI(dst, src, {j_start, j_end}, cl_sz, args);
    rhs_II(dst, src, {cl_sz, i_end}, {j_start, j_end}, args);
  } else if (idx_end[0] == nx) // RIGHT NOT BOTTOM OR TOP
  {
    rhs_RI(dst, src, {j_start, j_end}, cl_sz, args);
    rhs_II(dst, src, {i_start, nx - cl_sz}, {j_start, j_end}, args);
  } else // CENTER
  {
    rhs_II(dst, src, {i_start, i_end}, {j_start, j_end}, args);
  }

  return 0;
}

template <typename RhsLL,
          typename RhsLI,
          typename RhsLR,
          typename RhsIL,
          typename RhsII,
          typename RhsIR,
          typename RhsRL,
          typename RhsRI,
          typename RhsRR,
          typename... Args>
PetscErrorCode rhs_local(const RhsLL& rhs_LL,
                         const RhsLI& rhs_LI,
                         const RhsLR& rhs_LR,
                         const RhsIL& rhs_IL,
                         const RhsII& rhs_II,
                         const RhsIR& rhs_IR,
                         const RhsRL& rhs_RL,
                         const RhsRI& rhs_RI,
                         const RhsRR& rhs_RR,
                         grid::grid_function_2d<PetscScalar> dst,
                         const grid::grid_function_2d<PetscScalar> src,
                         const std::array<PetscInt,2> idx_start,
                         const std::array<PetscInt,2> idx_end,
                         const PetscInt halo_sz,
                         const PetscInt cl_sz,
                         Args... args)
{
  const PetscInt i_start = idx_start[0] + halo_sz; 
  const PetscInt j_start = idx_start[1] + halo_sz;
  const PetscInt i_end = idx_end[0] - halo_sz;
  const PetscInt j_end = idx_end[1] - halo_sz;
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();

  if (idx_start[1] == 0)  // BOTTOM
  {
    if (idx_start[0] == 0) // BOTTOM LEFT
    {
      rhs_LL(dst, src, cl_sz, args);
      rhs_IL(dst, src, {cl_sz, i_end}, cl_sz, args);
      rhs_LI(dst, src, {cl_sz, j_end}, cl_sz, args);
      rhs_II(dst, src, {cl_sz, i_end}, {cl_sz, j_end}, args); 
    } else if (idx_end[0] == nx) // BOTTOM RIGHT
    { 
      rhs_RL(dst, src, cl_sz, args);
      rhs_IL(dst, src, {i_start, nx-cl_sz}, cl_sz, args);
      rhs_RI(dst, src, {cl_sz, j_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, nx-cl_sz}, {cl_sz, j_end}, args); 
    } else // BOTTOM CENTER
    { 
      rhs_IL(dst, src, {i_start, i_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_end}, {cl_sz, j_end}, args); 
    }
  } else if (idx_end[1] == ny) // TOP
  {
    if (idx_start[0] == 0) // TOP LEFT
    {
      rhs_LR(dst, src, cl_sz, args);
      rhs_IR(dst, src, {cl_sz, i_end}, cl_sz, args);
      rhs_LI(dst, src, {j_start, ny - cl_sz}, cl_sz, args);
      rhs_II(dst, src, {cl_sz, i_end}, {j_start, ny-cl_sz}, args);  
    } else if (idx_end[0] == nx) // TOP RIGHT
    { 
      rhs_RR(dst, src, cl_sz, args);
      rhs_IR(dst, src, {i_start, nx-cl_sz} cl_sz, args);
      rhs_RI(dst, src, {j_start, ny - cl_sz}, cl_sz, args);
      rhs_II(dst, src, {i_start, nx-cl_sz}, {j_start, ny - cl_sz}, args);
    } else // TOP CENTER
    { 
      rhs_IR(dst, src, {i_start, i_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_end}, {j_start,  ny - cl_sz}, args);
    }
  } else if (idx_start[0] == 0) // LEFT NOT BOTTOM OR TOP
  { 
    rhs_LI(dst, src, {j_start, j_end}, cl_sz, args);
    rhs_II(dst, src, {cl_sz, i_end}, {j_start, j_end}, args);
  } else if (idx_end[0] == nx) // RIGHT NOT BOTTOM OR TOP
  {
    rhs_RI(dst, src, {j_start, j_end}, cl_sz, args);
    rhs_II(dst, src, {i_start, nx - cl_sz}, {j_start, j_end}, args);
  } else // CENTER
  {
    rhs_II(dst, src, {i_start, i_end}, {j_start, j_end}, args);
  }

  return 0;
}

template <typename RhsLI,
          typename RhsIL,
          typename RhsII,
          typename RhsIR,
          typename RhsRI,
          typename... Args>
PetscErrorCode rhs_overlap(const RhsLI& rhs_LI,
                           const RhsIL& rhs_IL,
                           const RhsII& rhs_II,
                           const RhsIR& rhs_IR,
                           const RhsRI& rhs_RI,
                           grid::grid_function_2d<PetscScalar> dst,
                           const grid::grid_function_2d<PetscScalar> src,
                           const std::array<PetscInt,2> idx_start,
                           const std::array<PetscInt,2> idx_end,
                           const PetscInt halo_sz,
                           const PetscInt cl_sz,
                           Args... args)
{
  const PetscInt i_start = idx_start[0]; 
  const PetscInt j_start = idx_start[1];
  const PetscInt i_end = idx_end[0];
  const PetscInt j_end = idx_end[1];
  const PetscInt nx = src.mapping().nx();
  const PetscInt nt = src.mapping().ny();

  if (j_start == 0)  // BOTTOM
  {
    if (i_start == 0) // BOTTOM LEFT
    {
      rhs_IL(dst, src, {i_end-halo_sz, i_end}, cl_sz, args);
      rhs_LI(dst, src, {j_end-halo_sz, j_end}, cl_sz, args);
      rhs_II(dst,src, {cl_sz, i_end-halo_sz}, {j_end-halo_sz, j_end}, args); 
      rhs_II(dst,src, {i_end-halo_sz, i_end}, {cl_sz, j_end}, args); 
    } else if (i_end == nx) // BOTTOM RIGHT
    { 
      rhs_IL(dst, src, {i_start, i_start+halo_sz}, cl_sz, args);
      rhs_RI(dst, src, {j_end-halo_sz, j_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_start+halo_sz}, {cl_sz, j_end}, args); 
      rhs_II(dst, src, {i_start+halo_sz, nx-cl_sz}, {j_end-halo_sz, j_end}, args); 
    } else // BOTTOM CENTER
    { 
      rhs_IL(dst, src, {i_start, i_start+halo_sz}, cl_sz, args);
      rhs_IL(dst, src, {i_end-halo_sz, i_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_start+halo_sz}, {cl_sz, j_end-halo_sz}, args); 
      rhs_II(dst, src, {i_end-halo_sz, i_end}, cl_sz, j_end-halo_sz, args); 
      rhs_II(dst, src, {i_start, i_end}, {j_end-halo_sz, j_end}, args); 
    }
  } else if (j_end == ny) // TOP
  {
    if (i_start == 0) // TOP LEFT
    {
      rhs_IR(dst, src, {i_end-halo_sz, i_end}, cl_sz, args);
      rhs_LI(dst, src, {j_start, j_start+halo_sz}, cl_sz, args);
      rhs_II(dst, src, {i_end-halo_sz, i_end}, {j_start, ny-cl_sz}, args);  
      rhs_II(dst, src, {cl_sz, i_end-halo_sz}, {j_start, j_start+halo_sz}, args);  
    } else if (i_end == nx) // TOP RIGHT
    { 
      rhs_IR(dst, src, {i_start, i_start+halo_sz}, cl_sz, args);
      rhs_RI(dst, src, {j_start, j_start+halo_sz}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_start+halo_sz}, {j_start, ny - cl_sz}, args);
      rhs_II(dst, src, {i_start+halo_sz, nx-cl_sz}, {j_start, j_start+halo_sz}, args);
    } else // TOP CENTER
    { 
      rhs_IR(dst, src, {i_start, i_start+halo_sz}, cl_sz, args);
      rhs_IR(dst, src, {i_end-halo_sz, i_end}, cl_sz, args);
      rhs_II(dst, src, {i_start, i_start+halo_sz}, {j_start, ny - cl_sz}, args);
      rhs_II(dst, src, {i_end-halo_sz, i_end}, {j_start, ny - cl_sz}, args);
      rhs_II(dst, src, {i_start+halo_sz, i_end-halo_sz}, {j_start, j_start+halo_sz}, args);
    }
  } else if (i_start == 0) // LEFT NOT BOTTOM OR TOP
  { 
    rhs_LI(dst, src, {j_start, j_start+halo_sz}, cl_sz, args);
    rhs_LI(dst, src, {j_end-halo_sz, j_end}, cl_sz, args);
    rhs_II(dst, src, {cl_sz, i_end}, {j_start, j_start+halo_sz}, args);
    rhs_II(dst, src, {cl_sz, i_end}, {j_end-halo_sz, j_end}, args);
    rhs_II(dst, src, {i_end-halo_sz, i_end}, {j_start+halo_sz, j_end-halo_sz}, args);
  } else if (i_end == nx) // RIGHT NOT BOTTOM OR TOP
  {
    rhs_RI(dst, src, {j_start, j_start+halo_sz}, cl_sz, args);
    rhs_RI(dst, src, {j_end-halo_sz, j_end}, cl_sz, args);
    rhs_II(dst, src, {i_start, nx - cl_sz}, {j_start, j_start+halo_sz}, args);
    rhs_II(dst, src, {i_start, nx - cl_sz}, {j_end-halo_sz, j_end}, args);
    rhs_II(dst, src, {i_start, i_start+halo_sz}, {j_start+halo_sz, j_end-halo_sz}, args);
  } else // CENTER
  {
    rhs_II(dst, src, {i_start, i_start+halo_sz}, {j_start, j_end}, args);
    rhs_II(dst, src, {i_end-halo_sz, i_end}, {j_start, j_end}, args);
    rhs_II(dst, src, {i_start+halo_sz, i_end-halo_sz}, {j_end-halo_sz, j_end}, args);
    rhs_II(dst, src, {i_start+halo_sz, i_end-halo_sz}, {j_start, j_start+halo_sz}, args);
  }
  return 0;
}


template <typename RhsLL,
          typename RhsLI,
          typename RhsLR,
          typename RhsIL,
          typename RhsII,
          typename RhsIR,
          typename RhsRL,
          typename RhsRI,
          typename RhsRR,
          typename... Args>
PetscErrorCode rhs_single_core(const RhsLL& rhs_LL,
                               const RhsLI& rhs_LI,
                               const RhsLR& rhs_LR,
                               const RhsIL& rhs_IL,
                               const RhsII& rhs_II,
                               const RhsIR& rhs_IR,
                               const RhsRL& rhs_RL,
                               const RhsRI& rhs_RI,
                               const RhsRR& rhs_RR,
                               grid::grid_function_2d<PetscScalar> dst,
                               const grid::grid_function_2d<PetscScalar> src,
                               const PetscInt cl_sz,
                               Args... args)
{
  const PetscInt nx = src.mapping.nx();
  const PetscInt ny = src.mapping.ny();
  rhs_LL(dst, src, cl_sz, args);
  rhs_LI(dst, src, {cl_sz,ny-cl_sz}, cl_sz, args);
  rhs_LR(dst, src, cl_sz, args);
  rhs_IL(dst, src, {cl_sz,nx-cl_sz}, cl_sz, args);
  rhs_II(dst, src, {cl_sz,nx-cl_sz}, {cl_sz,ny-cl_sz}, args);
  rhs_IR(dst, src, {cl_sz,nx-cl_sz}, cl_sz, args);
  rhs_RL(dst, src, cl_sz, args);
  rhs_RI(dst, src, {cl_sz,ny-cl_sz}, cl_sz, args);
  rhs_RR(dst, src, cl_sz, args);
}

template <typename BCWest,
          typename BCSouth,
          typename BCEast,
          typename BCNorth,
          typename... Args>
PetscErrorCode bc(const BCWest& bc_w,
                  const BCSouth& bc_s,
                  const BCEast& bc_e,
                  const BCNorth& bc_n,
                  grid::grid_function_2d<PetscScalar> dst,
                  const grid::grid_function_2d<PetscScalar> src,
                  const std::array<PetscInt,2> idx_start,
                  const std::array<PetscInt,2> idx_end,
                  Args... args)
{
  const PetscInt i_start = idx_start[0]; 
  const PetscInt j_start = idx_start[1];
  const PetscInt i_end = idx_end[0];
  const PetscInt j_end = idx_end[1];
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();
  if (idx_start[0] == 0) // West
    bc_w(dst,src,{j_start, j_end},args...);
  if (idx_start[0] == nx) // East
    bc_e(dst,src,{j_start, j_end},args...);
  if (idx_start[1] == 0) // South
    bc_s(dst,src,{i_start, i_end},args...); 
  if (idx_start[1] == ny) // North
    bc_n(dst,src,{i_start, i_end},args...);
  return 0;
}

template <typename BCWest,
          typename BCSouth,
          typename BCEast,
          typename BCNorth,
          typename... Args>
PetscErrorCode bc_single_core(const BCWest& bc_w,
                              const BCSouth& bc_s,
                              const BCEast& bc_e,
                              const BCNorth& bc_n,
                              grid::grid_function_2d<PetscScalar> dst,
                              const grid::grid_function_2d<PetscScalar> src
                              Args... args)
{
  const PetscInt nx = src.mapping.nx();
  const PetscInt ny = src.mapping.ny();
  bc_w(dst,src,{0,ny},args...);
  bc_s(dst,src,{0,nx},args...);
  bc_e(dst,src,{0,ny},args...);
  bc_n(dst,src,{0,nx},args...);
}

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