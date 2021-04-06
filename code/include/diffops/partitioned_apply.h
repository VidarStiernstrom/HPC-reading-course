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

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * LL *    *    *
  *   ****************
**/
template <typename RhsWestBC,
          typename RhsSouthBC,
          typename RhsLClosureX,
          typename RhsLClosureY,
          typename... Args>
inline PetscErrorCode rhs_LL(const RhsWestBC& rhs_wbc,
                             const RhsSouthBC& rhs_sbc,
                             const RhsClosure& rhs_cl,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  // Corner point
  rhs_wbc(dst,src,0,args...);
  rhs_sbc(dst,src,0,args...);
  rhs_cl(dst,src,0,0,args...);
  // south boundary (exluding corner)
  for (i = 1; i < cl_sz; i++) { 
    rhs_cl(dst,src,i,0,args...);
    rhs_sbc(dst,src,i,args...);
  }
  // west boundary (exluding corner)
  for (j = 1; j < cl_sz; j++) { 
    rhs_cl(dst,src,0,j,args...); 
    rhs_wbc(dst,src,j,args...);
  }
  // remaining closure region
  for (j = 1; j < cl_sz; j++) { 
    for (i = 1; i < cl_sz; i++) { 
      rhs_cl(dst,src,i,j,args...); 
    }
  }
  return 0;
}

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * LI *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
**/
template <typename RhsWestBC,
          typename RhsLClosureX,
          typename RhsInteriorY,
          typename... Args>
inline PetscErrorCode rhs_LI(const RhsWestBC& rhs_wbc,
                             const RhsClosure& rhs_cl,
                             const RhsInterior& rhs_int,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const std::array<PetscInt,2>& ind_j,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  // west boundary
  for (j = ind_j[0]; j < ind_j[1]; j++) {
    rhs_wbc(dst,src,j,args...);
    rhs_cl(dst,src,0,j,args...); 
    rhs_int(dst,src,0,j,args...)
  }
  // remaining points
  for (j = ind_j[0]; j < ind_j[1]; j++) { 
    for (i = 1; i < cl_sz; i++) { 
        rhs_cl(dst,src,i,j,args...); 
        rhs_int(dst,src,i,j,args...)
    }
  }
  return 0;
}

/**
  *   ****************
  *   * LR *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
**/
template <typename RhsWestBC,
          typename RhsNorthBC,
          typename RhsLClosureX,
          typename RhsRClosureY,
          typename... Args>
inline PetscErrorCode rhs_LR(const RhsWestBC& rhs_wbc,
                             const RhsNorthBC& rhs_nbc,
                             const RhsClosure& rhs_cl,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  const PetscInt j_end = src.mapping().ny-1;
  // Corner point
  rhs_wbc(dst,src,j_end,args...);
  rhs_nbc(dst,src,0,args...);
  rhs_cl(dst,src,0,j_end,args...);
  // north boundary (exluding corner)
  for (i = 1; i < cl_sz; i++) { 
    rhs_cl(dst,src,i,j_end,args...);
    rhs_nbc(dst,src,i,args...);
  }
  // west boundary (exluding corner)
  for (j = j_end-cl_sz+1; j < j_end; j++) { 
    rhs_cl(dst,src,0,j,args...); 
    rhs_wbc(dst,src,j,args...);
  }
  // remaining closure region
  for (j = j_end-cl_sz+1; j < j_end; j++) { 
    for (i = 1; i < cl_sz; i++) { 
      rhs_cl(dst,src,i,j,args...); 
    }
  }
  return 0;
}


/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * IL *    *
  *   ****************
**/
template <typename RhsInteriorX,
          typename RhsSouthBC,
          typename RhsLClosureY,
          typename... Args>
inline PetscErrorCode rhs_IL(const RhsInterior& rhs_int,
                             const RhsSouthBC& rhs_sbc,
                             const RhsClosure& rhs_cl,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const std::array<PetscInt,2>& ind_i,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  // south boundary
  for (i = ind_i[0]; i < ind_i[1]; i++) {
    rhs_sbc(dst,src,i,args...);
    rhs_cl(dst,src,i,0,args...); 
    rhs_int(dst,src,i,0,args...)
  }
  // remaining points
  for (j = 1; j < cl_sz; j++) { 
    for (i = ind_i[0]; i < ind_i[1]; i++) { 
        rhs_cl(dst,src,i,j,args...); 
        rhs_int(dst,src,i,j,args...)
    }
  }
  return 0;
}



/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * II *    *
  *   ****************
  *   *    *    *    *
  *   ****************
**/
template <typename RhsInteriorX,
          typename RhsInteriorY,
          typename... Args>
inline PetscErrorCode rhs_II(const RhsInterior& rhs_int,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const std::array<PetscInt,2>& ind_i,
                             const std::array<PetscInt,2>& ind_j,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  // Interior points points
  for (j = ind_j[0]; j < ind_j[1]; j++) { 
    for (i = ind_i[0]; i < ind_i[1]; i++) { 
        rhs_int(dst,src,i,j,args...)
    }
  }
  return 0;
}


/**
  *   ****************
  *   *    * IR *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
**/
template <typename RhsInteriorX,
          typename RhsNorthBC,
          typename RhsRClosureY,
          typename... Args>
inline PetscErrorCode rhs_IR(const RhsInterior& rhs_int,
                             const RhsNorthBC& rhs_nbc,
                             const RhsClosure& rhs_cl,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const std::array<PetscInt,2>& ind_i,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  const PetscInt j_end = src.mapping().ny-1;
  // north boundary
  for (i = ind_i[0]; i < ind_i[1]; i++) {
    rhs_int(dst,src,i,j_end,args...); 
    rhs_cl(dst,src,i,j_end,args...);
    rhs_nbc(dst,src,i,args...);
  }
  // remaining points
  for (j = j_end-cl_sz+1; j < j_end; j++) { 
    for (i = ind_i[0]; i < ind_i[1]; i++) { 
      rhs_int(dst,src,i,j,args...);
      rhs_cl(dst,src,i,j,args...); 
    }
  }
  return 0;
}


/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * RL *
  *   ****************
**/
template <typename RhsEastBC,
          typename RhsSouthBC,
          typename RhsRClosureX,
          typename RhsLClosureY,
          typename... Args>
inline PetscErrorCode rhs_RL(const RhsEastBC& rhs_ebc,
                             const RhsSouthBC& rhs_sbc,
                             const RhsClosure& rhs_cl,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  const PetscInt i_end = src.mapping().nx()-1;

  // Corner point
  rhs_ebc(dst,src,0,args...);
  rhs_sbc(dst,src,i_end,args...);
  rhs_cl(dst,src,i_end,0,args...);

  // east boundary (exluding corner)
  for (j = 1; j < cl_sz; j++) { 
    rhs_cl(dst,src,i_end,j,args...); 
    rhs_ebc(dst,src,j,args...);
  }

  // east boundary (exluding corner)
  for (j = 1; j < cl_sz; j++) { 
    rhs_cl(dst,src,i_end,j,args...); 
    rhs_ebc(dst,src,j,args...);
  }

  // remaining closure region
  for (j = 1; j < cl_sz; j++) { 
    for (i = i_end-cl_sz+1; i < i_end; i++) { 
      rhs_cl(dst,src,i,j,args...); 
    }
  }

  return 0;
}


/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * RI *
  *   ****************
  *   *    *    *    *
  *   ****************
**/
template <typename RhsEastBC,
          typename RhsRClosureX,
          typename RhsInteriorY,
          typename... Args>
inline PetscErrorCode rhs_RI(const RhsEastBC& rhs_ebc,
                             const RhsRClosureX& rhs_cl,
                             const RhsInteriorY& rhs_int,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const std::array<PetscInt,2>& ind_j,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  const PetscInt i_end = src.mapping().nx()-1;

  // east boundary
  for (j = ind_j[0]; j < ind_j[1]; j++) { 
    rhs_ebc(dst,src,j,args...);
    rhs_cl(dst,src,i_end,j,args...); 
    rhs_int(dst,src,i_end,j,args...);
  }
  // remaining points
  for (j = ind_j[0]; j < ind_j[0]; j++) { 
    for (i = i_end-cl_sz+1; i < i_end; i++) { 
      rhs_int(dst,src,i,j,args...);
      rhs_cl(dst,src,i,j,args...); 
    }
  }
  return 0;
}


/**
  *   ****************
  *   *    *    * RR *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
**/
template <typename RhsEastBC,
          typename RhsNorthBC,
          typename RhsRClosureX,
          typename RhsRClosureY,
          typename... Args>
inline PetscErrorCode rhs_RR(const RhsEastBC& rhs_ebc,
                             const RhsNorthBC& rhs_nbc,
                             const RhsClosure& rhs_cl,
                             grid::grid_function_2d<PetscScalar> dst,
                             const grid::grid_function_2d<PetscScalar> src,
                             const PetscInt cl_sz,
                             Args args...)
{
  PetscInt i,j;
  const PetscInt i_end = src.mapping().nx()-1;
  const PetscInt j_end = src.mapping().ny()-1;

  // Corner point
  rhs_ebc(dst,src,j_end,args...);
  rhs_nbc(dst,src,i_end,args...);
  rhs_cl(dst,src,i_end,j_end,args...);
  // north boundary (exluding corner)
  for (i = i_end-cl_sz+1; i < i_end; i++) { 
    rhs_cl(dst,src,i,j_end,args...);
    rhs_nbc(dst,src,i,args...);
  }
  // east boundary (exluding corner)
  for (j = j_end-cl_sz+1; j < j_end; j++) { 
    rhs_cl(dst,src,i_end,j,args...); 
    rhs_ebc(dst,src,j,args...);
  }
  // remaining closure region
  for (j = j_end-cl_sz+1; j < j_end; j++) { 
    for (i = i_end-cl_sz+1; i < i_end; i++) { 
      rhs_cl(dst,src,i,j,args...); 
    }
  }
  return 0;
}

template <typename RhsWestBC,
          typename RhsEastBC,
          typename RhsSouthBC,
          typename RhsNorthBC,
          typename RhsLClosureX,
          typename RhsLClosureY,
          typename RhsRClosureX,
          typename RhsRClosureY,
          typename RhsInteriorX,
          typename RhsInteriorY,
          typename... Args>
inline PetscErrorCode rhs_2D_local(const RhsWestBC&,
                                   const RhsEastBC&,
                                   const RhsSouthBC&,
                                   const RhsNorthBC&,
                                   const RhsLClosureX&,
                                   const RhsLClosureY&,
                                   const RhsRClosureX&,
                                   const RhsRClosureY&,
                                   const RhsInteriorX&,
                                   const RhsInteriorY&,
                                   ...
                                   Args... args)
{
}

template <typename RhsWestBC,
          typename RhsEastBC,
          typename RhsSouthBC,
          typename RhsNorthBC,
          typename RhsLClosureX,
          typename RhsLClosureY,
          typename RhsRClosureX,
          typename RhsRClosureY,
          typename RhsInteriorX,
          typename RhsInteriorY,
          typename... Args>
inline PetscErrorCode rhs_2D_overlap(const RhsWestBC&,
                                     const RhsEastBC&,
                                     const RhsSouthBC&,
                                     const RhsNorthBC&,
                                     const RhsLClosureX&,
                                     const RhsLClosureY&,
                                     const RhsRClosureX&,
                                     const RhsRClosureY&,
                                     const RhsInteriorX&,
                                     const RhsInteriorY&,
                                     ...
                                     Args... args)
{
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