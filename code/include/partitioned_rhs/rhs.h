#pragma once
#include<array>
#include<petscsystypes.h>
#include "grids/grid_function.h"

//=============================================================================
// 1D functions
//=============================================================================
/*
*   *************
*   * l * i * r *
*   *************
*/
template <typename RhsL, 
          typename RhsI, 
          typename RhsR,
          typename... Args>
void rhs_local(const RhsL& rhs_l,
               const RhsI& rhs_i,
               const RhsR& rhs_r,
                     grid::grid_function_1d<PetscScalar> dst, 
               const grid::grid_function_1d<PetscScalar> src,                                                          
               const std::array<PetscInt,2>& ind_i,
               const PetscInt cls_sz,
               const PetscInt halo_sz,
                     Args... args)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt i_start = ind_i[0]+halo_sz;
  const PetscInt i_end = ind_i[1]-halo_sz;
  if (ind_i[0] == 0) { // Left region
    rhs_l(dst,src,cls_sz,args...);
    rhs_i(dst,src,{cls_sz,i_end},args...);
  }
  else if (ind_i[1] == nx) { // Right region
    rhs_i(dst,src,{i_start,nx-cls_sz},args...);
    rhs_r(dst,src,cls_sz,args...);
  }
  else { // Interior region
    rhs_i(dst,src,{i_start,i_end},args...);
  }
};

template <typename RhsInterior, 
          typename... Args>
void rhs_overlap(const RhsInterior& rhs_i,
                        grid::grid_function_1d<PetscScalar> dst, 
                  const grid::grid_function_1d<PetscScalar> src,                                                          
                  const std::array<PetscInt,2>& ind_i,
                  const PetscInt halo_sz,
                        Args... args)
{
  const PetscInt nx = src.mapping().nx();
  if (ind_i[0]== 0) { // Left region
    rhs_i(dst,src,{ind_i[1]-halo_sz,ind_i[1]},args...);
  }
  else if (ind_i[1] == nx) { // Right region
    rhs_i(dst,src,{ind_i[0],ind_i[0]+halo_sz},args...);
  }
  else { // Interior region
    rhs_i(dst,src,{ind_i[0], ind_i[0]+halo_sz},args...);
    rhs_i(dst,src,{ind_i[1]-halo_sz, ind_i[1]},args...);
  }
};

template <typename RhsLeft, 
          typename RhsInterior, 
          typename RhsRight,
          typename... Args>
void rhs_serial(const RhsLeft& rhs_l,
                const RhsInterior& rhs_i,
                const RhsRight& rhs_r,                                                         
                      grid::grid_function_1d<PetscScalar> dst, 
                const grid::grid_function_1d<PetscScalar> src,
                const PetscInt cls_sz,
                Args... args)
{
  const PetscInt nx = src.mapping().nx(); 
  rhs_l(dst,src,cls_sz,args...);
  rhs_i(dst,src,{cls_sz,nx-cls_sz},args...);
  rhs_r(dst,src,cls_sz,args...);
};

// =============================================================================
// 2D functions
// =============================================================================
/**
 *   ****************
 *   * lr * ir * rr *
 *   ****************
 *   * li * ii * ri *
 *   ****************
 *   * ll * il * rl *
 *   ****************
 **/
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
void rhs_local(const RhsLL& rhs_ll,
               const RhsLI& rhs_li,
               const RhsLR& rhs_lr,
               const RhsIL& rhs_il,
               const RhsII& rhs_ii,
               const RhsIR& rhs_ir,
               const RhsRL& rhs_rl,
               const RhsRI& rhs_ri,
               const RhsRR& rhs_rr,
                     grid::grid_function_2d<PetscScalar> dst,
               const grid::grid_function_2d<PetscScalar> src,
               const std::array<PetscInt,2>& ind_i,
               const std::array<PetscInt,2>& ind_j,
               const PetscInt cls_sz,
               const PetscInt halo_sz,
               Args... args)
{
  const PetscInt i_start = ind_i[0] + halo_sz; 
  const PetscInt i_end = ind_i[1] - halo_sz;
  const PetscInt j_start = ind_j[0] + halo_sz;
  const PetscInt j_end = ind_j[1] - halo_sz;
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();

  if (ind_j[0] == 0)  // BOTTOM
  {
    if (ind_i[0] == 0) // BOTTOM LEFT
    {
      rhs_ll(dst, src, cls_sz, args...);
      rhs_il(dst, src, {cls_sz, i_end}, cls_sz, args...);
      rhs_li(dst, src, {cls_sz, j_end}, cls_sz, args...);
      rhs_ii(dst, src, {cls_sz, i_end}, {cls_sz, j_end}, args...); 
    } else if (ind_i[1] == nx) // BOTTOM RIGHT
    { 
      rhs_rl(dst, src, cls_sz, args...);
      rhs_il(dst, src, {i_start, nx-cls_sz}, cls_sz, args...);
      rhs_ri(dst, src, {cls_sz, j_end}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, nx-cls_sz}, {cls_sz, j_end}, args...); 
    } else // BOTTOM CENTER
    { 
      rhs_il(dst, src, {i_start, i_end}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, i_end}, {cls_sz, j_end}, args...); 
    }
  } else if (ind_j[1] == ny) // TOP
  {
    if (ind_i[0] == 0) // TOP LEFT
    {
      rhs_lr(dst, src, cls_sz, args...);
      rhs_ir(dst, src, {cls_sz, i_end}, cls_sz, args...);
      rhs_li(dst, src, {j_start, ny - cls_sz}, cls_sz, args...);
      rhs_ii(dst, src, {cls_sz, i_end}, {j_start, ny-cls_sz}, args...);  
    } else if (ind_i[1] == nx) // TOP RIGHT
    { 
      rhs_rr(dst, src, cls_sz, args...);
      rhs_ir(dst, src, {i_start, nx-cls_sz}, cls_sz, args...);
      rhs_ri(dst, src, {j_start, ny - cls_sz}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, nx-cls_sz}, {j_start, ny - cls_sz}, args...);
    } else // TOP CENTER
    { 
      rhs_ir(dst, src, {i_start, i_end}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, i_end}, {j_start,  ny - cls_sz}, args...);
    }
  } else if (ind_i[0] == 0) // LEFT NOT BOTTOM OR TOP
  { 
    rhs_li(dst, src, {j_start, j_end}, cls_sz, args...);
    rhs_ii(dst, src, {cls_sz, i_end}, {j_start, j_end}, args...);
  } else if (ind_i[1] == nx) // RIGHT NOT BOTTOM OR TOP
  {
    rhs_ri(dst, src, {j_start, j_end}, cls_sz, args...);
    rhs_ii(dst, src, {i_start, nx - cls_sz}, {j_start, j_end}, args...);
  } else // CENTER
  {
    rhs_ii(dst, src, {i_start, i_end}, {j_start, j_end}, args...);
  }
}

template <typename RhsLI,
          typename RhsIL,
          typename RhsII,
          typename RhsIR,
          typename RhsRI,
          typename... Args>
void rhs_overlap(const RhsLI& rhs_li,
                 const RhsIL& rhs_il,
                 const RhsII& rhs_ii,
                 const RhsIR& rhs_ir,
                 const RhsRI& rhs_ri,
                       grid::grid_function_2d<PetscScalar> dst,
                 const grid::grid_function_2d<PetscScalar> src,
                 const std::array<PetscInt,2>& ind_i,
                 const std::array<PetscInt,2>& ind_j,
                 const PetscInt cls_sz,
                 const PetscInt halo_sz,
                       Args... args)
{
  const PetscInt i_start = ind_i[0]; 
  const PetscInt i_end = ind_i[1];
  const PetscInt j_start = ind_j[0];
  const PetscInt j_end = ind_j[1];
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();

  if (j_start == 0)  // BOTTOM
  {
    if (i_start == 0) // BOTTOM LEFT
    {
      rhs_il(dst, src, {i_end-halo_sz, i_end}, cls_sz, args...);
      rhs_li(dst, src, {j_end-halo_sz, j_end}, cls_sz, args...);
      rhs_ii(dst,src, {cls_sz, i_end-halo_sz}, {j_end-halo_sz, j_end}, args...); 
      rhs_ii(dst,src, {i_end-halo_sz, i_end}, {cls_sz, j_end}, args...); 
    } else if (i_end == nx) // BOTTOM RIGHT
    { 
      rhs_il(dst, src, {i_start, i_start+halo_sz}, cls_sz, args...);
      rhs_ri(dst, src, {j_end-halo_sz, j_end}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, i_start+halo_sz}, {cls_sz, j_end}, args...); 
      rhs_ii(dst, src, {i_start+halo_sz, nx-cls_sz}, {j_end-halo_sz, j_end}, args...); 
    } else // BOTTOM CENTER
    { 
      rhs_il(dst, src, {i_start, i_start+halo_sz}, cls_sz, args...);
      rhs_il(dst, src, {i_end-halo_sz, i_end}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, i_start+halo_sz}, {cls_sz, j_end-halo_sz}, args...); 
      rhs_ii(dst, src, {i_end-halo_sz, i_end}, {cls_sz, j_end-halo_sz}, args...); 
      rhs_ii(dst, src, ind_i, {j_end-halo_sz, j_end}, args...); 
    }
  } else if (j_end == ny) // TOP
  {
    if (i_start == 0) // TOP LEFT
    {
      rhs_ir(dst, src, {i_end-halo_sz, i_end}, cls_sz, args...);
      rhs_li(dst, src, {j_start, j_start+halo_sz}, cls_sz, args...);
      rhs_ii(dst, src, {i_end-halo_sz, i_end}, {j_start, ny-cls_sz}, args...);  
      rhs_ii(dst, src, {cls_sz, i_end-halo_sz}, {j_start, j_start+halo_sz}, args...);  
    } else if (i_end == nx) // TOP RIGHT
    { 
      rhs_ir(dst, src, {i_start, i_start+halo_sz}, cls_sz, args...);
      rhs_ri(dst, src, {j_start, j_start+halo_sz}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, i_start+halo_sz}, {j_start, ny - cls_sz}, args...);
      rhs_ii(dst, src, {i_start+halo_sz, nx-cls_sz}, {j_start, j_start+halo_sz}, args...);
    } else // TOP CENTER
    { 
      rhs_ir(dst, src, {i_start, i_start+halo_sz}, cls_sz, args...);
      rhs_ir(dst, src, {i_end-halo_sz, i_end}, cls_sz, args...);
      rhs_ii(dst, src, {i_start, i_start+halo_sz}, {j_start, ny - cls_sz}, args...);
      rhs_ii(dst, src, {i_end-halo_sz, i_end}, {j_start, ny - cls_sz}, args...);
      rhs_ii(dst, src, {i_start+halo_sz, i_end-halo_sz}, {j_start, j_start+halo_sz}, args...);
    }
  } else if (i_start == 0) // LEFT NOT BOTTOM OR TOP
  { 
    rhs_li(dst, src, {j_start, j_start+halo_sz}, cls_sz, args...);
    rhs_li(dst, src, {j_end-halo_sz, j_end}, cls_sz, args...);
    rhs_ii(dst, src, {cls_sz, i_end}, {j_start, j_start+halo_sz}, args...);
    rhs_ii(dst, src, {cls_sz, i_end}, {j_end-halo_sz, j_end}, args...);
    rhs_ii(dst, src, {i_end-halo_sz, i_end}, {j_start+halo_sz, j_end-halo_sz}, args...);
  } else if (i_end == nx) // RIGHT NOT BOTTOM OR TOP
  {
    rhs_ri(dst, src, {j_start, j_start+halo_sz}, cls_sz, args...);
    rhs_ri(dst, src, {j_end-halo_sz, j_end}, cls_sz, args...);
    rhs_ii(dst, src, {i_start, nx - cls_sz}, {j_start, j_start+halo_sz}, args...);
    rhs_ii(dst, src, {i_start, nx - cls_sz}, {j_end-halo_sz, j_end}, args...);
    rhs_ii(dst, src, {i_start, i_start+halo_sz}, {j_start+halo_sz, j_end-halo_sz}, args...);
  } else // CENTER
  {
    rhs_ii(dst, src, {i_start, i_start+halo_sz}, ind_j, args...);
    rhs_ii(dst, src, {i_end-halo_sz, i_end}, ind_j, args...);
    rhs_ii(dst, src, {i_start+halo_sz, i_end-halo_sz}, {j_end-halo_sz, j_end}, args...);
    rhs_ii(dst, src, {i_start+halo_sz, i_end-halo_sz}, {j_start, j_start+halo_sz}, args...);
  }
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
void rhs_serial(const RhsLL& rhs_ll,
                const RhsLI& rhs_li,
                const RhsLR& rhs_lr,
                const RhsIL& rhs_il,
                const RhsII& rhs_ii,
                const RhsIR& rhs_ir,
                const RhsRL& rhs_rl,
                const RhsRI& rhs_ri,
                const RhsRR& rhs_rr,
                      grid::grid_function_2d<PetscScalar> dst,
                const grid::grid_function_2d<PetscScalar> src,
                const PetscInt cls_sz,
                      Args... args)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();
  rhs_ll(dst, src, cls_sz, args...);
  rhs_li(dst, src, {cls_sz,ny-cls_sz}, cls_sz, args...);
  rhs_lr(dst, src, cls_sz, args...);
  rhs_il(dst, src, {cls_sz,nx-cls_sz}, cls_sz, args...);
  rhs_ii(dst, src, {cls_sz,nx-cls_sz}, {cls_sz,ny-cls_sz}, args...);
  rhs_ir(dst, src, {cls_sz,nx-cls_sz}, cls_sz, args...);
  rhs_rl(dst, src, cls_sz, args...);
  rhs_ri(dst, src, {cls_sz,ny-cls_sz}, cls_sz, args...);
  rhs_rr(dst, src, cls_sz, args...);
}

// =============================================================================
// TODO: 3D functions
// ============================================================================= 