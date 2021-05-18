#pragma once
#include<array>
#include<petscsystypes.h>
#include "grids/grid_function.h"

//=============================================================================
// 1D functions
//=============================================================================
template <typename BcL,
          typename BcR,
          typename... Args>
void bc(const BcL& bc_l,
        const BcR& bc_r,
              grid::grid_function_1d<PetscScalar> dst,
        const grid::grid_function_1d<PetscScalar> src,
        const std::array<PetscInt,2>& ind_i,
              Args... args)
{
  const PetscInt nx = src.mapping().nx();
  if (ind_i[0] == 0) // left
    bc_l(dst,src,args...);
  if (ind_i[1] == nx) // right
    bc_r(dst,src,args...);
}

template <typename BcL,
          typename BcR,
          typename... Args>
void bc_serial(const BcL& bc_l,
               const BcR& bc_r,
                     grid::grid_function_1d<PetscScalar> dst,
               const grid::grid_function_1d<PetscScalar> src,
                     Args... args)
{
  bc_l(dst,src,args...);
  bc_r(dst,src,args...);
}

//=============================================================================
// 2D functions
//=============================================================================
template <typename BCWest,
          typename BCSouth,
          typename BCEast,
          typename BCNorth,
          typename... Args>
void bc(const BCWest& bc_w,
        const BCSouth& bc_s,
        const BCEast& bc_e,
        const BCNorth& bc_n,
              grid::grid_function_2d<PetscScalar> dst,
        const grid::grid_function_2d<PetscScalar> src,
        const std::array<PetscInt,2>& ind_i,
        const std::array<PetscInt,2>& ind_j,
              Args... args)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();
  if (ind_i[0] == 0) // West
    bc_w(dst,src,ind_j,args...);
  if (ind_i[1] == nx) // East
    bc_e(dst,src,ind_j,args...);
  if (ind_j[0] == 0) // South
    bc_s(dst,src,ind_i,args...); 
  if (ind_j[1] == ny) // North
    bc_n(dst,src,ind_i,args...);
}

template <typename BCWest,
          typename BCSouth,
          typename BCEast,
          typename BCNorth,
          typename... Args>
void bc_serial(const BCWest& bc_w,
               const BCSouth& bc_s,
               const BCEast& bc_e,
               const BCNorth& bc_n,
                     grid::grid_function_2d<PetscScalar> dst,
               const grid::grid_function_2d<PetscScalar> src,
                     Args... args)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();
  bc_w(dst,src,{0,ny},args...);
  bc_s(dst,src,{0,nx},args...);
  bc_e(dst,src,{0,ny},args...);
  bc_n(dst,src,{0,nx},args...);
}