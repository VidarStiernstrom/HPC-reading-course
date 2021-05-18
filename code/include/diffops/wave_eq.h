#pragma once

#include<petscsystypes.h>
#include <array>
#include "diffops/partitioned_apply.h"
#include "grids/grid_function.h"

using namespace sbp;
/**
* Functions for computing the righ-hand-side of the acoustic wave equation
* 
* F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
* where q = [u,v,p]^T, and 
* F1 = -1/rho(x,y) p_x + forcing_u
* F2 = -1/rho(x,y) p_y + forcing_v
* F3 = -(u_x + q_y)
*
* Subscripts _x, and _y denote partial derivates. These are approximated by the summation-by-parts (SBP) difference operator.
* The SBP difference operators have specialized stencils in the boundary regions (for each coordinate direction)
* For this reason, the domain is separated into 9 parts. See below.
*
*   L - left
*   I - interior
*   R - right
*   ****************
*   * LR * IR * RR *
*   ****************
*   * LI * II * RI *
*   ****************
*   * LL * IL * RL *
*   ****************
*   
* The RHS function F(t,q) are separated into the above regions, using the specialized stencils of the difference operator.
* Furthermore, along the boundary points, additional SBP operators are used to impose free surface boundary conditions,
* i.e, zero pressure conditions.
* 
* NOTE: The division of the grid into the above regions is not related to processor topology. Typically multiple processes will 
* act on the interior regions.
**/
  
/**
* Inverse of density rho(x,y) at grid point i,j
**/
PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0]; // multiplicera med h istället för division med hi
  PetscScalar y = xl[1] + j/hi[1];
  return 1./(2 + x*y);
};

/**
* Forcing function on u component
**/
PetscScalar forcing_u(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return -(3*PETSC_PI*cos(5*PETSC_PI*t)*cos(3*PETSC_PI*x)*sin(4*PETSC_PI*y)*(x*y + 1))/(x*y + 2);
};

/**
* Forcing function on v component
**/
PetscScalar forcing_v(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return -(4*PETSC_PI*cos(5*PETSC_PI*t)*cos(4*PETSC_PI*y)*sin(3*PETSC_PI*x)*(x*y + 1))/(x*y + 2);
};

 /**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * LL *    *    *
  *   ****************
  **/
template <class SbpDerivative>
void wave_eq_rhs_LL(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  for (PetscInt j = 0; j < cl_sz; j++) { 
    for (PetscInt i = 0; i < cl_sz; i++) { 
        F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
        F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
        F(j, i, 2) = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }
}


 /**
  * 
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * IL *    *
  *   ****************
  *
  **/
template <class SbpDerivative>
void wave_eq_rhs_IL(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const std::array<PetscInt,2> ind_i,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  for (PetscInt j = 0; j < cl_sz; j++) { 
    for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) {  
        F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
        F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
        F(j, i, 2) = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }
}

/**
  * 
  * 
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * RL *
  *   ****************
  *
  **/
template <class SbpDerivative>
void wave_eq_rhs_RL(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  const PetscInt nx = q.mapping().nx();
  for (PetscInt j = 0; j < cl_sz; j++) { 
    for (PetscInt i = nx-cl_sz; i < nx; i++) { 
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_right(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }
}

/**
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * LI *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  **/
template <class SbpDerivative>
void wave_eq_rhs_LI(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const std::array<PetscInt,2> ind_j,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
 
 for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    for (PetscInt i = 0; i < cl_sz; i++) { 
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
}

/**
  * 
  * 
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * II *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  **/
template <class SbpDerivative>
void wave_eq_rhs_II(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const std::array<PetscInt,2> ind_i,
                    const std::array<PetscInt,2> ind_j,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{

  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
}

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * RI *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  **/
template <class SbpDerivative>
void wave_eq_rhs_RI(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const std::array<PetscInt,2> ind_j,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  const PetscInt nx = q.mapping().nx();
  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    for (PetscInt i = nx-cl_sz; i < nx; i++) {
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_right(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
}


/**
  * 
  *   ****************
  *   * LR *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  **/
template <class SbpDerivative>
void wave_eq_rhs_LR(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  const PetscInt ny = q.mapping().ny();  
  for (PetscInt j = ny-cl_sz; j < ny; j++) { 
    for (PetscInt i = 0; i < cl_sz; i++) { 
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], i, j, 1);
    }
  }
}

/**
  *   ****************
  *   *    * IR  *   *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative>
void wave_eq_rhs_IR(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const std::array<PetscInt,2> ind_i,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  const PetscInt ny = q.mapping().ny();
  for (PetscInt j = ny-cl_sz; j < ny; j++) { 
    for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], i, j, 1);
    }
  }
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
template <class SbpDerivative>
void wave_eq_rhs_RR(grid::grid_function_2d<PetscScalar> F,
                    const grid::grid_function_2d<PetscScalar> q,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    const std::array<PetscScalar,2>& xl,
                    const PetscScalar t)
{
  const PetscInt nx = q.mapping().nx();
  const PetscInt ny = q.mapping().ny();
  for (PetscInt j = ny-cl_sz; j < ny; j++) { 
    for (PetscInt i = nx-cl_sz; i < nx; i++) { 
      F(j, i, 0) = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F(j, i, 1) = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F(j, i, 2) = -D1.apply_x_right(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], i, j, 1);
    }
  }
}

template <class SbpDerivative>
void wave_eq_single_core(grid::grid_function_2d<PetscScalar> F,
                          const grid::grid_function_2d<PetscScalar> q,
                          const PetscInt cl_sz,
                          const SbpDerivative& D1,
                          const std::array<PetscScalar,2>& hi,
                          const std::array<PetscScalar,2>& xl,
                          const PetscScalar t)
{
  return rhs_single_core(wave_eq_rhs_LL<decltype(D1)>,
                         wave_eq_rhs_LI<decltype(D1)>,
                         wave_eq_rhs_LR<decltype(D1)>,
                         wave_eq_rhs_IL<decltype(D1)>,
                         wave_eq_rhs_II<decltype(D1)>,
                         wave_eq_rhs_IR<decltype(D1)>,
                         wave_eq_rhs_RL<decltype(D1)>,
                         wave_eq_rhs_RI<decltype(D1)>,
                         wave_eq_rhs_RR<decltype(D1)>,
                         F,q,cl_sz,D1,hi,xl,t);
}

/**
* Free surface boundary condition functions
**/
template<class SbpInvQuad>
void free_surface_bc_west(grid::grid_function_2d<PetscScalar> F,
                           const grid::grid_function_2d<PetscScalar> q,
                           const std::array<PetscInt,2>& ind_j,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi)
{
  const PetscInt i = 0;
  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    F(j, i, 0) -= HI.apply_x_left(q, hi[0], i, j, 2);
  }
};

template<class SbpInvQuad>
void free_surface_bc_south(grid::grid_function_2d<PetscScalar> F,
                           const grid::grid_function_2d<PetscScalar> q,
                           const std::array<PetscInt,2>& ind_i,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi)
{
  const PetscInt j = 0;
  for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
    F(j, i, 1) -= HI.apply_y_left(q, hi[1], i, j, 2);
  }
};

template<class SbpInvQuad>
void free_surface_bc_east(grid::grid_function_2d<PetscScalar> F,
                           const grid::grid_function_2d<PetscScalar> q,
                           const std::array<PetscInt,2>& ind_j,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi)
{
  const PetscInt nx = q.mapping().nx();
  const PetscInt i = nx-1;
  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    F(j, i, 0) += HI.apply_x_right(q, hi[0], nx, i, j, 2);
  }
};

template<class SbpInvQuad>
void free_surface_bc_north(grid::grid_function_2d<PetscScalar> F,
                           const grid::grid_function_2d<PetscScalar> q,
                           const std::array<PetscInt,2>& ind_i,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi)
{
  const PetscInt ny = q.mapping().ny();
  const PetscInt j = ny-1;
  for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
    F(j, i, 1) += HI.apply_y_right(q, hi[1], ny, i, j, 2);
  }
};

template <class SbpInvQuad>
void wave_eq_free_surface_bc_single_core(grid::grid_function_2d<PetscScalar> F,
                                 const grid::grid_function_2d<PetscScalar> q,
                                 const SbpInvQuad& HI,
                                 const std::array<PetscScalar,2>& hi)
{
  bc_single_core(free_surface_bc_west<decltype(HI)>,
                 free_surface_bc_south<decltype(HI)>,
                 free_surface_bc_east<decltype(HI)>,
                 free_surface_bc_north<decltype(HI)>,F,q,HI,hi);
};