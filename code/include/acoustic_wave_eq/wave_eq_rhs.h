#pragma once

#include<petscsystypes.h>
#include <array>


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
PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl);

/**
* Forcing functions
**/
PetscScalar forcing_u(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl);
PetscScalar forcing_v(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl);
/**
* Free surface boundary condition functions
**/
template<class SbpInvQuad>
void free_surface_bc_west(const SbpInvQuad& HI,
                          const PetscScalar *const *const *const q,
                          PetscScalar *const *const *const F,
                          const PetscInt j_start, 
                          const PetscInt j_end,
                          const PetscScalar hix);
template<class SbpInvQuad>
void free_surface_bc_south(const SbpInvQuad& HI,
                           const PetscScalar *const *const *const q,
                           PetscScalar *const *const *const F,
                           const PetscInt i_start, 
                           const PetscInt i_end,
                           const PetscScalar hiy);
template<class SbpInvQuad>
void free_surface_bc_east(const SbpInvQuad& HI,
                          const PetscScalar *const *const *const q,
                          PetscScalar *const *const *const F,
                          const PetscInt j_start, 
                          const PetscInt j_end,
                          const PetscInt Nx,
                          const PetscScalar hix);
template<class SbpInvQuad>
void free_surface_bc_north(const SbpInvQuad& HI,
                           const PetscScalar *const *const *const q,
                           PetscScalar *const *const *const F,
                           const PetscInt i_start, 
                           const PetscInt i_end,
                           const PetscInt Ny,
                           const PetscScalar hiy);

 /**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the lower-left block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: Free surface boundary condition on west and south boundary
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * LL *    *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * xl         - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_LL(const PetscScalar t, 
                              const SbpDerivative& D1,
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
  
  // Apply rhs of wave equation with forcing in LL block
  for (PetscInt j = 0; j < cl_sz; j++)
  { 
    for (PetscInt i = 0; i < cl_sz; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }

  // Apply boundary conditions on west and south boundary
  free_surface_bc_west(HI, q, F , 0, cl_sz, hi[0]);
  free_surface_bc_south(HI, q, F , 0, cl_sz, hi[1]);
  return 0;
}


 /**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the interior-left block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: Free surface boundary condition on south boundary
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * IL *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * i_start  - Starting index for the process in the x-direction
  * i_end    - End index for the process in the x-direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_IL(const PetscScalar t,
                              const SbpDerivative& D1, 
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt i_start, 
                              const PetscInt i_end,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
  // Apply rhs of wave equation with forcing in IL block
  for (PetscInt j = 0; j < cl_sz; j++)
  { 
    for (PetscInt i = i_start; i < i_end; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }

  // Apply boundary conditions on south boundary
  free_surface_bc_south(HI, q, F , i_start, i_end, hi[0]);
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the right-left block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: Free surface boundary conditions on east and south boundary
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * RL *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * N         - Grid points per coordinate direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_RL(const PetscScalar t, 
                              const SbpDerivative& D1, 
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const std::array<PetscInt,2>& N,
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

  // Apply rhs of wave equation with forcing in RL block
  for (PetscInt j = 0; j < cl_sz; j++)
  { 
    for (PetscInt i = N[0]-cl_sz; i < N[0]; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }
  // Apply boundary conditions on east and south boundary
  free_surface_bc_east(HI, q, F , 0, cl_sz, N[0], hi[0]);
  free_surface_bc_south(HI, q, F ,  N[0]-cl_sz,  N[0], hi[1]);
  return 0;
}


/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the left-interior block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: Free surface boundary condition on west boundary
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * LI *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * j_start  - Starting index for the process in the y-direction
  * j_end    - End index for the process in the y-direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_LI(const PetscScalar t,
                              const SbpDerivative& D1,
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt j_start, 
                              const PetscInt j_end,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
 
  // Apply rhs of wave equation with forcing in LI block
  for (PetscInt j = j_start; j < j_end; j++)
  { 
    for (PetscInt i = 0; i < cl_sz; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }

  // Apply boundary conditions on west boundary
  free_surface_bc_west(HI, q, F , j_start, j_end, hi[0]);
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the interior-interior block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
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
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * i_start  - Starting index for the process in the x-direction
  * i_end    - End index for the process in the x-direction
  * j_start  - Starting index for the process in the y-direction
  * j_end    - End index for the process in the y-direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative>
PetscErrorCode wave_eq_rhs_II(const PetscScalar t, 
                              const SbpDerivative& D1,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt i_start, 
                              const PetscInt i_end,
                              const PetscInt j_start, 
                              const PetscInt j_end,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  // Apply rhs of wave equation with forcing in II block
  for (PetscInt j = j_start; j < j_end; j++)
  { 
    for (PetscInt i = i_start; i < i_end; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  } 
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the right-interior block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  *
  * BC: Free surface boundary conditions on east boundary
  *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * RI *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * j_start  - Starting index for the process in the y-direction
  * j_end    - End index for the process in the y-direction
  * N         - Grid points per coordinate direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_RI(const PetscScalar t, 
                              const SbpDerivative& D1, 
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt j_start, 
                              const PetscInt j_end,
                              const std::array<PetscInt,2>& N,
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

  // Apply rhs of wave equation with forcing in RI block
  for (PetscInt j = j_start; j < j_end; j++)
  { 
    for (PetscInt i = N[0]-cl_sz; i < N[0]; i++) 
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
  // Apply boundary conditions on east boundary
  free_surface_bc_east(HI, q, F , j_start, j_end, N[0], hi[0]);
  return 0;
}


/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the left-right block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: free surface boundary conditions on west and north boundary
  *
  *   ****************
  *   * LR *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * N         - Grid points per coordinate direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_LR(const PetscScalar t,
                              const SbpDerivative& D1,
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const std::array<PetscInt,2>& N, 
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

  // Apply rhs of wave equation with forcing in LR block
  for (PetscInt j = N[1]-cl_sz; j < N[1]; j++) 
  { 
    for (PetscInt i = 0; i < cl_sz; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
    }
  }
  // Apply boundary conditions on west and north boundary
  free_surface_bc_west(HI, q, F , N[1]-cl_sz, N[1], hi[0]);
  free_surface_bc_north(HI, q, F , 0, cl_sz, N[1], hi[1]);
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the interior-right block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: free surface boundary conditions on north boundary
  *
  *   ****************
  *   *    * IR  *   *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * i_start  - Starting index for the process in the x-direction
  * i_end    - End index for the process in the x-direction
  * N         - Grid points per coordinate direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_IR(const PetscScalar t,
                              const SbpDerivative& D1, 
                              const SbpInvQuad& HI,
                              const PetscScalar * const * const * const q,
                              PetscScalar * const * const * const F,
                              const PetscInt i_start,
                              const PetscInt i_end,
                              const std::array<PetscInt,2>& N,
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

  // Apply rhs of wave equation with forcing in IR block
  for (PetscInt j = N[1]-cl_sz; j < N[1]; j++)
  { 
    for (PetscInt i = i_start; i < i_end; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
    }
  }
  // Apply boundary conditions on north boundary
  free_surface_bc_north(HI, q, F , i_start, i_end, N[1], hi[1]);
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the right-right block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * BC: free surface boundary conditions on east and north boundary
  *
  *   ****************
  *   *    *    * RR *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * N         - Grid points per coordinate direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_RR(const PetscScalar t,
                              const SbpDerivative& D1,
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const std::array<PetscInt,2>& N,
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

 // Apply rhs of wave equation with forcing in RR block
  for (PetscInt j = N[1]-cl_sz; j < N[1]; j++)
  { 
    for (PetscInt i = N[0]-cl_sz; i < N[0]; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
    }
  }

  // Apply boundary conditions on east and north boundary
  free_surface_bc_east(HI, q, F , N[1]-cl_sz, N[1], N[0], hi[0]);
  free_surface_bc_north(HI, q, F , N[0]-cl_sz, N[0], N[1], hi[1]);
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation for local indices for a process (i.e avoiding ghost points)
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1 + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2 + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
  *
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * idx_start - Starting indices per coordinate direction for the process
  * idx_end   - End indices per coordinate direction for the process
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_local(const PetscScalar t,
                                 const SbpDerivative& D1,
                                 const SbpInvQuad& HI,
                                 const PetscScalar *const *const *const q,
                                 PetscScalar *const *const *const F,
                                 const std::array<PetscInt,2>& idx_start,
                                 const std::array<PetscInt,2>& idx_end,
                                 const std::array<PetscInt,2>& N,
                                 const std::array<PetscScalar,2>& xl, 
                                 const std::array<PetscScalar,2>& hi)
{
  const PetscScalar sw = (D1.interior_stencil_size()-1)/2;
  const PetscInt cl_sz = D1.closure_size();
  const PetscInt i_start = idx_start[0] + sw; 
  const PetscInt j_start = idx_start[1] + sw;
  const PetscInt i_end = idx_end[0] - sw;
  const PetscInt j_end = idx_end[1] - sw;

  if (idx_start[1] == 0)  // BOTTOM
  {
    if (idx_start[0] == 0) // BOTTOM LEFT
    {
      wave_eq_rhs_LL(t, D1, HI, q, F, xl, hi);
      wave_eq_rhs_IL(t, D1, HI, q, F, cl_sz, i_end, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, cl_sz, j_end, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end, cl_sz, j_end, xl, hi); 
    } else if (idx_end[0] == N[0]) // BOTTOM RIGHT
    { 
      wave_eq_rhs_RL(t, D1, HI, q, F, N, xl, hi);
      wave_eq_rhs_IL(t, D1, HI, q, F, i_start, N[0]-cl_sz, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, cl_sz, j_end, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, N[0]-cl_sz, cl_sz, j_end, xl, hi); 
    } else // BOTTOM CENTER
    { 
      wave_eq_rhs_IL(t, D1, HI, q, F, i_start, i_end, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, i_end, cl_sz, j_end, xl, hi); 
    }
  } else if (idx_end[1] == N[1]) // TOP
  {
    if (idx_start[0] == 0) // TOP LEFT
    {
      wave_eq_rhs_LR(t, D1, HI, q, F, N, xl, hi);
      wave_eq_rhs_IR(t, D1, HI, q, F, cl_sz, i_end, N, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, j_start, N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end, j_start, N[1]-cl_sz, xl, hi);  
    } else if (idx_end[0] == N[0]) // TOP RIGHT
    { 
      wave_eq_rhs_RR(t, D1, HI, q, F, N, xl, hi);
      wave_eq_rhs_IR(t, D1, HI, q, F, i_start, N[0]-cl_sz, N, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, j_start, N[1] - cl_sz, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, N[0]-cl_sz, j_start, N[1] - cl_sz, xl, hi);
    } else // TOP CENTER
    { 
      wave_eq_rhs_IR(t, D1, HI, q, F, i_start, i_end, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, i_end, j_start,  N[1] - cl_sz, xl, hi);
    }
  } else if (idx_start[0] == 0) // LEFT NOT BOTTOM OR TOP
  { 
    wave_eq_rhs_LI(t, D1, HI, q, F, j_start, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end, j_start, j_end, xl, hi);
  } else if (idx_end[0] == N[0]) // RIGHT NOT BOTTOM OR TOP
  {
    wave_eq_rhs_RI(t, D1, HI, q, F, j_start, j_end, N, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_start, N[0] - cl_sz, j_start, j_end, xl, hi);
  } else // CENTER
  {
    wave_eq_rhs_II(t, D1, q, F, i_start, i_end, j_start, j_end, xl, hi);
  }

  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation for overlapping indices for a process (i.e only ghost points)
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1 + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2 + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * idx_start - Starting indices per coordinate direction for the process
  * idx_end   - End indices per coordinate direction for the process
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_overlap(const PetscScalar t,
                                   const SbpDerivative& D1,
                                   const SbpInvQuad& HI,
                                   const PetscScalar *const *const *const q,
                                   PetscScalar *const *const *const F,
                                   const std::array<PetscInt,2>& idx_start,
                                   const std::array<PetscInt,2>& idx_end,
                                   const std::array<PetscInt,2>& N,
                                   const std::array<PetscScalar,2>& xl, 
                                   const std::array<PetscScalar,2>& hi)
{
  const PetscScalar sw = (D1.interior_stencil_size()-1)/2;
  const PetscInt cl_sz = D1.closure_size();
  const PetscInt i_start = idx_start[0]; 
  const PetscInt j_start = idx_start[1];
  const PetscInt i_end = idx_end[0];
  const PetscInt j_end = idx_end[1];

  if (j_start == 0)  // BOTTOM
  {
    if (i_start == 0) // BOTTOM LEFT
    {
      wave_eq_rhs_IL(t, D1, HI, q, F, i_end-sw, i_end, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, j_end-sw, j_end, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end-sw, j_end-sw, j_end, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_end-sw, i_end, cl_sz, j_end, xl, hi); 
    } else if (i_end == N[0]) // BOTTOM RIGHT
    { 
      wave_eq_rhs_IL(t, D1, HI, q, F, i_start, i_start+sw, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, j_end-sw, j_end, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, i_start+sw, cl_sz, j_end, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_start+sw, N[0]-cl_sz, j_end-sw, j_end, xl, hi); 
    } else // BOTTOM CENTER
    { 
      wave_eq_rhs_IL(t, D1, HI, q, F, i_start, i_start+sw, xl, hi);
      wave_eq_rhs_IL(t, D1, HI, q, F, i_end-sw, i_end, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, i_start+sw, cl_sz, j_end-sw, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_end-sw, i_end, cl_sz, j_end-sw, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_start, i_end, j_end-sw, j_end, xl, hi); 
    }
  } else if (j_end == N[1]) // TOP
  {
    if (i_start == 0) // TOP LEFT
    {
      wave_eq_rhs_IR(t, D1, HI, q, F, i_end-sw, i_end, N, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, j_start, j_start+sw, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_end-sw, i_end, j_start, N[1]-cl_sz, xl, hi);  
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end-sw, j_start, j_start+sw, xl, hi);  
    } else if (i_end == N[0]) // TOP RIGHT
    { 
      wave_eq_rhs_IR(t, D1, HI, q, F, i_start, i_start+sw, N, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, j_start, j_start+sw, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, i_start+sw, j_start, N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start+sw, N[0]-cl_sz, j_start, j_start+sw, xl, hi);
    } else // TOP CENTER
    { 
      wave_eq_rhs_IR(t, D1, HI, q, F, i_start, i_start+sw, N, xl, hi);
      wave_eq_rhs_IR(t, D1, HI, q, F, i_end-sw, i_end, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start, i_start+sw, j_start,  N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_end-sw, i_end, j_start,  N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_start+sw, i_end-sw, j_start,  j_start+sw, xl, hi);
    }
  } else if (i_start == 0) // LEFT NOT BOTTOM OR TOP
  { 
    wave_eq_rhs_LI(t, D1, HI, q, F, j_start, j_start+sw, xl, hi);
    wave_eq_rhs_LI(t, D1, HI, q, F, j_end-sw, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end, j_start, j_start+sw, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, cl_sz, i_end, j_end-sw, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_end-sw, i_end, j_start+sw, j_end-sw, xl, hi);
  } else if (i_end == N[0]) // RIGHT NOT BOTTOM OR TOP
  {
    wave_eq_rhs_RI(t, D1, HI, q, F, j_start, j_start+sw, N, xl, hi);
    wave_eq_rhs_RI(t, D1, HI, q, F, j_end-sw, j_end, N, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_start, N[0] - cl_sz, j_start, j_start+sw, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_start, N[0] - cl_sz, j_end-sw, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_start, i_start+sw, j_start+sw, j_end-sw, xl, hi);
  } else // CENTER
  {
    wave_eq_rhs_II(t, D1, q, F, i_start, i_start+sw, j_start, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_end-sw, i_end, j_start, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_start+sw, i_end-sw, j_end-sw, j_end, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_start+sw, i_end-sw, j_start, j_start+sw, xl, hi);
  }

  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation on a single process
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1 + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2 + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
  *
  *
  *
  * Input: 
  * t         - time
  * D1        - SBP difference operator used to approximate d/dx and d/dy
  * HI        - SBP inverse quadrature used to impose boundary conditions
  * q         - Current solution vector. Stored as a 3D-array q[j][i][k] i,j are grid point indices 
  *             and where k is the component.
  * F         - Result of function evaluation. Stored as a 3D-array q[j][i][k] i,j are grid point indices
  *             and where k is the component.
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_serial(const PetscScalar t,
                                  const SbpDerivative& D1,
                                  const SbpInvQuad& HI,
                                  const PetscScalar *const *const *const q,
                                  PetscScalar *const *const *const F,
                                  const std::array<PetscInt,2>& N,
                                  const std::array<PetscScalar,2>& xl, 
                                  const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

  wave_eq_rhs_LL(t, D1, HI, q, F, xl, hi);
  wave_eq_rhs_RL(t, D1, HI, q, F, N, xl, hi);
  wave_eq_rhs_LR(t, D1, HI, q, F, N, xl, hi);
  wave_eq_rhs_RR(t, D1, HI, q, F, N, xl, hi);
  wave_eq_rhs_IL(t, D1, HI, q, F, cl_sz, N[0]-cl_sz, xl, hi);
  wave_eq_rhs_IR(t, D1, HI, q, F, cl_sz, N[0]-cl_sz, N, xl, hi);
  wave_eq_rhs_LI(t, D1, HI, q, F, cl_sz, N[1]-cl_sz, xl, hi);
  wave_eq_rhs_RI(t, D1, HI, q, F, cl_sz, N[1]-cl_sz, N, xl, hi);
  wave_eq_rhs_II(t, D1, q, F, cl_sz, N[0]-cl_sz, cl_sz, N[1]-cl_sz, xl, hi);

  return 0;
}

/**
* Inverse of density rho(x,y) at grid point i,j
**/
// inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) __attribute__((pure));
inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return 1./(2 + x*y);
}

// /**
// * Inverse of density rho(x,y) at grid point i,j
// **/
// inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& h, const std::array<PetscScalar,2>& xl) {
//   PetscScalar x = xl[0] + i * h[0];
//   PetscScalar y = xl[1] + j * h[1];
//   return 1./(2 + x*y);
// }

// /**
// * Inverse of density rho(x,y) at grid point i,j
// **/
// inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& h, const std::array<PetscScalar,2>& xl) {
//   PetscScalar x = xl[0] + i * h[0];
//   PetscScalar y = xl[1] + j * h[1];
//   return (2 + x*y);
// }

/**
* TODO WIP optimized version
*
* Forcing function on u component
**/
PetscScalar forcing_u(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return -(3*PETSC_PI*cos(5*PETSC_PI*t)*cos(3*PETSC_PI*x)*sin(4*PETSC_PI*y)*(x*y + 1))/(x*y + 2);
}

/**
* Forcing function on v component
**/
PetscScalar forcing_v(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return -(4*PETSC_PI*cos(5*PETSC_PI*t)*cos(4*PETSC_PI*y)*sin(3*PETSC_PI*x)*(x*y + 1))/(x*y + 2);
}

/**
* Free surface boundary condition on west boundary
**/
template<class SbpInvQuad>
void free_surface_bc_west(const SbpInvQuad& HI,
                          const PetscScalar *const *const *const q,
                          PetscScalar *const *const *const F,
                          const PetscInt j_start, 
                          const PetscInt j_end,
                          const PetscScalar hix)
{
  PetscInt i = 0;
  for (PetscInt j = j_start; j < j_end; j++)
  { 
    F[j][i][0] -= HI.apply_x_left(q, hix, i, j, 2);
  }
}

/**
* Free surface boundary condition on south boundary
**/
template<class SbpInvQuad>
void free_surface_bc_south(const SbpInvQuad& HI,
                           const PetscScalar *const *const *const q,
                           PetscScalar *const *const *const F,
                           const PetscInt i_start, 
                           const PetscInt i_end,
                           const PetscScalar hiy)
{

  PetscInt j = 0;
  for (PetscInt i = i_start; i < i_end; i++)
  { 
    F[j][i][1] -= HI.apply_y_left(q, hiy, i, j, 2);
  }
}

/**
* Free surface boundary condition on east boundary
**/
template<class SbpInvQuad>
void free_surface_bc_east(const SbpInvQuad& HI,
                          const PetscScalar *const *const *const q,
                          PetscScalar *const *const *const F,
                          const PetscInt j_start, 
                          const PetscInt j_end,
                          const PetscInt Nx,
                          const PetscScalar hix)
{
  PetscInt i = Nx-1;
  for (PetscInt j = j_start; j < j_end; j++) 
  { 
    F[j][i][0] += HI.apply_x_right(q, hix, Nx, i, j, 2);
  }
}

/**
* Free surface boundary condition on north boundary
**/
template<class SbpInvQuad>
void free_surface_bc_north(const SbpInvQuad& HI,
                           const PetscScalar *const *const *const q,
                           PetscScalar *const *const *const F,
                           const PetscInt i_start, 
                           const PetscInt i_end,
                           const PetscInt Ny,
                           const PetscScalar hiy)
{

  PetscInt j = Ny-1;
  for (PetscInt i = i_start; i < i_end; i++)
  { 
    F[j][i][1] += HI.apply_y_right(q, hiy, Ny, i, j, 2);
  }
}
