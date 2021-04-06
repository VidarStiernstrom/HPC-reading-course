#pragma once

#include<petscsystypes.h>
#include <array>


/**
* Function computing the righ-hand-side of the acoustic wave equation
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
* Furthermore, along the boundary points, additional SBP operators are used to impose boundary conditions, using 
* simulatenous approxmiation terms (SAT). 
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
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the lower-left block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1w + SAT1s + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2w + SAT2s + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
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
  PetscInt i,j;

  // Corner point
  i = 0; 
  j = 0;
  F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) - HI.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
  F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) - HI.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
  F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);

  // South boundary
  i = 0;
  for (j = 1; j < cl_sz; j++)
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) - HI.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
  }

  // West boundary
  j = 0;
  for (i = 1; i < cl_sz; i++) 
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) - HI.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
  }

  // Remaining points in closure region
  for (j = 1; j < cl_sz; j++)
  { 
    for (i = 1; i < cl_sz; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }

  return 0;
}


 /**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the interior-left block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2s + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
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
  * i_xstart  - Starting index for the process in the x-direction
  * i_xend    - End index for the process in the x-direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_IL(const PetscScalar t,
                              const SbpDerivative& D1, 
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt i_xstart, 
                              const PetscInt i_xend,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
  PetscInt i,j;

  // South boundary
  j = 0;
  for (i = i_xstart; i < i_xend; i++)
  {
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) - HI.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
  }

  // Remanining points
  for (j = 1; j < cl_sz; j++)
  { 
    for (i = i_xstart; i < i_xend; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
    }
  }

  return 0;
}


/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the left-interior block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1w + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
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
  * i_ystart  - Starting index for the process in the y-direction
  * i_yend    - End index for the process in the y-direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_LI(const PetscScalar t,
                              const SbpDerivative& D1,
                              const SbpInvQuad& HI,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt i_ystart, 
                              const PetscInt i_yend,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
  PetscInt i,j;

  // West boundary
  i = 0;
  for (j = i_ystart; j < i_yend; j++)
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) - HI.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);;
  }

  // Remaining points
  for (j = i_ystart; j < i_yend; j++)
  { 
    for (i = 1; i < cl_sz; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }

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
  * i_xstart  - Starting index for the process in the x-direction
  * i_xend    - End index for the process in the x-direction
  * i_ystart  - Starting index for the process in the y-direction
  * i_yend    - End index for the process in the y-direction
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative>
PetscErrorCode wave_eq_rhs_II(const PetscScalar t, 
                              const SbpDerivative& D1,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt i_xstart, 
                              const PetscInt i_xend,
                              const PetscInt i_ystart, 
                              const PetscInt i_yend,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  PetscInt i,j;
  for (j = i_ystart; j < i_yend; j++)
  { 
    for (i = i_xstart; i < i_xend; i++)
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
  * Function computing the righ-hand-side of the acoustic wave equation in the right-left block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1e + SAT1s + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2e + SAT2s + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
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
  PetscInt i,j;

  // Corner point
  i = N[0]-1; 
  j = 0;
  F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + HI.apply_x_right(q, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
  F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) - HI.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
  F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);

  // East boundary
  i = N[0]-1;
  for (j = 1; j < cl_sz; j++)
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + HI.apply_x_right(q, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
  }

  // South boundary
  j = 0;
  for (i = N[0]-cl_sz; i < N[0]-1; i++) 
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) - HI.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
  }

  // Remaining points in block
  for (j = 1; j < cl_sz; j++)
  { 
    for (i = N[0]-cl_sz; i < N[0]-1; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_left(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_left(q, hi[1], i, j, 1);
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
  * F1 = -1/rho(x,y) p_x + SAT1e  + forcing_u 
  * F2 = -1/rho(x,y) p_y + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
  * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
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
  * i_ystart  - Starting index for the process in the y-direction
  * i_yend    - End index for the process in the y-direction
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
                              const PetscInt i_ystart, 
                              const PetscInt i_yend,
                              const std::array<PetscInt,2>& N,
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
  PetscInt i,j;


   // East boundary
  i = N[0]-1;
  for (j = i_ystart; j < i_yend; j++)
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + HI.apply_x_right(q, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
  }

  // Remaining points
  for (j = i_ystart; j < i_yend; j++)
  { 
    for (i = N[0]-cl_sz; i < N[0]-1; i++) 
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
  return 0;
}


/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the left-right block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1w + SAT1n + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2w + SAT2n + forcing_v
  * F3 = -(u_x + q_y)
  * 
  * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
  * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
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
  PetscInt i,j;

  // Corner point 
  i = 0; 
  j = N[1]-1;
  F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) - HI.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
  F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + HI.apply_y_right(q, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
  F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);

  // West boundary
  i = 0;
  for (j = N[1]-cl_sz; j < N[1]-1; j++) 
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) - HI.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
  }

  // North boundary
  j = N[1]-1;
  for (i = 1; i < cl_sz; i++) 
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + HI.apply_y_right(q, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
  }

  // Remaining points in block
  for (j = N[1]-cl_sz; j < N[1]-1; j++) 
  { 
    for (i = 1; i < cl_sz; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_left(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_left(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
    }
  }
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in a part of the interior-right block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2n + forcing_v
  * F3 = -(u_x + q_y)
  * 
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
  * i_xstart  - Starting index for the process in the x-direction
  * i_xend    - End index for the process in the x-direction
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
                              const PetscInt i_xstart,
                              const PetscInt i_xend,
                              const std::array<PetscInt,2>& N,
                              const std::array<PetscScalar,2>& xl, 
                              const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();
  PetscInt i,j;


  // Interior points
  for (j = N[1]-cl_sz; j < N[1]-1; j++)
  { 
    for (i = i_xstart; i < i_xend; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
    }
  }

  // North boundary
  j = N[1]-1;
  for (i = i_xstart; i < i_xend; i++)
  {
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + HI.apply_y_right(q, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
  }
  return 0;
}

/**
  * 
  * Function computing the righ-hand-side of the acoustic wave equation in the right-right block (see below).
  * 
  * F(t,q) = [F1(t,q),F2(t,q),F3(t,q)]^T
  * where q = [u,v,p]^T, and 
  * F1 = -1/rho(x,y) p_x + SAT1e + SAT1n + forcing_u 
  * F2 = -1/rho(x,y) p_y + SAT2e + SAT2n + forcing_v
  * F3 = -(u_x + q_y)
  * 
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
  PetscInt i,j;

  // Interior points
  for (j = N[1]-cl_sz; j < N[1]-1; j++)
  { 
    for (i = N[0]-cl_sz; i < N[0]-1; i++) 
    { 
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
    }
  }

  // East boundary
  i = N[0]-1; 
  for (j = N[1]-cl_sz; j < N[1]-1; j++)
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + HI.apply_x_right(q, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
  }

  // North boundary
  j = N[1]-1;
  for (i = N[0]-cl_sz; i < N[0]-1; i++) 
  { 
    F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
    F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + HI.apply_y_right(q, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);
  }

  // Corner point
  i = N[0]-1; 
  j = N[1]-1;
  F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_right(q,hi[0],N[0],i,j,2) + HI.apply_x_right(q, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
  F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_right(q,hi[1],N[1],i,j,2) + HI.apply_y_right(q, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
  F[j][i][2] = -D1.apply_x_right(q, hi[0], N[0], i, j, 0) - D1.apply_y_right(q, hi[1], N[1], i, j, 1);

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
  * i_start   - Starting indices per coordinate direction for the process
  * i_end     - End indices per coordinate direction for the process
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_local(const PetscScalar t,
                                 const SbpDerivative& D1,
                                 const SbpInvQuad& HI,
                                 const PetscScalar *const *const *const q,
                                 PetscScalar *const *const *const F,
                                 const std::array<PetscInt,2>& i_start,
                                 const std::array<PetscInt,2>& i_end,
                                 const std::array<PetscInt,2>& N,
                                 const std::array<PetscScalar,2>& xl, 
                                 const std::array<PetscScalar,2>& hi)
{
  const PetscScalar sw = (D1.interior_stencil_size()-1)/2;
  const PetscInt cl_sz = D1.closure_size();
  const PetscInt i_xstart = i_start[0] + sw; 
  const PetscInt i_ystart = i_start[1] + sw;
  const PetscInt i_xend = i_end[0] - sw;
  const PetscInt i_yend = i_end[1] - sw;

  if (i_start[1] == 0)  // BOTTOM
  {
    if (i_start[0] == 0) // BOTTOM LEFT
    {
      wave_eq_rhs_LL(t, D1, HI, q, F, xl, hi);
      wave_eq_rhs_IL(t, D1, HI, q, F, cl_sz, i_xend, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, cl_sz, i_yend, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend, cl_sz, i_yend, xl, hi); 
    } else if (i_end[0] == N[0]) // BOTTOM RIGHT
    { 
      wave_eq_rhs_RL(t, D1, HI, q, F, N, xl, hi);
      wave_eq_rhs_IL(t, D1, HI, q, F, i_xstart, N[0]-cl_sz, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, cl_sz, i_yend, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, N[0]-cl_sz, cl_sz, i_yend, xl, hi); 
    } else // BOTTOM CENTER
    { 
      wave_eq_rhs_IL(t, D1, HI, q, F, i_xstart, i_xend, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xend, cl_sz, i_yend, xl, hi); 
    }
  } else if (i_end[1] == N[1]) // TOP
  {
    if (i_start[0] == 0) // TOP LEFT
    {
      wave_eq_rhs_LR(t, D1, HI, q, F, N, xl, hi);
      wave_eq_rhs_IR(t, D1, HI, q, F, cl_sz, i_xend, N, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, i_ystart, N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend, i_ystart, N[1]-cl_sz, xl, hi);  
    } else if (i_end[0] == N[0]) // TOP RIGHT
    { 
      wave_eq_rhs_RR(t, D1, HI, q, F, N, xl, hi);
      wave_eq_rhs_IR(t, D1, HI, q, F, i_xstart, N[0]-cl_sz, N, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, i_ystart, N[1] - cl_sz, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, N[0]-cl_sz, i_ystart, N[1] - cl_sz, xl, hi);
    } else // TOP CENTER
    { 
      wave_eq_rhs_IR(t, D1, HI, q, F, i_xstart, i_xend, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xend, i_ystart,  N[1] - cl_sz, xl, hi);
    }
  } else if (i_start[0] == 0) // LEFT NOT BOTTOM OR TOP
  { 
    wave_eq_rhs_LI(t, D1, HI, q, F, i_ystart, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend, i_ystart, i_yend, xl, hi);
  } else if (i_end[0] == N[0]) // RIGHT NOT BOTTOM OR TOP
  {
    wave_eq_rhs_RI(t, D1, HI, q, F, i_ystart, i_yend, N, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xstart, N[0] - cl_sz, i_ystart, i_yend, xl, hi);
  } else // CENTER
  {
    wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xend, i_ystart, i_yend, xl, hi);
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
  * i_start   - Starting indices per coordinate direction for the process
  * i_end     - End indices per coordinate direction for the process
  * xl        - Lower-left grid points per coordinate direction
  * hi        - Inverse spacing spacing per coordinate direction
  **/
template <class SbpDerivative, class SbpInvQuad>
PetscErrorCode wave_eq_rhs_overlap(const PetscScalar t,
                                   const SbpDerivative& D1,
                                   const SbpInvQuad& HI,
                                   const PetscScalar *const *const *const q,
                                   PetscScalar *const *const *const F,
                                   const std::array<PetscInt,2>& i_start,
                                   const std::array<PetscInt,2>& i_end,
                                   const std::array<PetscInt,2>& N,
                                   const std::array<PetscScalar,2>& xl, 
                                   const std::array<PetscScalar,2>& hi)
{
  const PetscScalar sw = (D1.interior_stencil_size()-1)/2;
  const PetscInt cl_sz = D1.closure_size();
  const PetscInt i_xstart = i_start[0]; 
  const PetscInt i_ystart = i_start[1];
  const PetscInt i_xend = i_end[0];
  const PetscInt i_yend = i_end[1];

  if (i_ystart == 0)  // BOTTOM
  {
    if (i_xstart == 0) // BOTTOM LEFT
    {
      wave_eq_rhs_IL(t, D1, HI, q, F, i_xend-sw, i_xend, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, i_yend-sw, i_yend, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend-sw, i_yend-sw, i_yend, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_xend-sw, i_xend, cl_sz, i_yend, xl, hi); 
    } else if (i_xend == N[0]) // BOTTOM RIGHT
    { 
      wave_eq_rhs_IL(t, D1, HI, q, F, i_xstart, i_xstart+sw, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, i_yend-sw, i_yend, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xstart+sw, cl_sz, i_yend, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_xstart+sw, N[0]-cl_sz, i_yend-sw, i_yend, xl, hi); 
    } else // BOTTOM CENTER
    { 
      wave_eq_rhs_IL(t, D1, HI, q, F, i_xstart, i_xstart+sw, xl, hi);
      wave_eq_rhs_IL(t, D1, HI, q, F, i_xend-sw, i_xend, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xstart+sw, cl_sz, i_yend-sw, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_xend-sw, i_xend, cl_sz, i_yend-sw, xl, hi); 
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xend, i_yend-sw, i_yend, xl, hi); 
    }
  } else if (i_yend == N[1]) // TOP
  {
    if (i_xstart == 0) // TOP LEFT
    {
      wave_eq_rhs_IR(t, D1, HI, q, F, i_xend-sw, i_xend, N, xl, hi);
      wave_eq_rhs_LI(t, D1, HI, q, F, i_ystart, i_ystart+sw, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xend-sw, i_xend, i_ystart, N[1]-cl_sz, xl, hi);  
      wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend-sw, i_ystart, i_ystart+sw, xl, hi);  
    } else if (i_xend == N[0]) // TOP RIGHT
    { 
      wave_eq_rhs_IR(t, D1, HI, q, F, i_xstart, i_xstart+sw, N, xl, hi);
      wave_eq_rhs_RI(t, D1, HI, q, F, i_ystart, i_ystart+sw, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xstart+sw, i_ystart, N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart+sw, N[0]-cl_sz, i_ystart, i_ystart+sw, xl, hi);
    } else // TOP CENTER
    { 
      wave_eq_rhs_IR(t, D1, HI, q, F, i_xstart, i_xstart+sw, N, xl, hi);
      wave_eq_rhs_IR(t, D1, HI, q, F, i_xend-sw, i_xend, N, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xstart+sw, i_ystart,  N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xend-sw, i_xend, i_ystart,  N[1] - cl_sz, xl, hi);
      wave_eq_rhs_II(t, D1, q, F, i_xstart+sw, i_xend-sw, i_ystart,  i_ystart+sw, xl, hi);
    }
  } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
  { 
    wave_eq_rhs_LI(t, D1, HI, q, F, i_ystart, i_ystart+sw, xl, hi);
    wave_eq_rhs_LI(t, D1, HI, q, F, i_yend-sw, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend, i_ystart, i_ystart+sw, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, cl_sz, i_xend, i_yend-sw, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xend-sw, i_xend, i_ystart+sw, i_yend-sw, xl, hi);
  } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
  {
    wave_eq_rhs_RI(t, D1, HI, q, F, i_ystart, i_ystart+sw, N, xl, hi);
    wave_eq_rhs_RI(t, D1, HI, q, F, i_yend-sw, i_yend, N, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xstart, N[0] - cl_sz, i_ystart, i_ystart+sw, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xstart, N[0] - cl_sz, i_yend-sw, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xstart+sw, i_ystart+sw, i_yend-sw, xl, hi);
  } else // CENTER
  {
    wave_eq_rhs_II(t, D1, q, F, i_xstart, i_xstart+sw, i_ystart, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xend-sw, i_xend, i_ystart, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xstart+sw, i_xend-sw, i_yend-sw, i_yend, xl, hi);
    wave_eq_rhs_II(t, D1, q, F, i_xstart+sw, i_xend-sw, i_ystart, i_ystart+sw, xl, hi);
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
PetscScalar rho_inv(const PetscInt i, const PetscInt j,  const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return 1./(2 + x*y);
}

/**
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
