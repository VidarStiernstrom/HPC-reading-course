#pragma once

#include<petscsystypes.h>
#include <array>

  inline PetscScalar forcing_u(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
    PetscScalar x = xl[0] + i/hi[0];
    PetscScalar y = xl[1] + j/hi[1];
    return -(3*PETSC_PI*cos(5*PETSC_PI*t)*cos(3*PETSC_PI*x)*sin(4*PETSC_PI*y)*(x*y + 1))/(x*y + 2);
  }

  inline PetscScalar forcing_v(const PetscInt i, const PetscInt j, const PetscScalar t, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
    PetscScalar x = xl[0] + i/hi[0];
    PetscScalar y = xl[1] + j/hi[1];
    return -(4*PETSC_PI*cos(5*PETSC_PI*t)*cos(4*PETSC_PI*y)*sin(3*PETSC_PI*x)*(x*y + 1))/(x*y + 2);
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_LL(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw, const PetscInt n_closures)
  {
    int i,j;

    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/

    // Set dst on unaffected points
    for (j = 1; j < n_closures; j++)
    { 
      for (i = 1; i < n_closures; i++) 
      { 
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
      }
    }

    // Set dst on affected points
    i = 0; 
    j = 0;
    dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) - HI.apply_x_left(src, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
    dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) - HI.apply_y_left(src, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);

    i = 0;
    for (j = 1; j < n_closures; j++)
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) - HI.apply_x_left(src, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
    }

    j = 0;
    for (i = 1; i < n_closures; i++) 
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);;
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) - HI.apply_y_left(src, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_CL(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const PetscInt & i_xstart, const PetscInt & i_xend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;


    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/ 

    // Set dst on unaffected points
    for (j = 1; j < n_closures; j++)
    { 
      for (i = i_xstart; i < i_xend; i++)
      {
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_interior(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);;
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_interior(src,hi[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
      }
    }

    // Set dst on affected points
    j = 0;
    for (i = i_xstart; i < i_xend; i++)
    {
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_interior(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);;
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) - HI.apply_y_left(src, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_interior(src,hi[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_LC(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const PetscInt & i_ystart, const PetscInt & i_yend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;


    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/ 

    // Set dst on unaffected points
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = 1; i < n_closures; i++)
      {
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_interior(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_interior(src,hi[1],i,j,1);
      }
    }

    // Set dst on affected points
    i = 0;
    for (j = i_ystart; j < i_yend; j++)
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) - HI.apply_x_left(src, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);;
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_interior(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_interior(src,hi[1],i,j,1);;
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_CC(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const PetscInt & i_xstart, const PetscInt & i_xend,
                                           const PetscInt & i_ystart, const PetscInt & i_yend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = i_xstart; i < i_xend; i++)
      {
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_interior(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_interior(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_interior(src,hi[0],i,j,0) - D1.apply_y_interior(src,hi[1],i,j,1);
      }
    } 
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_RL(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;

    // Set dst on unaffected points
    for (j = 1; j < n_closures; j++)
    { 
      for (i = N[0]-n_closures; i < N[0]-1; i++) 
      { 
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
      }
    }

    // Set dst on affected points
    i = N[0]-1; 
    j = 0;
    dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + HI.apply_x_right(src, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) - HI.apply_y_left(src, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);

    i = N[0]-1;
    for (j = 1; j < n_closures; j++)
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + HI.apply_x_right(src, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
    }

    j = 0;
    for (i = N[0]-n_closures; i < N[0]-1; i++) 
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_left(src,hi[1],i,j,2) - HI.apply_y_left(src, hi[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_left(src,hi[1],i,j,1);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_RC(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const PetscInt & i_ystart, const PetscInt & i_yend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;


    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/ 

    // Set dst on unaffected points
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = N[0]-n_closures; i < N[0]-1; i++) 
      {
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_interior(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_interior(src,hi[1],i,j,1);
      }
    }

    // Set dst on affected points
    i = N[0]-1;
    for (j = i_ystart; j < i_yend; j++)
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + HI.apply_x_right(src, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_interior(src,hi[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_interior(src,hi[1],i,j,1);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_LR(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;


    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/ 

    // Set dst on unaffected points
    for (j = N[1]-n_closures; j < N[1]-1; j++) 
    { 
      for (i = 1; i < n_closures; i++) 
      { 
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
      }
    }

    // Set dst on affected points
    i = 0; 
    j = N[1]-1;
    dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) - HI.apply_x_left(src, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + HI.apply_y_right(src, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);

    i = 0;
    for (j = N[1]-n_closures; j < N[1]-1; j++) 
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) - HI.apply_x_left(src, hi[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
    }

    j = N[1]-1;
    for (i = 1; i < n_closures; i++) 
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_left(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + HI.apply_y_right(src, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_left(src,hi[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_CR(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const PetscInt & i_xstart, const PetscInt & i_xend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j;

    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/ 

    // Set dst on unaffected points
    for (j = N[1]-n_closures; j < N[1]-1; j++)
    { 
      for (i = i_xstart; i < i_xend; i++)
      {
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_interior(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_interior(src,hi[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
      }
    }

    // Set dst on affected points
    j = N[1]-1;
    for (i = i_xstart; i < i_xend; i++)
    {
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_interior(src,hi[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + HI.apply_y_right(src, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_interior(src,hi[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_RR(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw, const PetscInt n_closures)
  {
    int i,j;

    /**
    * BC using SAT, vt = D*v + SATw*v + SATe*v + SATs*v + SATn*v.
    * SATw = kron(tauw,HIx) ew kron(e3,ew'), tauw = [-1;0;0]
    * SATe = kron(taue,HIx) ee kron(e3,ee'), taue = [1;0;0]
    * SATs = kron(taus,HIy) es kron(e3,es'), taus = [0;-1;0]
    * SATn = kron(taun,HIy) en kron(e3,en'), taun = [0;1;0]
    **/ 

    // Set dst on unaffected points
    for (j = N[1]-n_closures; j < N[1]-1; j++)
    { 
      for (i = N[0]-n_closures; i < N[0]-1; i++) 
      { 
        dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
        dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
        dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
      }
    }

    // Set dst on affected points
    i = N[0]-1; 
    j = N[1]-1;
    dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + HI.apply_x_right(src, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
    dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + HI.apply_y_right(src, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
    dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);

    i = N[0]-1; 
    for (j = N[1]-n_closures; j < N[1]-1; j++)
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + HI.apply_x_right(src, hi[0], N[0], i, j, 2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
    }

    j = N[1]-1;
    for (i = N[0]-n_closures; i < N[0]-1; i++) 
    { 
      dst[j][i][0] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_x_right(src,hi[0],N[0],i,j,2) + forcing_u(i, j, t, hi, xl);
      dst[j][i][1] = -std::forward<CoefficientFunction>(rho_inv)(i,j)*D1.apply_y_right(src,hi[1],N[1],i,j,2) + HI.apply_y_right(src, hi[1], N[1], i, j, 2) + forcing_v(i, j, t, hi, xl);
      dst[j][i][2] = -D1.apply_x_right(src,hi[0],N[0],i,j,0) - D1.apply_y_right(src,hi[1],N[1],i,j,1);
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_interior(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           CoefficientFunction&& rho_inv,
                                           const PetscScalar *const *const *const src,
                                           PetscScalar *const *const *const dst,
                                           const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    const PetscInt i_xstart = i_start[0] + sw; 
    const PetscInt i_ystart = i_start[1] + sw;
    const PetscInt i_xend = i_end[0] - sw;
    const PetscInt i_yend = i_end[1] - sw;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start[1] == 0)  // BOTTOM
    {
      if (i_start[0] == 0) // BOTTOM LEFT
      {
        acowave_apply_LL(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_LC(t, D1, HI, rho_inv, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else if (i_end[0] == N[0]) // BOTTOM RIGHT
      { 
        acowave_apply_RL(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_RC(t, D1, HI, rho_inv, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, N[0]-n_closures, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_end[1] == N[1]) // TOP
    {
      if (i_start[0] == 0) // TOP LEFT
      {
        acowave_apply_LR(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_LC(t, D1, HI, rho_inv, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
      } else if (i_end[0] == N[0]) // TOP RIGHT
      { 
        acowave_apply_RR(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_RC(t, D1, HI, rho_inv, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, N[0]-n_closures, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
      }
    } else if (i_start[0] == 0) // LEFT NOT BOTTOM OR TOP
    { 
      acowave_apply_LC(t, D1, HI, rho_inv, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else if (i_end[0] == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      acowave_apply_RC(t, D1, HI, rho_inv, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_overlap(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                               CoefficientFunction&& rho_inv,
                                               const PetscScalar *const *const *const src,
                                               PetscScalar *const *const *const dst,
                                               const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end,
                                               const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                               const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    const PetscInt i_xstart = i_start[0]; 
    const PetscInt i_ystart = i_start[1];
    const PetscInt i_xend = i_end[0];
    const PetscInt i_yend = i_end[1];
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_ystart == 0)  // BOTTOM
    {
      if (i_xstart == 0) // BOTTOM LEFT
      {
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_LC(t, D1, HI, rho_inv, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend-sw, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_RC(t, D1, HI, rho_inv, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, n_closures, i_yend, N, xl, hi, sw, n_closures); 
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart+sw, N[0]-n_closures, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_CL(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, n_closures, i_yend-sw, N, xl, hi, sw, n_closures); 
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, n_closures, i_yend-sw, N, xl, hi, sw, n_closures); 
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xend, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_LC(t, D1, HI, rho_inv, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend-sw, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_RC(t, D1, HI, rho_inv, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart+sw, N[0]-n_closures, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_CR(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart+sw, i_xend-sw, i_ystart,  i_ystart+sw, N, xl, hi, sw, n_closures);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      acowave_apply_LC(t, D1, HI, rho_inv, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_LC(t, D1, HI, rho_inv, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, i_xend, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, i_ystart+sw, i_yend-sw, N, xl, hi, sw, n_closures);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      acowave_apply_RC(t, D1, HI, rho_inv, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_RC(t, D1, HI, rho_inv, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, N[0] - n_closures, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, i_ystart+sw, i_yend-sw, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart, i_xstart+sw, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xend-sw, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart+sw, i_xend-sw, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_CC(t, D1, HI, rho_inv, src, dst, i_xstart+sw, i_xend-sw, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename CoefficientFunction>
  inline PetscErrorCode acowave_apply_serial(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                             CoefficientFunction&& rho_inv,
                                             const PetscScalar *const *const *const src,
                                             PetscScalar *const *const *const dst,
                                             const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                             const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    acowave_apply_LL(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_RL(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_LR(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_RR(t, D1, HI, rho_inv, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_CL(t, D1, HI, rho_inv, src, dst, n_closures, N[0]-n_closures, N, xl, hi, sw, n_closures);
    acowave_apply_CR(t, D1, HI, rho_inv, src, dst, n_closures, N[0]-n_closures, N, xl, hi, sw, n_closures);
    acowave_apply_LC(t, D1, HI, rho_inv, src, dst, n_closures, N[1]-n_closures, N, xl, hi, sw, n_closures);
    acowave_apply_RC(t, D1, HI, rho_inv, src, dst, n_closures, N[1]-n_closures, N, xl, hi, sw, n_closures);
    acowave_apply_CC(t, D1, HI, rho_inv, src, dst, n_closures, N[0]-n_closures, n_closures, N[1]-n_closures, N, xl, hi, sw, n_closures);

    return 0;
  }
