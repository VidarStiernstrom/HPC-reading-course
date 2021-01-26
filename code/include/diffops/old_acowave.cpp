#pragma once

#include<petscsystypes.h>
#include <array>
#include "grids/grid_function.h"

namespace sbp{

  inline PetscScalar aco_imp_apply_D_time(const PetscScalar D_time[4][4], PetscScalar ***src, PetscInt i, PetscInt j, PetscInt tcomp, PetscInt dof)
  {
    return D_time[tcomp][0]*src[j][i][0+dof] + D_time[tcomp][1]*src[j][i][3+dof] + D_time[tcomp][2]*src[j][i][6+dof] + D_time[tcomp][3]*src[j][i][9+dof];
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_LL(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw, const PetscInt n_closures, const PetscInt closure_width)
  {
    int i,j,tcomp;

    // Compute third component at boundary points, only time derivative
    i = 0;
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
    }

    i = 0;
    for (j = 1; j < n_closures; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    j = 0;
    for (i = 1; i < n_closures; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    i = 0;
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      src[j][i][3*tcomp+2] = 0;
    }

    i = 0;
    for (j = 1; j < closure_width; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    j = 0;
    for (i = 1; i < closure_width; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    i = 0;
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
    }

    i = 0;
    for (j = 1; j < n_closures; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
      }
    }

    j = 0;
    for (i = 1; i < n_closures; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
      }
    }

    // Compute inner points
    for (j = 1; j < n_closures; j++) {
      for (i = 1; i < n_closures; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,0) + D1.apply_2D_y_left(src,hi[1],i,j,1);
        }
      }
    }


    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_CL(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const PetscInt & i_xstart, const PetscInt & i_xend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures, const PetscInt iw)
  {
    int i,j,tcomp;

    // Compute third component at boundary points, only time derivative
    // printf("i_xstart: %d, i_xend: %d, iw: %d\n",i_xstart,i_xend,iw);
    j = 0;
    for (i = i_xstart; i < i_xend; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
        // dst[j][i][3*tcomp+2] = -100;
      }
    }

    // Set third comp in src to zero at boundary points
    j = 0;
    for (i = i_xstart-sw; i < i_xend+sw; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    j = 0;
    for (i = i_xstart; i < i_xend; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        // printf("i: %d, tcomp: %d\n",i,tcomp);
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
      }
    }

    // Compute inner points
    for (j = 1; j < n_closures; j++) {
      for (i = i_xstart; i < i_xend; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_LC(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const PetscInt & i_ystart, const PetscInt & i_yend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures, const PetscInt iw)
  {
    int i,j,tcomp;

    // printf("i_ystart: %d, i_yend: %d iw: %d\n",i_ystart,i_yend,iw);

    // Compute third component at boundary points, only time derivative
    i = 0;
    for (j = i_ystart; j < i_yend; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        // printf("src[j=%d][tcomp=%d] = %f\n",j,tcomp,src[j][i][3*tcomp+2]);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    i = 0;
    for (j = i_ystart-sw; j < i_yend+sw; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    i = 0;
    for (j = i_ystart; j < i_yend; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
      }
    }

    // Compute inner points
    for (j = i_ystart; j < i_yend; j++) {
      for (i = 1; i < n_closures; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_CC(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const PetscInt & i_xstart, const PetscInt & i_xend,
                                           const PetscInt & i_ystart, const PetscInt & i_yend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl,
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j,tcomp;

    for (j = i_ystart; j < i_yend; j++) {
      for (i = i_xstart; i < i_xend; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }
    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_RL(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j,tcomp;

    // Compute third component at boundary points, only time derivative
    i = N[0]-1;
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
    }

    i = N[0]-1;
    for (j = 1; j < n_closures; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    j = 0;
    for (i = N[0]-n_closures; i < N[0]-1; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    i = N[0]-1;
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      src[j][i][3*tcomp+2] = 0;
    }

    i = N[0]-1;
    for (j = 1; j < n_closures; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    j = 0;
    for (i = N[0]-n_closures; i < N[0]-1; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    i = N[0]-1;
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
    }

    i = N[0]-1;
    for (j = 1; j < n_closures; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
      }
    }

    j = 0;
    for (i = N[0]-n_closures; i < N[0]-1; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
      }
    }

    // Compute inner points
    for (j = 1; j < n_closures; j++) {
      for (i = N[0]-n_closures; i < N[0]-1; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) + D1.apply_2D_y_left(src,hi[1],i,j,1);
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_RC(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const PetscInt & i_ystart, const PetscInt & i_yend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j,tcomp;

    // Compute third component at boundary points, only time derivative
    i = N[0]-1;
    for (j = i_ystart; j < i_yend; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    i = N[0]-1;
    for (j = i_ystart; j < i_yend; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    i = N[0]-1;
    for (j = i_ystart; j < i_yend; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,2);
      }
    }

    // Compute inner points
    for (j = i_ystart; j < i_yend; j++) {
      for (i = N[0]-n_closures; i < N[0]-1; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) + D1.apply_2D_y_interior(src,hi[1],i,j,1);
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_LR(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j,tcomp;

    // Compute third component at boundary points, only time derivative
    i = 0;
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
    }

    i = 0;
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    j = N[1]-1;
    for (i = 1; i < n_closures; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    i = 0;
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      src[j][i][3*tcomp+2] = 0;
    }

    i = 0;
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    j = N[1]-1;
    for (i = 1; i < n_closures; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    i = 0;
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
    }

    i = 0;
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
      }
    }

    j = N[1]-1;
    for (i = 1; i < n_closures; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
      }
    }

    // Compute inner points
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (i = 1; i < n_closures; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,0) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_CR(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const PetscInt & i_xstart, const PetscInt & i_xend,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt& sw, const PetscInt& n_closures)
  {
    int i,j,tcomp;


    // Compute third component at boundary points, only time derivative
    j = N[1]-1;
    for (i = i_xstart; i < i_xend; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    j = N[1]-1;
    for (i = i_xstart; i < i_xend; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    j = N[1]-1;
    for (i = i_xstart; i < i_xend; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
      }
    }

    // Compute inner points
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (i = i_xstart; i < i_xend; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,0) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
        }
      }
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_RR(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw, const PetscInt n_closures)
  {
    int i,j,tcomp;



    // Compute third component at boundary points, only time derivative
    i = N[0]-1;
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
    }

    i = N[0]-1;
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    j = N[1]-1;
    for (i = N[0]-n_closures; i < N[0]-1; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2);
      }
    }

    // Set third comp in src to zero at boundary points
    i = N[0]-1;
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      src[j][i][3*tcomp+2] = 0;
    }

    i = N[0]-1;
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    j = N[1]-1;
    for (i = N[0]-n_closures; i < N[0]-1; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        src[j][i][3*tcomp+2] = 0;
      }
    }

    // Compute first and second component at boundary points
    i = N[0]-1;
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
    }

    i = N[0]-1;
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
      }
    }

    j = N[1]-1;
    for (i = N[0]-n_closures; i < N[0]-1; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
      }
    }

    // Compute inner points
    for (j = N[1]-n_closures; j < N[1]-1; j++) {
      for (i = N[0]-n_closures; i < N[0]-1; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
        }
      }
    }

    return 0;
  }




  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode aco_imp_apply_all(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
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
        acowave_apply_2D_LL(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures, closure_width);
        // acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures, iw);
        // acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures, iw);
        // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
        // printf("Hej\n");
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        // acowave_apply_2D_RL(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        // acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        // acowave_apply_2D_LR(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        // acowave_apply_2D_RR(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        // acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      // acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      // acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_inner(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
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
        acowave_apply_2D_LL(t, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else if (i_end[0] == N[0]) // BOTTOM RIGHT
      { 
        acowave_apply_2D_RL(t, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_end[1] == N[1]) // TOP
    {
      if (i_start[0] == 0) // TOP LEFT
      {
        acowave_apply_2D_LR(t, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
      } else if (i_end[0] == N[0]) // TOP RIGHT
      { 
        acowave_apply_2D_RR(t, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
      }
    } else if (i_start[0] == 0) // LEFT NOT BOTTOM OR TOP
    { 
      acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else if (i_end[0] == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D_outer(const PetscScalar t, const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
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
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend-sw, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 

      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        // acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, n_closures, i_yend, N, xl, hi, sw, n_closures); 
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, N[0]-n_closures, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        // acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, n_closures, i_yend-sw, N, xl, hi, sw, n_closures); 
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, n_closures, i_yend-sw, N, xl, hi, sw, n_closures); 
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xend, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        // acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend-sw, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        // acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, N[0]-n_closures, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        // acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
        // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, i_xend-sw, i_ystart,  i_ystart+sw, N, xl, hi, sw, n_closures);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      // acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart+sw, i_yend-sw, N, xl, hi, sw, n_closures);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart+sw, i_yend-sw, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, i_xend-sw, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      // acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, i_xend-sw, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
    }

    return 0;
  }


} //namespace sbp
