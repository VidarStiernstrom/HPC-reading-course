#pragma once

#include<petscsystypes.h>
#include "grids/grid_function.h"
#include <tuple>

namespace sbp {
  
  /**
  * Norm (mass-matrix) for central D1 SBP operator. The class holds the quadrature weights for
  * the boundary closures and contains methods for applying the operator to a grid function
  * vector. 
  **/
  template <typename Quadrature, PetscInt n_closures>
  class H_central{
  public:
    constexpr H_central(){};
    /**
    * Convenience function returning the ranges number of boundary closures n_closures.
    **/
    inline constexpr PetscInt get_n_closures() const
    {
      return n_closures;
    };

    //=============================================================================
    // 1D functions
    //=============================================================================

    /**
    * Returns H_ii * v_i for i within left closure points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         i     - Grid index. Index must be within the set of left closure points.
    *         comp  - Grid function component.
    *
    * Output: H[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_left(const PetscScalar *const *const v, const PetscScalar h, const PetscInt i, const PetscInt comp) const
    {
      return h*static_cast<const Quadrature&>(*this).closure_quad[i]*v[i][comp];
    };

    /**
    * Returns H_ii * v_i for i within right closure points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         N     - Number of global grid points.
    *         i     - Grid index. Index must be within the set of right closure points.
    *         comp  - Grid function component.
    *
    * Output: H[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_right(const PetscScalar *const *const v, const PetscScalar h, const PetscInt N, const PetscInt i, const PetscInt comp) const
    {
      return h*static_cast<const Quadrature&>(*this).closure_quad[N-i-1]*v[i][comp];
    };

    // *
    // * Computes the H-norm of each component of v and adds them together.
    // * Input:  v       - Multicomponent 1D grid function (typically obtained via DMDAVecGetArrayDOF).
    // *         h       - grid spacing.
    // *         N       - number of global grid points.
    // *         i_start - current processors start index.
    // *         i_end   - current processors end index.
    // *         dofs    - number of degrees of freedom of v.
    // *
    // * Output: computed H-norm, sqrt(v1'*H*v1 + v2'*H*v2 + ...).
    // *
    inline PetscScalar get_norm_1D(const PetscScalar *const *const v, const PetscScalar h, const PetscInt N, const PetscInt i_start, const PetscInt i_end, const PetscInt dofs) const
    {
      PetscInt is, comp;
      PetscReal u = 0, sum;

      for (comp = 0; comp < dofs; comp++) { // for each degree of freedom
        if (i_start == 0) { // Left block
          for (is = n_closures; is < i_end; is++) {
            u += h*v[is][comp]*v[is][comp];  
          }
          for (is = 0; is < n_closures; is++) {
            u += apply_left(v, h, is, comp)*v[is][comp];
          }
        } else if(i_end == N) { // Right block
          for (is = i_start; is < N-n_closures; is++) {
            u += h*v[is][comp]*v[is][comp];  
          }
          for (is = N-n_closures; is < N; is++) {
            u += apply_right(v, h, N, is, comp)*v[is][comp];
          }
        } else { // center block
          for (is = i_start; is < i_end; is++) {
            u += h*v[is][comp]*v[is][comp];  
          }
        }
      }

      // Sum all blocks
      MPIU_Allreduce(&u, &sum, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
      sum = PetscSqrtReal(sum);

      return sum;
    };

    //=============================================================================
    // 2D functions
    //=============================================================================
    /**
    * Returns HI_ii * v_ij for i within left closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         i     - Grid index. Index must be within the set of left closure points.
    *         comp  - Grid function component.
    *
    * Output: HI[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_2D_x_left(const PetscScalar *const *const *const v, const PetscScalar h, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return h*static_cast<const Quadrature&>(*this).closure_quad[i]*v[j][i][comp];
    };

    /**
    * Returns HI_jj * v_ij for j within left closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         j     - Grid index. Index must be within the set of left closure points.
    *         comp  - Grid function component.
    *
    * Output: HI[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_2D_y_left(const PetscScalar *const *const *const v, const PetscScalar h, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return h*static_cast<const Quadrature&>(*this).closure_quad[j]*v[j][i][comp];
    };

    /**
    * Returns HI_ii * v_i for i within right closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         N     - Number of global grid points.
    *         i     - Grid index. Index must be within the set of right closure points.
    *         comp  - Grid function component.
    *
    * Output: HI[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_2D_x_right(const PetscScalar *const *const *const v, const PetscScalar h, const PetscInt N, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return h*static_cast<const Quadrature&>(*this).closure_quad[N-i-1]*v[j][i][comp];
    };

    /**
    * Returns HI_ii * v_i for i within right closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         N     - Number of global grid points.
    *         i     - Grid index. Index must be within the set of right closure points.
    *         comp  - Grid function component.
    *
    * Output: HI[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_2D_y_right(const PetscScalar *const *const *const v, const PetscScalar h, const PetscInt N, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return h*static_cast<const Quadrature&>(*this).closure_quad[N-j-1]*v[j][i][comp];
    };

    // *
    // * Computes the H-norm of v. 
    // * Input:  v       - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF).
    // *         h       - grid spacing.
    // *         N       - number of global grid points.
    // *         i_start - current processors start index.
    // *         i_end   - current processors end index.
    // *         comp    - grid function component.
    // *
    // * Output: computed H-norm, sqrt(v'*H*v).
    // *
    inline PetscScalar get_norm_2D(const PetscScalar *const *const *const v, 
      const std::array<PetscInt,2>& i_start, std::array<PetscInt,2>& i_end, 
      const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& h, const PetscInt comp) const
    {
      // TODO!
      return 0;
    };
  };

  //=============================================================================
  // Operator definitions
  //=============================================================================

  /** 
  * Standard 2nd order central
  **/
  struct Quadrature_2nd : public H_central<Quadrature_2nd,1>
  {
    static constexpr double closure_quad[1] = {1./2};
  };

  /** 
  * Standard 4th order central
  **/
  struct Quadrature_4th : public H_central<Quadrature_4th,4>
  {
    static constexpr double closure_quad[4] = {17./48, 59./48, 43./48, 49./48};
  };

  /** 
  * Standard 6th order central
  **/
  struct Quadrature_6th : public H_central<Quadrature_6th,6>
  {
    static constexpr double closure_quad[6] = {13649./43200, 12013./8640, 2711./4320, 5359./4320, 7877./8640, 43801./43200};
  };
} //End namespace sbp