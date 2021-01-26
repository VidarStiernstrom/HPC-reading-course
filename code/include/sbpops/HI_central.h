#pragma once

#include<petscsystypes.h>
#include "grids/grid_function.h"
#include <tuple>

namespace sbp {
  
  /**
  * Inverse of norm (mass-matrix) for central D1 SBP operator. The class holds the inverse of quadrature 
  * weights for the boundary closures and contains methods for applying the operator to a grid function
  * vector. 
  **/
  template <typename InverseQuadrature, PetscInt n_closures>
  class HI_central{
  public:
    constexpr HI_central(){};
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
    * Returns HI_ii * v_i for i within left closure points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         i     - Grid index. Index must be within the set of left closure points.
    *         comp  - Grid function component.
    *
    * Output: HI[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_left(const grid::grid_function_1d<PetscScalar> v, const PetscScalar hi, const PetscInt i, const PetscInt comp) const
    {
      return hi*static_cast<const InverseQuadrature&>(*this).closure_invquad[i]*v(i,comp);
    };

    /**
    * Returns HI_ii * v_i for i within right closure points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         h     - Grid spacing
    *         N     - Number of global grid points.
    *         i     - Grid index. Index must be within the set of right closure points.
    *         comp  - Grid function component.
    *
    * Output: HI[i][i]*v[i][comp]
    **/
    inline PetscScalar apply_right(const grid::grid_function_1d<PetscScalar> v, const PetscScalar hi, const PetscInt N, const PetscInt i, const PetscInt comp) const
    {
      return hi*static_cast<const InverseQuadrature&>(*this).closure_invquad[N-i-1]*v(i,comp);
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
    inline PetscScalar apply_2D_x_left(PetscScalar ***v, const PetscScalar hi, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return hi*static_cast<const InverseQuadrature&>(*this).closure_invquad[i]*v[j][i][comp];
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
    inline PetscScalar apply_2D_y_left(PetscScalar ***v, const PetscScalar hi, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return hi*static_cast<const InverseQuadrature&>(*this).closure_invquad[j]*v[j][i][comp];
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
    inline PetscScalar apply_2D_x_right(PetscScalar ***v, const PetscScalar hi, const PetscInt N, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return hi*static_cast<const InverseQuadrature&>(*this).closure_invquad[N-i-1]*v[j][i][comp];
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
    inline PetscScalar apply_2D_y_right(PetscScalar ***v, const PetscScalar hi, const PetscInt N, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      return hi*static_cast<const InverseQuadrature&>(*this).closure_invquad[N-j-1]*v[j][i][comp];
    };
  };

  
  //=============================================================================
  // Operator definitions
  //=============================================================================

  /** 
  * Standard 2nd order central
  **/
  struct InverseQuadrature_2nd : public HI_central<InverseQuadrature_2nd,1>
  {
    static constexpr double closure_invquad[1] = {2.};
  };

  /** 
  * Standard 4th order central
  **/
  struct InverseQuadrature_4th : public HI_central<InverseQuadrature_4th,4>
  {
    static constexpr double closure_invquad[4] = {48./17, 48./59, 48./43, 48./49};
  };

  /** 
  * Standard 6th order central
  **/
  struct InverseQuadrature_6th : public HI_central<InverseQuadrature_6th,6>
  {
    static constexpr double closure_invquad[6] = {43200./13649, 8640./12013, 4320./2711, 4320./5359, 8640./7877, 43200./43801};
  };
} //End namespace sbp