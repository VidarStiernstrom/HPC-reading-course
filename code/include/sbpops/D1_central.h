#pragma once

#include <tuple>
#include<petscsystypes.h>
#include "grids/grid_function.h"


namespace sbp {
  
  /**
  * Central first derivative SBP operator. The class holds the stencil weights for the interior
  * stencil and the closure stencils and contains method for applying the operator to a grid function
  * vector. The stencils of the operator are all declared at compile time, which (hopefully) should
  * allow the compiler to perform extensive optimization on the apply methods.
  **/
  template <typename Stencils, PetscInt int_width, PetscInt cls_sz, PetscInt cls_width>
  class D1_central{
  public:
    constexpr D1_central(){};
    /**
    * Convenience function returning the ranges (int_width,cls_sz,cls_width)
    **/
    inline constexpr std::tuple<PetscInt,PetscInt,PetscInt> get_ranges() const
    {
      return std::make_tuple(int_width,cls_sz,cls_width);
    };

    /**
    * Convenience function returning the size of the closure cls_sz (number of points with a closure stencil)
    **/
    inline constexpr PetscInt closure_size() const
    {
      return cls_sz;
    };

    //=============================================================================
    // 1D functions
    //=============================================================================

    /**
    * Computes the derivative in x-direction of a multicomponent 1D grid function v[i][comp] for an index i within the set of left closure points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hi    - inverse grid spacing
    *         i     - Grid index in x-direction. Index must be within the set of left closure points
    *         comp  - grid function component.
    *
    * Output: derivative v_x[i][comp]
    **/
    inline PetscScalar apply_left(const grid::grid_function_1d<PetscScalar> v, const PetscScalar hi, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<cls_width; is++)
      {
        u += static_cast<const Stencils&>(*this).closure_stencils[i][is]*v(is, comp);
      }
      return hi*u;
    };

    /**
    * Computes the derivative in x-direction of a multicomponent 1D grid function v[i][comp] for an index i within the set of interior points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hi    - inverse grid spacing
    *         i     - Grid index in x-direction. Index must be within the set of interior points
    *         comp  - grid function component.
    *
    * Output: derivative v_x[i][comp]
    **/
    inline PetscScalar apply_interior(const grid::grid_function_1d<PetscScalar> v, const PetscScalar hi, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<int_width; is++)
      {
        u += static_cast<const Stencils&>(*this).interior_stencil[is]*v(i-(int_width-1)/2+is, comp);
      }
      return hi*u;
    };

    /**
    * Computes the derivative in x-direction of a multicomponent 1D grid function v[i][comp] for an index i within the set of right closure points.
    * Input:  v     - Multicomponent 1D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hi    - inverse grid spacing
    *         i     - Grid index in x-direction. Index must be within the set of right closure points.
    *         comp  - grid function component.
    *
    * Output: derivative v_x[i][comp]
    **/
    inline PetscScalar apply_right(const grid::grid_function_1d<PetscScalar> v, const PetscScalar hi, const PetscInt i, const PetscInt comp) const
    {
      const PetscInt N = v.mapping().nx();
      PetscScalar u = 0;
      for (PetscInt is = 0; is < cls_width; is++)
      {
        u -= static_cast<const Stencils&>(*this).closure_stencils[N-i-1][cls_width-is-1]*v(N-cls_width+is, comp);
      }
      return hi*u;
    };

    //=============================================================================
    // 2D functions
    //=============================================================================
    /**
    * Computes the derivative in x-direction of a multicomponent 2D grid function v[j][i][comp] for an index i within the set of left closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hix    - inverse grid spacing
    *         i     - Grid index in x-direction. Index must be within the set of left closure points
    *         j     - Grid index in y-direction.
    *         comp  - grid function component.
    *
    * Output: derivative v_x[j][i][comp]
    **/
    inline PetscScalar apply_x_left(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hix, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<cls_width; is++)
      {
        u += static_cast<const Stencils&>(*this).closure_stencils[i][is]*v(j,is,comp);
      }
      return hix*u;
    };

    /**
    * Computes the derivative in y-direction of a multicomponent 2D grid function v[j][i][comp] for an index j within the set of left closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hiy    - inverse grid spacing
    *         i     - Grid index in x-direction.
    *         j     - Grid index in y-direction. Index must be within the set of left closure points
    *         comp  - grid function component.
    *
    * Output: derivative v_x[j][i][comp]
    **/
    inline PetscScalar apply_y_left(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hiy, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<cls_width; is++)
      {
        u += static_cast<const Stencils&>(*this).closure_stencils[j][is]*v(is,i,comp);
      }
      return hiy*u;
    };

    /**
    * Computes the derivative in x-direction of a multicomponent 2D grid function v[j][i][comp] for an index i within the set of interior points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hix    - inverse grid spacing in x-direction
    *         i     - Grid index in x-direction. Must be within the set of interior points.
    *         j     - Grid index in y-direction.
    *         comp  - grid function component.
    *
    * Output: derivative v_x[j][i][comp]
    **/
    inline PetscScalar apply_x_interior(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hix, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<int_width; is++)
      {
        u += static_cast<const Stencils&>(*this).interior_stencil[is]*v(j,i-(int_width-1)/2+is,comp);
      }
      return hix*u;
    };

    /**
    * Computes the derivative in y-direction of a multicomponent 2D grid function v[j][i][comp] for an index j within the set of interior points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hiy    - inverse grid spacing in y-direction
    *         i     - Grid index in x-direction.
    *         j     - Grid index in y-direction. Must be within the set of interior points.
    *         comp  - grid function component.
    *
    * Output: derivative v_x[j][i][comp]
    **/
    inline PetscScalar apply_y_interior(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hiy, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<int_width; is++)
      {
        u += static_cast<const Stencils&>(*this).interior_stencil[is]*v(j-(int_width-1)/2+is,i,comp);
      }
      return hiy*u;
    };

    /**
    * Computes the derivative in x-direction of a multicomponent 2D grid function v[j][i][comp] for an index i within the set of right closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hix   - inverse grid spacing
    *         Nx    - Points in x-direction
    *         i     - Grid index in x-direction. Index must be within the set of right closure points.
    *         j     - Grid index in x-direction.
    *         comp  - grid function component.
    *
    * Output: derivative v_x[j][i][comp]
    **/
    inline PetscScalar apply_x_right(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hix, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      const PetscInt Nx = v.mapping().nx();
      PetscScalar u = 0;
      for (PetscInt is = 0; is < cls_width; is++)
      {
        u -= static_cast<const Stencils&>(*this).closure_stencils[Nx-i-1][cls_width-is-1]*v(j,Nx-cls_width+is,comp);
      }
      return hix*u;
    };

    /**
    * Computes the derivative in x-direction of a multicomponent 2D grid function v[j][i][comp] for an index j within the set of right closure points.
    * Input:  v     - Multicomponent 2D grid function v (typically obtained via DMDAVecGetArrayDOF)
    *         hiy   - inverse grid spacing
    *         Ny    - Points in y-direction
    *         i     - Grid index in x-direction.
    *         j     - Grid index in y-direction. Index must be within the set of right closure points.
    *         comp  - grid function component.
    *
    * Output: derivative v_x[j][i][comp]
    **/
    inline PetscScalar apply_y_right(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hiy, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      const PetscInt Ny = v.mapping().ny();
      PetscScalar u = 0;
      for (PetscInt is = 0; is < cls_width; is++)
      {
        u -= static_cast<const Stencils&>(*this).closure_stencils[Ny-j-1][cls_width-is-1]*v(Ny-cls_width+is,i,comp);
      }
      return hiy*u;
    };

  };

  
  //=============================================================================
  // Operator definitions
  //=============================================================================

  /** 
  * Standard 2nd order central
  **/
  struct Stencils_2nd : public D1_central<Stencils_2nd,3,1,2>
  {
    static constexpr double interior_stencil[3] = {-1./2, 0, 1./2};
    static constexpr double closure_stencils[1][2] = {{-1., 1}};
  };

  /** 
  * Standard 4th order central
  **/
  struct Stencils_4th : public D1_central<Stencils_4th,5,4,6>
  {
    static constexpr double interior_stencil[5] = {1./12, -2./3, 0., 2./3, -1./12};
    static constexpr double closure_stencils[4][6] = {{-24./17, 59./34, -4./17,  -3./34, 0, 0},
                                                      {-1./2, 0, 1./2,  0, 0, 0},
                                                      {4./43, -59./86, 0, 59./86, -4./43, 0},
                                                      {3./98, 0, -59./98, 0, 32./49, -4./49}};
  };

  /** 
  * Standard 6th order central
  **/
  struct Stencils_6th : public D1_central<Stencils_6th,7,6,9>
  {
    static constexpr double interior_stencil[7] = {-1./60, 3./20, -3./4, 0, 3./4, -3./20, 1./60};
    static constexpr double closure_stencils[6][9] = {{-21600./13649, 104009./54596,  30443./81894,  -33311./27298, 16863./27298, -15025./163788, 0, 0, 0},
                                                      {-104009./240260, 0, -311./72078,  20229./24026, -24337./48052, 36661./360390, 0, 0, 0},
                                                      {-30443./162660, 311./32532, 0, -11155./16266, 41287./32532, -21999./54220, 0, 0, 0},
                                                      {33311./107180, -20229./21436, 485./1398, 0, 4147./21436, 25427./321540, 72./5359, 0, 0},
                                                      {-16863./78770, 24337./31508, -41287./47262, -4147./15754, 0, 342523./472620, -1296./7877, 144./7877, 0},
                                                      {15025./525612, -36661./262806, 21999./87602, -25427./262806, -342523./525612, 0, 32400./43801, -6480./43801, 720./43801}};
  };
} //End namespace sbp