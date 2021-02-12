#pragma once

#include<petscsystypes.h>
#include "grids/grid_function.h"
#include <tuple>

namespace sbp {
  
  /**
  * Central first derivative SBP operator. The class holds the stencil weights for the interior
  * stencil and the closure stencils and contains method for applying the operator to a grid function
  * vector. The stencils of the operator are all declared at compile time, which (hopefully) should
  * allow the compiler to perform extensive optimization on the apply methods.
  **/
  template <typename Interp, PetscInt F2C_iw, PetscInt F2C_nc, PetscInt F2C_cw, PetscInt C2F_odd_iw, PetscInt C2F_even_iw, PetscInt C2F_nc, PetscInt C2F_cw>
  class Int_central{
  public:
    constexpr Int_central(){};
    /**
    * Convenience function returning the ranges (interior_width,n_closures,closure_width)
    **/
    inline constexpr std::tuple<PetscInt,PetscInt> get_ranges() const
    {
      return std::make_tuple(F2C_nc,C2F_nc);
    };

    //TODO: The pointer to pointer layout may prevent compiler optimization for e.g vectorization since it is not clear whether
    //      the memory is contiguous or not. We should switch to a flat array layout, once we get something running for systems in 2D.
    
    //=============================================================================
    // 1D functions
    //=============================================================================

    inline PetscScalar F2C_apply_left(PetscScalar **v, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < F2C_cw; is++)
      {
        u += static_cast<const Interp&>(*this).F2C_closure_stencils[i][is]*v[is][comp];
      }
      return u;
    };

    inline PetscScalar C2F_apply_left(PetscScalar **v, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++)
      {
        u += static_cast<const Interp&>(*this).C2F_closure_stencils[i][is]*v[is][comp];
      }
      return u;
    };

    inline PetscScalar F2C_apply_interior(PetscScalar **v, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is<F2C_iw; is++)
      {
        u += static_cast<const Interp&>(*this).F2C_interior_stencil[is]*v[i+is+i-(F2C_iw-1)/2][comp];
      }
      return u;
    };

    inline PetscScalar C2F_odd_apply_interior(PetscScalar **v, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      PetscInt icoar;
      for (PetscInt is = 0; is<C2F_odd_iw; is++)
      {
        icoar = (i-1)/2 + is;
        u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[is]*v[icoar][comp];
      }
      return u;
    };

    inline PetscScalar C2F_even_apply_interior(PetscScalar **v, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      PetscInt icoar;
      for (PetscInt is = 0; is<C2F_even_iw; is++)
      {
        icoar = i/2;
        u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[is]*v[icoar][comp];
      }
      return u;
    };

    inline PetscScalar F2C_apply_right(PetscScalar **v, const PetscInt N, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      PetscInt ifine;
      for (PetscInt is = 0; is < F2C_cw; is++)
      {
        ifine = N+i-F2C_cw+is;
        u += static_cast<const Interp&>(*this).F2C_closure_stencils[N-i-1+is][F2C_cw-is-1]*v[ifine][comp];
      }
      
      return u;
    };

    inline PetscScalar C2F_apply_right(PetscScalar **v, const PetscInt N, const PetscInt i, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++)
      {
        u += static_cast<const Interp&>(*this).C2F_closure_stencils[N-i-1][C2F_cw-is-1]*v[(N-1)/2+1-C2F_cw+is][comp];
      }
      return u;
    };

    //=============================================================================
    // 2D functions
    //=============================================================================
    inline PetscScalar F2C_apply_2D_LL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < F2C_cw; is++) {
        for (PetscInt js = 0; js < F2C_cw; js++) {
          u += static_cast<const Interp&>(*this).F2C_closure_stencils[0][is]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][js]*v[j+js][i+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_CL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt js = 0; js < F2C_cw; js++) {
        for (PetscInt is = 0; is < F2C_iw; is++) {
          u += v[j+js][2*i+is-(F2C_iw-1)/2][comp]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][js]*static_cast<const Interp&>(*this).F2C_interior_stencil[is];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_LC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt js = 0; js < F2C_iw; js++) {
        for (PetscInt is = 0; is < F2C_cw; is++) {
          u += v[2*j+js-(F2C_iw-1)/2][i+is][comp]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][is]*static_cast<const Interp&>(*this).F2C_interior_stencil[js];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_CC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt js = 0; js < F2C_iw; js++) {
        for (PetscInt is = 0; is < F2C_iw; is++) {
          u += v[2*j+js-(F2C_iw-1)/2][2*i+is-(F2C_iw-1)/2][comp]*static_cast<const Interp&>(*this).F2C_interior_stencil[is]*static_cast<const Interp&>(*this).F2C_interior_stencil[js];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_RL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < F2C_cw; is++) {
        for (PetscInt js = 0; js < F2C_cw; js++) {
          u += static_cast<const Interp&>(*this).F2C_closure_stencils[0][js]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][F2C_cw-is-1]*v[j+js][2*i+1-F2C_cw+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_RC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt js = 0; js < F2C_iw; js++) {
        for (PetscInt is = 0; is < F2C_cw; is++) {
          u += static_cast<const Interp&>(*this).F2C_interior_stencil[js]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][F2C_cw-is-1]*v[2*j+js-(F2C_iw-1)/2][2*i+1-F2C_cw+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_LR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < F2C_cw; is++) {
        for (PetscInt js = 0; js < F2C_cw; js++) {
          u += static_cast<const Interp&>(*this).F2C_closure_stencils[0][is]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][F2C_cw-js-1]*v[2*j+1-F2C_cw+js][i+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_CR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < F2C_iw; is++) {
        for (PetscInt js = 0; js < F2C_cw; js++) {
          u += static_cast<const Interp&>(*this).F2C_interior_stencil[is]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][F2C_cw-js-1]*v[2*j+1-F2C_cw+js][2*i+is-(F2C_iw-1)/2][comp];
        }
      }
      return u;
    };

    inline PetscScalar F2C_apply_2D_RR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < F2C_cw; is++) {
        for (PetscInt js = 0; js < F2C_cw; js++) {
          u += static_cast<const Interp&>(*this).F2C_closure_stencils[0][F2C_cw-is-1]*static_cast<const Interp&>(*this).F2C_closure_stencils[0][F2C_cw-js-1]*v[2*j+1-F2C_cw+js][2*i+1-F2C_cw+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_apply_2D_LL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_closure_stencils[0][is]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][js]*v[j+js][i+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_even_apply_2D_CL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_even_iw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[is]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][js]*v[j+js][i/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_odd_apply_2D_CL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_odd_iw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[is]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][js]*v[j+js][(i-1)/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_even_apply_2D_LC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_even_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[js]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][is]*v[j/2+js][i+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_odd_apply_2D_LC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_odd_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[js]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][is]*v[(j-1)/2+js][i+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_even_even_apply_2D_CC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_even_iw; is++) {
        for (PetscInt js = 0; js < C2F_even_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[js]*static_cast<const Interp&>(*this).C2F_interior_stencil_even[is]*v[j/2+js][i/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_even_odd_apply_2D_CC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_even_iw; is++) {
        for (PetscInt js = 0; js < C2F_odd_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[js]*static_cast<const Interp&>(*this).C2F_interior_stencil_even[is]*v[(j-1)/2+js][i/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_odd_even_apply_2D_CC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_odd_iw; is++) {
        for (PetscInt js = 0; js < C2F_even_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[js]*static_cast<const Interp&>(*this).C2F_interior_stencil_odd[is]*v[j/2+js][(i-1)/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_odd_odd_apply_2D_CC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_odd_iw; is++) {
        for (PetscInt js = 0; js < C2F_odd_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[js]*static_cast<const Interp&>(*this).C2F_interior_stencil_odd[is]*v[(j-1)/2+js][(i-1)/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_apply_2D_RL(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-is-1]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][js]*v[j+js][i/2+1-C2F_cw+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_apply_2D_LR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_closure_stencils[0][is]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-js-1]*v[j/2+1-C2F_cw+js][i+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_even_apply_2D_RC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_even_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[js]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-is-1]*v[j/2+js][i/2+1-C2F_cw+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_odd_apply_2D_RC(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_odd_iw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[js]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-is-1]*v[(j-1)/2+js][i/2+1-C2F_cw+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_even_apply_2D_CR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_even_iw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_even[is]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-js-1]*v[j/2+1-C2F_cw+js][i/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_odd_apply_2D_CR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_odd_iw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_interior_stencil_odd[is]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-js-1]*v[j/2+1-C2F_cw+js][(i-1)/2+is][comp];
        }
      }
      return u;
    };

    inline PetscScalar C2F_apply_2D_RR(PetscScalar ***v, const PetscInt i, const PetscInt j, const PetscInt comp) const
    {
      PetscScalar u = 0;
      for (PetscInt is = 0; is < C2F_cw; is++) {
        for (PetscInt js = 0; js < C2F_cw; js++) {
          u += static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-is-1]*static_cast<const Interp&>(*this).C2F_closure_stencils[0][C2F_cw-js-1]*v[j/2+1-C2F_cw+js][i/2+1-C2F_cw+is][comp];
        }
      }
      return u;
    };
  };
  
  //=============================================================================
  // Operator definitions
  //=============================================================================

  // FOR HIGHER ORDER: FIX CLOSURE STENCILS FIRST ARGUMENT, CHECK D1
  /** 
  * Standard 2nd order central
  **/
  struct Interp_2nd : public Int_central<Interp_2nd,3,1,2,2,1,1,1>  // F2C_iw,  F2C_nc,  F2C_cw,  C2F_1_iw,  C2F_2_iw,  C2F_nc,  C2F_cw
  {
    static constexpr double C2F_closure_stencils[1][1] = {{1}};
    static constexpr double F2C_closure_stencils[1][2] = {{1./2,1./2}};
    static constexpr double F2C_interior_stencil[3] = {1./4, 1./2, 1./4};
    static constexpr double C2F_interior_stencil_odd[2] = {1./2,1./2};
    static constexpr double C2F_interior_stencil_even[1] = {1};
  };
  /** 
  * Standard 4th order central
  **/
  // struct Stencils_4th : public Int_central<Stencils_4th,5,4,6>
  // {
  // };

  /** 
  * Standard 6th order central
  **/
  // struct Stencils_6th : public Int_central<Stencils_6th,7,6,9>
  // {
  // };

} //End namespace sbp