#pragma once

#include<petscsystypes.h>
#include <array>
#include "grids/grid_function.h"

// TODO: remove conditionals in loops

namespace sbp{

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_LL(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;
    
    for (jdst = 0; jdst < F2C_nc; jdst++) { 
      for (idst = 0; idst < F2C_nc; idst++) { 
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_LL(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_LL(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;
    
    for (jdst = 0; jdst < C2F_nc; jdst++) { 
      for (idst = 0; idst < C2F_nc; idst++) { 
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.C2F_apply_2D_LL(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_CL(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_xstart, const PetscInt & i_xend, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = 0; jdst < F2C_nc; jdst++) { 
      for (idst = i_xstart; idst < i_xend; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_CL(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_CL(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_xstart, const PetscInt & i_xend, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = 0; jdst < C2F_nc; jdst++) { 
      for (idst = i_xstart; idst < i_xend; idst++) {
        if (idst % 2 == 0) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_even_apply_2D_CL(src, idst, jdst, dof);
          }
        } else {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_odd_apply_2D_CL(src, idst, jdst, dof);
          }
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_LC(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_ystart, const PetscInt & i_yend, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {

    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = i_ystart; jdst < i_yend; jdst++) {
      for (idst = 0; idst < F2C_nc; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_LC(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_LC(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_ystart, const PetscInt & i_yend, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = i_ystart; jdst < i_yend; jdst++) {
      if (jdst % 2 == 0) {
        for (idst = 0; idst < C2F_nc; idst++) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_even_apply_2D_LC(src, idst, jdst, dof);
          }
        }
      } else {
        for (idst = 0; idst < C2F_nc; idst++) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_odd_apply_2D_LC(src, idst, jdst, dof);
          }
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_CC(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_xstart, const PetscInt & i_xend, const PetscInt & i_ystart, const PetscInt & i_yend, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = i_ystart; jdst < i_yend; jdst++) {
        for (idst = i_xstart; idst < i_xend; idst++) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.F2C_apply_2D_CC(src, idst, jdst, dof);
          }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_CC(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_xstart, const PetscInt & i_xend, const PetscInt & i_ystart, const PetscInt & i_yend, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = i_ystart; jdst < i_yend; jdst++) {
      if (jdst % 2 == 0) {                                                                  // j even
        for (idst = i_xstart; idst < i_xend; idst++) {
          if (idst % 2 == 0) {                                                              // i even
              for (dof = 0; dof < ndofs; dof++) {
                dst[jdst][idst][dof] = ICF.C2F_even_even_apply_2D_CC(src, idst, jdst, dof);
              }
          } else {                                                                          // i odd        
            for (dof = 0; dof < ndofs; dof++) {
              dst[jdst][idst][dof] = ICF.C2F_odd_even_apply_2D_CC(src, idst, jdst, dof);
            }
          }
        }
      } else {                                                                              // j odd
        for (idst = i_xstart; idst < i_xend; idst++) {
          if (idst % 2 == 0) {                                                              // i even
              for (dof = 0; dof < ndofs; dof++) {
                dst[jdst][idst][dof] = ICF.C2F_even_odd_apply_2D_CC(src, idst, jdst, dof);
              }
          } else {                                                                          // i odd        
            for (dof = 0; dof < ndofs; dof++) {
              dst[jdst][idst][dof] = ICF.C2F_odd_odd_apply_2D_CC(src, idst, jdst, dof);
            }
          }
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_RL(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = 0; jdst < F2C_nc; jdst++) { 
      for (idst = N[0]-F2C_nc; idst < N[0]; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_RL(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_RL(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = 0; jdst < C2F_nc; jdst++) { 
      for (idst = N[0]-C2F_nc; idst < N[0]; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.C2F_apply_2D_RL(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_RC(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_ystart, const PetscInt & i_yend, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;
    
    for (jdst = i_ystart; jdst < i_yend; jdst++) {
      for (idst = N[0]-F2C_nc; idst < N[0]; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_RC(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_RC(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_ystart, const PetscInt & i_yend, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = i_ystart; jdst < i_yend; jdst++) {
      if (jdst % 2 == 0) {
        for (idst = N[0]-C2F_nc; idst < N[0]; idst++) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_even_apply_2D_RC(src, idst, jdst, dof);
          }
        }
      } else {
        for (idst = N[0]-C2F_nc; idst < N[0]; idst++) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_odd_apply_2D_RC(src, idst, jdst, dof);
          }
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_LR(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = N[1]-F2C_nc; jdst < N[1]; jdst++) {
      for (idst = 0; idst < F2C_nc; idst++) { 
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_LR(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_LR(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = N[1]-C2F_nc; jdst < N[1]; jdst++) {
      for (idst = 0; idst < C2F_nc; idst++) { 
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.C2F_apply_2D_LR(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_CR(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_xstart, const PetscInt & i_xend, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = N[1]-F2C_nc; jdst < N[1]; jdst++) {
      for (idst = i_xstart; idst < i_xend; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_CR(src, idst, jdst, dof);
        }
      }
    }
    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_CR(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const PetscInt & i_xstart, const PetscInt & i_xend, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = N[1]-C2F_nc; jdst < N[1]; jdst++) {
      for (idst = i_xstart; idst < i_xend; idst++) {
        if (idst % 2 == 0) {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_even_apply_2D_CR(src, idst, jdst, dof);
          }
        } else {
          for (dof = 0; dof < ndofs; dof++) {
            dst[jdst][idst][dof] = ICF.C2F_odd_apply_2D_CR(src, idst, jdst, dof);
          }
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode F2C_2D_RR(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt F2C_nc)
  {

    int idst,jdst,dof;
    const PetscInt ndofs = 12;
    
    for (jdst = N[1]-F2C_nc; jdst < N[1]; jdst++) {
      for (idst = N[0]-F2C_nc; idst < N[0]; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.F2C_apply_2D_RR(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode C2F_2D_RR(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& N, const PetscInt C2F_nc)
  {
    int idst,jdst,dof;
    const PetscInt ndofs = 12;

    for (jdst = N[1]-C2F_nc; jdst < N[1]; jdst++) {
      for (idst = N[0]-C2F_nc; idst < N[0]; idst++) {
        for (dof = 0; dof < ndofs; dof++) {
          dst[jdst][idst][dof] = ICF.C2F_apply_2D_RR(src, idst, jdst, dof);
        }
      }
    }

    return 0;
  }
 
  template <class SbpInterpolator>
  inline PetscErrorCode apply_F2C(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end, const std::array<PetscInt,2>& N)
  {
    const PetscInt i_xstart = i_start[0]; 
    const PetscInt i_ystart = i_start[1];
    const PetscInt i_xend = i_end[0];
    const PetscInt i_yend = i_end[1];
    const auto [F2C_nc, C2F_nc] = ICF.get_ranges();

    if (i_ystart == 0)  // BOTTOM
    {
      if (i_xstart == 0) // BOTTOM LEFT
      {
        F2C_2D_LL(ICF, src, dst, N, F2C_nc);
        F2C_2D_CL(ICF, src, dst, F2C_nc, i_xend, N, F2C_nc);
        F2C_2D_LC(ICF, src, dst, F2C_nc, i_yend, N, F2C_nc);
        F2C_2D_CC(ICF, src, dst, F2C_nc, i_xend, F2C_nc, i_yend, N, F2C_nc); 
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        F2C_2D_RL(ICF, src, dst, N, F2C_nc);
        F2C_2D_CL(ICF, src, dst, i_xstart, N[0]-F2C_nc, N, F2C_nc);
        F2C_2D_RC(ICF, src, dst, F2C_nc, i_yend, N, F2C_nc);
        F2C_2D_CC(ICF, src, dst, i_xstart, N[0]-F2C_nc, F2C_nc, i_yend, N, F2C_nc); 
      } else // BOTTOM CENTER
      { 
        F2C_2D_CL(ICF, src, dst, i_xstart, i_xend, N, F2C_nc);
        F2C_2D_CC(ICF, src, dst, i_xstart, i_xend, F2C_nc, i_yend, N, F2C_nc); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        F2C_2D_LR(ICF, src, dst, N, F2C_nc);
        F2C_2D_CR(ICF, src, dst, F2C_nc, i_xend, N, F2C_nc);
        F2C_2D_LC(ICF, src, dst, i_ystart, N[1] - F2C_nc, N, F2C_nc);
        F2C_2D_CC(ICF, src, dst, F2C_nc, i_xend, i_ystart, N[1]-F2C_nc, N, F2C_nc);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        F2C_2D_RR(ICF, src, dst, N, F2C_nc);
        F2C_2D_CR(ICF, src, dst, i_xstart, N[0]-F2C_nc, N, F2C_nc);
        F2C_2D_RC(ICF, src, dst, i_ystart, N[1] - F2C_nc, N, F2C_nc);
        F2C_2D_CC(ICF, src, dst, i_xstart, N[0]-F2C_nc, i_ystart, N[1] - F2C_nc, N, F2C_nc);
      } else // TOP CENTER
      { 
        F2C_2D_CR(ICF, src, dst, i_xstart, i_xend, N, F2C_nc);
        F2C_2D_CC(ICF, src, dst, i_xstart, i_xend, i_ystart,  N[1] - F2C_nc, N, F2C_nc);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      F2C_2D_LC(ICF, src, dst, i_ystart, i_yend, N, F2C_nc);
      F2C_2D_CC(ICF, src, dst, F2C_nc, i_xend, i_ystart, i_yend, N, F2C_nc);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      F2C_2D_RC(ICF, src, dst, i_ystart, i_yend, N, F2C_nc);
      F2C_2D_CC(ICF, src, dst, i_xstart, N[0] - F2C_nc, i_ystart, i_yend, N, F2C_nc);
    } else // CENTER
    {
      F2C_2D_CC(ICF, src, dst, i_xstart, i_xend, i_ystart, i_yend, N, F2C_nc);
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode apply_F2C_1p(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end, const std::array<PetscInt,2>& N)
  {
    const auto [F2C_nc, C2F_nc] = ICF.get_ranges();

    F2C_2D_LL(ICF, src, dst, N, F2C_nc);
    F2C_2D_RL(ICF, src, dst, N, F2C_nc);
    F2C_2D_LR(ICF, src, dst, N, F2C_nc);
    F2C_2D_RR(ICF, src, dst, N, F2C_nc);

    F2C_2D_CL(ICF, src, dst, F2C_nc, N[0]-F2C_nc, N, F2C_nc);
    F2C_2D_LC(ICF, src, dst, F2C_nc, N[1]-F2C_nc, N, F2C_nc);
    F2C_2D_CR(ICF, src, dst, F2C_nc, N[0]-F2C_nc, N, F2C_nc);
    F2C_2D_RC(ICF, src, dst, F2C_nc, N[1]-F2C_nc, N, F2C_nc);

    F2C_2D_CC(ICF, src, dst, F2C_nc, N[0]-F2C_nc, F2C_nc, N[1]-F2C_nc, N, F2C_nc); 

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode apply_C2F(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end, const std::array<PetscInt,2>& N)
  {
    const PetscInt i_xstart = i_start[0]; 
    const PetscInt i_ystart = i_start[1];
    const PetscInt i_xend = i_end[0];
    const PetscInt i_yend = i_end[1];
    const auto [F2C_nc, C2F_nc] = ICF.get_ranges();

    if (i_ystart == 0)  // BOTTOM
    {
      if (i_xstart == 0) // BOTTOM LEFT
      {
        C2F_2D_LL(ICF, src, dst, N, C2F_nc);
        C2F_2D_CL(ICF, src, dst, C2F_nc, i_xend, N, C2F_nc);
        C2F_2D_LC(ICF, src, dst, C2F_nc, i_yend, N, C2F_nc);
        C2F_2D_CC(ICF, src, dst, C2F_nc, i_xend, C2F_nc, i_yend, N, C2F_nc); 
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        C2F_2D_RL(ICF, src, dst, N, C2F_nc);
        C2F_2D_CL(ICF, src, dst, i_xstart, N[0]-C2F_nc, N, C2F_nc);
        C2F_2D_RC(ICF, src, dst, C2F_nc, i_yend, N, C2F_nc);
        C2F_2D_CC(ICF, src, dst, i_xstart, N[0]-C2F_nc, C2F_nc, i_yend, N, C2F_nc); 
      } else // BOTTOM CENTER
      { 
        C2F_2D_CL(ICF, src, dst, i_xstart, i_xend, N, C2F_nc);
        C2F_2D_CC(ICF, src, dst, i_xstart, i_xend, C2F_nc, i_yend, N, C2F_nc); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        C2F_2D_LR(ICF, src, dst, N, C2F_nc);
        C2F_2D_CR(ICF, src, dst, C2F_nc, i_xend, N, C2F_nc);
        C2F_2D_LC(ICF, src, dst, i_ystart, N[1] - C2F_nc, N, C2F_nc);
        C2F_2D_CC(ICF, src, dst, C2F_nc, i_xend, i_ystart, N[1]-C2F_nc, N, C2F_nc);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        C2F_2D_RR(ICF, src, dst, N, C2F_nc);
        C2F_2D_CR(ICF, src, dst, i_xstart, N[0]-C2F_nc, N, C2F_nc);
        C2F_2D_RC(ICF, src, dst, i_ystart, N[1] - C2F_nc, N, C2F_nc);
        C2F_2D_CC(ICF, src, dst, i_xstart, N[0]-C2F_nc, i_ystart, N[1] - C2F_nc, N, C2F_nc);
      } else // TOP CENTER
      { 
        C2F_2D_CR(ICF, src, dst, i_xstart, i_xend, N, C2F_nc);
        C2F_2D_CC(ICF, src, dst, i_xstart, i_xend, i_ystart,  N[1] - C2F_nc, N, C2F_nc);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      C2F_2D_LC(ICF, src, dst, i_ystart, i_yend, N, C2F_nc);
      C2F_2D_CC(ICF, src, dst, C2F_nc, i_xend, i_ystart, i_yend, N, C2F_nc);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      C2F_2D_RC(ICF, src, dst, i_ystart, i_yend, N, C2F_nc);
      C2F_2D_CC(ICF, src, dst, i_xstart, N[0] - C2F_nc, i_ystart, i_yend, N, C2F_nc);
    } else // CENTER
    {
      C2F_2D_CC(ICF, src, dst, i_xstart, i_xend, i_ystart, i_yend, N, C2F_nc);
    }

    return 0;
  }

  template <class SbpInterpolator>
  inline PetscErrorCode apply_C2F_1p(const SbpInterpolator& ICF, PetscScalar ***src, PetscScalar ***dst, const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end, const std::array<PetscInt,2>& N)
  {
    const auto [F2C_nc, C2F_nc] = ICF.get_ranges();

    C2F_2D_LL(ICF, src, dst, N, C2F_nc);
    C2F_2D_RL(ICF, src, dst, N, C2F_nc);
    C2F_2D_LR(ICF, src, dst, N, C2F_nc);
    C2F_2D_RR(ICF, src, dst, N, C2F_nc);

    C2F_2D_CL(ICF, src, dst, C2F_nc, N[0]-C2F_nc, N, C2F_nc);
    C2F_2D_LC(ICF, src, dst, C2F_nc, N[1]-C2F_nc, N, C2F_nc);
    C2F_2D_CR(ICF, src, dst, C2F_nc, N[0]-C2F_nc, N, C2F_nc);
    C2F_2D_RC(ICF, src, dst, C2F_nc, N[1]-C2F_nc, N, C2F_nc);

    C2F_2D_CC(ICF, src, dst, C2F_nc, N[0]-C2F_nc, C2F_nc, N[1]-C2F_nc, N, C2F_nc); 

    return 0;
  }

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
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw, const PetscInt n_closures)
  {
    int i,j,tcomp;

    // Set dst on unaffected points
    for (j = 1; j < n_closures; j++) { 
      for (i = 1; i < n_closures; i++) { 
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);  
        }
      }
    }

    // Set dst on affected points
    i = 0; 
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp]   = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2) + HI.apply_2D_x_left(src, hi[0], i, j, 3*tcomp+2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2) + HI.apply_2D_y_left(src, hi[1], i, j, 3*tcomp+2);
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
    }

    i = 0;
    for (j = 1; j < n_closures; j++)
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2) + HI.apply_2D_x_left(src, hi[0], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
      }
    }

    j = 0;
    for (i = 1; i < n_closures; i++) 
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2) + HI.apply_2D_y_left(src, hi[1], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
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

    // Set dst on unaffected points
    for (j = 1; j < n_closures; j++) { 
      for (i = i_xstart; i < i_xend; i++) {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }

    // Set dst on affected points
    j = 0;
    for (i = i_xstart; i < i_xend; i++) {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2) + HI.apply_2D_y_left(src, hi[1], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
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

    // Set dst on unaffected points
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = 1; i < n_closures; i++)
      {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }

    // Set dst on affected points
    i = 0;
    for (j = i_ystart; j < i_yend; j++) {
      for (tcomp = 0; tcomp < 4; tcomp++) { 
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2) + HI.apply_2D_x_left(src, hi[0], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+1);;
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

     // Set dst on unaffected points
    for (j = 1; j < n_closures; j++)
    { 
      for (i = N[0]-n_closures; i < N[0]-1; i++) 
      { 
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }

    // Set dst on affected points
    i = N[0]-1; 
    j = 0;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2) - HI.apply_2D_x_right(src, hi[0], N[0], i, j, 3*tcomp+2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2) + HI.apply_2D_y_left(src, hi[1], i, j, 3*tcomp+2);
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
    }

    i = N[0]-1;
    for (j = 1; j < n_closures; j++)
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2) - HI.apply_2D_x_right(src, hi[0], N[0], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
      }
    }

    j = 0;
    for (i = N[0]-n_closures; i < N[0]-1; i++) 
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+2) + HI.apply_2D_y_left(src, hi[1], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_left(src,hi[1],i,j,3*tcomp+1);
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

    // Set dst on unaffected points
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = N[0]-n_closures; i < N[0]-1; i++) 
      {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+1);
        }
      }
    }

    // Set dst on affected points
    i = N[0]-1;
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2) - HI.apply_2D_x_right(src, hi[0], N[0], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_interior(src,hi[1],i,j,3*tcomp+1);
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

    // Set dst on unaffected points
    for (j = N[1]-n_closures; j < N[1]-1; j++) 
    { 
      for (i = 1; i < n_closures; i++) 
      { 
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
        }
      }
    }

    // Set dst on affected points
    i = 0; 
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
      dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2) + HI.apply_2D_x_left(src, hi[0], i, j, 3*tcomp+2);
      dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2) - HI.apply_2D_y_right(src, hi[1], N[1], i, j, 3*tcomp+2);
      dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
    }

    i = 0;
    for (j = N[1]-n_closures; j < N[1]-1; j++) 
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2) + HI.apply_2D_x_left(src, hi[0], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
      }
    }

    j = N[1]-1;
    for (i = 1; i < n_closures; i++) 
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2) - HI.apply_2D_y_right(src, hi[1], N[1], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_left(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
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

    // Set dst on unaffected points
    for (j = N[1]-n_closures; j < N[1]-1; j++)
    { 
      for (i = i_xstart; i < i_xend; i++)
      {
        for (tcomp = 0; tcomp < 4; tcomp++) {
          dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2);
          dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
        }
      }
    }

    // Set dst on affected points
    j = N[1]-1;
    for (i = i_xstart; i < i_xend; i++)
    {
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2) - HI.apply_2D_y_right(src, hi[1], N[1], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_interior(src,hi[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
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


    // Set dst on unaffected points
    for (j = N[1]-n_closures; j < N[1]-1; j++)
    { 
      for (i = N[0]-n_closures; i < N[0]-1; i++) 
      { 
        for (tcomp = 0; tcomp < 4; tcomp++) {
         dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2);
         dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2);
         dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
       }
      }
    }

    // Set dst on affected points
    i = N[0]-1; 
    j = N[1]-1;
    for (tcomp = 0; tcomp < 4; tcomp++) {
     dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2) - HI.apply_2D_x_right(src, hi[0], N[0], i, j, 3*tcomp+2);
     dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2) - HI.apply_2D_y_right(src, hi[1], N[1], i, j, 3*tcomp+2);
     dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
   }

    i = N[0]-1; 
    for (j = N[1]-n_closures; j < N[1]-1; j++)
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
       dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2) - HI.apply_2D_x_right(src, hi[0], N[0], i, j, 3*tcomp+2);
       dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2);
       dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
      }
    }

    j = N[1]-1;
    for (i = N[0]-n_closures; i < N[0]-1; i++) 
    { 
      for (tcomp = 0; tcomp < 4; tcomp++) {
        dst[j][i][3*tcomp] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 0) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp+2);
        dst[j][i][3*tcomp+1] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 1) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+2) - HI.apply_2D_y_right(src, hi[1], N[1], i, j, 3*tcomp+2);
        dst[j][i][3*tcomp+2] = aco_imp_apply_D_time(D_time, src, i, j, tcomp, 2) + D1.apply_2D_x_right(src,hi[0],N[0],i,j,3*tcomp) + D1.apply_2D_y_right(src,hi[1],N[1],i,j,3*tcomp+1);
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
        acowave_apply_2D_LL(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures, iw);
        acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures, iw);
        acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        acowave_apply_2D_RL(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures, iw);
        acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, n_closures, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures, iw);
        acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, n_closures, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        acowave_apply_2D_LR(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures, iw);
        acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        acowave_apply_2D_RR(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, N[0]-n_closures, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures, iw);
      acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, i_xstart, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
    }

    return 0;
  }

  template <class SbpDerivative, class SbpInvQuad, typename VelocityFunction>
  inline PetscErrorCode aco_imp_apply_1p(const PetscScalar D_time[4][4], const SbpDerivative& D1, const SbpInvQuad& HI,
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           PetscScalar ***src, 
                                           PetscScalar ***dst,
                                           const std::array<PetscInt,2>& i_start, const std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& xl, 
                                           const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    acowave_apply_2D_LL(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_2D_RL(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_2D_LR(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);
    acowave_apply_2D_RR(D_time, D1, HI, a, b, src, dst, N, xl, hi, sw, n_closures);

    acowave_apply_2D_CL(D_time, D1, HI, a, b, src, dst, n_closures, N[0]-n_closures, N, xl, hi, sw, n_closures, iw);
    acowave_apply_2D_LC(D_time, D1, HI, a, b, src, dst, n_closures, N[1] - n_closures, N, xl, hi, sw, n_closures, iw);
    acowave_apply_2D_CR(D_time, D1, HI, a, b, src, dst, n_closures, N[0]-n_closures, N, xl, hi, sw, n_closures);
    acowave_apply_2D_RC(D_time, D1, HI, a, b, src, dst, n_closures, N[1]-n_closures, N, xl, hi, sw, n_closures);

    acowave_apply_2D_CC(D_time, D1, HI, a, b, src, dst, n_closures, N[0]-n_closures, n_closures, N[1]-n_closures, N, xl, hi, sw, n_closures); 

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
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, n_closures, i_yend, N, xl, hi, sw, n_closures); 
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, N[0]-n_closures, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
      } else // BOTTOM CENTER
      { 
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CL(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, n_closures, i_yend-sw, N, xl, hi, sw, n_closures); 
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, n_closures, i_yend-sw, N, xl, hi, sw, n_closures); 
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xend, i_yend-sw, i_yend, N, xl, hi, sw, n_closures); 
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart, N[1]-n_closures, N, xl, hi, sw, n_closures);  
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend-sw, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);  
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart, N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, N[0]-n_closures, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      } else // TOP CENTER
      { 
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CR(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart,  N[1] - n_closures, N, xl, hi, sw, n_closures);
        acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, i_xend-sw, i_ystart,  i_ystart+sw, N, xl, hi, sw, n_closures);
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_2D_LC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, n_closures, i_xend, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart+sw, i_yend-sw, N, xl, hi, sw, n_closures);
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_2D_RC(t, D1, HI, a, b, src, dst, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, N[0] - n_closures, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart+sw, i_yend-sw, N, xl, hi, sw, n_closures);
    } else // CENTER
    {
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart, i_xstart+sw, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xend-sw, i_xend, i_ystart, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, i_xend-sw, i_yend-sw, i_yend, N, xl, hi, sw, n_closures);
      acowave_apply_2D_CC(t, D1, HI, a, b, src, dst, i_xstart+sw, i_xend-sw, i_ystart, i_ystart+sw, N, xl, hi, sw, n_closures);
    }

    return 0;
  }


} //namespace sbp