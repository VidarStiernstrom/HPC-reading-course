#pragma once

#include<petscsystypes.h>
#include <array>
#include "grids/grid_function.h"

namespace sbp{

    /**
  * Approximate RHS of 2D acoustic wave equation, w_t = A*wx + B*wy, between indices in i_start, i_end. Using direct looping, likely bug free.
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function in x-direction, parametrized on indices i.e a(x(i),y(j))
  a         b         -  velocity field function in y-direction, parametrized on indices i.e b(x(i),y(j))
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[j][i][component] (obtained using DMDAVecGetArrayDOF).
  *         array_dst - 2D array containing multi-component output data. Ordered as array_dst[j][i][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting indices (global indexing) [ix_start, iy_start]
  *         i_end     - Final indices (global indexing) [ix_end, iy_end]
  *         N         - Global number of points excluding ghost points [Nx, Ny]
  *         hi        - Inverse step length [hix, hiy]
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode acowave_apply_2D(const SbpDerivative& D1, 
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           const grid::grid_function_2d<PetscScalar> src,
                                           grid::grid_function_2d<PetscScalar> dst,
                                           const std::array<PetscInt,2>& i_start, std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    PetscInt i,j;
    //Could pass use i_start and i_end using lvalue references and move semantics here aswell. Not sure what is best.
    const PetscInt i_xstart = i_start[0]; 
    const PetscInt i_ystart = i_start[1];
    const PetscInt i_xend = i_end[0];
    const PetscInt i_yend = i_end[1];
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    if (i_ystart == 0)  // BOTTOM
    {
      if (i_xstart == 0) // BOTTOM LEFT
      {
        // x: closure left, y: closure left
        for (j = 0; j < n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            dst(j,i,0) = -D1.apply_2D_x_left(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_left(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_left(src,hi[0],i,j,0) - D1.apply_2D_y_left(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: closure left
        for (j = 0; j < n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_left(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_left(src,hi[0],i,j,1);
          }
        }
        // x: closure left, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_left(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_left(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }    
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        // x: closure right, y: closure left
        for (j = 0; j < n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            dst(j,i,0) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_left(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) - D1.apply_2D_y_left(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: closure left
        for (j = 0; j < n_closures; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_left(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_left(src,hi[0],i,j,1);
          }
        }
        // x: closure right, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            dst(j,i,0) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        } 
      } else // BOTTOM CENTER
      { 
        // x: inner, y: closure left
        for (j = 0; j < n_closures; j++) 
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_left(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_left(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        // x: closure left, y: closure right
        for (j = N[1]-n_closures; j < N[1]; j++) 
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            dst(j,i,0) = -D1.apply_2D_x_left(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_left(src,hi[0],i,j,0) - D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
          }
        }
        // x: inner, y: closure right
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
          }
        }
        // x: closure left, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_left(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_left(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);

          }
        }    
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        // x: closure right, y: closure right
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            dst(j,i,0) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) - D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
          }
        }
        // x: inner, y: closure right
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
          }
        }
        // x: closure right, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            dst(j,i,0) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        } 
      } else // TOP CENTER
      { 
        // x: inner, y: closure right
        for (j = N[1]-n_closures; j < N[1]; j++)
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_right(src,hi[1],N[1],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_right(src,hi[1],N[1],i,j,1);
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
          }
        }
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      // x: closure left, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
            dst(j,i,0) = -D1.apply_2D_x_left(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_left(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = n_closures; i < i_xend; i++) 
        { 
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
        }
      }
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // x: closure right, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++) 
        { 
            dst(j,i,0) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_right(src,hi[0],N[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend - n_closures; i++)
        { 
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
        }
      }
    } else // CENTER
    {
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend; i++) 
        { 
            dst(j,i,0) = -D1.apply_2D_x_interior(src,hi[0],i,j,2);
            dst(j,i,1) = -D1.apply_2D_y_interior(src,hi[0],i,j,2);
            dst(j,i,2) = -D1.apply_2D_x_interior(src,hi[0],i,j,0) - D1.apply_2D_y_interior(src,hi[0],i,j,1);
        }
      }
    }
    return 0;
  };
} //namespace sbp