#pragma once

#include<petscsystypes.h>
#include <array>
#include "grids/grid_function.h"

namespace sbp{

  /**
  * Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x], between indices i_start <= i < i_end. "Smart" looping.
  * Inputs: D1        - SBP D1 operator
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative>
  inline PetscErrorCode reflection_apply1(const SbpDerivative& D1, const grid::grid_function_1d<PetscScalar> src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, PetscInt N, PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) {
      for (i = 0; i < n_closures; i++) 
      { 
        dst(i,1) = D1.apply_left(src,hi,i,0);
        dst(i,0) = D1.apply_left(src,hi,i,1);
      }
      i_start = n_closures;
    }

    if (i_end == N) {
      for (i = N-n_closures; i < N; i++)
      {
          dst(i,1) = D1.apply_right(src,hi,N,i,0);
          dst(i,0) = D1.apply_right(src,hi,N,i,1);
      }
      i_end = N-n_closures;
    }

    for (i = i_start; i < i_end; i++)
    {
      dst(i,1) = D1.apply_interior(src,hi,i,0);
      dst(i,0) = D1.apply_interior(src,hi,i,1);
    }

    return 0;
  };

  /**
  * Approximate RHS of reflection problem, [u;v]_t = [v_x;u_x], between indices i_start <= i < i_end. Direct looping.
  * Inputs: D1        - SBP D1 operator
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative>
  inline PetscErrorCode reflection_apply2(const SbpDerivative& D1, const grid::grid_function_1d<PetscScalar> src, grid::grid_function_1d<PetscScalar> dst, PetscInt i_start, PetscInt i_end, PetscInt N, PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      for (i = 0; i < n_closures; i++) 
      { 
        dst(i,1) = D1.apply_left(src,hi,i,0);
        dst(i,0) = D1.apply_left(src,hi,i,1);
      }
      
      for (i = n_closures; i < i_end; i++)
      {
        dst(i,1) = D1.apply_interior(src,hi,i,0);
        dst(i,0) = D1.apply_interior(src,hi,i,1);
      }
    } else if (i_end == N) 
    {
      for (i = N-n_closures; i < N; i++)
      {
        dst(i,1) = D1.apply_right(src,hi,N,i,0);
        dst(i,0) = D1.apply_right(src,hi,N,i,1);
      }

      for (i = i_start; i < N-n_closures; i++)
      {
        dst(i,1) = D1.apply_interior(src,hi,i,0);
        dst(i,0) = D1.apply_interior(src,hi,i,1);
      }
    } else 
    {
      for (i = i_start; i < i_end; i++)
      {
        dst(i,1) = D1.apply_interior(src,hi,i,0);
        dst(i,0) = D1.apply_interior(src,hi,i,1);
      }
    }
    return 0;
  };

    /**
  * Approximate RHS of 2D a general advection problem, u_t = -(au_x+bu_y), between indices in i_start, i_end. Using direct looping, likely bug free.
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
  inline PetscErrorCode reflection_apply_2D_2(const SbpDerivative& D1, 
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           const PetscScalar *const *const *const array_src,
                                           PetscScalar *const *const *const array_dst,
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
        // x: closure, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,1) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0);  
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0); 
          }
        }
        // x: closure, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,1) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0); 
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0); 
          }
        }    
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        // x: closure, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0);
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0);
          }
        }
        // x: closure, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        } 
      } else // BOTTOM CENTER
      { 
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++) 
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0);
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        }
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        // x: closure, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++) 
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0);
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0);
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);

          }
        }    
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        // x: closure, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0);
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0);
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        } 
      } else // TOP CENTER
      { 
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0);
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
          }
        }
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      // x: closure, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = n_closures; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
        }
      }
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // x: closure, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++) 
        { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend - n_closures; i++)
        { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
        }
      }
    } else // CENTER
    {
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,1) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,1);

            array_dst[j][i][1] = std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) + 
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0);
        }
      }
    }
    return 0;
  };
} //namespace sbp