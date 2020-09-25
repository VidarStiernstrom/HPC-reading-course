#pragma once

#include<petscsystypes.h>
#include <array>


namespace sbp{

  /**
  * Approximate RHS of advection problem, u_t = -au_x, between indices i_start <= i < i_end. "Smart" looping.
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function parametrized on indices, i.e a(x(i)),
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode advection_apply1(const SbpDerivative& D1, VelocityFunction&& a, const PetscScalar *const *const array_src, PetscScalar **array_dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      for (i = 0; i < n_closures; i++) 
      { 
        array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_left(array_src,hi,i,0);
      }
      i_start = n_closures;
    }

    if (i_end == N) 
    {
      for (i = N-n_closures; i < N; i++)
      {
          array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_right(array_src,hi,N,i,0);
      }
      i_end = N-n_closures;
    }

    for (i = i_start; i < i_end; i++)
    {
      array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_interior(array_src,hi,i,0);
    }

    return 0;
  };

  /**
  * Approximate RHS of advection problem, u_t = -au_x, between indices i_start <= i < i_end. Direct looping.
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function parametrized on indices, i.e a(x(i)),
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         array_src - 2D array containing multi-component output data. Ordered as array_src[index][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index to compute
  *         i_end     - Final index to compute, index < i_end
  *         N         - Global number of points excluding ghost points
  *         hi        - Inverse step length
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode advection_apply2(const SbpDerivative& D1, VelocityFunction&& a, const PetscScalar *const *const array_src, PetscScalar **array_dst, PetscInt i_start, PetscInt i_end, const PetscInt N, const PetscScalar hi)
  {
    PetscInt i;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start == 0) 
    {
      for (i = 0; i < n_closures; i++) 
      { 
        array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_left(array_src,hi,i,0);
      }
      
      for (i = n_closures; i < i_end; i++)
      {
        array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_interior(array_src,hi,i,0);
      }
    } else if (i_end == N) 
    {
      for (i = N-n_closures; i < N; i++)
      {
          array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_right(array_src,hi,N,i,0);
      }

      for (i = i_start; i < N-n_closures; i++)
      {
        array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_interior(array_src,hi,i,0);
      }
    } else 
    {
      for (i = i_start; i < i_end; i++)
      {
        array_dst[i][0] = -std::forward<VelocityFunction>(a)(i)*D1.apply_interior(array_src,hi,i,0);
      }
    }
    return 0;
  };


  /**
  * Approximate RHS of 2D advection problem, u_t = -au_x, between indices [i_start, i_end], [j_start, j_end].
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function, parametrized on indices i.e a(x(i),y(j))
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[j][i][component] (obtained using DMDAVecGetArrayDOF).
  *         array_dst - 2D array containing multi-component output data. Ordered as array_dst[j][i][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index in x-direction (global indexing)
  *         i_end     - Final index in x-direction (global indexing)
  *         j_start   - Starting index in y-direction (global indexing)
  *         j_end     - Final index in y-direction (global indexing)
  *         Nx        - Global number of points excluding ghost points in x-direction
  *         hix       - Inverse step length in x-direction
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode advection_apply_2D_x(const SbpDerivative& D1, VelocityFunction&& a,
                                             const PetscScalar *const *const *const array_src,
                                             PetscScalar *const *const *const array_dst,
                                             PetscInt i_start, const PetscInt j_start,
                                             PetscInt i_end, const PetscInt j_end,
                                             const PetscInt Nx, const PetscScalar hix)
  {
    PetscInt i,j;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    if (i_start == 0) 
    {
      for (j = j_start; j < j_end; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
          array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hix,i,j,0);
        }
      }
      i_start = n_closures;
    }

    if (i_end == Nx) 
    {
      for (j = j_start; j < j_end; j++)
      { 
        for (i = Nx-n_closures; i < Nx; i++)
        {
            array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hix,Nx,i,j,0);
        }
      }
      i_end = Nx-n_closures;
    }
    for (j = j_start; j < j_end; j++)
      { 
      for (i = i_start; i < i_end; i++)
      {
        array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hix,i,j,0);
      }
    }
    return 0;
  };

  /**
  * Approximate RHS of 2D advection problem, u_t = -au_y, between indices [i_start, i_end], [j_start, j_end].
  * Inputs: D1        - SBP D1 operator
  *         a         -  velocity field function, parametrized on indices i.e a(x(i),y(j))
  *         array_src - 2D array containing multi-component input data. Ordered as array_src[j][i][component] (obtained using DMDAVecGetArrayDOF).
  *         array_dst - 2D array containing multi-component output data. Ordered as array_dst[j][i][component] (obtained using DMDAVecGetArrayDOF).
  *         i_start   - Starting index in x-direction (global indexing)
  *         i_end     - Final index in x-direction (global indexing)
  *         j_start   - Starting index in y-direction (global indexing)
  *         j_end     - Final index in y-direction (global indexing)
  *         Ny        - Global number of points excluding ghost points in y-direction
  *         hiy       - Inverse step length in y-direction
  **/
  template <class SbpDerivative, typename VelocityFunction>
  inline PetscErrorCode advection_apply_2D_y(const SbpDerivative& D1, VelocityFunction&& a,
                                             const PetscScalar *const *const *const array_src,
                                             PetscScalar *const *const *const array_dst,
                                             const PetscInt i_start, PetscInt j_start,
                                             const PetscInt i_end, PetscInt j_end,
                                             const PetscInt Ny, const PetscScalar hiy)
  {
    PetscInt i,j;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    if (j_start == 0) 
    {
      for (j = 0; j < n_closures; j++)
      { 
        for (i = i_start; i < i_end; i++) 
        { 
          array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_y_left(array_src,hiy,i,j,0);
        }
      }
      j_start = n_closures;
    }

    if (j_end == Ny) 
    {
      for (j = Ny-n_closures; j < Ny; j++)
      { 
        for (i = i_start; i < i_end; i++)
        {
            array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_y_right(array_src,hiy,Ny,i,j,0);
        }
      }
      j_end = Ny-n_closures;
    }
    for (j = j_start; j < j_end; j++)
      { 
      for (i = i_start; i < i_end; i++)
      {
        array_dst[j][i][0] = -std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_y_interior(array_src,hiy,i,j,0);
      }
    }
    return 0;
  };

  /**
  * Approximate RHS of 2D a general advection problem, u_t = -(au_x+bu_y), between indices in i_start, i_end. Using "smart" looping, likely contains bug.
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
  inline PetscErrorCode advection_apply_2D_1(const SbpDerivative& D1, 
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           const PetscScalar *const *const *const array_src,
                                           PetscScalar *const *const *const array_dst,
                                           const std::array<PetscInt,2>& i_start, std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& hi)
  {
    PetscInt i,j;
    //Could pass use i_start and i_end using lvalue references and move semantics here aswell. Not sure what is best.
    PetscInt i_xstart = i_start[0]; 
    PetscInt i_ystart = i_start[1];
    PetscInt i_xend = i_end[0];
    PetscInt i_yend = i_end[1];
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    // Left y 
    if (i_ystart == 0) 
    {
      // Left x
      if (i_xstart == 0)
      {
 
        for (j = 0; j < n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        i_xstart = n_closures;
      }
      // Right x
      if (i_xend == N[0])
      { 
 
        for (j = 0; j < n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++)
          {
              array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                     std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        i_xend = N[0]-n_closures;
      }
      // Interior x
      for (j = 0; j < n_closures; j++)
      { 
        for (i = i_xstart; i < i_xend; i++)
        {
          array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
        }
      }
      i_ystart = n_closures;
    }
    // Right y
    if (i_end[1] == N[1]) 
    {
      // Left x
      if (i_xstart == 0)
      { 
 
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
              array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                     std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        i_xstart = n_closures;
      }
      // Right x
      if (i_xend == N[0])
      {
 
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++)
          {
              array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                     std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        i_xend = N[0]-n_closures;
      }
      // Interior x
      for (j = N[1]-n_closures; j < N[1]; j++)
      { 
        for (i = i_xstart; i < i_xend; i++)
        {
          array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
        }
      }
      i_yend = N[1]-n_closures;
    }
    // Interior y
    // Left x
    if (i_xstart == 0)
    { 
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = 0; i < n_closures; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      i_xstart = n_closures;
    }
    // Right x
    if (i_xend == N[0])
    {
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++)
        {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      i_xend = N[0]-n_closures;
    }
    // Interior 
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = i_xstart; i < i_xend; i++)
      {
        array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                               std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
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
  inline PetscErrorCode advection_apply_2D_2(const SbpDerivative& D1, 
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           const PetscScalar *const *const *const array_src,
                                           PetscScalar *const *const *const array_dst,
                                           const std::array<PetscInt,2>& i_start, std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& hi)
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
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }    
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        // x: closure, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        } 
      } else // BOTTOM CENTER
      { 
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++) 
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
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
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }    
      } else if (i_xend == N[0]) // TOP RIGHT
      { 
        // x: closure, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        } 
      } else // TOP CENTER
      { 
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
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
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = n_closures; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // x: closure, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend - n_closures; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    } else // CENTER
    {
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    }
    return 0;
  };

  /**
  * Approximate RHS of 2D a general advection problem, u_t = -(au_x+bu_y), between indices in i_start, i_end. Only applies to indices not requiring ghost points. Using "smart" looping.
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
  inline PetscErrorCode advection_apply_2D_1_inner(const SbpDerivative& D1, 
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           const PetscScalar *const *const *const array_src,
                                           PetscScalar *const *const *const array_dst,
                                           const std::array<PetscInt,2>& i_start, std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    PetscInt i,j;
    //Could pass use i_start and i_end using lvalue references and move semantics here aswell. Not sure what is best.
    PetscInt i_xstart = i_start[0] + sw; 
    PetscInt i_ystart = i_start[1] + sw;
    PetscInt i_xend = i_end[0] - sw;
    PetscInt i_yend = i_end[1] - sw;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();
    // Left y 
    if (i_start[1] == 0) 
    {
      // Left x
      if (i_start[0] == 0)
      {
        for (j = 0; j < n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        i_xstart = n_closures;
      }
      // Right x
      if (i_end[0] == N[0])
      { 
        for (j = 0; j < n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++)
          {
              array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                     std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        i_xend = N[0]-n_closures;
      }
      // Interior x
      for (j = 0; j < n_closures; j++)
      { 
        for (i = i_xstart; i < i_xend; i++)
        {
          array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
        }
      }
      i_ystart = n_closures;
    }
    // Right y
    if (i_end[1] == N[1]) 
    {
      // Left x
      if (i_start[0] == 0)
      {
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
              array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                     std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        i_xstart = n_closures;
      }
      // Right x
      if (i_end[0] == N[0])
      {
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++)
          {
              array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                     std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        i_xend = N[0]-n_closures;
      }
      // Interior x
      for (j = N[1]-n_closures; j < N[1]; j++)
      { 
        for (i = i_xstart + sw; i < i_xend - sw; i++)
        {
          array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                 std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
        }
      }
      i_yend = N[1]-n_closures;
    }
    // Interior y
    // Left x
    if (i_start[0] == 0)
    { 
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = 0; i < n_closures; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      i_xstart = n_closures;
    }
    // Right x
    if (i_xend == N[0])
    {
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++)
        {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      i_xend = N[0]-n_closures;
    }
    // Interior 
    for (j = i_ystart; j < i_yend; j++)
    { 
      for (i = i_xstart; i < i_xend; i++)
      {
        array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                               std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
      }
    }
    return 0;
  };

  /**
  * Approximate RHS of 2D a general advection problem, u_t = -(au_x+bu_y), between indices in i_start, i_end. Only applies to indices not requiring ghost points. Using direct looping.
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
  inline PetscErrorCode advection_apply_2D_2_inner(const SbpDerivative& D1, 
                                           VelocityFunction&& a,
                                           VelocityFunction&& b,
                                           const PetscScalar *const *const *const array_src,
                                           PetscScalar *const *const *const array_dst,
                                           const std::array<PetscInt,2>& i_start, std::array<PetscInt,2>& i_end,
                                           const std::array<PetscInt,2>& N, const std::array<PetscScalar,2>& hi, const PetscInt sw)
  {
    PetscInt i,j;
    //Could pass use i_start and i_end using lvalue references and move semantics here aswell. Not sure what is best.
    const PetscInt i_xstart = i_start[0] + sw; 
    const PetscInt i_ystart = i_start[1] + sw;
    const PetscInt i_xend = i_end[0] - sw;
    const PetscInt i_yend = i_end[1] - sw;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    if (i_start[1] == 0)  // BOTTOM
    {
      if (i_start[0] == 0) // BOTTOM LEFT
      {
        // x: closure, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = n_closures; i < i_xend ; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }    
      } else if (i_end[0] == N[0]) // BOTTOM RIGHT
      { 
        // x: closure, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = i_xstart ; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        } 
      } else // BOTTOM CENTER
      { 
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++) 
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend ; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
      }
    } else if (i_end[1] == N[1]) // TOP
    {
      if (i_start[0] == 0) // TOP LEFT
      {
        // x: closure, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++) 
        { 
          for (i = 0; i < n_closures; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < N[1] - n_closures; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < N[1] - n_closures; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }    
      } else if (i_end[0] == N[0]) // TOP RIGHT
      { 
        // x: closure, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) + 
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < N[1] - n_closures; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < N[1] - n_closures; j++)
        { 
          for (i = i_xstart; i < N[0] - n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        } 
      } else // TOP CENTER
      { 
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        {
          for (i = i_xstart; i < i_xend; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < N[1] - n_closures; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
      }
    } else if (i_start[0] == 0) // LEFT NOT BOTTOM OR TOP
    { 
      // x: closure, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = n_closures; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    } else if (i_end[0] == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // x: closure, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < N[0]-n_closures; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    } else // CENTER
    {
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    }
    return 0;
  };

  /**
  * Approximate RHS of 2D a general advection problem, u_t = -(au_x+bu_y), between indices in i_start, i_end. Only applies to indices requiring ghost points. Using "direct looping.
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
  inline PetscErrorCode advection_apply_2D_outer(const SbpDerivative& D1, 
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
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = i_xend - sw; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_yend - sw; j < i_yend; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_yend - sw; j < i_yend; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend - sw; j++)
        { 
          for (i = i_xend - sw; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }      
      } else if (i_xend == N[0]) // BOTTOM RIGHT
      { 
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++)
        { 
          for (i = i_xstart; i < i_xstart + sw; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_yend - sw; j < i_yend; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_yend - sw; j < i_yend; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend - sw; j++)
        { 
          for (i = i_xstart; i < i_xstart + sw; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }  
      } else // BOTTOM CENTER
      { 
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++) 
        {
          for (i = i_xstart; i < i_xstart + sw; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = 0; j < n_closures; j++) 
        {
          for (i = i_xend - sw; i < i_xend; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_left(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_yend - sw; j < i_yend; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend - sw; j++)
        { 
          for (i = i_xstart; i < i_xstart + sw; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = n_closures; j < i_yend - sw; j++)
        { 
          for (i = i_xend - sw; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
      }
    } else if (i_yend == N[1]) // TOP
    {
      if (i_xstart == 0) // TOP LEFT
      {
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = i_xend - sw; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < i_ystart + sw; j++)
        { 
          for (i = 0; i < n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_ystart + sw; j++)
        { 
          for (i = n_closures; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart + sw; j < i_yend - n_closures; j++)
        { 
          for (i = i_xend - sw; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }      
      } else if (i_xend == N[0]) // TOP RIGHT /////
      { 
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        { 
          for (i = i_xstart; i < i_xstart + sw; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: closure, y: inner
        for (j = i_ystart; j < i_ystart + sw; j++)
        { 
          for (i = N[0]-n_closures; i < N[0]; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_ystart + sw; j++)
        { 
          for (i = i_xstart; i < N[0]-n_closures; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        } 
        // x: inner, y: inner
        for (j = i_ystart + sw; j < N[1] - n_closures; j++)
        { 
          for (i = i_xstart; i < i_xstart + sw; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        } 
      } else // TOP CENTER
      { 
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        {
          for (i = i_xstart; i < i_xstart + sw; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: closure
        for (j = N[1]-n_closures; j < N[1]; j++)
        {
          for (i = i_xend - sw; i < i_xend; i++) 
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_right(array_src,hi[1],N[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart; j < i_ystart + sw; j++)
        { 
          for (i = i_xstart; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart + sw; j < i_yend - n_closures; j++)
        { 
          for (i = i_xstart; i < i_xstart + sw; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
        // x: inner, y: inner
        for (j = i_ystart + sw; j < i_yend - n_closures; j++)
        { 
          for (i = i_xend - sw; i < i_xend; i++)
          {
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
          }
        }
      }
    } else if (i_xstart == 0) // LEFT NOT BOTTOM OR TOP
    { 
      // x: closure, y: inner
      for (j = i_ystart; j < i_ystart + sw; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: closure, y: inner
      for (j = i_yend - sw; j < i_yend; j++)
      { 
        for (i = 0; i < n_closures; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_left(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xend - sw; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_ystart + sw; j++)
      { 
        for (i = n_closures; i < i_xend - sw; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_yend - sw; j < i_yend; j++)
      { 
        for (i = n_closures; i < i_xend - sw; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    } else if (i_xend == N[0]) // RIGHT NOT BOTTOM OR TOP
    {
      // x: closure, y: inner
      for (j = i_ystart; j < i_ystart + sw; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: closure, y: inner
      for (j = i_yend - sw; j < i_yend; j++)
      { 
        for (i = N[0]-n_closures; i < N[0]; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_right(array_src,hi[0],N[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xstart + sw; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart; j < i_ystart + sw; j++)
      { 
        for (i = i_xstart + sw; i < i_xend - n_closures; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_yend - sw; j < i_yend; j++)
      { 
        for (i = i_xstart + sw; i < i_xend - n_closures; i++)
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    } else // CENTER
    {
      // x: inner, y: inner
      for (j = i_ystart; j < i_ystart + sw; j++)
      { 
        for (i = i_xstart; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_yend - sw; j < i_yend; j++)
      { 
        for (i = i_xstart; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart + sw; j < i_yend - sw; j++)
      { 
        for (i = i_xstart; i < i_xstart + sw; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
      // x: inner, y: inner
      for (j = i_ystart + sw; j < i_yend - sw; j++)
      { 
        for (i = i_xend - sw; i < i_xend; i++) 
        { 
            array_dst[j][i][0] = -(std::forward<VelocityFunction>(a)(i,j)*D1.apply_2D_x_interior(array_src,hi[0],i,j,0) +
                                   std::forward<VelocityFunction>(b)(i,j)*D1.apply_2D_y_interior(array_src,hi[1],i,j,0));
        }
      }
    }
    return 0;
  };

} //namespace sbp