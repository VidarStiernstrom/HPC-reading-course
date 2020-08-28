#pragma once

#include<petscdmda.h>
#include<petscvec.h>

namespace sbp{
  template <class SbpDerivative, typename Lambda>
  inline double advection_variable_left(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int i);

  template <class SbpDerivative, typename Lambda>
  inline double advection_variable_interior(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int i);

  template <class SbpDerivative, typename Lambda>
  inline double advection_variable_right(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int n, const int i);

  template <class SbpDerivative, typename Lambda>
  inline void advection_variable(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int n, double *v_x);

  template <class SbpDerivative, typename Lambda>
  inline PetscErrorCode advection_variable_distributed(const SbpDerivative& D1, Lambda&& velocity_field, const DM& da, const Vec& v, const double hi, const int N, Vec& v_x);

  //=============================================================================
  // Implementations
  //=============================================================================
  template <class SbpDerivative, typename Lambda>
  inline double advection_variable_left(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int i)
  {
    return std::forward<Lambda>(velocity_field)(i)*D1.apply_left(v,hi,i);
  }

  template <class SbpDerivative, typename Lambda>
  inline double advection_variable_interior(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int i)
  {
    return std::forward<Lambda>(velocity_field)(i)*D1.apply_interior(v,hi,i);
  }

  template <class SbpDerivative, typename Lambda>
  inline double advection_variable_right(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int n, const int i)
  {
    return std::forward<Lambda>(velocity_field)(i)*D1.apply_right(v,hi,n,i);
  }


  template <class SbpDerivative, typename Lambda>
  inline void advection_variable(const SbpDerivative& D1, Lambda&& velocity_field, const double *v, const double hi, const int n, double *v_x)
  { 
    const auto [iw, n_closures, cw] = D1.get_ranges();
    for (int i = 0; i < n_closures; i++){
      v_x[i] = advection_variable_left(D1, velocity_field, v, hi, i);
    }
    for (int i = n_closures; i < n-n_closures; i++){
      v_x[i] = advection_variable_interior(D1, velocity_field, v, hi, i);
    }
    for (int i = n-n_closures; i < n; i++){
      v_x[i] = advection_variable_right(D1, velocity_field, v, hi, n,  i);
    }
  }

  template <class SbpDerivative, typename Lambda>
  inline PetscErrorCode advection_variable_distributed(const SbpDerivative& D1, Lambda&& velocity_field, const DM& da, const Vec& v, const double hi, const int N, Vec& v_x)
  {
    PetscErrorCode ierr = 0;
    PetscScalar *array_src, *array_dst;
    PetscInt i_start, i_end, n;
    const auto [iw, n_closures, closure_width] = D1.get_ranges();

    DMDAGetCorners(da,&i_start,0,0,&n,0,0);
    // Perform bounds check.
    // No communication is to be made over the closures, i.e a processor at least has number of nodes equal
    // to the closure width closure_width.
    if (n < closure_width)
    {
      ierr = PETSC_ERR_MIN_VALUE;
      std::string err_msg("Number of nodes per process must be greater or equal than " + std::to_string(closure_width) + ".");
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MIN_VALUE,err_msg.c_str());
      return ierr;
    }
    // Extract underlying arrays and perform differentation.
    DMDAVecGetArray(da,v_x,&array_dst);
    DMDAVecGetArray(da,v,&array_src);
    if (n == N) // Single process. Perform standard apply
    {
      advection_variable(D1, velocity_field, array_src, hi, N, array_dst);
    }
    else // Multiple processes
    { 
      // TODO: The index ranges and applies performed by a process is given by
      // the problem size and number of processors, as well as the stencil widths of the derivative
      // We should consider creating a new class/struct which bundles together the index ranges and the applies for a processor
      // in a nice way.
      i_end = i_start+n;
      if (i_start < n_closures){
        for (PetscInt i = i_start; i < n_closures; i++) 
        { 
          array_dst[i] = advection_variable_left(D1, velocity_field, array_src, hi, i);
        }
        for (PetscInt i = n_closures; i < i_end; i++)
        {
          array_dst[i] = advection_variable_interior(D1, velocity_field, array_src, hi, i);
        }
      }
      if ((n_closures < i_start) && (i_end < N-n_closures))
      {
        for (PetscInt i = i_start; i < i_end; i++)
        {
          array_dst[i] = advection_variable_interior(D1, velocity_field, array_src, hi, i);
        }
      }
      if ((i_start < N-n_closures) && (N-n_closures < i_end))
      {
        for (PetscInt i = i_start; i < N-n_closures; i++)
        {
          array_dst[i] = advection_variable_interior(D1, velocity_field, array_src, hi, i);
        }
        for (PetscInt i = N-n_closures; i < N; i++)
        {
          array_dst[i] = advection_variable_right(D1, velocity_field, array_src, hi, N, i);
        }
      }
    }
    DMDAVecRestoreArray(da,v,array_src);
    DMDAVecRestoreArray(da,v_x,array_dst);
    return 0;
  }
} //namespace sbp