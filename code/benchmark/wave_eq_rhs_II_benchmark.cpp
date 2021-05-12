static char help[] ="Solves the 2D acoustic wave equation on first order form: u_t = A*u_x + B*u_y, A = [0,0,-1;0,0,0;-1,0,0], B = [0,0,0;0,0,-1;0,-1,0].";

#include <array>
#include "sbpops/op_defs.h"
#include <petsc.h>
#include <stdlib.h>


/**
* Inverse of density rho(x,y) at grid point i,j
**/
PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0]; // multiplicera med h istället för division med hi
  PetscScalar y = xl[1] + j/hi[1];
  return 1./(2 + x*y);
}

template <class SbpDerivative>
void wave_eq_rhs_II_benchmark(
                              const SbpDerivative& D1,
                              const double *const *const *const q,
                              double *const *const *const F,
                              const int i_xstart,
                              const int i_xend,
                              const int i_ystart,
                              const int i_yend,
                              const std::array<double,2>& xl,
                              const std::array<double,2>& hi)
{
  int i,j;
  for (j = i_ystart; j < i_yend; j++)
  {
    for (i = i_xstart; i < i_xend; i++)
    {
      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
}

template <class SbpDerivative>
void wave_eq_rhs_serial_benchmark(
                                  const SbpDerivative& D1,
                                  const double *const *const *const q,
                                  double *const *const *const F,
                                  const int N,
                                  const std::array<double,2>& xl,
                                  const std::array<double,2>& hi)
{
  const int cl_sz = D1.closure_size();

  wave_eq_rhs_II_benchmark(D1, q, F, cl_sz, N-cl_sz, cl_sz, N-cl_sz, xl, hi);
}

/**
* Computes the analytic (exact) solution to the problem at time t.
* da - Distributed array context
* appctx - application context, contains necessary information
* q - vector to store analytic solution
**/
PetscErrorCode setup(DM da, int n, Vec q) {
  PetscInt i, j;
  PetscScalar ***q_arr;

  DMDAVecGetArrayDOF(da,q,&q_arr);

  for (j = 0; j < n; j++)
  {
    for (i = 0; i < n; i++)
    {
      // change rhs to random doubles
      q_arr[j][i][0] = (double)rand() / RAND_MAX;
      q_arr[j][i][1] = (double)rand() / RAND_MAX;
      q_arr[j][i][2] = (double)rand() / RAND_MAX;
    }
  }

  DMDAVecRestoreArrayDOF(da,q,&q_arr);
  return 0;
}

int main(int argc,char **argv)
{
  DM da;
  PetscMPIInt    size, rank;
  const DifferenceOp D1;
  int turns = 2000;
  int N = 101;
  int dofs = 3;
  double hx = (2.)/(N-1);
  double hy = (2.)/(N-1);
  std::array<double,2> xl = {-1,-1};
  std::array<double,2> hi = {1./hx, 1./hy};
  Vec q,F;
  double*** q_arr;
  double*** F_arr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  double sw = (D1.interior_stencil_size()-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               N,N,PETSC_DECIDE,PETSC_DECIDE,dofs,sw,NULL,NULL,&da);
  DMSetFromOptions(da);
  DMSetUp(da);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&q);
  VecDuplicate(q,&F);

  setup(da, N, q);

  DMDAVecGetArrayDOF(da,q,&q_arr);
  DMDAVecGetArrayDOF(da,F,&F_arr);

  for (int i = 0; i < turns; i++) {
    wave_eq_rhs_serial_benchmark(D1, q_arr, F_arr, N, xl, hi);
  }

  DMDAVecRestoreArrayDOF(da,q,&q_arr);
  DMDAVecRestoreArrayDOF(da,F,&F_arr);

  VecDestroy(&q);
  VecDestroy(&F);
  DMDestroy(&da);

  PetscFinalize();

  return 0;
}
