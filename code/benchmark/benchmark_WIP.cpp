static char help[] ="Solves the 2D acoustic wave equation on first order form: u_t = A*u_x + B*u_y, A = [0,0,-1;0,0,0;-1,0,0], B = [0,0,0;0,0,-1;0,-1,0].";

/**
 * Program to benchmark isolated functions that use the majority of the total runtime
 *
 *
 *
 **/
#include <array>
#include "sbpops/op_defs.h"
#include <petsc.h>
#include <stdlib.h>
#include "acoustic_wave_eq/wave_eq_rhs.h"
#include <sys/time.h>

#define H_INV 1


// create struct for arrays with time




static double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
    return seconds;
}



// #if H_INV
// /**
// * Inverse of density rho(x,y) at grid point i,j
// **/
// PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
//   printf("Using rho_inv from inside benchmark\n");
//   PetscScalar x = xl[0] + i/hi[0]; // multiplicera med h istället för division med hi
//   PetscScalar y = xl[1] + j/hi[1];
//   return 1./(2 + x*y);
// }

// #else

// /**
// * Inverse of density rho(x,y) at grid point i,j
// **/
// PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& h, const std::array<PetscScalar,2>& xl) {
//   PetscScalar x = xl[0] + i*h[0]; // Multiply with h instead of h inverse for less expensive operation
//   PetscScalar y = xl[1] + j*h[1];
//   return 1./(2 + x*y);
// }

// #endif

template <class SbpDerivative>
void wave_eq_rhs_II_benchmark(
                              const SbpDerivative& D1,
                              const PetscScalar *const *const *const q,
                              PetscScalar *const *const *const F,
                              const PetscInt i_xstart,
                              const PetscInt i_xend,
                              const PetscInt i_ystart,
                              const PetscInt i_yend,
                              const std::array<PetscScalar,2>& xl,
                              const std::array<PetscScalar,2>& hi)
{
  PetscScalar inv, D1_y, D1_x, D1_x0, D1_y1, inv_t, D1_t;
  PetscInt j,i;
  D1_t = get_wall_seconds();
  for (j = i_ystart; j < i_yend; j++)
  {
    for (i = i_xstart; i < i_xend; i++)
    {
      inv_t = get_wall_seconds();
      inv = -rho_inv(i, j, hi, xl);
      inv_t = get_wall_seconds()-inv_t;
      D1_x = D1.apply_x_interior(q, hi[0], i, j, 2);
      D1_y = D1.apply_y_interior(q, hi[1], i, j, 2);
      D1_x0 = D1.apply_x_interior(q, hi[0], i, j, 0);
      D1_y1 = D1.apply_y_interior(q, hi[1], i, j, 1);

      F[j][i][0] = -inv * D1_x;
      F[j][i][1] = -inv * D1_y;
      F[j][i][2] = -D1_x0 - D1_y1;

      // F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2);
      // F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2);
      // F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
  printf("cummulative time is %7.3f wall seconds.\n", inv_t);
}

template <class SbpDerivative>
void wave_eq_rhs_serial_benchmark(
                                  const SbpDerivative& D1,
                                  const PetscScalar *const *const *const q,
                                  PetscScalar *const *const *const F,
                                  const PetscInt N,
                                  const std::array<PetscScalar,2>& xl,
                                  const std::array<PetscScalar,2>& hi)
{
  const PetscInt cl_sz = D1.closure_size();

  wave_eq_rhs_II_benchmark(D1, q, F, cl_sz, N-cl_sz, cl_sz, N-cl_sz, xl, hi);

}

/**
* Computes the analytic (exact) solution to the problem at time t.
* da - Distributed array context
* appctx - application context, contains necessary information
* q - vector to store analytic solution
**/
PetscErrorCode setup(DM da, PetscInt n, Vec q) {
  PetscInt i, j;
  PetscScalar ***q_arr;

  DMDAVecGetArrayDOF(da,q,&q_arr);

  for (j = 0; j < n; j++)
  {
    for (i = 0; i < n; i++)
    {
      q_arr[j][i][0] = (PetscScalar)rand() / RAND_MAX;
      q_arr[j][i][1] = (PetscScalar)rand() / RAND_MAX;
      q_arr[j][i][2] = (PetscScalar)rand() / RAND_MAX;
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
  PetscInt turns = 1000;
  PetscInt N = 401;
  PetscInt dofs = 3;
  PetscScalar hx = (2.)/(N-1);
  PetscScalar hy = (2.)/(N-1);
  std::array<PetscScalar,2> xl = {-1,-1};
  std::array<PetscScalar,2> hi = {1./hx, 1./hy};
  // h for rho_inv
  std::array<PetscScalar,2> h = {hx, hy};
  Vec q,F;
  PetscScalar*** q_arr;
  PetscScalar*** F_arr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar sw = (D1.interior_stencil_size()-1)/2;
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

  PetscScalar time = get_wall_seconds();
  for (PetscInt i = 0; i < turns; i++) {
    wave_eq_rhs_serial_benchmark(D1, q_arr, F_arr, N, xl, hi);
  }
  printf("Loop of wave_eq_rhs_serial_benchmark took %7.3f wall seconds.\n", get_wall_seconds()-time);

  DMDAVecRestoreArrayDOF(da,q,&q_arr);
  DMDAVecRestoreArrayDOF(da,F,&F_arr);

  VecDestroy(&q);
  VecDestroy(&F);
  DMDestroy(&da);

  PetscFinalize();

  return 0;
}
