/**
 * Program to benchmark isolated functions that use the majority of the total runtime
 *
 *
 *
 **/
#include <array>
#include "sbpops/op_defs.h"
#include <petsc.h>
#include "benchmark_utils.h"
#include "iostream"

/**
* Inverse of density rho(x,y) at grid point i,j
**/
// inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) __attribute__((pure));
inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& hi, const std::array<PetscScalar,2>& xl) {
  PetscScalar x = xl[0] + i/hi[0];
  PetscScalar y = xl[1] + j/hi[1];
  return 1./(2 + x*y);
}

// /**
// * Inverse of density rho(x,y) at grid point i,j
// **/
// inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& h, const std::array<PetscScalar,2>& xl) {
//   PetscScalar x = xl[0] + i * h[0];
//   PetscScalar y = xl[1] + j * h[1];
//   return 1./(2 + x*y);
// }

// /**
// * Inverse of density rho(x,y) at grid point i,j
// **/
// inline PetscScalar rho_inv(const PetscInt i, const PetscInt j, const std::array<PetscScalar,2>& h, const std::array<PetscScalar,2>& xl) {
//   PetscScalar x = xl[0] + i * h[0];
//   PetscScalar y = xl[1] + j * h[1];
//   return (2 + x*y);
// }

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
                              const std::array<PetscScalar,2>& hi,
                              const std::array<PetscScalar,2>& h)
{
  // PetscScalar inv;
  PetscInt i,j;
  for (j = i_ystart; j < i_yend; j++)
  {
    for (i = i_xstart; i < i_xend; i++)
    {
      // inv = rho_inv(i, j, hi, xl);
      // F[j][i][0] = -inv*D1.apply_x_interior(q, hi[0], i, j, 2);
      // F[j][i][1] = -inv*D1.apply_y_interior(q, hi[1], i, j, 2);
      // F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);

      F[j][i][0] = -rho_inv(i, j, hi, xl)*D1.apply_x_interior(q, hi[0], i, j, 2);
      F[j][i][1] = -rho_inv(i, j, hi, xl)*D1.apply_y_interior(q, hi[1], i, j, 2);
      F[j][i][2] = -D1.apply_x_interior(q, hi[0], i, j, 0) - D1.apply_y_interior(q, hi[1], i, j, 1);
    }
  }
}

template <class SbpDerivative>
void wave_eq_rhs_serial_benchmark(
                                  const SbpDerivative& D1,
                                  const PetscScalar *const *const *const q,
                                  PetscScalar *const *const *const F,
                                  const PetscInt N,
                                  const std::array<PetscScalar,2>& xl,
                                  const std::array<PetscScalar,2>& hi,
                                  const std::array<PetscScalar,2>& h)
{
  const PetscInt cl_sz = D1.closure_size();

  wave_eq_rhs_II_benchmark(D1, q, F, cl_sz, N-cl_sz, cl_sz, N-cl_sz, xl, hi, h);

}

int main(int argc,char **argv)
{
  PetscInt turns = 2000;
  PetscInt N = 801;
  PetscInt dofs = 3;
  const DifferenceOp D1;
  PetscScalar hx = (2.)/(N-1);
  PetscScalar hy = (2.)/(N-1);
  std::array<PetscScalar,2> xl = {-1,-1};
  std::array<PetscScalar,2> h = {hx, hy};
  std::array<PetscScalar,2> hi = {1./hx, 1./hy};
  PetscScalar ***q, *q_flat;
  PetscScalar ***F, *F_flat;

  PetscScalar Ntot = N*N*dofs;
  q_flat = new PetscScalar[Ntot];
  F_flat = new PetscScalar[Ntot];
  // Initialize q_flat with random values
  for  (int i = 0; i<Ntot; i++)
    q_flat[i] = (PetscScalar)rand() / RAND_MAX;

  // Poit q into q_flat and F into F_flat
  petsc_triple_ptr_layout(q_flat, N, dofs, &q);
  petsc_triple_ptr_layout(F_flat, N, dofs, &F);

  // Run benchmark
  PetscScalar time = get_wall_seconds();
  for (PetscInt i = 0; i < turns; i++) {
    wave_eq_rhs_serial_benchmark(D1, q, F, N, xl, hi, h);
  }
  time = get_wall_seconds() - time;
  std::cout << "Total time of benchmark: " << time << " s." << std::endl;
  std::cout << "Average time per iteration of benchmark: " << time/turns << " s/iteration." << std::endl;

  free_triple_ptr(&q);
  free_triple_ptr(&F);
  delete[] q_flat;
  delete[] F_flat;

  return 0;
}
