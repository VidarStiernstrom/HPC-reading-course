#include "benchmark_utils.h"
void petsc_triple_ptr_layout(PetscScalar *q, PetscInt N, PetscInt dofs, PetscScalar ****q_3ptr) {
  PetscInt i, j;
  PetscScalar **b;
  PetscMalloc1(N*sizeof(PetscScalar**)+N*N,q_3ptr);
  b = (PetscScalar**)((*q_3ptr) + N);
  for (j = 0; j < N; j++)
    (*q_3ptr)[j] = b + j*N;

  for (j = 0; j < N; j++) 
    for (i = 0; i < N; i++)
      b[j*N+i] = q + j*N*dofs + i*dofs;
};

void free_triple_ptr(PetscScalar ****q_3ptr) {
  void * dummy = (void *)(*q_3ptr);
  PetscFree(dummy);
};