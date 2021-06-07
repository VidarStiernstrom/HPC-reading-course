#pragma once
#include <petsc.h>
/**
 * Assigns q_3ptr to the memory of q, q is a flat array holding N*N*dofs elements. The elements are accessed as
 * q_3ptr[j][i][dof]. q_3ptr should be freed after use.
 * 
 * The code is stripped from the sequential parts of VecGetArray3d() PETSc v3.15
 **/
void petsc_triple_ptr_layout(PetscScalar *q, PetscInt N, PetscInt dofs, PetscScalar ****q_3ptr);

/**
 * Frees memory of q_3ptr
 * 
 * The code is stripped from the sequential parts of VecRestoreArray3d() PETSc v3.15
 **/
void free_triple_ptr(PetscScalar ****q_3ptr);