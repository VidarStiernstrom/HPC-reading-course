/**
 * Program to verify the data layout obtained using DMDA
 *
 *
 *
 **/
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "benchmark_utils.h"
#include <petscdmda.h>

// Initialization util
PetscScalar* init_array(PetscInt Ntot) {
  PetscScalar *q = new PetscScalar[Ntot];
  for  (int i = 0; i<Ntot; i++)
    q[i] = (PetscScalar)rand() / RAND_MAX;
  return q;
};


void init_DMDADOF_array(DM da, PetscInt N, PetscInt dofs, Vec q) {
  PetscScalar ***q_arr;
  DMDAVecGetArrayDOF(da,q,&q_arr);

  for (PetscInt j = 0; j < N; j++) {
    for (PetscInt i = 0; i < N; i++) {
      for (PetscInt k = 0; k < dofs; k++) {
        q_arr[j][i][k] = (PetscScalar)rand() / RAND_MAX;
      }
    }
  }
  DMDAVecRestoreArrayDOF(da,q,&q_arr);
};

// Test util
void print_test_status(bool status, std::string test_name)
{
  if (status)
    std::cout << test_name << ": Passed!" << std::endl;
  else
    std::cout << test_name  << ": Failed!" << std::endl;
};

bool cmp_arrays(PetscScalar *q, PetscScalar ***q_3ptr, PetscInt N, PetscInt dofs) {
  bool passed = true;
  PetscInt counter = 0;
  for (PetscInt j = 0; j < N; j++) {
    for (PetscInt i = 0; i < N; i++) {
      for (PetscInt k = 0; k < dofs; k++) {
        passed *= q[counter] == q_3ptr[j][i][k];
        counter++;
      }
    }
  }
  return passed;
}

// Tests
void test_array_layout(PetscInt N, PetscInt dofs) {
  PetscScalar *q;
  PetscScalar ***q_3ptr;

  q = init_array(N*N*dofs);
  petsc_triple_ptr_layout(q, N, dofs, &q_3ptr);

  bool status = cmp_arrays(q, q_3ptr, N, dofs);
  print_test_status(status,"test_array_layout");

  delete[] q;
  free_triple_ptr(&q_3ptr);
};


void test_DMDADOF_layout(PetscInt N, PetscInt dofs) {
  DM da;
  Vec q;
  PetscScalar * q_arr;
  PetscScalar ***q_3ptr;
  // Create DMDA
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               N,N,PETSC_DECIDE,PETSC_DECIDE,dofs,0,NULL,NULL,&da);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMCreateGlobalVector(da,&q);
  // Initialize array using DOF layout
  init_DMDADOF_array(da, N, dofs, q);

  // Cmp arrays
  VecGetArray(q,&q_arr);
  petsc_triple_ptr_layout(q_arr, N, dofs, &q_3ptr);
  bool status = cmp_arrays(q_arr, q_3ptr, N, dofs);
  print_test_status(status,"test_DMDADOF_layout");
  VecRestoreArray(q,&q_arr);
  free_triple_ptr(&q_3ptr);
  VecDestroy(&q);
  DMDestroy(&da);
};


int main(int argc,char **argv) {
  PetscMPIInt    size, rank;
  PetscInitialize(&argc,&argv,(char*)0,NULL);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  assert(size == 1); //Program is illformed when running more than 1 process.

  PetscInt N = 801;
  PetscInt dofs = 3;
  test_array_layout(N, dofs);
  test_DMDADOF_layout(N, dofs);
  PetscFinalize();
  return 0;
}
