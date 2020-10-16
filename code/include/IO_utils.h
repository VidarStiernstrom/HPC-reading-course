#pragma once

#include<petsc.h>

void print_usage_2d(char* exec_name);

int get_inputs_2d(int argc, char *argv[], PetscInt *Nx, PetscInt *Ny, PetscScalar *Tend, PetscScalar *CFL, PetscBool *use_custom_ts, PetscBool *use_custom_sc);

void print_usage_1d(char* exec_name);

int get_inputs_1d(int argc, char *argv[], PetscInt *N, PetscScalar *Tend, PetscScalar *CFL, PetscBool *use_custom_ts, PetscBool *use_custom_sc);