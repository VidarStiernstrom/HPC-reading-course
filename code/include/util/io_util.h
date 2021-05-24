#pragma once

#include<petsc.h>
#include<string>

PetscErrorCode write_vector_to_binary(const Vec, const std::string, const std::string);

PetscErrorCode write_data_to_file(const std::string data_string, const std::string folder, const std::string file);

void print_usage_1d(char* exec_name);

int get_inputs(int argc, char *argv[], PetscInt& N, PetscScalar& Tend, PetscScalar& CFL, PetscBool& use_custom_sc);

void print_usage_2d(char* exec_name);

int get_inputs(int argc, char *argv[], PetscInt& Nx, PetscInt& Ny, PetscScalar& Tend, PetscScalar& CFL, PetscBool& use_custom_sc);