#pragma once

#include<petsc.h>

PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);

PetscErrorCode write_data_to_file(const std::string data_string, const std::string folder, const std::string file);

void print_usage(char* exec_name);

int get_inputs(int argc, char *argv[], PetscInt *Nx, PetscInt *Ny, PetscScalar *Tend, PetscScalar *CFL);