#pragma once

#include<petsc.h>

// TODO: combine into one function

PetscErrorCode build_ltol_1D(DM da, VecScatter *ltol);
PetscErrorCode build_ltol_2D(DM da, VecScatter *ltol);