#pragma once

#include<petsc.h>

PetscErrorCode scatter_ctx_ltol(DM da, VecScatter& ltol);