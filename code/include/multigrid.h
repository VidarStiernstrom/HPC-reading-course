#pragma once

#include<petsc.h>
#include "appctx.h"

extern PetscErrorCode mgsolver(Mat& Afine, Vec& v, PetscInt nvlevels);