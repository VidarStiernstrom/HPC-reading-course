#pragma once

#include<petsc.h>
#include "appctx.h"

// TODO: combine into one function

extern PetscErrorCode setup_mgsolver(KSP& ksp_fine, PetscInt nlevels, Mat& Afine, Mat Acoarses[], Mat P[], Mat R[]);