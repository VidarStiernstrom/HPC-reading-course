#pragma once

#include<petsc.h>
#include "appctx.h"

PetscErrorCode standard_solver(Mat &A, Vec& v, std::string filename_reshist);