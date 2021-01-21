#pragma once

#include<petsc.h>
#include "appctx.h"

PetscErrorCode setup_timestepper(TimeCtx& timectx, PetscScalar tau) ;