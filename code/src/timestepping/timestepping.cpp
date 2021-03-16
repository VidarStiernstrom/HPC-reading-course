#include "timestepping/timestepping.h"
#include <petscts.h>

PetscErrorCode time_integrate_rk4(const DM& da, const PetscScalar Tend, const PetscScalar dt, Vec& v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* appctx)
{
  TS             ts;
  TSAdapt        adapt;

  TSCreate(PETSC_COMM_WORLD, &ts);
  
  // Problem type and RHS function
  TSSetProblemType(ts, TS_LINEAR);
  TSSetRHSFunction(ts, NULL, rhs, appctx);
  
  // Integrator
  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK4);
  TSGetAdapt(ts, &adapt);
  TSAdaptSetType(adapt,TSADAPTNONE);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  // DM context
  TSSetDM(ts,da);

  TSSetSolution(ts, v);
  TSSetTime(ts,0);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,Tend);

  // Set all options
  TSSetFromOptions(ts);

  // Simulate
  TSSolve(ts,v);

  TSDestroy(&ts);

  return 0;
}