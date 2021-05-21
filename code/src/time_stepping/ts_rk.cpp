#include "time_stepping/ts_rk.h"
#include <array>

/**
* Utility function for setting up Runge-Kutta TS context with user defined rhs function describing a linear system of ODE:s
* Inputs: ts          - TS context
*         rk_type     - The type of RK method
*         adapt_type  - Adaptivity method
*         da          - DMDA context
*         t_span      - array holding initial and final times
*         dt          - Time step
*         rhs         - RHS function. Inputs (required by petsc): (TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) 
*         ctx         - User defined context
**/
PetscErrorCode ts_rk_setup(TS& ts, const TSRKType rk_type, const TSAdaptType adapt_type, const DM da, const std::array<PetscScalar,2>& t_span, const PetscScalar dt, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* ctx)
{
  TSAdapt        adapt;
  TSCreate(PETSC_COMM_WORLD, &ts);
  // Problem type and RHS function
  TSSetProblemType(ts, TS_LINEAR);
  TSSetRHSFunction(ts, NULL, rhs, ctx);
  
  // Specify Integrator
  TSSetType(ts,TSRK);
  TSRKSetType(ts,rk_type);
  TSGetAdapt(ts, &adapt);
  TSAdaptSetType(adapt,adapt_type);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  // Set DM context
  TSSetDM(ts,da);
  TSSetTime(ts,t_span[0]);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,t_span[1]);

  // Set all options
  TSSetFromOptions(ts);
  return 0;
};

PetscErrorCode ts_rk45(const DM da, const PetscScalar t_end, const PetscScalar dt, Vec v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* ctx)
{
  TS             ts;
  // Setup context
  ts_rk_setup(ts, TSRK5F, TSADAPTBASIC, da, {0,t_end}, dt, rhs, ctx);
  // Set initial condition and solve
  TSSetSolution(ts, v);
  TSSolve(ts,v);

  TSDestroy(&ts);
  return 0;
}

PetscErrorCode ts_rk4(const DM da, const PetscScalar t_end, const PetscScalar dt, Vec v, PetscErrorCode (*rhs)(TS, PetscReal, Vec, Vec, void *), void* ctx)
{
  TS             ts;
  // Setup context
  ts_rk_setup(ts, TSRK4, TSADAPTNONE, da, {0,t_end}, dt, rhs, ctx);
  // Set initial condition and solve
  TSSetSolution(ts, v);
  TSSolve(ts,v);

  TSDestroy(&ts);
  return 0;
}