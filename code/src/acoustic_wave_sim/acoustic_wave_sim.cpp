static char help[] ="Solves the 2D acoustic wave equation on first order form: u_t = A*u_x + B*u_y, A = [0,0,-1;0,0,0;-1,0,0], B = [0,0,0;0,0,-1;0,-1,0].";


/**
* Solves 2D acoustic wave equation on first order form. See http://sepwww.stanford.edu/sep/prof/bei/fdm/paper_html/node40.html
* Variables:
* u(x,y) - x-velocity
* v(x,y) - y-velocity
* p(x,y) - pressure
* 
* Coefficients: 
* rho(x,y) - density
* K(x,y) - incompressibility. At the moment K(x) is assumed to be 1
* 
* Equations:
* u_t = -1/rho(x,y) p_x + F1
* v_t = -1/rho(x,y) p_y + F2
* p_t = -K(x,y)*(u_x + q_y)
* 
* F1, F2 - velocity forcing data
*
* The unkowns are stored in a vector q = [u,v,p]^T.
* 
**/
#include <functional>
#include <petsc.h>
#include "sbpops/op_defs.h"
#include "acoustic_wave_eq/wave_eq_rhs.h"
#include "timestepping/timestepping.h"
#include "io/IO_utils.h"

/** 
* A user defined application context, storing the relevant information used by
* PETSc and application routines.
**/
struct AppCtx{
    std::array<PetscInt,2> N, i_start, i_end;
    std::array<PetscScalar,2> hi, xl;
    const DifferenceOp D1; // Difference operator
    const InverseNormOp HI; // Inverse norm operator
};

/* Functions used by the PETSc time stepping routines */
PetscErrorCode rhs(TS, PetscReal, Vec, Vec, void *);
PetscErrorCode rhs_serial(TS, PetscReal, Vec, Vec, void *);
/* Utility functions used to initialize the problem and compute errors */
PetscErrorCode analytic_solution(DM, PetscScalar, AppCtx&, Vec);
PetscErrorCode set_initial_condition(DM, AppCtx&, Vec);
PetscScalar compute_l2_error(Vec, Vec, PetscScalar, PetscScalar);
PetscScalar compute_max_error(Vec, Vec);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            q, q_analytic;
  PetscInt       sw, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, procx, procy, dofs;
  PetscScalar    xl, xr, yl, yr, hx, hy, dt, Tend, n_steps, CFL;
  PetscReal      l2_error, max_error;

  AppCtx         appctx;
  PetscLogDouble t1,t2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

if (get_inputs(argc, argv, &Nx, &Ny, &n_steps, &CFL) == -1) {
    PetscEnd();
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  dofs = 3;
  xl = -1;
  xr = 1;
  yl = -1;
  yr = 1;
  hx = (xr-xl)/(Nx-1);
  hy = (yr-yl)/(Ny-1);
  dt = CFL*(std::min(hx,hy)); // Time step
  Tend = n_steps*dt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  sw = (appctx.D1.interior_stencil_size()-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,dofs,sw,NULL,NULL,&da);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDAGetCorners(da,&i_xstart,&i_ystart,NULL,&nx,&ny,NULL);
  i_xend = i_xstart + nx;
  i_yend = i_ystart + ny;

  DMDAGetInfo(da,NULL,NULL,NULL,NULL,&procx,&procy,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"Processor topology dimensions: [%d,%d]\n",procx,procy);
  if ((procx == 1) || (procy == 1)) {
    PetscPrintf(PETSC_COMM_WORLD,"--- Warning ---\nOne dimensional topology\n");
  }
  
  if (i_xend < appctx.D1.closure_stencil_size() || i_yend < appctx.D1.closure_stencil_size() ||
      i_xstart > (Nx-appctx.D1.closure_stencil_size()) || i_ystart > (Ny-appctx.D1.closure_stencil_size())) {
    PetscPrintf(PETSC_COMM_WORLD,"--- Error ---\nSubdividing closure region is not supported\n");
    PetscEnd();
  }



  // Populate application context.
  appctx.N = {Nx, Ny};
  appctx.hi = {1./hx, 1./hy};
  appctx.xl = {xl, yl};
  appctx.i_start = {i_xstart,i_ystart};
  appctx.i_end = {i_xend,i_yend};

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&q);
  VecDuplicate(q,&q_analytic);
  
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Set initial condition and create local vector used in the stencil computations
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  set_initial_condition(da, appctx, q);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) q);
  if (rank == 0) {
    PetscTime(&t1); // Start timer
  }

  if (size == 1) // If single processor, run the serial version
    time_integrate_rk4(da, Tend, dt, q, rhs_serial, (void *)&appctx);  
  else
    time_integrate_rk4(da, Tend, dt, q, rhs, (void *)&appctx);  

  PetscBarrier((PetscObject) q);
  if (rank == 0) {
    PetscTime(&t2); // Stop timer
    elapsed_time = t2 - t1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);
  analytic_solution(da, Tend, appctx, q_analytic);

  l2_error = compute_l2_error(q, q_analytic, hx, hy);
  max_error = compute_max_error(q, q_analytic);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, and the maximum error is %g\n",l2_error,max_error);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&q);
  VecDestroy(&q_analytic);
  DMDestroy(&da);
  
  ierr = PetscFinalize();
  return ierr;
}

/**
* Interface for the PETSc timestepping (TS) user-defined right-hand-side function.
* ts - Timestepping context
* t - current time
* q - global source vector (read from)
* F - global dst vector (written to)
* ctx - user defined application context.
**/
PetscErrorCode rhs(TS ts, PetscReal t, Vec q, Vec F, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;
  Vec q_local;
  PetscScalar       ***q_arr, ***F_arr;

  TSGetDM(ts,&da);
  DMGetLocalVector(da, &q_local);
  DMGlobalToLocalBegin(da,q,INSERT_VALUES,q_local);  
  DMGlobalToLocalEnd(da,q,INSERT_VALUES,q_local);

  // Get the underlying array for the local src vector
  // and the global dst vector.
  DMDAVecGetArrayDOF(da,q_local,&q_arr);
  DMDAVecGetArrayDOF(da,F,&F_arr);
  // Apply stencil (here no overlap in communication and computations are used)
  wave_eq_rhs_local(t, appctx->D1, appctx->HI, q_arr, F_arr, appctx->i_start, appctx->i_end, appctx->N, appctx->xl, appctx->hi);
  wave_eq_rhs_overlap(t, appctx->D1, appctx->HI, q_arr, F_arr, appctx->i_start, appctx->i_end, appctx->N, appctx->xl, appctx->hi);
  // Restore arrays
  DMRestoreLocalVector(da,&q_local);
  DMDAVecRestoreArrayDOF(da,F,&F_arr);
  return 0;
}

/**
* Interface for the PETSc timestepping (TS) user-defined right-hand-side function for serial runs.
* ts - Timestepping context
* t - current time
* q - global source vector (read from)
* F - global dst vector (written to)
* ctx - user defined application context.
**/
PetscErrorCode rhs_serial(TS ts, PetscReal t, Vec q, Vec F, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;
  PetscScalar       ***q_arr, ***F_arr;

  TSGetDM(ts,&da);
  DMDAVecGetArrayDOFRead(da,q,&q_arr);
  DMDAVecGetArrayDOF(da,F,&F_arr);

  wave_eq_rhs_serial(t, appctx->D1, appctx->HI, q_arr, F_arr, appctx->N, appctx->xl, appctx->hi);

  // Restore arrays
  DMDAVecRestoreArrayDOFRead(da,q,&q_arr);
  DMDAVecRestoreArrayDOF(da,F,&F_arr);
  return 0;
}

/**
* Computes the analytic (exact) solution to the problem at time t.
* da - Distributed array context
* appctx - application context, contains necessary information
* q - vector to store analytic solution
**/
PetscErrorCode analytic_solution(DM da, PetscScalar t, AppCtx &appctx, Vec q) {
  PetscInt i, j, n, m; 
  PetscScalar ***q_arr, x, y;

  n = 3; // n = 1,2,3,4,...
  m = 4; // m = 1,2,3,4,...

  DMDAVecGetArrayDOF(da,q,&q_arr);    

  for (j = appctx.i_start[1]; j < appctx.i_end[1]; j++)
  {
    y = appctx.xl[1] + j/appctx.hi[1];
    for (i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
    {
      x = appctx.xl[0] + i/appctx.hi[0];
      q_arr[j][i][0] = -n*cos(n*PETSC_PI*x)*sin(m*PETSC_PI*y)*sin(PETSC_PI*sqrt(n*n + m*m)*t)/sqrt(n*n + m*m);
      q_arr[j][i][1] = -m*sin(n*PETSC_PI*x)*cos(m*PETSC_PI*y)*sin(PETSC_PI*sqrt(n*n + m*m)*t)/sqrt(n*n + m*m);
      q_arr[j][i][2] = sin(PETSC_PI*n*x)*sin(PETSC_PI*m*y)*cos(PETSC_PI*sqrt(n*n + m*m)*t);
    }
  }
  DMDAVecRestoreArrayDOF(da,q,&q_arr);  
  return 0;
}

/**
* Set initial condition. The inital condition is computed using the analytic solution at t = t0
* da - Distributed array context
* appctx - application context, contains necessary information
* q - vector to store initial data
**/
PetscErrorCode set_initial_condition(DM da, AppCtx& appctx, Vec q) 
{
  analytic_solution(da, 0, appctx, q);
  return 0;
}

/**
* Computes the l2 error between the vectors q1 and a2
* q1, q2 - vectors being compared.
* h - array storing grid spacings
**/
PetscScalar compute_l2_error(Vec q1, Vec q2, PetscScalar hx, PetscScalar hy) {
  PetscScalar l2_error;
  Vec q_error;
  VecDuplicate(q1,&q_error);
  VecWAXPY(q_error,-1,q1,q2);
  VecNorm(q_error,NORM_2,&l2_error);
  VecDestroy(&q_error);
  return l2_error = sqrt(hx*hy)*(l2_error);
}

/**
* Computes the max error between the vectors q1 and a2
* q1, q2 - vectors being compared.
**/
PetscScalar compute_max_error(Vec q1, Vec q2) {
  PetscScalar max_error;
  Vec q_error;
  VecDuplicate(q1,&q_error);
  VecWAXPY(q_error,-1,q1,q2);
  VecNorm(q_error,NORM_INFINITY,&max_error);
  VecDestroy(&q_error);
  return max_error;
}

