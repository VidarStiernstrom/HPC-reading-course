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
* p_t = -K(x,y)*(u_x + v_y)
* 
* F1, F2 - velocity forcing data
* 
**/

#define SBP_OPERATOR_ORDER 2

#include <petsc.h>
#include "sbpops/op_defs.h"
#include "acoustic_wave_eq/acowave.h"
#include "timestepping/timestepping.h"
#include "io/IO_utils.h"
#include "scatter_ctx/scatter_ctx.h"


struct AppCtx{
    std::array<PetscInt,2> N, i_start, i_end;
    std::array<PetscScalar,2> hi, h, xl;
    PetscInt dofs;
    PetscScalar sw;
    std::function<PetscScalar(PetscInt, PetscInt)> rho_inv;
    const DifferenceOp D1;
    const NormOp H;
    const InverseNormOp HI;
    VecScatter scatctx;
};

extern PetscErrorCode analytic_solution(DM, PetscScalar, AppCtx&, Vec&);
extern PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx& appctx);
extern PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx);
extern PetscErrorCode rhs(DM, PetscReal, Vec, Vec, AppCtx *);
extern PetscErrorCode rhs_TS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode rhs_serial(DM, PetscReal, Vec, Vec, AppCtx *);
extern PetscErrorCode rhs_TS_serial(TS, PetscReal, Vec, Vec, void *);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, procx, procy, dofs;
  PetscScalar    xl, xr, yl, yr, hix, hiy, dt, t0, Tend, CFL;
  PetscReal      l2_error, max_error, H_error;

  AppCtx         appctx;
  PetscBool      write_data;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (get_inputs(argc, argv, &Nx, &Ny, &Tend, &CFL) == -1) {
    PetscFinalize();
    return -1;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // Space
  dofs = 3;
  xl = -1;
  xr = 1;
  yl = -1;
  yr = 1;
  hix = (Nx-1)/(xr-xl);
  hiy = (Ny-1)/(yr-yl);
  
  // Time
  t0 = 0;
  dt = CFL/(std::min(hix,hiy));

  // Velocity field a(i,j) = 1
  auto rho_inv = [xl,yl,hix,hiy](const PetscInt i, const PetscInt j){  // 1/rho.
    PetscScalar x = xl + i/hix;
    PetscScalar y = yl + j/hiy;
    return 1.0/(2 + x*y);
  };

  // Set if data should be written.
  write_data = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,dofs,stencil_radius,NULL,NULL,&da);
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

  // Populate application context.
  appctx.N = {Nx, Ny};
  appctx.hi = {hix, hiy};
  appctx.h = {1./hix, 1./hiy};
  appctx.xl = {xl, yl};
  appctx.i_start = {i_xstart,i_ystart};
  appctx.i_end = {i_xend,i_yend};
  appctx.dofs = dofs;
  appctx.rho_inv = rho_inv;
  appctx.sw = stencil_radius;

  // Extract local to local scatter context
  scatter_ctx_ltol(da, &appctx.scatctx);

  
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);
  
  // Initial solution, starting time and end time.
  set_initial_condition(da, v, appctx);

  if (write_data) write_vector_to_binary(v,"data/acowave_2D","v_init");

  ierr = DMCreateLocalVector(da,&vlocal);CHKERRQ(ierr);
  DMGlobalToLocalBegin(da,v,INSERT_VALUES,vlocal);  
  DMGlobalToLocalEnd(da,v,INSERT_VALUES,vlocal);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  if (size == 1)
    time_integrate_rk4(da, Tend, dt, vlocal, rhs_TS_serial, (void *)&appctx);  
  else
    time_integrate_rk4(da, Tend, dt, vlocal, rhs_TS, (void *)&appctx);  

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  DMLocalToGlobalBegin(da,vlocal,INSERT_VALUES,v);
  DMLocalToGlobalEnd(da,vlocal,INSERT_VALUES,v);

  analytic_solution(da, Tend, appctx, v_analytic);
  get_error(da, v, v_analytic, &v_error, &H_error, &l2_error, &max_error, appctx);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, the H-error is: %g and the maximum error is %g\n",l2_error,H_error,max_error);

  // Write solution to file
  if (write_data)
  {
    write_vector_to_binary(v,"data/acowave_2D","v");
    write_vector_to_binary(v_error,"data/acowave_2D","v_error");

    char tmp_str[200];
    std::string data_string;
    sprintf(tmp_str,"%d\t%d\t%d\t%e\t%f\t%f\t%e\t%e\t%e\n",size,Nx,Ny,dt,Tend,elapsed_time,l2_error,H_error,max_error);
    data_string.assign(tmp_str);
    write_data_to_file(data_string, "data/acowave_2D", "data.tsv");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_analytic);
  VecDestroy(&v_error);
  DMDestroy(&da);
  
  ierr = PetscFinalize();
  return ierr;
}

/**
* Set initial condition.
* Inputs: v      - vector to place initial data
*         appctx - application context, contains necessary information
**/
PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx& appctx) 
{
  analytic_solution(da, 0, appctx, v);
  return 0;
}

PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx) {
  PetscScalar ***arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);
  *l2_error = sqrt(appctx.h[0]*appctx.h[1])*(*l2_error);

  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(da, *v_error, &arr);
  *H_error = appctx.H.get_norm_2D(arr, appctx.h, appctx.N, appctx.i_start, appctx.i_end, appctx.dofs);

  DMDAVecRestoreArrayDOF(da, *v_error, &arr);

  return 0;
}

PetscErrorCode analytic_solution(DM da, PetscScalar t, AppCtx &appctx, Vec& v) {
  PetscInt i, j, n, m; 
  PetscScalar ***varr, x, y;

  n = 3; // n = 1,2,3,4,...
  m = 4; // m = 1,2,3,4,...

  DMDAVecGetArrayDOF(da,v,&varr);    

  for (j = appctx.i_start[1]; j < appctx.i_end[1]; j++)
  {
    y = appctx.xl[1] + j/appctx.hi[1];
    for (i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
    {
      x = appctx.xl[0] + i/appctx.hi[0];
      varr[j][i][0] = -n*cos(n*PETSC_PI*x)*sin(m*PETSC_PI*y)*sin(PETSC_PI*sqrt(n*n + m*m)*t)/sqrt(n*n + m*m);
      varr[j][i][1] = -m*sin(n*PETSC_PI*x)*cos(m*PETSC_PI*y)*sin(PETSC_PI*sqrt(n*n + m*m)*t)/sqrt(n*n + m*m);
      varr[j][i][2] = sin(PETSC_PI*n*x)*sin(PETSC_PI*m*y)*cos(PETSC_PI*sqrt(n*n + m*m)*t);
    }
  }
  DMDAVecRestoreArrayDOF(da,v,&varr);  
  return 0;
}

PetscErrorCode rhs(DM da, PetscReal t, Vec v_src, Vec v_dst, AppCtx *appctx)
{
  PetscScalar       ***array_src, ***array_dst;

  DMDAVecGetArrayDOFRead(da,v_src,&array_src);
  DMDAVecGetArrayDOF(da,v_dst,&array_dst);

  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  acowave_apply_interior(t, appctx->D1, appctx->HI, appctx->rho_inv, array_src, array_dst, appctx->i_start, appctx->i_end, appctx->N, appctx->xl, appctx->hi, appctx->sw);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  acowave_apply_overlap(t, appctx->D1, appctx->HI, appctx->rho_inv, array_src, array_dst, appctx->i_start, appctx->i_end, appctx->N, appctx->xl, appctx->hi, appctx->sw);
  
  // Restore arrays
  DMDAVecRestoreArrayDOFRead(da,v_src,&array_src);
  DMDAVecRestoreArrayDOF(da,v_dst,&array_dst);
  return 0;
}

PetscErrorCode rhs_TS(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) // Function to utilize PETSc TS.
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;

  TSGetDM(ts,&da);
  rhs(da, t, v_src, v_dst, appctx);
  return 0;
}

//
// Serial versions of rhs and rhs_TS used for single processor runs
// 

PetscErrorCode rhs_serial(DM da, PetscReal t, Vec v_src, Vec v_dst, AppCtx *appctx)
{
  PetscScalar       ***array_src, ***array_dst;

  DMDAVecGetArrayDOFRead(da,v_src,&array_src);
  DMDAVecGetArrayDOF(da,v_dst,&array_dst);

  acowave_apply_serial(t, appctx->D1, appctx->HI, appctx->rho_inv, array_src, array_dst, appctx->N, appctx->xl, appctx->hi, appctx->sw);

  // Restore arrays
  DMDAVecRestoreArrayDOFRead(da,v_src,&array_src);
  DMDAVecRestoreArrayDOF(da,v_dst,&array_dst);
  return 0;
}

PetscErrorCode rhs_TS_serial(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) // Function to utilize PETSc TS.
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;

  TSGetDM(ts,&da);
  rhs_serial(da, t, v_src, v_dst, appctx);
  return 0;
}

