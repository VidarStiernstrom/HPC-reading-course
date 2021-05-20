static char help[] ="Solves the 2D acoustic wave equation on first order form: u_t = A*u_x + B*u_y, A = [0,0,-1;0,0,0;-1,0,0], B = [0,0,0;0,0,-1;0,-1,0].";


/**
* Solves 2D acoustic wave equation on first order form. See http://sepwww.stanford.edu/sep/prof/bei/fdm/paper_html/node40.html
* Variables:
* u - x-velocity
* v - y-velocity
* p - pressure
* 
* Coefficients: 
* rho - density
* K - incompressibility
* 
* Equations:
* ut = -1/rho(x) px + F1
* vt = -1/rho(x) py + F2
* pt = -K(x)*(ux + vy)
* 
* F1, F2 - velocity forcing data
* 
**/

#define PROBLEM_TYPE_2D_O6

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/wave_eq.h"
#include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

extern PetscErrorCode analytic_solution(DM, PetscScalar, AppCtx&, Vec&);
extern PetscErrorCode rhs_TS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode rhs(DM, PetscReal, Vec, Vec, AppCtx *);
extern PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx& appctx);
extern PetscErrorCode get_error(const DM& da, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, procx, procy, dofs;
  PetscScalar    xl, xr, yl, yr, hix, hiy, dt, t0, Tend, CFL;
  PetscReal      l2_error, max_error, H_error;

  AppCtx         appctx;
  PetscBool      write_data, use_custom_ts, use_custom_sc;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (get_inputs_2d(argc, argv, &Nx, &Ny, &Tend, &CFL, &use_custom_ts, &use_custom_sc) == -1) {
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
  auto a = [xl,yl,hix,hiy](const PetscInt i, const PetscInt j){  // 1/rho.
    PetscScalar x = xl + i/hix;
    PetscScalar y = yl + j/hiy;
    return 1.0/(2 + x*y);
  };
  auto b = [](const PetscInt i, const PetscInt j){ return 1;}; // unused at the moment, K = 1.

  // Set if data should be written.
  write_data = PETSC_TRUE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  stencil_radius = (appctx.D1.interior_stencil_width()-1)/2;
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
  appctx.a = a;
  appctx.b = b;
  appctx.sw = stencil_radius;
  appctx.layout = grid::create_layout_2d(da);

  // Extract local to local scatter context
  if (use_custom_sc) {
    build_ltol_2D(da, &appctx.scatctx);
  } else {
    DMDAGetScatter(da, NULL, &appctx.scatctx);
  }
  
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

  if (use_custom_ts) {
    RK4_custom(da, appctx, Tend, dt, vlocal, rhs);  
  } else {
    RK4_petsc(da, appctx, Tend, dt, vlocal, rhs_TS);  
  }
  
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

PetscErrorCode rhs_TS(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx) // Function to utilize PETSc TS.
{
  AppCtx *appctx = (AppCtx*) ctx;
  DM                da;

  TSGetDM(ts,&da);
  rhs(da, t, v_src, v_dst, appctx);
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

PetscErrorCode rhs(DM da, PetscReal t, Vec v_src, Vec v_dst, AppCtx *appctx)
{
  PetscScalar       *array_src, *array_dst;

  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);

  auto gf_src = grid::grid_function_2d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_2d<PetscScalar>(array_dst, appctx->layout);
  const std::array<PetscInt,2> ind_i = {appctx->i_start[0], appctx->i_end[0]};
  const std::array<PetscInt,2> ind_j = {appctx->i_start[1], appctx->i_end[1]};
  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  wave_eq_local(gf_dst, gf_src, ind_i, ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->xl, t);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  wave_eq_overlap(gf_dst, gf_src, ind_i, ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->xl, t);
  wave_eq_free_surface_bc(gf_dst, gf_src, ind_i, ind_j, appctx->HI, appctx->hi);
  //wave_eq_serial(gf_dst, gf_src, appctx->D1, appctx->hi, appctx->xl, t);
  //wave_eq_free_surface_bc_serial(gf_dst, gf_src, appctx->HI, appctx->hi);

  // Restore arrays
  VecRestoreArray(v_src,&array_src);
  VecRestoreArray(v_dst,&array_dst);
  return 0;
}

