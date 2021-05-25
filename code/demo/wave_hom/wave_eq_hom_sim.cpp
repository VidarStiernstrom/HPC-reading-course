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
* Material parameters and forcing functions are defined in wave_eq_rhs.h
* 
**/

#include <petsc.h>
#include <array>
#include "wave_eq_hom_rhs.h"
#include "sbpops/op_defs.h"
#include "time_stepping/ts_rk.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "scatter_ctx/scatter_ctx.h"
#include "util/io_util.h"
#include "util/vec_util.h"

struct AppCtx{
    std::array<PetscInt,2> N, ind_i, ind_j;
    std::array<PetscScalar,2> hi, h, xl;
    PetscInt dofs;
    PetscScalar sw;
    const FirstDerivativeOp D1;
    const InverseNormOp HI;
    VecScatter scatctx;
    grid::partitioned_layout_2d layout;
};

PetscErrorCode initial_condition(const DM da, Vec v, const AppCtx& appctx);
PetscErrorCode analytic_solution(const DM, const PetscScalar, const AppCtx&, Vec);
PetscErrorCode rhs(TS, PetscReal, Vec, Vec, void *);
PetscErrorCode rhs_non_overlapping(TS, PetscReal, Vec, Vec, void *);
PetscErrorCode rhs_serial(TS, PetscReal, Vec, Vec, void *);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, procx, procy, dofs;
  PetscScalar    xl, xr, yl, yr, hix, hiy, dt, t0, Tend, CFL;
  PetscReal      l2_error, max_error;

  AppCtx         appctx;
  PetscBool      write_data, use_custom_sc;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (get_inputs(argc, argv, Nx, Ny, Tend, CFL, use_custom_sc) == -1) {
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

  // Set if data should be written.
  write_data = PETSC_FALSE;

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

  if (procx != procy) {
    PetscPrintf(PETSC_COMM_WORLD,"--- Warning ---\nNon-square topology\n");
  }

  // Populate application context.
  appctx.N = {Nx, Ny};
  appctx.hi = {hix, hiy};
  appctx.h = {1./hix, 1./hiy};
  appctx.xl = {xl, yl};
  appctx.ind_i = {i_xstart,i_xend};
  appctx.ind_j = {i_ystart,i_yend};
  appctx.dofs = dofs;
  appctx.sw = stencil_radius;
  appctx.layout = grid::create_layout_2d(da);

  // Extract local to local scatter context
  if (use_custom_sc) {
    scatter_ctx_ltol(da, appctx.scatctx);
  } else {
    DMDAGetScatter(da, NULL, &appctx.scatctx);
  }
  
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);  
  // Initial solution, starting time and end time.
  initial_condition(da, v, appctx);

  if (write_data) write_vector_to_binary(v,"data/wave","v_init");

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

  // TODO: Add runtime flag checking for overlapping or non-overlapping RHS
  if (size == 1) {
    ts_rk4(da, Tend, dt, vlocal, rhs_serial, &appctx);  
  }
  else {
    ts_rk4(da, Tend, dt, vlocal, rhs, &appctx); 
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
  l2_error = error_l2(v,v_analytic, appctx.h);
  max_error = error_max(v,v_analytic);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %g, and the maximum error is %g\n",l2_error,max_error);

  // Print outs to generate .csv file. Headers: mode, order, custom sc, size, Nx, nx, error, elap time
  // PetscPrintf(PETSC_COMM_WORLD,"local,%s,%d,%d,%d,%d,%f,%f\n",getenv("order"),use_custom_sc,size,Nx,nx,l2_error,elapsed_time);

  // Write solution to file
  if (write_data) {
    write_vector_to_binary(v,"data/wave","v");
    Vec v_error = compute_error(v,v_analytic);
    write_vector_to_binary(v_error,"data/wave","v_error");
    VecDestroy(&v_error);
    char tmp_str[200];
    std::string data_string;
    sprintf(tmp_str,"%d\t%d\t%d\t%e\t%f\t%f\t%e\t%e\n",size,Nx,Ny,dt,Tend,elapsed_time,l2_error,max_error);
    data_string.assign(tmp_str);
    write_data_to_file(data_string, "data/wave", "data.tsv");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_analytic);
  DMDestroy(&da);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode analytic_solution(const DM da, const PetscScalar t, const AppCtx &appctx, Vec v) {
  PetscInt i, j, n, m; 
  PetscScalar ***varr, x, y;

  n = 3; // n = 1,2,3,4,...
  m = 4; // m = 1,2,3,4,...

  DMDAVecGetArrayDOF(da,v,&varr);    

  for (j = appctx.ind_j[0]; j < appctx.ind_j[1]; j++)
  {
    y = appctx.xl[1] + j*appctx.h[1];
    for (i = appctx.ind_i[0]; i < appctx.ind_i[1]; i++)
    {
      x = appctx.xl[0] + i*appctx.h[0];
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
PetscErrorCode initial_condition(const DM da, Vec v, const AppCtx& appctx) 
{
  analytic_solution(da, 0, appctx, v);
  return 0;
}

PetscErrorCode rhs(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  PetscScalar       *array_src, *array_dst;

  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);

  auto gf_src = grid::grid_function_2d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_2d<PetscScalar>(array_dst, appctx->layout);

  // Overlapping
  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  wave_eq_hom_local(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->xl, t);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  wave_eq_hom_overlap(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->xl, t);
  wave_eq_hom_free_surface_bc(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->HI, appctx->hi);

  // Restore arrays
  VecRestoreArray(v_src,&array_src);
  VecRestoreArray(v_dst,&array_dst);
  return 0;
}

PetscErrorCode rhs_non_overlapping(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  PetscScalar       *array_src, *array_dst;

  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);

  auto gf_src = grid::grid_function_2d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_2d<PetscScalar>(array_dst, appctx->layout);
  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  wave_eq_hom_all(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->xl, t);
  wave_eq_hom_free_surface_bc(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->HI, appctx->hi);

  // Restore arrays
  VecRestoreArray(v_src,&array_src);
  VecRestoreArray(v_dst,&array_dst);
  return 0;
}

PetscErrorCode rhs_serial(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;
  PetscScalar       *array_src, *array_dst;

  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);

  auto gf_src = grid::grid_function_2d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_2d<PetscScalar>(array_dst, appctx->layout);
  wave_eq_hom_serial(gf_dst, gf_src, appctx->D1, appctx->hi, appctx->xl, t);
  wave_eq_hom_free_surface_bc_serial(gf_dst, gf_src, appctx->HI, appctx->hi);

  // Restore arrays
  VecRestoreArray(v_src,&array_src);
  VecRestoreArray(v_dst,&array_dst);
  return 0;
}

