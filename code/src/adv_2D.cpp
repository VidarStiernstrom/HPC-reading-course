
static char help[] ="Solves the 2D advection equation u_t + au_x +bu_y = 0.";

#include <petsc.h>
#include "sbpops/op_defs.h"
#include "diffops/advection.h"
#include "time_stepping/ts_rk.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx/scatter_ctx.h"

struct AppCtx{
    std::array<PetscInt,2> N, ind_i, ind_j;
    std::array<PetscScalar,2> hi, h, xl;
    PetscInt dofs;
    PetscScalar sw;
    std::function<double(int, int)> a, b;
    const FirstDerivativeOp D1;
    const NormOp H;
    const InverseNormOp HI;
    VecScatter scatctx;
    grid::partitioned_layout_2d layout;
};

extern PetscErrorCode analytic_solution(const DM, const PetscScalar, const AppCtx&, Vec);
extern PetscErrorCode rhs_TS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode rhs(DM, PetscReal, Vec, Vec, void *);
extern PetscScalar gaussian_2D(PetscScalar, PetscScalar);
extern PetscErrorCode get_error(const DM da, const Vec v1, const Vec v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, procx, procy, dofs;
  PetscScalar    xl, xr, yl, yr, hix, hiy, dt, t0, Tend, CFL;
  PetscReal      l2_error, max_error, H_error;

  AppCtx         appctx;
  PetscBool      write_data, use_custom_sc;
  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (get_inputs_2d(argc, argv, &Nx, &Ny, &Tend, &CFL, &use_custom_sc) == -1) {
    PetscFinalize();
    return -1;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // Space
  dofs = 1;
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
  auto a = [](const PetscInt i, const PetscInt j){ return 1.5;};
  auto b = [](const PetscInt i, const PetscInt j){ return -1;};

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

  // Populate application context.
  appctx.N = {Nx, Ny};
  appctx.hi = {hix, hiy};
  appctx.h = {1./hix, 1./hiy};
  appctx.xl = {xl, yl};
  appctx.ind_i = {i_xstart,i_xend};
  appctx.ind_j = {i_ystart,i_yend};
  appctx.a = a;
  appctx.b = b;
  appctx.sw = stencil_radius;
  appctx.layout = grid::create_layout_2d(da);

  // Extract local to local scatter context
  if (use_custom_sc) {
    scatter_ctx_ltol(da, &appctx.scatctx);
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
  analytic_solution(da, 0, appctx, v);
  if (write_data) write_vector_to_binary(v,"data/adv_2D","v_init");

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


  ts_rk4(da, Tend, dt, vlocal, rhs_TS, &appctx);
  
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
    write_vector_to_binary(v,"data/adv_2D","v");
    write_vector_to_binary(v_error,"data/adv_2D","v_error");

    char tmp_str[200];
    std::string data_string;
    sprintf(tmp_str,"%d\t%d\t%d\t%e\t%f\t%f\t%e\t%e\t%e\n",size,Nx,Ny,dt,Tend,elapsed_time,l2_error,H_error,max_error);
    data_string.assign(tmp_str);
    write_data_to_file(data_string, "data/adv_2D", "data.tsv");
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
  DM                da;
  TSGetDM(ts,&da);
  rhs(da, t, v_src, v_dst, ctx);
  return 0;
}

PetscErrorCode get_error(const DM da, const Vec v1, const Vec v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error, const AppCtx& appctx) {
  PetscScalar ***arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);
  *l2_error = sqrt(appctx.h[0]*appctx.h[1])*(*l2_error);

  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(da, *v_error, &arr);
  *H_error = appctx.H.get_norm_2D(arr, appctx.h, appctx.N, appctx.ind_i, appctx.ind_j, appctx.dofs);

  DMDAVecRestoreArrayDOF(da, *v_error, &arr);

  return 0;
}

PetscErrorCode analytic_solution(const DM da, const PetscScalar t, const AppCtx& appctx, Vec v_analytic)
{ 
  PetscScalar x,y, **array_analytic;
  DMDAVecGetArray(da,v_analytic,&array_analytic);
  for (PetscInt j = appctx.ind_j[0]; j < appctx.ind_j[1]; j++)
  {
    y = appctx.xl[1] + j*appctx.h[1];
    for (PetscInt i = appctx.ind_i[0]; i < appctx.ind_i[1]; i++)
    {
      x = appctx.xl[0] + i*appctx.h[0];
      array_analytic[j][i] = gaussian_2D(x-appctx.a(i,j)*t,y-appctx.b(i,j)*t);
    }
  }
  DMDAVecRestoreArray(da,v_analytic,&array_analytic);  

  return 0;
};

PetscScalar gaussian_2D(PetscScalar x, PetscScalar y) {
  PetscScalar rstar = 0.1;
  return std::exp(-(x*x+y*y)/(rstar*rstar));
}

PetscErrorCode rhs(DM da, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  PetscScalar       *array_src, *array_dst;
  AppCtx *appctx = (AppCtx*) ctx;
  VecGetArray(v_src,&array_src);
  VecGetArray(v_dst,&array_dst);

  auto gf_src = grid::grid_function_2d<PetscScalar>(array_src, appctx->layout);
  auto gf_dst = grid::grid_function_2d<PetscScalar>(array_dst, appctx->layout);
  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  sbp::advection_local(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->a, appctx->b);
  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  sbp::advection_overlap(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->sw, appctx->D1, appctx->hi, appctx->a, appctx->b);
  sbp::advection_bc(gf_dst, gf_src, appctx->ind_i, appctx->ind_j, appctx->HI, appctx->hi, appctx->a, appctx->b);

  // sbp::advection_serial(gf_dst, gf_src, appctx->D1, appctx->hi, appctx->a, appctx->b);
  // sbp::advection_bc_serial(gf_dst, gf_src, appctx->HI, appctx->hi, appctx->a, appctx->b);

  // Restore arrays
  VecRestoreArray(v_src,&array_src);
  VecRestoreArray(v_dst,&array_dst);
  return 0;
}