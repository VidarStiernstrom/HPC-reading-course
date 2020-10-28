static char help[] ="Solves the 2D acoustic wave equation on first order form: u_t = A*u_x + B*u_y, A = [0,0,-1;0,0,0;-1,0,0], B = [0,0,0;0,0,-1;0,-1,0].";

#define PROBLEM_TYPE_2D_O6

#include <algorithm>
#include <cmath>
#include <string>
// #include <filesystem>
#include <petscsys.h>
#include <petscdmda.h>
#include <petscvec.h>
#include <functional>
#include <petscts.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/acowave.h"
#include "timestepping.h"
#include <petsc/private/dmdaimpl.h> 
#include "appctx.h"
#include "IO_utils.h"

extern PetscErrorCode analytic_solution(DM, PetscScalar, AppCtx*, Vec&);
extern PetscErrorCode rhs_TS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode rhs(DM, PetscReal, Vec, Vec, AppCtx *);
extern PetscScalar theta1(PetscScalar x, PetscScalar y, PetscScalar t);
extern PetscScalar theta2(PetscScalar x, PetscScalar y, PetscScalar t);
extern PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx *appctx);
extern PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);
extern PetscErrorCode build_ltol_2D(DM da, VecScatter *ltol);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error, vlocal;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, procx, procy;
  PetscScalar    xl, xr, yl, yr, hix, hiy, dt, t0, Tend, CFL;

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
  auto a = [](const PetscInt i, const PetscInt j){ return 1;};
  auto b = [](const PetscInt i, const PetscInt j){ return 1;};

  // Set if data should be written.
  write_data = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,3,stencil_radius,NULL,NULL,&da);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDAGetCorners(da,&i_xstart,&i_ystart,NULL,&nx,&ny,NULL);
  i_xend = i_xstart + nx;
  i_yend = i_ystart + ny;

  DMDAGetInfo(da,NULL,NULL,NULL,NULL,&procx,&procy,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"Processor topology dimensions: [%d,%d]\n",procx,procy);

  // Populate application context.
  appctx.N = {Nx, Ny};
  appctx.hi = {hix, hiy};
  appctx.xl = xl;
  appctx.yl = yl;
  appctx.i_start = {i_xstart,i_ystart};
  appctx.i_end = {i_xend,i_yend};
  appctx.a = a;
  appctx.b = b;
  appctx.sw = stencil_radius;

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
  set_initial_condition(da, v, &appctx);

  // if (write_data) write_vector_to_binary(v,"data/sim_adv_ts","v_init");

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

  analytic_solution(da, Tend, &appctx, v_analytic);
  VecSet(v_error,0);
  VecWAXPY(v_error,-1,v,v_analytic);
  PetscReal l2_error, max_error;
  VecNorm(v_error,NORM_2,&l2_error);
  VecNorm(v_error,NORM_INFINITY,&max_error);
  l2_error = sqrt(1./(hix*hiy))*l2_error;
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error error is: %g and the maximum error is %g\n",l2_error,max_error);

  // Write solution to file
  // if (write_data)
  // {
  //   write_vector_to_binary(v,"data/sim_adv_ts","v");
  //   write_vector_to_binary(v_error,"data/sim_adv_ts","v_error");
  // }

  if (rank == 0) {
    FILE *f;
    if (use_custom_ts) {
      if (use_custom_sc) {
        f = fopen("data/timings_tsC_scC.txt", "a");
      } else {
        f = fopen("data/timings_tsC_scP.txt", "a");
      }
    } else {
      if (use_custom_sc) {
        f = fopen("data/timings_tsP_scC.txt", "a");
      } else {
        f = fopen("data/timings_tsP_scP.txt", "a");
      }
    }
    
    fprintf(f,"Size: %d, Nx: %d, Ny: %d, dt: %e, Tend: %f, elapsed time: %f, l2-error: %e, max-error: %e\n",size,Nx,Ny,dt,Tend,elapsed_time,l2_error,max_error);
    fclose(f);
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

PetscErrorCode analytic_solution(DM da, PetscScalar t, AppCtx *appctx, Vec& v) {
  PetscInt i, j, n, m; 
  PetscScalar ***varr, x, y;

  n = 3; // n = 1,2,3,4,...
  m = 4; // m = 1,2,3,4,...

  DMDAVecGetArrayDOF(da,v,&varr);    

  for (j = appctx->i_start[1]; j < appctx->i_end[1]; j++)
  {
    y = appctx->yl + j/appctx->hi[1];
    for (i = appctx->i_start[0]; i < appctx->i_end[0]; i++)
    {
      x = appctx->xl + i/appctx->hi[0];
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
PetscErrorCode set_initial_condition(DM da, Vec v, AppCtx *appctx) 
{
  analytic_solution(da, 0, appctx, v);
  return 0;
}

/**
* Required function for initial and analytical solution
**/
PetscScalar theta1(PetscScalar x, PetscScalar y, PetscScalar t) 
{
  PetscScalar rstar = 0.1;
  return exp(-(x - t)*(x - t)/(rstar*rstar) - (y - t)*(y - t)/(rstar*rstar));
}

/**
* Required function for initial and analytical solution
**/
PetscScalar theta2(PetscScalar x, PetscScalar y, PetscScalar t) 
{
  return -theta1(x,y,t);
}

PetscErrorCode rhs(DM da, PetscReal t, Vec v_src, Vec v_dst, AppCtx *appctx)
{
  PetscScalar       ***array_src, ***array_dst;

  DMDAVecGetArrayDOF(da, v_src, &array_src);
  DMDAVecGetArrayDOF(da, v_dst, &array_dst);

  VecScatterBegin(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);
  
  sbp::acowave_apply_2D_inner(appctx->D1, appctx->HI, appctx->a, appctx->b, array_src, array_dst, appctx->i_start, appctx->i_end, appctx->N, appctx->hi, appctx->sw);

  VecScatterEnd(appctx->scatctx,v_src,v_src,INSERT_VALUES,SCATTER_FORWARD);

  sbp::acowave_apply_2D_outer(appctx->D1, appctx->HI, appctx->a, appctx->b, array_src, array_dst, appctx->i_start, appctx->i_end, appctx->N, appctx->hi, appctx->sw);
  
  // sbp::acowave_apply_2D_all(appctx->D1, appctx->HI, appctx->a, appctx->b, gf_src, gf_dst, appctx->i_start, appctx->i_end, appctx->N, appctx->hi, appctx->sw);

  // Restore arrays
  DMDAVecRestoreArrayDOF(da, v_src, &array_src);
  DMDAVecRestoreArrayDOF(da, v_dst, &array_dst);
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

// PetscErrorCode write_vector_to_binary(const Vec& v, const std::string folder, const std::string file)
// { 
//   std::filesystem::create_directories(folder);
//   PetscErrorCode ierr;
  // PetscViewer viewer;
  // ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(folder+"/"+file).c_str(),FILE_MODE_WRITE,&viewer);
  // ierr = VecView(v,viewer);
  // ierr = PetscViewerDestroy(&viewer);
//   CHKERRQ(ierr);
//   return 0;
// }

/**
* Build local to local scatter context containing only ghost point communications
* Inputs: da        - DMDA object
*         ltol      - pointer to local to local scatter context
**/
PetscErrorCode build_ltol_2D(DM da, VecScatter *ltol)
{
  AO          ao;
  PetscInt    stencil_radius, i_xstart, i_xend, i_ystart, i_yend, ig_xstart, ig_xend, ig_ystart, ig_yend, nx, ny, i, j, l, lnx, lny, no_com_vals, count, Nx, Ny, dof;
  IS          ix, iy;
  Vec         vglobal, vlocal;
  VecScatter  gtol;

  DMDAGetStencilWidth(da, &stencil_radius);
  DMDAGetCorners(da,&i_xstart,&i_ystart,NULL,&nx,&ny,NULL);
  DMDAGetGhostCorners(da,&ig_xstart,&ig_ystart,NULL,&lnx,&lny,NULL);
  DMDAGetInfo(da, NULL, &Nx, &Ny,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);

  i_xend = i_xstart + nx;
  i_yend = i_ystart + ny;
  ig_xend = ig_xstart + lnx;
  ig_yend = ig_ystart + lny;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute how many elements to receive
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  no_com_vals = 0;
  if (i_ystart != 0)  // NOT BOTTOM, RECEIVE BELOW
  {
    no_com_vals += nx;
  }
  if (i_yend != Ny) // NOT TOP, RECEIVE ABOVE
  {
    no_com_vals += nx;
  }
  if (i_xstart != 0)  // NOT LEFT BOUNDARY, RECEIVE LEFT
  {
    no_com_vals += ny;
  }
  if (i_xend != Nx) // NOT RIGHT BOUNDARY, RECEIVE RIGHT
  {
    no_com_vals += ny;
  }
  no_com_vals *= stencil_radius*dof;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define communication pattern, from global index ixx[i] to local index iyy[i]
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt ixx[no_com_vals], iyy[no_com_vals];
  count = 0;
  for (i = i_xstart; i < i_xend; i++) { // UP
    for (j = i_yend; j < ig_yend; j++) {
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  for (i = i_xstart; i < i_xend; i++) { // DOWN
    for (j = ig_ystart; j < i_ystart; j++) { 
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  for (i = ig_xstart; i < i_xstart; i++) { // LEFT
    for (j = i_ystart; j < i_yend; j++) {
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  for (i = i_xend; i < ig_xend; i++) { // RIGHT
    for (j = i_ystart; j < i_yend; j++) {
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  // Map global indices from natural ordering to petsc application ordering
  DMDAGetAO(da,&ao);
  AOApplicationToPetsc(ao,no_com_vals,ixx);
  AODestroy(&ao);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Build global to local scatter context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,ixx,PETSC_COPY_VALUES,&ix);  
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,iyy,PETSC_COPY_VALUES,&iy);  

  DMGetGlobalVector(da, &vglobal);
  DMGetLocalVector(da, &vlocal);

  VecScatterCreate(vglobal,ix,vlocal,iy, &gtol);  
  VecScatterSetUp(gtol);

  DMRestoreGlobalVector(da, &vglobal);
  DMRestoreLocalVector(da, &vlocal);

  ISDestroy(&ix);
  ISDestroy(&iy);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Map 2D global to local scatter context to local to local (petsc source code)
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecScatterCopy(gtol,ltol);
  VecScatterDestroy(&gtol);

  PetscInt *idx,left,up,down;
  DM_DA *dd = (DM_DA*) da->data;
  left  = dd->xs - dd->Xs; down  = dd->ys - dd->Ys; up = down + dd->ye-dd->ys;
  PetscMalloc1((dd->xe-dd->xs)*(up - down),&idx);
  count = 0;
  for (i=down; i<up; i++) {
    for (j=0; j<dd->xe-dd->xs; j++) {
      idx[count++] = left + i*(dd->Xe-dd->Xs) + j;
    }
  }
  VecScatterRemap(*ltol,idx,NULL);

  return 0;
}