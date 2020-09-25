
static char help[] ="Solves the 2D advection equation u_t + au_x +bu_y = 0, using the PETSc time stepping contexts.";

#include <algorithm>
#include <cmath>
#include <string>
// #include <filesystem>
#include <functional>
#include <petscsys.h>
#include <petscdmda.h>
#include <petscvec.h>
#include <petscts.h>
#include "sbpops/D1_central.h"
#include "diffops/advection.h"

struct AppCtx{
  std::array<PetscInt,2> N, i_start, i_end;
  std::array<PetscScalar,2> hi;
  PetscScalar xl, yl;
  std::function<double(int, int)> a;
  std::function<double(int, int)> b;
  const sbp::D1_central<3,1,2> D1;
  // const sbp::D1_central<5,4,6> D1;
    // const sbp::D1_central<7,6,9> D1;
};

extern PetscErrorCode analytic_solution(const DM&, const PetscScalar, const AppCtx&, Vec&);
extern PetscErrorCode rhs(TS,PetscReal,Vec,Vec,void *);
extern PetscScalar gaussian_2D(PetscScalar, PetscScalar);
extern PetscErrorCode write_vector_to_binary(const Vec&, const std::string, const std::string);

int main(int argc,char **argv)
{ 
  DM             da;
  Vec            v, v_analytic, v_error;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny;
  PetscScalar    xl, xr, yl, yr, hix, hiy, dt, t0, Tend;
  TS             ts;
  TSAdapt        adapt;
  AppCtx         appctx;
  PetscBool       write_data;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  // Space
  xl = -1;
  xr = 1;
  yl = -1;
  yr = 1;
  Nx = 401;
  Ny = 401;
  hix = (Nx-1)/(xr-xl);
  hiy = (Ny-1)/(yr-yl);
  
  // Time
  t0 = 0;
  Tend = 0.4;
  dt = 0.1/(std::min(hix,hiy));

  // Velocity field a(i,j) = 1
  auto a = [](const PetscInt i, const PetscInt j){ return 1.5;};
  auto b = [](const PetscInt i, const PetscInt j){ return -1;};

  // Set if data should be written.
  write_data = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,stencil_radius,NULL,NULL,&da);
  DMSetFromOptions(da);
  DMSetUp(da);
  DMDAGetCorners(da,&i_xstart,&i_ystart,NULL,&nx,&ny,NULL);
  i_xend = i_xstart + nx;
  i_yend = i_ystart + ny;
  PetscInt procx, procy;
  DMDAGetInfo(da,NULL,NULL,NULL,NULL,&procx,&procy,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
  PetscPrintf(PETSC_COMM_WORLD,"Processor topology dimensions: [%d,%d]\n",procx,procy);
  // printf("Local info: My rank: %d, i_xstart: %d, i_xend: %d, i_ystart: %d, i_yend: %d\n",rank,i_xstart,i_xend,i_ystart,i_yend);

  // Populate application context.
  appctx.N = {Nx, Ny};
  appctx.hi = {hix, hiy};
  appctx.xl = xl;
  appctx.yl = yl;
  appctx.i_start = {i_xstart,i_ystart};
  appctx.i_end = {i_xend,i_yend};
  appctx.a = a;
  appctx.b = b;

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v);
  VecDuplicate(v,&v_analytic);
  VecDuplicate(v,&v_error);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Setup time stepping context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TSCreate(PETSC_COMM_WORLD, &ts);
  
  // Problem type and RHS function
  TSSetProblemType(ts, TS_LINEAR);
  TSSetRHSFunction(ts, NULL, rhs, &appctx);
  
  // Integrator
  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK4);
  TSGetAdapt(ts, &adapt);
  TSAdaptSetType(adapt,TSADAPTNONE);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  // DM context
  TSSetDM(ts,da);

  // Initial solution, starting time and end time.
  analytic_solution(da, 0, appctx, v);
  // if (write_data) write_vector_to_binary(v,"data/sim_adv_ts","v_init");
  TSSetSolution(ts, v);
  TSSetTime(ts,t0);
  TSSetTimeStep(ts,dt);
  TSSetMaxTime(ts,Tend);
  // Set all options
  TSSetFromOptions(ts);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,v);CHKERRQ(ierr);
  analytic_solution(da, Tend, appctx, v_analytic);
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_analytic);
  VecDestroy(&v_error);
  TSDestroy(&ts);
  DMDestroy(&da);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode analytic_solution(const DM& da, const PetscScalar t, const AppCtx& appctx, Vec& v_analytic)
{ 
  PetscScalar x,y, **array_analytic;
  DMDAVecGetArray(da,v_analytic,&array_analytic);
  for (PetscInt j = appctx.i_start[1]; j < appctx.i_end[1]; j++)
  {
    y = appctx.yl + j/appctx.hi[1];
    for (PetscInt i = appctx.i_start[0]; i < appctx.i_end[0]; i++)
    {
      x = appctx.xl + i/appctx.hi[0];
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

PetscErrorCode rhs(TS ts, PetscReal t, Vec v_src, Vec v_dst, void *ctx)
{
  AppCtx *appctx = (AppCtx*) ctx;     /* user-defined application context */
  DM                da;
  Vec               v_local;
  PetscScalar       ***array_src, ***array_dst;

  TSGetDM(ts,&da);
  DMGetLocalVector(da, &v_local);
  // Scatter global vector to local vectors, communicating ghost points
  DMGlobalToLocalBegin(da,v_src,INSERT_VALUES,v_local);
  DMGlobalToLocalEnd(da,v_src,INSERT_VALUES,v_local);
  
  // Extract arrays
  DMDAVecGetArrayDOFRead(da,v_local,&array_src);
  DMDAVecGetArrayDOF(da,v_dst,&array_dst);
  
  sbp::advection_apply_2D_2(appctx->D1, appctx->a, appctx->b, array_src, array_dst, 
                          appctx->i_start, appctx->i_end, appctx->N, appctx->hi);

  //Apply homogeneous Dirichlet BC via injection on west and north boundary
  if (appctx->i_start[0] == 0)
  {
   for (PetscInt j = appctx->i_start[1]; j < appctx->i_end[1]; j++)
    {
      array_dst[j][0][0] = 0;
    }   
  }

  if (appctx->i_end[1] == appctx->N[1])
  {
   for (PetscInt i = appctx->i_start[0]; i < appctx->i_end[0]; i++)
    {
      array_dst[appctx->N[1]-1][i][0] = 0;
    }   
  }

  // Restore arrays
  DMDAVecRestoreArrayDOFRead(da, v_local, &array_src);
  DMRestoreLocalVector(da,&v_local);
  DMDAVecRestoreArrayDOF(da, v_dst, &array_dst);
  return 0;
}

// PetscErrorCode write_vector_to_binary(const Vec& v, const std::string folder, const std::string file)
// { 
//   std::filesystem::create_directories(folder);
//   PetscErrorCode ierr;
//   PetscViewer viewer;
//   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,(folder+"/"+file).c_str(),FILE_MODE_WRITE,&viewer);
//   ierr = VecView(v,viewer);
//   ierr = PetscViewerDestroy(&viewer);
//   CHKERRQ(ierr);
//   return 0;
// }