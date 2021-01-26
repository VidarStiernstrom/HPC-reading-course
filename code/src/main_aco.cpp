static char help[] = "Solves advection 2D aco wave problem.\n";

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

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "sbpops/ICF_central.h"
#include "diffops/acowave.h"
// #include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"
// #include "multigrid.h"
#include "standard.h"
#include "imp_timestepping.h"
#include "aco_2D.h"

int main(int argc,char **argv)
{ 
  Vec            v;
  Mat            Dfine;
  PetscInt       stencil_radius, i_xstart, i_xend, i_ystart, i_yend, Nx, Ny, nx, ny, Nt, dofs, tblocks;
  PetscScalar    xl, xr, yl, yr, dx, dxi, dy, dyi, dt, dti, t0, Tend, Tpb, tau;
  
  MatCtx         F_matctx;
  PetscInt       nlevels;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  xl = -1;
  xr = 1;
  yl = -1;
  yr = 1;
  tau = 1.0; // Initial condition SAT parameter

  // Fine space grid
  Nx = 501;
  Ny = 501;
  dx = (xr - xl)/(Nx-1);
  dxi = 1./dx;
  dy = (yr - yl)/(Ny-1);
  dyi = 1./dy;

  // Time
  tblocks = 10;
  t0 = 0;
  Tend = 0.1;
  Tpb = Tend/tblocks;
  Nt = 4;
  dt = Tpb/(Nt-1);
  dti = 1./dt;

  nlevels = 4;

  dofs = 3;

  auto a = [xl,yl,dx,dy](const PetscInt i, const PetscInt j){  // 1/rho.
  PetscScalar x = xl + dx*i;
  PetscScalar y = yl + dy*j;
  return 1.0/(2 + x*y);
  };
  auto b = [](const PetscInt i, const PetscInt j){ return 1;}; // unused at the moment, K = 1.


  F_matctx.gridctx.N = {Nx,Nx};
  F_matctx.gridctx.hi = {dxi, dyi};
  F_matctx.gridctx.h = {dx,dy};
  F_matctx.gridctx.xl = {xl,yl};
  F_matctx.gridctx.xr = {xr,yr};
  F_matctx.gridctx.dofs = dofs;
  F_matctx.gridctx.a = a; 
  F_matctx.gridctx.b = b; 

  F_matctx.timectx.N = Nt;
  F_matctx.timectx.Tend = Tend;
  F_matctx.timectx.Tpb = Tpb;
  F_matctx.timectx.tblocks = tblocks;

  auto [stencil_width, nc, cw] = F_matctx.gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,Nt*dofs,stencil_radius,NULL,NULL,&F_matctx.gridctx.da_xt);
  
  DMSetFromOptions(F_matctx.gridctx.da_xt);
  DMSetUp(F_matctx.gridctx.da_xt);
  DMDAGetCorners(F_matctx.gridctx.da_xt,&i_xstart,&i_ystart,NULL,&nx,&ny,NULL);
  i_xend = i_xstart + nx;
  i_yend = i_ystart + ny;

  F_matctx.gridctx.n = {nx,ny};
  F_matctx.gridctx.i_start = {i_xstart,i_ystart};
  F_matctx.gridctx.i_end = {i_xend,i_yend};
  F_matctx.gridctx.sw = stencil_radius;

  PetscPrintf(PETSC_COMM_WORLD,"System size: %d\n",dofs*Nt*Nx*Nx);
  printf("Rank: %d, number of unknowns: %d\n",rank,dofs*Nt*nx*ny);

  DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
               Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,dofs,stencil_radius,NULL,NULL,&F_matctx.gridctx.da_x);
  DMSetFromOptions(F_matctx.gridctx.da_x);
  DMSetUp(F_matctx.gridctx.da_x);

  MatCreateShell(PETSC_COMM_WORLD,nx*ny*Nt*dofs,nx*ny*Nt*dofs,Nx*Ny*Nt*dofs,Nx*Ny*Nt*dofs,&F_matctx,&Dfine);
  MatShellSetOperation(Dfine,MATOP_MULT,(void(*)(void))LHS);
  MatSetDM(Dfine, F_matctx.gridctx.da_xt);
  MatShellSetContext(Dfine, &F_matctx);
  MatSetUp(Dfine);

  setup_timestepper(F_matctx.timectx, tau);

  DMCreateGlobalVector(F_matctx.gridctx.da_xt,&v);

  // mgsolver(Dfine, v, nlevels);
  standard_solver(Dfine, v);

   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // Free work space.  All PETSc objects should be destroyed when they
      // are no longer needed.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  VecDestroy(&v);
  MatDestroy(&Dfine);
  
  ierr = PetscFinalize();
  return ierr;
}