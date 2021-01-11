static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#define PROBLEM_TYPE_1D_O2

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "sbpops/ICF_central.h"
#include "diffops/advection.h"
// #include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

#include <unistd.h>

struct TimeCtx {
  PetscScalar Tend, Tpb;
  PetscScalar er[4], HI_el[4], D[4][4];
};

struct MatCtx {
  GridCtx gridctx;
  TimeCtx timectx;
};

struct PCCtx {
  MatCtx F_matctx, C_matctx;
};

extern PetscErrorCode setup_timestepper(TimeCtx& timectx, PetscScalar tau);
extern PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) ;
extern PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) ;
extern PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) ;
extern PetscErrorCode analytic_solution( const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic);
extern PetscScalar gaussian(PetscScalar x) ;
extern PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0);
extern PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst);
extern PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, PCCtx pcctx) ;
extern PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, PCCtx pcctx) ;
extern PetscErrorCode print_vec(GridCtx gridctx, Vec v, PetscInt comp);

int main(int argc,char **argv)
{ 
  Vec            v, v_analytic, v_error, v_final;
  PC             pc;
  PetscInt       stencil_radius, stencil_radius_pc, i_xstart, i_xend, Nx, nx, Nt, ngx, Nx_pc, nx_pc, dofs, tblocks, ig_xstart, ig_xend, blockidx;
  PetscScalar    xl, xr, dx, dxi, dx_pc, dxi_pc, dt, dti, t0, Tend, Tpb, tau;
  PetscReal      l2_error, max_error, H_error;
  PCCtx          pcctx;

  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  xl = -1;
  xr = 1;
  tau = 1.0; // SAT parameter

  // Fine space grid
  Nx = 2001;
  dx = (xr - xl)/(Nx-1);
  dxi = 1./dx;
  
  // Coarse space grid
  Nx_pc = 0.5*(Nx + 1);
  dx_pc = (xr - xl)/(Nx_pc-1);
  dxi_pc = 1./dx_pc;
  
  // Time
  tblocks = 10000;
  t0 = 0;
  Tend = 1;
  Tpb = Tend/tblocks;
  Nt = 4;
  dt = Tpb/(Nt-1);
  dti = 1./dt;

  dofs = 1;

  auto a = [](const PetscInt i){ return 1;};

  PetscPrintf(PETSC_COMM_WORLD,"dx: %f, dt: %f\n",dx,dt);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Fill application context
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  pcctx.F_matctx.gridctx.N = {Nx};
  pcctx.F_matctx.gridctx.hi = {dxi};
  pcctx.F_matctx.gridctx.h = {dx};
  pcctx.F_matctx.gridctx.xl = {xl};
  pcctx.F_matctx.gridctx.dofs = dofs;
  pcctx.F_matctx.gridctx.a = a;

  pcctx.F_matctx.timectx.Tend = Tend;
  pcctx.F_matctx.timectx.Tpb = Tpb;

  pcctx.C_matctx.gridctx.N = {Nx_pc};
  pcctx.C_matctx.gridctx.hi = {dxi_pc};
  pcctx.C_matctx.gridctx.h = {dx_pc};
  pcctx.C_matctx.gridctx.xl = {xl};
  pcctx.C_matctx.gridctx.dofs = dofs;
  pcctx.C_matctx.gridctx.a = a;

  pcctx.C_matctx.timectx.Tend = Tend;
  pcctx.C_matctx.timectx.Tpb = Tpb;

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Create distributed array (DMDA) to manage parallel grid and vectors of FINE grid
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = pcctx.F_matctx.gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx, Nt*dofs, stencil_radius, NULL, &pcctx.F_matctx.gridctx.da_xt);
  
  DMSetFromOptions(pcctx.F_matctx.gridctx.da_xt);
  DMSetUp(pcctx.F_matctx.gridctx.da_xt);
  DMDAGetCorners(pcctx.F_matctx.gridctx.da_xt,&i_xstart,NULL,NULL,&nx,NULL,NULL);
  i_xend = i_xstart + nx;

  pcctx.F_matctx.gridctx.i_start = {i_xstart};
  pcctx.F_matctx.gridctx.i_end = {i_xend};
  pcctx.F_matctx.gridctx.sw = stencil_radius;

  printf("Rank: %d, number of unknowns: %d\n",rank,Nt*nx);

  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx, dofs, stencil_radius, NULL, &pcctx.F_matctx.gridctx.da_x);
  DMSetFromOptions(pcctx.F_matctx.gridctx.da_x);
  DMSetUp(pcctx.F_matctx.gridctx.da_x);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Create distributed array (DMDA) to manage parallel grid and vectors of COARSE grid
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width_pc, nc_pc, cw_pc] = pcctx.C_matctx.gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx_pc, Nt*dofs, stencil_radius, NULL, &pcctx.C_matctx.gridctx.da_xt);
  
  DMSetFromOptions(pcctx.C_matctx.gridctx.da_xt);
  DMSetUp(pcctx.C_matctx.gridctx.da_xt);
  DMDAGetCorners(pcctx.C_matctx.gridctx.da_xt,&i_xstart,NULL,NULL,&nx_pc,NULL,NULL);
  i_xend = i_xstart + nx_pc;

  pcctx.C_matctx.gridctx.i_start = {i_xstart};
  pcctx.C_matctx.gridctx.i_end = {i_xend};
  pcctx.C_matctx.gridctx.sw = stencil_radius;

  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx_pc, dofs, stencil_radius, NULL, &pcctx.C_matctx.gridctx.da_x);
  DMSetFromOptions(pcctx.C_matctx.gridctx.da_x);
  DMSetUp(pcctx.C_matctx.gridctx.da_x);


  PetscReal xend = xl + pcctx.F_matctx.gridctx.i_end[0]*pcctx.F_matctx.gridctx.h[0];
  PetscReal xend_pc = xl + pcctx.C_matctx.gridctx.i_end[0]*pcctx.C_matctx.gridctx.h[0];

  assert(abs(xend - xend_pc) < 2*pcctx.F_matctx.gridctx.h[0]);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Setup solver
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */  
  MatCreateShell(PETSC_COMM_WORLD,nx*Nt,nx*Nt,Nx*Nt,Nx*Nt,&pcctx.F_matctx.gridctx,&pcctx.F_matctx.gridctx.D);
  MatShellSetOperation(pcctx.F_matctx.gridctx.D,MATOP_MULT,(void(*)(void))LHS);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define matrix object
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // MatCreateAIJ(PETSC_COMM_WORLD, n, n, N, N, 0, NULL, 0, NULL, &D_sparse);

  // MatSetOption(D_sparse, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  // MatSetUp(D_sparse);

  // DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x,&v);
  // DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x,&b);

  // int j = 0, i, count;
  // PetscInt idx[n];
  // PetscInt col_idx[1];
  // PetscScalar val[n];
  // VecSet(v,0.0);
  // for (j = 0; j < N; j++) {
  //   VecSetValue(v,j,1,INSERT_VALUES);
  //   MatMult(D,v,b);
  //   VecSetValue(v,j,0,INSERT_VALUES);

  //   PetscScalar *arr;
  //   VecGetArray(b,&arr); 
    
  //   col_idx[0] = j;

  //   // sleep(rank);
  //   count = 0;
  //   for (i = 0; i < n; i++) {
  //     if (arr[i] != 0) {
  //       idx[count] = i + i_xstart;
  //       val[count] = arr[i];
  //       count++;
  //     }
  //     // printf("v[%d] = %f\n",i,arr[i]);
  //     // idx[i] = i + i_xstart;
  //   }

  //   MatSetValues(D_sparse, count , idx , 1, col_idx, val , INSERT_VALUES);

  //   VecRestoreArray(b,&arr); 
  // }

  // MatAssemblyBegin(D_sparse, MAT_FINAL_ASSEMBLY);
  // MatAssemblyEnd(D_sparse, MAT_FINAL_ASSEMBLY);


  // PetscViewer viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  // PetscViewerPushFormat(viewer,   PETSC_VIEWER_ASCII_MATLAB );
  // MatView(D_sparse, viewer); 
  // PetscViewerPopFormat(viewer);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // MatDestroy(&pcctx.F_matctx.gridctx.D);
  // MatDestroy(&pcctx.C_matctx.gridctx.D);
  // PCDestroy(&pc);
  // VecDestroy(&v);
  // VecDestroy(&v_error);
  // VecDestroy(&v_analytic);
  // DMDestroy(&pcctx.C_matctx.gridctx.da_x);
  // DMDestroy(&pcctx.F_matctx.gridctx.da_x);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst)
{
  PetscScalar       **array_src, **array_dst;
  Vec               v_src_local;
  MatCtx            *matctx;

  MatShellGetContext(D, &matctx);

  DMGetLocalVector(matctx->gridctx.da_xt, &v_src_local);

  DMGlobalToLocalBegin(matctx->gridctx.da_xt,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(matctx->gridctx.da_xt,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(matctx->gridctx.da_xt,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(matctx->gridctx.da_xt,v_src_local,&array_src); 

  sbp::adv_imp_apply_all(matctx->timectx.D, matctx->gridctx.D1, matctx->gridctx.HI, matctx->gridctx.a, array_src, array_dst, matctx->gridctx.i_start[0], matctx->gridctx.i_end[0], matctx->gridctx.N[0], matctx->gridctx.hi[0], matctx->timectx.Tpb);

  DMDAVecRestoreArrayDOF(matctx->gridctx.da_xt,v_dst,&array_dst); 
  DMDAVecRestoreArrayDOF(matctx->gridctx.da_xt,v_src_local,&array_src); 

  DMRestoreLocalVector(matctx->gridctx.da_xt,&v_src_local);
  
  return 0;
}