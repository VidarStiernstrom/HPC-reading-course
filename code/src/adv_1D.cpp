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

  MatCreateShell(PETSC_COMM_WORLD,nx*Nt,nx*Nt,Nx*Nt,Nx*Nt,&pcctx.F_matctx.gridctx,&pcctx.F_matctx.gridctx.D_presmo);
  MatShellSetOperation(pcctx.F_matctx.gridctx.D_presmo,MATOP_MULT,(void(*)(void))LHS);

  MatCreateShell(PETSC_COMM_WORLD,nx_pc*Nt,nx_pc*Nt,Nx_pc*Nt,Nx_pc*Nt,&pcctx.C_matctx.gridctx,&pcctx.C_matctx.gridctx.D);
  MatShellSetOperation(pcctx.C_matctx.gridctx.D,MATOP_MULT,(void(*)(void))LHS);

  // ---------------------------------------------------------------------------------------- //
  KSPCreate(PETSC_COMM_WORLD, &pcctx.F_matctx.gridctx.ksp);
  KSPSetOperators(pcctx.F_matctx.gridctx.ksp, pcctx.F_matctx.gridctx.D, pcctx.F_matctx.gridctx.D);
  KSPSetTolerances(pcctx.F_matctx.gridctx.ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetPCSide(pcctx.F_matctx.gridctx.ksp, PC_RIGHT);
  KSPSetType(pcctx.F_matctx.gridctx.ksp, KSPPIPEFGMRES);
  KSPGMRESSetRestart(pcctx.F_matctx.gridctx.ksp, 20);
  KSPSetInitialGuessNonzero(pcctx.F_matctx.gridctx.ksp, PETSC_TRUE);
  KSPSetFromOptions(pcctx.F_matctx.gridctx.ksp);
  KSPSetUp(pcctx.F_matctx.gridctx.ksp);

  KSPGetPC(pcctx.F_matctx.gridctx.ksp,&pc);
  // PCSetType(pc,PCSHELL);
  PCSetType(pc,PCNONE);
  PCShellSetContext(pc, &pcctx);
  PCShellSetApply(pc, apply_pc);
  PCSetUp(pc);

  // ---------------------------------------------------------------------------------------- //
  KSPCreate(PETSC_COMM_WORLD, &pcctx.F_matctx.gridctx.ksp_smo);
  KSPSetOperators(pcctx.F_matctx.gridctx.ksp_smo, pcctx.F_matctx.gridctx.D_presmo, pcctx.F_matctx.gridctx.D_presmo);
  KSPSetTolerances(pcctx.F_matctx.gridctx.ksp_smo, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 16);
  KSPSetType(pcctx.F_matctx.gridctx.ksp_smo, KSPGMRES);
  KSPGMRESSetRestart(pcctx.F_matctx.gridctx.ksp_smo, 4);
  KSPSetInitialGuessNonzero(pcctx.F_matctx.gridctx.ksp_smo, PETSC_TRUE);
  // KSPSetFromOptions(pcctx.F_matctx.gridctx.ksp_smo);
  KSPSetUp(pcctx.F_matctx.gridctx.ksp_smo);
  // KSPGetPC(pcctx.F_matctx.gridctx.ksp_smo,&pc);
  // PCSetType(pc,PCNONE);
  // PCSetUp(pc);

  // ---------------------------------------------------------------------------------------- //

  KSPCreate(PETSC_COMM_WORLD, &pcctx.C_matctx.gridctx.ksp);
  KSPSetOperators(pcctx.C_matctx.gridctx.ksp, pcctx.C_matctx.gridctx.D, pcctx.C_matctx.gridctx.D);
  KSPSetTolerances(pcctx.C_matctx.gridctx.ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 40);
  KSPSetType(pcctx.C_matctx.gridctx.ksp, KSPGMRES);
  KSPGMRESSetRestart(pcctx.C_matctx.gridctx.ksp, 10);
  // KSPSetInitialGuessNonzero(pcctx.C_matctx.gridctx.ksp, PETSC_TRUE);
  // KSPSetFromOptions(pcctx.C_matctx.gridctx.ksp);
  KSPSetUp(pcctx.C_matctx.gridctx.ksp);

  // KSPGetPC(pcctx.C_matctx.gridctx.ksp,&pc);
  // PCSetType(pc,PCNONE);
  // PCSetUp(pc);

  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x, &pcctx.F_matctx.gridctx.v_curr);
  DMCreateGlobalVector(pcctx.C_matctx.gridctx.da_x, &pcctx.C_matctx.gridctx.v_curr);

  analytic_solution(pcctx.F_matctx.gridctx, 0, pcctx.F_matctx.gridctx.v_curr);
  analytic_solution(pcctx.C_matctx.gridctx, 0, pcctx.C_matctx.gridctx.v_curr);

  setup_timestepper(pcctx.F_matctx.timectx, tau);
  setup_timestepper(pcctx.C_matctx.timectx, tau);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_xt,&v);
  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_xt,&pcctx.F_matctx.gridctx.b);
  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_xt,&pcctx.F_matctx.gridctx.b_smo);
  DMCreateGlobalVector(pcctx.C_matctx.gridctx.da_xt,&pcctx.C_matctx.gridctx.b);

  VecDuplicate(pcctx.F_matctx.gridctx.v_curr,&v_error);
  VecDuplicate(pcctx.F_matctx.gridctx.v_curr,&v_analytic);

  

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  for (blockidx = 0; blockidx < tblocks; blockidx++) 
  {
    RHS(pcctx.F_matctx.gridctx, pcctx.F_matctx.timectx, pcctx.F_matctx.gridctx.b, pcctx.F_matctx.gridctx.v_curr);
    RHS(pcctx.F_matctx.gridctx, pcctx.F_matctx.timectx, pcctx.F_matctx.gridctx.b_smo, pcctx.F_matctx.gridctx.v_curr);

    KSPSolve(pcctx.F_matctx.gridctx.ksp, pcctx.F_matctx.gridctx.b, v);
    get_solution(pcctx.F_matctx.gridctx.v_curr, v, pcctx.F_matctx.gridctx, pcctx.F_matctx.timectx); 
  }

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  write_vector_to_binary(v,"data/adv_1D","v");
  write_vector_to_binary(pcctx.F_matctx.gridctx.v_curr,"data/adv_1D","v_final");

  analytic_solution(pcctx.F_matctx.gridctx, Tend, v_analytic);

  get_solution(pcctx.F_matctx.gridctx.v_curr, v, pcctx.F_matctx.gridctx, pcctx.F_matctx.timectx); 
  get_error(pcctx.F_matctx.gridctx, pcctx.F_matctx.gridctx.v_curr, v_analytic, &v_error, &H_error, &l2_error, &max_error);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %.8e, the H-error is: %.9e and the maximum error is %.9e\n",l2_error,H_error,max_error);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&pcctx.F_matctx.gridctx.v_curr);
  VecDestroy(&v_error);
  VecDestroy(&v_analytic);
  VecDestroy(&v_final);
  PCDestroy(&pc);
  DMDestroy(&pcctx.F_matctx.gridctx.da_xt);
  DMDestroy(&pcctx.F_matctx.gridctx.da_x);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode print_vec(GridCtx gridctx, Vec v, PetscInt comp)
{
  PetscInt i, rank;
  PetscScalar **v_arr;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 
  sleep(rank);
  printf("Rank %d printing...\n", rank);
  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    printf("v(%d) = %.16f;\n",i+1,v_arr[i][comp]);
  }
  printf("Rank %d done...\n", rank);
  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 

  return 0;
}

// Error equation:
// 1: 
// A*v1 = b
// A*v0 = b + resi
// err = v1 - v0
// A*err = -resi
// v1 = v0 + err
// 
// 
// 

PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) 
{
  Vec               xcoarse;
  PCCtx             *pcctx;
  Vec               F_resi, F_err, C_resi, C_err;

  PCShellGetContext(pc, (void**) &pcctx);

  DMGetGlobalVector(pcctx->F_matctx.gridctx.da_xt, &F_resi);
  DMGetGlobalVector(pcctx->F_matctx.gridctx.da_xt, &F_err);
  DMGetGlobalVector(pcctx->C_matctx.gridctx.da_xt, &C_resi);
  DMGetGlobalVector(pcctx->C_matctx.gridctx.da_xt, &C_err);

  // Presmoother
  // KSPSolve(pcctx->F_matctx.gridctx.ksp_smo, pcctx->F_matctx.gridctx.b_smo, xin);

  // Compute fine residual
  MatMult(pcctx->F_matctx.gridctx.D, xin, F_resi); // F_resi = Df*xin
  VecAXPY(F_resi, -1.0, pcctx->F_matctx.gridctx.b); // F_resi = F_resi - b = Df*xin - b

  // Restrict residual to coarse grid
  apply_F2C(F_resi, C_resi, *pcctx); // C_resi = R*F_resi

  // Solve error equation 
  VecSet(C_err,0.0);
  KSPSolve(pcctx->C_matctx.gridctx.ksp, C_resi, C_err); // C_err = Dc^-1*C_resi

  // Prolong error to fine grid
  apply_C2F(F_err, C_err, *pcctx); // F_err = P*C_err

  // Correct fine grid solution
  VecWAXPY(xout, -1.0, F_err, xin);

  // PetscPrintf(PETSC_COMM_WORLD,"Postsmoothing\n");
  // KSPSolve(pcctx->F_matctx.gridctx.ksp_smo, pcctx->F_matctx.gridctx.b_smo, xout);

  DMRestoreGlobalVector(pcctx->F_matctx.gridctx.da_xt, &F_resi);
  DMRestoreGlobalVector(pcctx->F_matctx.gridctx.da_xt, &F_err);
  DMRestoreGlobalVector(pcctx->C_matctx.gridctx.da_xt, &C_resi);
  DMRestoreGlobalVector(pcctx->C_matctx.gridctx.da_xt, &C_err);

  return 0;
}

PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, PCCtx pcctx) 
{
  PetscScalar       **array_src, **array_dst;
  Vec               xfine_local;

  DMGetLocalVector(pcctx.F_matctx.gridctx.da_xt, &xfine_local);

  DMGlobalToLocalBegin(pcctx.F_matctx.gridctx.da_xt,xfine,INSERT_VALUES,xfine_local);
  DMGlobalToLocalEnd(pcctx.F_matctx.gridctx.da_xt,xfine,INSERT_VALUES,xfine_local);

  DMDAVecGetArrayDOF(pcctx.C_matctx.gridctx.da_xt,xcoarse,&array_dst);
  DMDAVecGetArrayDOF(pcctx.F_matctx.gridctx.da_xt,xfine_local,&array_src);

  sbp::apply_F2C(pcctx.F_matctx.gridctx.ICF, array_src, array_dst, pcctx.C_matctx.gridctx.i_start[0], pcctx.C_matctx.gridctx.i_end[0], pcctx.C_matctx.gridctx.N[0]);

  DMDAVecRestoreArrayDOF(pcctx.C_matctx.gridctx.da_xt,xcoarse,&array_dst);
  DMDAVecRestoreArrayDOF(pcctx.F_matctx.gridctx.da_xt,xfine_local,&array_src);

  DMRestoreLocalVector(pcctx.F_matctx.gridctx.da_xt, &xfine_local);

  return 0;
}

PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, PCCtx pcctx) 
{
  PetscScalar       **array_src, **array_dst;
  Vec               xcoarse_local;

  DMGetLocalVector(pcctx.C_matctx.gridctx.da_xt, &xcoarse_local);

  DMGlobalToLocalBegin(pcctx.C_matctx.gridctx.da_xt,xcoarse,INSERT_VALUES,xcoarse_local);
  DMGlobalToLocalEnd(pcctx.C_matctx.gridctx.da_xt,xcoarse,INSERT_VALUES,xcoarse_local);

  DMDAVecGetArrayDOF(pcctx.F_matctx.gridctx.da_xt,xfine,&array_dst); 
  DMDAVecGetArrayDOF(pcctx.C_matctx.gridctx.da_xt,xcoarse_local,&array_src); 

  sbp::apply_C2F(pcctx.F_matctx.gridctx.ICF, array_src, array_dst, pcctx.F_matctx.gridctx.i_start[0], pcctx.F_matctx.gridctx.i_end[0], pcctx.F_matctx.gridctx.N[0]);

  DMDAVecRestoreArrayDOF(pcctx.F_matctx.gridctx.da_xt,xfine,&array_dst);
  DMDAVecRestoreArrayDOF(pcctx.C_matctx.gridctx.da_xt,xcoarse_local,&array_src); 

  DMRestoreLocalVector(pcctx.C_matctx.gridctx.da_xt, &xcoarse_local);
  return 0;
}

PetscErrorCode setup_timestepper(TimeCtx& timectx, PetscScalar tau) 
{
  int i,j;
  
  timectx.HI_el[0] = tau*4.389152966531085*2/timectx.Tpb;
  timectx.HI_el[1] = tau*-1.247624770988935*2/timectx.Tpb;
  timectx.HI_el[2] = tau*0.614528095966794*2/timectx.Tpb;
  timectx.HI_el[3] = tau*-0.327484862937516*2/timectx.Tpb;

  timectx.er[0] = -0.113917196281990;
  timectx.er[1] = 0.400761520311650;
  timectx.er[2] = -0.813632449486927;
  timectx.er[3] = 1.526788125457267;

  PetscScalar D1_time[4][4] = {
    {-3.3320002363522817*2./timectx.Tpb,4.8601544156851962*2./timectx.Tpb,-2.1087823484951789*2./timectx.Tpb,0.5806281691622644*2./timectx.Tpb},
    {-0.7575576147992339*2./timectx.Tpb,-0.3844143922232086*2./timectx.Tpb,1.4706702312807167*2./timectx.Tpb,-0.3286982242582743*2./timectx.Tpb},
    {0.3286982242582743*2./timectx.Tpb,-1.4706702312807167*2./timectx.Tpb,0.3844143922232086*2./timectx.Tpb,0.7575576147992339*2./timectx.Tpb},
    {-0.5806281691622644*2./timectx.Tpb,2.1087823484951789*2./timectx.Tpb,-4.8601544156851962*2./timectx.Tpb,3.3320002363522817*2./timectx.Tpb}
  };

  PetscScalar HI_BL_time[4][4] = {
    {6.701306630115196*2./timectx.Tpb,-3.571157279331500*2./timectx.Tpb,1.759003615747388*2./timectx.Tpb,-0.500000000000000*2./timectx.Tpb},
    {-1.904858685372247*2./timectx.Tpb,1.015107998460294*2./timectx.Tpb,-0.500000000000000*2./timectx.Tpb,0.142125915923019*2./timectx.Tpb},
    {0.938254199681965*2./timectx.Tpb,-0.500000000000000*2./timectx.Tpb,0.246279214013876*2./timectx.Tpb,-0.070005317729047*2./timectx.Tpb},
    {-0.500000000000000*2./timectx.Tpb,0.266452311201742*2./timectx.Tpb,-0.131243331549891*2./timectx.Tpb,0.037306157410634*2./timectx.Tpb}
  };

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      timectx.D[j][i] = D1_time[j][i] + tau*HI_BL_time[j][i];
    }
  }

  return 0;
}

PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) 
{
  PetscInt i;
  PetscScalar **vfinal_arr, **v_arr;

  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    vfinal_arr[i][0] = timectx.er[0]*v_arr[i][0] + timectx.er[1]*v_arr[i][1] + timectx.er[2]*v_arr[i][2] + timectx.er[3]*v_arr[i][3];
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,v,&v_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x, v_final, &vfinal_arr); 

  return 0;
}

PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) 
{
  PetscScalar **arr;

  VecWAXPY(*v_error,-1,v1,v2);

  VecNorm(*v_error,NORM_2,l2_error);

  *l2_error = sqrt(gridctx.h[0])*(*l2_error);
  VecNorm(*v_error,NORM_INFINITY,max_error);
  
  DMDAVecGetArrayDOF(gridctx.da_x, *v_error, &arr);
  *H_error = gridctx.H.get_norm_1D(arr, gridctx.h[0], gridctx.N[0], gridctx.i_start[0], gridctx.i_end[0], gridctx.dofs);
  DMDAVecRestoreArrayDOF(gridctx.da_x, *v_error, &arr);

  return 0;
}

PetscErrorCode analytic_solution(const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic)
{ 
  PetscScalar x, **array_analytic;
  PetscInt i;

  DMDAVecGetArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    x = gridctx.xl[0] + i*gridctx.h[0];
    array_analytic[i][0] = gaussian(x-gridctx.a(i)*t);
  }

  DMDAVecRestoreArrayDOF(gridctx.da_x, v_analytic, &array_analytic); 

  return 0;
};

PetscScalar gaussian(PetscScalar x) 
{
  PetscScalar rstar = 0.1;
  return exp(-x*x/(rstar*rstar));
}

PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0)
{ 
  PetscScalar       **b_arr, **v0_arr;
  PetscInt          i;

  DMDAVecGetArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x,v0,&v0_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    b_arr[i][0] = timectx.HI_el[0]*v0_arr[i][0];
    b_arr[i][1] = timectx.HI_el[1]*v0_arr[i][0];
    b_arr[i][2] = timectx.HI_el[2]*v0_arr[i][0];
    b_arr[i][3] = timectx.HI_el[3]*v0_arr[i][0];
  }

  DMDAVecRestoreArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecRestoreArrayDOF(gridctx.da_x,v0,&v0_arr); 

  return 0;
};

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
