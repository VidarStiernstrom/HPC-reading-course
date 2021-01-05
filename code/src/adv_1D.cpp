static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#define PROBLEM_TYPE_1D_O2

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/advection.h"
// #include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"

struct TimeCtx {
  PetscScalar Tend, Tpb;
  PetscScalar er[4], HI_el[4], D[4][4];
};

struct AppCtx {
  GridCtx F_gridctx, C_gridctx;
  TimeCtx timectx;
};

extern PetscErrorCode setup_timestepper(TimeCtx& timectx, PetscScalar tau);
extern PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) ;
extern PetscErrorCode get_solution(Vec& v_final, Vec& v, const AppCtx& appctx) ;
extern PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) ;
extern PetscErrorCode analytic_solution( const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic);
extern PetscScalar gaussian(PetscScalar x) ;
extern PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0);
extern PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst);

int main(int argc,char **argv)
{ 
  Mat D;
  Vec            v, v_analytic, v_error, v_final, b;
  PC             pc;
  PetscInt       stencil_radius, i_xstart, i_xend, Nx, nx, Nt, ngx, dofs, tblocks, ig_xstart, ig_xend, blockidx;
  PetscScalar    xl, xr, dx, dxi, dt, dti, t0, Tend, Tpb, tau;
  PetscReal      l2_error, max_error, H_error;
  AppCtx         appctx;

  PetscLogDouble v1,v2,elapsed_time = 0;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  // Space
  Nx = 2001;
  xl = -1;
  xr = 1;
  dx = (xr - xl)/(Nx-1);
  dxi = 1./dx;
  tau = 1.0; // SAT parameter
  
  // Time
  tblocks = 100;
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
  appctx.F_gridctx.N = {Nx};
  appctx.F_gridctx.hi = {dxi};
  appctx.F_gridctx.h = {dx};
  appctx.F_gridctx.xl = {xl};
  appctx.F_gridctx.dofs = dofs;
  appctx.F_gridctx.a = a;

  appctx.timectx.Tend = Tend;
  appctx.timectx.Tpb = Tpb;

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Create distributed array (DMDA) to manage parallel grid and vectors
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = appctx.F_gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx, Nt*dofs, stencil_radius, NULL, &appctx.F_gridctx.da_xt);
  
  DMSetFromOptions(appctx.F_gridctx.da_xt);
  DMSetUp(appctx.F_gridctx.da_xt);
  DMDAGetCorners(appctx.F_gridctx.da_xt,&i_xstart,NULL,NULL,&nx,NULL,NULL);
  i_xend = i_xstart + nx;
  DMDAGetGhostCorners(appctx.F_gridctx.da_xt,&ig_xstart,NULL,NULL,&ngx,NULL,NULL);
  ig_xend = ig_xstart + ngx;

  appctx.F_gridctx.i_start = {i_xstart};
  appctx.F_gridctx.i_end = {i_xend};
  appctx.F_gridctx.sw = stencil_radius;

  printf("Rank: %d, number of unknowns: %d\n",rank,Nt*nx);

  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx, dofs, stencil_radius, NULL, &appctx.F_gridctx.da_x);
  DMSetFromOptions(appctx.F_gridctx.da_x);
  DMSetUp(appctx.F_gridctx.da_x);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Setup solver
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(appctx.F_gridctx.da_xt,&v);
  DMCreateGlobalVector(appctx.F_gridctx.da_xt,&b);

  MatCreateShell(PETSC_COMM_WORLD,nx*Nt,nx*Nt,Nx*Nt,Nx*Nt,&appctx.F_gridctx,&D);
  MatShellSetOperation(D,MATOP_MULT,(void(*)(void))LHS);

  KSPCreate(PETSC_COMM_WORLD, &appctx.F_gridctx.ksp);
  KSPSetOperators(appctx.F_gridctx.ksp, D, D);
  KSPSetTolerances(appctx.F_gridctx.ksp, 1e-14, PETSC_DEFAULT, PETSC_DEFAULT, 1e5);
  KSPSetPCSide(appctx.F_gridctx.ksp, PC_RIGHT);
  KSPSetType(appctx.F_gridctx.ksp, KSPPIPEFGMRES);
  KSPGMRESSetRestart(appctx.F_gridctx.ksp, 10);
  KSPSetInitialGuessNonzero(appctx.F_gridctx.ksp, PETSC_TRUE);
  KSPSetFromOptions(appctx.F_gridctx.ksp);

  KSPGetPC(appctx.F_gridctx.ksp,&pc);
  PCSetType(pc,PCSHELL);
  // PCSetType(pc,PCNONE);
  PCShellSetContext(pc, &appctx);
  PCShellSetApply(pc, apply_pc);
  PCSetUp(pc);

  DMCreateGlobalVector(appctx.F_gridctx.da_x, &v_final);
  analytic_solution(appctx.F_gridctx, 0, v_final);

  setup_timestepper(appctx.timectx, tau);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run simulation and compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  for (blockidx = 0; blockidx < tblocks; blockidx++) {
    RHS(appctx.F_gridctx, appctx.timectx, b, v_final);
    KSPSolve(appctx.F_gridctx.ksp, b, v);
    get_solution(v_final, v, appctx); 
  }

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  VecDuplicate(v_final,&v_error);
  VecDuplicate(v_final,&v_analytic);

  write_vector_to_binary(v,"data/adv_1D","v");
  write_vector_to_binary(v_final,"data/adv_1D","v_final");

  analytic_solution(appctx.F_gridctx, Tend, v_analytic);

  get_error(appctx.F_gridctx, v_final, v_analytic, &v_error, &H_error, &l2_error, &max_error);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %.8e, the H-error is: %.9e and the maximum error is %.9e\n",l2_error,H_error,max_error);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v);
  VecDestroy(&v_final);
  VecDestroy(&v_error);
  VecDestroy(&v_final);
  PCDestroy(&pc);
  MatDestroy(&D);
  DMDestroy(&appctx.F_gridctx.da_xt);
  DMDestroy(&appctx.F_gridctx.da_x);
  
  ierr = PetscFinalize();
  return ierr;
}


PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) 
{
  // Vec               xcoarse;

  // PCShellGetContext(pc, (void**) &gridctxs);

  // DMGetGlobalVector(gridctxs->C_appctx.da, &xcoarse);

  // apply_F2C(xin, xcoarse, gridctxs);
  // KSPSolve(gridctxs->C_appctx.ksp, gridctxs->C_appctx.b, xcoarse);
  // apply_C2F(xout, xcoarse, gridctxs);

  // DMRestoreGlobalVector(gridctxs->C_appctx.da, &xcoarse);
  VecCopy(xin,xout);

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

PetscErrorCode get_solution(Vec& v_final, Vec& v, const AppCtx& appctx) 
{
  PetscInt i;
  PetscScalar **vfinal_arr, **v_arr;

  DMDAVecGetArrayDOF(appctx.F_gridctx.da_xt,v,&v_arr); 
  DMDAVecGetArrayDOF(appctx.F_gridctx.da_x, v_final, &vfinal_arr); 

  for (i = appctx.F_gridctx.i_start[0]; i < appctx.F_gridctx.i_end[0]; i++) {
    vfinal_arr[i][0] = appctx.timectx.er[0]*v_arr[i][0] + appctx.timectx.er[1]*v_arr[i][1] + appctx.timectx.er[2]*v_arr[i][2] + appctx.timectx.er[3]*v_arr[i][3];
  }

  DMDAVecRestoreArrayDOF(appctx.F_gridctx.da_xt,v,&v_arr); 
  DMDAVecRestoreArrayDOF(appctx.F_gridctx.da_x, v_final, &vfinal_arr); 

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
  PetscScalar       **b_arr, val, x, **v0_arr;
  PetscInt          i;

  DMDAVecGetArrayDOF(gridctx.da_xt,b,&b_arr); 
  DMDAVecGetArrayDOF(gridctx.da_x,v0,&v0_arr); 

  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    x = gridctx.xl[0] + i*gridctx.h[0];
    val = gaussian(x);
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
  AppCtx            *appctx;

  MatShellGetContext(D, &appctx);

  DMGetLocalVector(appctx->F_gridctx.da_xt, &v_src_local);

  DMGlobalToLocalBegin(appctx->F_gridctx.da_xt,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(appctx->F_gridctx.da_xt,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(appctx->F_gridctx.da_xt,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(appctx->F_gridctx.da_xt,v_src_local,&array_src); 

  sbp::adv_imp_apply_all(appctx->timectx.D, appctx->F_gridctx.D1, appctx->F_gridctx.HI, appctx->F_gridctx.a, array_src, array_dst, appctx->F_gridctx.i_start[0], appctx->F_gridctx.i_end[0], appctx->F_gridctx.N[0], appctx->F_gridctx.hi[0], appctx->timectx.Tpb);

  DMDAVecRestoreArrayDOF(appctx->F_gridctx.da_xt,v_dst,&array_dst); 
  DMDAVecRestoreArrayDOF(appctx->F_gridctx.da_xt,v_src_local,&array_src); 

  DMRestoreLocalVector(appctx->F_gridctx.da_xt,&v_src_local);
  
  return 0;
}
