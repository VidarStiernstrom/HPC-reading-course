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
#include "multigrid.h"

#include <unistd.h>

struct TimeCtx {
  PetscInt N;
  PetscScalar Tend, Tpb;
  PetscScalar er[4], HI_el[4], D[4][4];
};

struct MatCtx {
  GridCtx gridctx;
  TimeCtx timectx;
};

struct InterpCtx {
  GridCtx *F_gridctx, *C_gridctx;
};

extern PetscErrorCode setup_timestepper(TimeCtx& timectx, PetscScalar tau);
extern PetscErrorCode apply_pc(PC pc, Vec xin, Vec xout) ;
extern PetscErrorCode get_solution(Vec& v_final, Vec& v, const GridCtx& gridctx, const TimeCtx& timectx) ;
extern PetscErrorCode get_error(const GridCtx& gridctx, const Vec& v1, const Vec& v2, Vec *v_error, PetscReal *H_error, PetscReal *l2_error, PetscReal *max_error) ;
extern PetscErrorCode analytic_solution( const GridCtx& gridctx, const PetscScalar t, Vec& v_analytic);
extern PetscScalar gaussian(PetscScalar x) ;
extern PetscErrorCode RHS(const GridCtx& gridctx, const TimeCtx& timectx, Vec& b, Vec v0);
extern PetscErrorCode LHS(Mat D, Vec v_src, Vec v_dst);
// extern PetscErrorCode apply_F2C(Vec &xfine, Vec &xcoarse, PCCtx pcctx) ;
// extern PetscErrorCode apply_C2F(Vec &xfine, Vec &xcoarse, PCCtx pcctx) ;
extern PetscErrorCode print_vec(GridCtx gridctx, Vec v, PetscInt comp);
extern PetscErrorCode shell2sparse(Mat D_shell, Mat& D_sparse, MatCtx matctx);
extern PetscErrorCode shell2dense(Mat D_shell, Mat& D_sparse, MatCtx matctx);
extern PetscErrorCode setup_operators(MatCtx *C_matctx, MatCtx *F_matctx, InterpCtx *prolctx, InterpCtx *restctx);
extern PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse);
extern PetscErrorCode apply_P(Mat P, Vec xcoarse, Vec xfine);

int main(int argc,char **argv)
{ 
  Vec            v, v_analytic, v_error, v_final;
  Mat            D_sparse, D_dense, Dfine;
  PC             pc;
  PetscInt       stencil_radius, stencil_radius_pc, i_xstart, i_xend, Nx, nx, Nt, ngx, Nx_pc, nx_pc, dofs, tblocks, ig_xstart, ig_xend, blockidx;
  PetscScalar    xl, xr, dx, dxi, dx_pc, dxi_pc, dt, dti, t0, Tend, Tpb, tau;
  PetscReal      l2_error, max_error, H_error;
  MatCtx         F_matctx;
  PetscInt       nlevels, level;

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
  Nx = 5121;
  dx = (xr - xl)/(Nx-1);
  dxi = 1./dx;

  // Time
  tblocks = 1;
  t0 = 0;
  Tend = 0.01;
  Tpb = Tend/tblocks;
  Nt = 4;
  dt = Tpb/(Nt-1);
  dti = 1./dt;

  nlevels = 9;

  dofs = 1;

  auto a = [](const PetscInt i){ return 1;};

  F_matctx.gridctx.N = {Nx};
  F_matctx.gridctx.hi = {dxi};
  F_matctx.gridctx.h = {dx};
  F_matctx.gridctx.xl = {xl};
  F_matctx.gridctx.xr = {xr};
  F_matctx.gridctx.dofs = dofs;
  F_matctx.gridctx.a = a; 

  F_matctx.timectx.N = Nt;
  F_matctx.timectx.Tend = Tend;
  F_matctx.timectx.Tpb = Tpb;

  auto [stencil_width, nc, cw] = F_matctx.gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx, Nt*dofs, stencil_radius, NULL, &F_matctx.gridctx.da_xt);
  
  DMSetFromOptions(F_matctx.gridctx.da_xt);
  DMSetUp(F_matctx.gridctx.da_xt);
  DMDAGetCorners(F_matctx.gridctx.da_xt,&i_xstart,NULL,NULL,&nx,NULL,NULL);
  i_xend = i_xstart + nx;

  F_matctx.gridctx.n = {nx};
  F_matctx.gridctx.i_start = {i_xstart};
  F_matctx.gridctx.i_end = {i_xend};
  F_matctx.gridctx.sw = stencil_radius;

  printf("Rank: %d, number of unknowns: %d\n",rank,Nt*nx);

  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx, dofs, stencil_radius, NULL, &F_matctx.gridctx.da_x);
  DMSetFromOptions(F_matctx.gridctx.da_x);
  DMSetUp(F_matctx.gridctx.da_x);

  MatCreateShell(PETSC_COMM_WORLD,F_matctx.gridctx.n[0]*Nt,F_matctx.gridctx.n[0]*Nt,F_matctx.gridctx.N[0]*Nt,F_matctx.gridctx.N[0]*Nt,&F_matctx,&Dfine);
  MatShellSetOperation(Dfine,MATOP_MULT,(void(*)(void))LHS);
  MatSetDM(Dfine, F_matctx.gridctx.da_xt);

  setup_timestepper(F_matctx.timectx, tau);

  // COARSE
  MatCtx C_matctxs[nlevels-1];
  InterpCtx restctx[nlevels-1], prolctx[nlevels-1];
  Mat    Dcoarses[nlevels-1];
  Mat    P[nlevels-1], R[nlevels-1];

  setup_operators(&C_matctxs[nlevels-2], &F_matctx, &prolctx[nlevels-2], &restctx[nlevels-2]);
  for (level = nlevels-2; level > 0; level--) {
    setup_operators(&C_matctxs[level-1], &C_matctxs[level], &prolctx[level-1], &restctx[level-1]);    
  }

  MatCreateShell(PETSC_COMM_WORLD,C_matctxs[nlevels-2].gridctx.n[0]*Nt,F_matctx.gridctx.n[0]*Nt,C_matctxs[nlevels-2].gridctx.N[0]*Nt,F_matctx.gridctx.N[0]*Nt,&restctx[nlevels-2],&R[nlevels-2]);
  MatShellSetOperation(R[nlevels-2],MATOP_MULT,(void(*)(void))apply_R);
  for (level = 1; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level-1].gridctx.n[0]*Nt,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level-1].gridctx.N[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,&restctx[level-1],&R[level-1]);
    MatShellSetOperation(R[level-1],MATOP_MULT,(void(*)(void))apply_R);    
  }

  MatCreateShell(PETSC_COMM_WORLD,F_matctx.gridctx.n[0]*Nt,C_matctxs[nlevels-2].gridctx.n[0]*Nt,F_matctx.gridctx.N[0]*Nt,C_matctxs[nlevels-2].gridctx.N[0]*Nt,&prolctx[nlevels-2],&P[nlevels-2]);
  MatShellSetOperation(P[nlevels-2],MATOP_MULT,(void(*)(void))apply_P);  
  for (level = 1; level < nlevels-1; level++) {
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level-1].gridctx.n[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,C_matctxs[level-1].gridctx.N[0]*Nt,&prolctx[level-1],&P[level-1]);
    MatShellSetOperation(P[level-1],MATOP_MULT,(void(*)(void))apply_P);    
  }

  for (level = 0; level < nlevels-1; level++) {
    PetscPrintf(PETSC_COMM_WORLD, "Level: %d, Nx: %d, nx: %d\n",level,C_matctxs[level].gridctx.N[0],C_matctxs[level].gridctx.n[0]);
    MatCreateShell(PETSC_COMM_WORLD,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level].gridctx.n[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,C_matctxs[level].gridctx.N[0]*Nt,&C_matctxs[level],&Dcoarses[level]);
    MatShellSetOperation(Dcoarses[level],MATOP_MULT,(void(*)(void))LHS);
    MatSetDM(Dcoarses[level], C_matctxs[level].gridctx.da_xt);
  }
  PetscPrintf(PETSC_COMM_WORLD, "Level: %d, Nx: %d, nx: %d\n",nlevels-1,F_matctx.gridctx.N[0],F_matctx.gridctx.n[0]);

  KSP ksp;
  setup_mgsolver(ksp, nlevels, Dfine, Dcoarses, P, R);

  MatCreateShell(PETSC_COMM_WORLD,C_matctxs[0].gridctx.n[0]*Nt,C_matctxs[0].gridctx.n[0]*Nt,C_matctxs[0].gridctx.N[0]*Nt,C_matctxs[0].gridctx.N[0]*Nt,&C_matctxs[0],&Dcoarses[0]);
  MatShellSetOperation(Dcoarses[0],MATOP_MULT,(void(*)(void))LHS);
  MatSetDM(Dcoarses[0], C_matctxs[0].gridctx.da_xt);

  /////////////////// SOLVE

  DMCreateGlobalVector(F_matctx.gridctx.da_x, &F_matctx.gridctx.v_curr);
  DMCreateGlobalVector(F_matctx.gridctx.da_xt,&F_matctx.gridctx.b);
  DMCreateGlobalVector(F_matctx.gridctx.da_xt,&v);

  analytic_solution(F_matctx.gridctx, 0, F_matctx.gridctx.v_curr);

  VecDuplicate(F_matctx.gridctx.v_curr,&v_error);
  VecDuplicate(F_matctx.gridctx.v_curr,&v_analytic);

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v1);
  }

  for (blockidx = 0; blockidx < tblocks; blockidx++) 
  {
    RHS(F_matctx.gridctx, F_matctx.timectx, F_matctx.gridctx.b, F_matctx.gridctx.v_curr);

    KSPSolve(ksp, F_matctx.gridctx.b, v);
    get_solution(F_matctx.gridctx.v_curr, v, F_matctx.gridctx, F_matctx.timectx); 
  }

  PetscBarrier((PetscObject) v);
  if (rank == 0) {
    PetscTime(&v2);
    elapsed_time = v2 - v1; 
  }

  PetscPrintf(PETSC_COMM_WORLD,"Elapsed time: %f seconds\n",elapsed_time);

  ///////////// CHECK SOLUTION

  // write_vector_to_binary(v,"data/adv_1D","v");
  // write_vector_to_binary(pcctx.F_matctx.gridctx.v_curr,"data/adv_1D","v_final");

  analytic_solution(F_matctx.gridctx, Tend, v_analytic);

  get_solution(F_matctx.gridctx.v_curr, v, F_matctx.gridctx, F_matctx.timectx); 
  get_error(F_matctx.gridctx, F_matctx.gridctx.v_curr, v_analytic, &v_error, &H_error, &l2_error, &max_error);
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error is: %.8e, the H-error is: %.9e and the maximum error is %.9e\n",l2_error,H_error,max_error);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // VecDestroy(&v);
  // VecDestroy(&pcctx.F_matctx.gridctx.v_curr);
  // VecDestroy(&v_error);
  // VecDestroy(&v_analytic);
  // VecDestroy(&v_final);
  // PCDestroy(&pc);
  // DMDestroy(&pcctx.F_matctx.gridctx.da_xt);
  // DMDestroy(&pcctx.F_matctx.gridctx.da_x);
  
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode apply_P(Mat P, Vec xcoarse, Vec xfine)
{
  PetscScalar       **array_src, **array_dst;
  Vec               xcoarse_local;
  InterpCtx         *prolctx;
  DM                dmda_fine, dmda_coarse;

  MatShellGetContext(P, &prolctx);

  dmda_fine = prolctx->F_gridctx->da_xt;
  dmda_coarse = prolctx->C_gridctx->da_xt;

  // printf("hejda: %d\n",prolctx->Nx);
  // printf("hej: %d\n",interpctx->C_gridctx->n[0]);

  // PetscInt m, n;

  dmda_fine = prolctx->F_gridctx->da_xt;
  dmda_coarse = prolctx->C_gridctx->da_xt;

  DMGetLocalVector(dmda_coarse, &xcoarse_local);

  DMGlobalToLocalBegin(dmda_coarse, xcoarse, INSERT_VALUES, xcoarse_local);
  DMGlobalToLocalEnd(dmda_coarse, xcoarse, INSERT_VALUES, xcoarse_local);

  DMDAVecGetArrayDOF(dmda_fine, xfine, &array_dst);
  DMDAVecGetArrayDOF(dmda_coarse, xcoarse_local, &array_src);

  sbp::apply_C2F(prolctx->F_gridctx->ICF, array_src, array_dst, prolctx->F_gridctx->i_start[0], prolctx->F_gridctx->i_end[0], prolctx->F_gridctx->N[0]);

  DMDAVecRestoreArrayDOF(dmda_fine, xfine, &array_dst);
  DMDAVecRestoreArrayDOF(dmda_coarse, xcoarse_local, &array_src);

  DMRestoreLocalVector(dmda_coarse, &xcoarse_local);

  // VecSet(xfine,0.0);
  return 0;
}

PetscErrorCode apply_R(Mat R, Vec xfine, Vec xcoarse)
{
  PetscScalar       **array_src, **array_dst;
  Vec               xfine_local;
  InterpCtx         *restctx;
  DM                dmda_fine, dmda_coarse;

  MatShellGetContext(R, &restctx);

  dmda_fine = restctx->F_gridctx->da_xt;
  dmda_coarse = restctx->C_gridctx->da_xt;

  // printf("hejda:\n");
  // printf("hej: %d\n",interpctx->C_gridctx->n[0]);

  // PetscInt m, n;

  // MatGetSize(R, &m, &n);
  // printf("m: %d, n: %d\n",m,n);

  DMGetLocalVector(dmda_fine, &xfine_local);

  DMGlobalToLocalBegin(dmda_fine, xfine, INSERT_VALUES, xfine_local);
  DMGlobalToLocalEnd(dmda_fine, xfine, INSERT_VALUES, xfine_local);

  DMDAVecGetArrayDOF(dmda_coarse, xcoarse, &array_dst);
  DMDAVecGetArrayDOF(dmda_fine, xfine_local, &array_src);

  sbp::apply_F2C(restctx->F_gridctx->ICF, array_src, array_dst, restctx->C_gridctx->i_start[0], restctx->C_gridctx->i_end[0], restctx->C_gridctx->N[0]);

  DMDAVecRestoreArrayDOF(dmda_coarse, xcoarse, &array_dst);
  DMDAVecRestoreArrayDOF(dmda_fine, xfine_local, &array_src);

  DMRestoreLocalVector(dmda_fine, &xfine_local);

  // VecSet(xcoarse, 0.0);
  return 0;
}

PetscErrorCode setup_operators(MatCtx *C_matctx, MatCtx *F_matctx, InterpCtx *prolctx, InterpCtx *restctx)
{
  PetscInt       stencil_radius, stencil_radius_pc, i_xstart, i_xend, Nx, nx, Nt, ngx, Nx_pc, nx_pc, dofs, tblocks, ig_xstart, ig_xend, blockidx;
  PetscScalar    xl, xr, dx, dxi, dx_pc, dxi_pc, dt, dti, t0, Tend, Tpb, tau;

  // MatCtx
  Nx = F_matctx->gridctx.N[0];
  dxi = F_matctx->gridctx.hi[0];
  dx = F_matctx->gridctx.h[0];
  xl = F_matctx->gridctx.xl[0];
  xr = F_matctx->gridctx.xr[0];
  dofs = F_matctx->gridctx.dofs;
  auto a = F_matctx->gridctx.a;
  Nt = F_matctx->timectx.N;
  Tend = F_matctx->timectx.Tend;
  Tpb = F_matctx->timectx.Tpb;

  // Nx_pc = 0.5*(Nx + 1);
  Nx_pc = (Nx - 1)/2 + 1;
  dx_pc = (xr - xl)/(Nx_pc-1);
  dxi_pc = 1./dx_pc;
  Nt = 4;
  tau = 1;

  C_matctx->gridctx.N = {Nx_pc};
  C_matctx->gridctx.hi = {dxi_pc};
  C_matctx->gridctx.h = {dx_pc};
  C_matctx->gridctx.xl = {xl};
  C_matctx->gridctx.dofs = dofs;
  C_matctx->gridctx.a = a;

  C_matctx->timectx.N = Nt;
  C_matctx->timectx.Tend = Tend;
  C_matctx->timectx.Tpb = Tpb;

  auto [stencil_width_pc, nc_pc, cw_pc] = C_matctx->gridctx.D1.get_ranges();
  stencil_radius = (stencil_width_pc-1)/2;
  DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, Nx_pc, Nt*dofs, stencil_radius, NULL, &C_matctx->gridctx.da_xt);
  
  DMSetFromOptions(C_matctx->gridctx.da_xt);
  DMSetUp(C_matctx->gridctx.da_xt);
  DMDAGetCorners(C_matctx->gridctx.da_xt,&i_xstart,NULL,NULL,&nx_pc,NULL,NULL);
  i_xend = i_xstart + nx_pc;

  C_matctx->gridctx.n = {nx_pc};
  C_matctx->gridctx.i_start = {i_xstart};
  C_matctx->gridctx.i_end = {i_xend};
  C_matctx->gridctx.sw = stencil_radius;

  setup_timestepper(C_matctx->timectx, tau);

  // InterpCtx
  restctx->F_gridctx = &F_matctx->gridctx;
  restctx->C_gridctx = &C_matctx->gridctx;

  prolctx->F_gridctx = &F_matctx->gridctx;
  prolctx->C_gridctx = &C_matctx->gridctx;

  return 0;
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
  DM                da;

  MatShellGetContext(D, &matctx);
  MatGetDM(D, &da);

  DMGetLocalVector(da, &v_src_local);

  DMGlobalToLocalBegin(da,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(da,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(da,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(da,v_src_local,&array_src); 

  sbp::adv_imp_apply_all(matctx->timectx.D, matctx->gridctx.D1, matctx->gridctx.HI, matctx->gridctx.a, array_src, array_dst, matctx->gridctx.i_start[0], matctx->gridctx.i_end[0], matctx->gridctx.N[0], matctx->gridctx.hi[0], matctx->timectx.Tpb);

  DMDAVecRestoreArrayDOF(da,v_dst,&array_dst); 
  DMDAVecRestoreArrayDOF(da,v_src_local,&array_src); 

  DMRestoreLocalVector(da,&v_src_local);
  
  return 0;
}

PetscErrorCode shell2sparse(Mat D_shell, Mat& D_sparse, MatCtx matctx) {
  Vec v, b;

  MatCreateAIJ(PETSC_COMM_WORLD, matctx.gridctx.n[0]*matctx.timectx.N, matctx.gridctx.n[0]*matctx.timectx.N, matctx.gridctx.N[0]*matctx.timectx.N, matctx.gridctx.N[0]*matctx.timectx.N, 0, NULL, 0, NULL, &D_sparse);

  MatSetOption(D_sparse, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  MatSetUp(D_sparse);

  DMCreateGlobalVector(matctx.gridctx.da_xt,&v);
  DMCreateGlobalVector(matctx.gridctx.da_xt,&b);

  int j = 0, i, k, count;
  PetscInt idx[matctx.gridctx.n[0]*matctx.timectx.N];
  PetscInt col_idx[1];
  PetscScalar val[matctx.gridctx.n[0]*matctx.timectx.N];
  VecSet(v,0.0);

  // Loop over all columns
  for (j = 0; j < matctx.gridctx.N[0]*matctx.timectx.N; j++) {

    // Set v[j] = 1
    VecSetValue(v,j,1,INSERT_VALUES);

    VecAssemblyBegin(v);
    VecAssemblyEnd(v);

    // Compute D*v
    MatMult(D_shell,v,b);
    VecSetValue(v,j,0,INSERT_VALUES);

    // Loop over elements in b, save non-zero in arr
    PetscScalar **arr;
    DMDAVecGetArrayDOF(matctx.gridctx.da_xt, b, &arr); 
    
    col_idx[0] = j;

    count = 0;
    for (i = matctx.gridctx.i_start[0]; i < matctx.gridctx.i_end[0]; i++) {
      for (k = 0; k < matctx.timectx.N; k++) {
        if (arr[i][k] != 0) {
          idx[count] = k + matctx.timectx.N*i;
          val[count] = arr[i][k];
          count++;
        }
      }
    }

    // Insert non-zero b-values in column j of Dpsarse
    MatSetValues(D_sparse, count , idx , 1, col_idx, val , INSERT_VALUES);

    DMDAVecRestoreArrayDOF(matctx.gridctx.da_xt, b,&arr); 
  }

  MatAssemblyBegin(D_sparse, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D_sparse, MAT_FINAL_ASSEMBLY);

  return 0;
}

PetscErrorCode shell2dense(Mat D_shell, Mat& D_dense, MatCtx matctx) {
  Vec v, b;

  MatCreateDense(PETSC_COMM_WORLD, matctx.gridctx.n[0]*matctx.timectx.N, matctx.gridctx.n[0]*matctx.timectx.N, matctx.gridctx.N[0]*matctx.timectx.N, matctx.gridctx.N[0]*matctx.timectx.N, NULL, &D_dense);

  MatSetOption(D_dense, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  MatSetUp(D_dense);

  DMCreateGlobalVector(matctx.gridctx.da_xt,&v);
  DMCreateGlobalVector(matctx.gridctx.da_xt,&b);

  int j = 0, i, k, count;
  PetscInt idx[matctx.gridctx.n[0]*matctx.timectx.N];

  PetscInt col_idx[1];
  PetscScalar val[matctx.gridctx.n[0]*matctx.timectx.N];
  VecSet(v,0.0);

  // Loop over all columns
  for (j = 0; j < matctx.gridctx.N[0]*matctx.timectx.N; j++) {

    // Set v[j] = 1
    VecSetValue(v,j,1,INSERT_VALUES);

    VecAssemblyBegin(v);
    VecAssemblyEnd(v);

    // Compute D*v
    MatMult(D_shell,v,b);
    VecSetValue(v,j,0,INSERT_VALUES);

    // Loop over elements in b, save non-zero in arr
    PetscScalar **arr;
    DMDAVecGetArrayDOF(matctx.gridctx.da_xt, b, &arr); 
    
    col_idx[0] = j;

    count = 0;
    for (i = matctx.gridctx.i_start[0]; i < matctx.gridctx.i_end[0]; i++) {
      for (k = 0; k < matctx.timectx.N; k++) {
        idx[count] = k + matctx.timectx.N*i;
        val[count] = arr[i][k];
        count++;
      }
    }

    // Insert non-zero b-values in column j of Dpsarse
    MatSetValues(D_dense, count , idx , 1, col_idx, val , INSERT_VALUES);

    DMDAVecRestoreArrayDOF(matctx.gridctx.da_xt, b,&arr); 
  }

  MatAssemblyBegin(D_dense, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D_dense, MAT_FINAL_ASSEMBLY);

  return 0;
}





















