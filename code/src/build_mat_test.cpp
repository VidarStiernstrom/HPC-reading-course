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
  Vec            v, b;
  Mat            D, D_sparse;
  PetscInt       stencil_radius, i_xstart, i_xend, Nx, nx, Nt, dofs, tblocks;
  PetscScalar    xl, xr, dx, dxi, dt, dti, t0, Tend, Tpb, tau;
  PCCtx          pcctx;

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
  
  // Time
  tblocks = 1;
  t0 = 0;
  Tend = 0.01;
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

  setup_timestepper(pcctx.F_matctx.timectx, tau);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Create distributed array (DMDA) to manage parallel grid and vectors of COARSE grid
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Setup solver
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */  
  MatCreateShell(PETSC_COMM_WORLD,nx*Nt,nx*Nt,Nx*Nt,Nx*Nt,&pcctx.F_matctx.gridctx,&D);
  MatShellSetOperation(D,MATOP_MULT,(void(*)(void))LHS);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define matrix object
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatCreateAIJ(PETSC_COMM_WORLD, nx*Nt, nx*Nt, Nx*Nt, Nx*Nt, 0, NULL, 0, NULL, &D_sparse);

  MatSetOption(D_sparse, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  MatSetUp(D_sparse);

  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_xt,&v);
  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_xt,&b);

  int j = 0, i, k, count;
  PetscInt idx[nx*Nt];
  PetscInt col_idx[1];
  PetscScalar val[nx*Nt];
  VecSet(v,0.0);
  for (j = 0; j < Nx*Nt; j++) {
  // j = 10;
    VecSetValue(v,j,1,INSERT_VALUES);

    VecAssemblyBegin(v);
    VecAssemblyEnd(v);

    // VecView(v,   PETSC_VIEWER_STDOUT_WORLD   ); 
    MatMult(D,v,b);
    VecSetValue(v,j,0,INSERT_VALUES);
    // VecView(b,   PETSC_VIEWER_STDOUT_WORLD   ); 

    PetscScalar **arr;
    DMDAVecGetArrayDOF(pcctx.F_matctx.gridctx.da_xt, b,&arr); 
    
    col_idx[0] = j;

    // sleep(rank);
    count = 0;
    for (i = i_xstart; i < i_xend; i++) {
      for (k = 0; k < Nt; k++) {
        if (arr[i][k] != 0) {
          idx[count] = k + 4*i;
          val[count] = arr[i][k];
          count++;
        }
        // printf("v[%d,%d] = %f\n",i,j,arr[i][j]);
        // idx[i] = i + i_xstart;
      }
    }


    MatSetValues(D_sparse, count , idx , 1, col_idx, val , INSERT_VALUES);

    DMDAVecRestoreArrayDOF(pcctx.F_matctx.gridctx.da_xt, b,&arr); 
  }

  MatAssemblyBegin(D_sparse, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D_sparse, MAT_FINAL_ASSEMBLY);


  // PetscViewer viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  // PetscViewerPushFormat(viewer,   PETSC_VIEWER_ASCII_MATLAB );
  // MatView(D_sparse, viewer); 
  // PetscViewerPopFormat(viewer);

  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  PetscRandom prand;
  PetscRandomCreate(PETSC_COMM_WORLD, &prand);
  VecSetRandom(v, prand);

  // VecView(v,  PETSC_VIEWER_STDOUT_WORLD   );
  Vec b2;
  VecDuplicate(b,&b2);

  MatMult(D_sparse  ,v,b);
  MatMult(D  ,v,b2);

  VecAXPY(b, -1.0, b2);

  PetscScalar val2;
  VecNorm(b, NORM_INFINITY, &val2);



  VecView(b,  PETSC_VIEWER_STDOUT_WORLD   );

  printf("norm: %e\n",val2);


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