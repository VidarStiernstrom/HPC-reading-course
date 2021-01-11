static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#define PROBLEM_TYPE_1D_O2

#include <petsc.h>
#include "sbpops/D1_central.h"
#include "sbpops/ICF_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "diffops/first_order_ode.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"
#include <unistd.h>

struct MatCtx {
  GridCtx gridctx;
};

struct PCCtx {
  MatCtx F_matctx, C_matctx;
};

extern PetscErrorCode LHS(Mat, Vec, Vec);

int main(int argc,char **argv)
{ 
  Vec            v, b;
  Mat            D, D_sparse;
  PetscInt       stencil_radius, i_xstart, i_xend, N, n, dofs;
  PetscScalar    xl, xr, hi, h;
  PCCtx          pcctx;

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  // if (get_inputs_1d(argc, argv, &N, &Tend, &CFL, &use_custom_ts, &use_custom_sc) == -1) {
  //   PetscFinalize();
  //   return -1;
  // }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Problem setup
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // Space
  N = 204;
  xl = -1;
  xr = 1;
  hi = (N-1)/(xr-xl);
  h = 1.0/hi;
  dofs = 1;

  auto a = [](const PetscInt i){ return 1.5;};

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  auto [stencil_width, nc, cw] = pcctx.F_matctx.gridctx.D1.get_ranges();
  stencil_radius = (stencil_width-1)/2;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, dofs, stencil_radius, NULL, &pcctx.F_matctx.gridctx.da_x);CHKERRQ(ierr);
  DMSetFromOptions(pcctx.F_matctx.gridctx.da_x);
  DMSetUp(pcctx.F_matctx.gridctx.da_x);
  DMDAGetCorners(pcctx.F_matctx.gridctx.da_x,&i_xstart,NULL,NULL,&n,NULL,NULL);
  i_xend = i_xstart + n;

  printf("Rank: %d, istart: %d, iend: %d\n",rank,i_xstart,i_xend);

  // Populate application context.
  pcctx.F_matctx.gridctx.N = {N};
  pcctx.F_matctx.gridctx.hi = {hi};
  pcctx.F_matctx.gridctx.h = {h};
  pcctx.F_matctx.gridctx.xl = {xl};
  pcctx.F_matctx.gridctx.i_start = {i_xstart};
  pcctx.F_matctx.gridctx.i_end = {i_xend};
  pcctx.F_matctx.gridctx.dofs = dofs;
  pcctx.F_matctx.gridctx.a = a;
  pcctx.F_matctx.gridctx.sw = stencil_radius;

  MatCreateShell(PETSC_COMM_WORLD,n,n,N,N,&pcctx.F_matctx.gridctx,&D);
  MatShellSetOperation(D,MATOP_MULT,(void(*)(void))LHS);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define matrix object
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatCreateAIJ(PETSC_COMM_WORLD, n, n, N, N, 0, NULL, 0, NULL, &D_sparse);

  MatSetOption(D_sparse, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  MatSetUp(D_sparse);

  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x,&v);
  DMCreateGlobalVector(pcctx.F_matctx.gridctx.da_x,&b);

  int j = 0, i, count;
  PetscInt idx[n];
  PetscInt col_idx[1];
  PetscScalar val[n];
  VecSet(v,0.0);
  for (j = 0; j < N; j++) {
    VecSetValue(v,j,1,INSERT_VALUES);
    MatMult(D,v,b);
    VecSetValue(v,j,0,INSERT_VALUES);

    PetscScalar *arr;
    VecGetArray(b,&arr); 
    
    col_idx[0] = j;

    // sleep(rank);
    count = 0;
    for (i = 0; i < n; i++) {
      if (arr[i] != 0) {
        idx[count] = i + i_xstart;
        val[count] = arr[i];
        count++;
      }
      // printf("v[%d] = %f\n",i,arr[i]);
      // idx[i] = i + i_xstart;
    }

    MatSetValues(D_sparse, count , idx , 1, col_idx, val , INSERT_VALUES);

    VecRestoreArray(b,&arr); 
  }

  MatAssemblyBegin(D_sparse, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D_sparse, MAT_FINAL_ASSEMBLY);


  PetscViewer viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
  PetscViewerPushFormat(viewer,   PETSC_VIEWER_ASCII_MATLAB );
  MatView(D_sparse, viewer); 
  PetscViewerPopFormat(viewer);

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
  GridCtx            *gridctx;

  MatShellGetContext(D, &gridctx);

  DMGetLocalVector(gridctx->da_x, &v_src_local);

  DMGlobalToLocalBegin(gridctx->da_x,v_src,INSERT_VALUES,v_src_local);
  DMGlobalToLocalEnd(gridctx->da_x,v_src,INSERT_VALUES,v_src_local);

  DMDAVecGetArrayDOF(gridctx->da_x,v_dst,&array_dst); 
  DMDAVecGetArrayDOF(gridctx->da_x,v_src_local,&array_src); 

  sbp::ode_apply_all(gridctx->D1, gridctx->HI, gridctx->a, array_src, array_dst, gridctx->i_start[0], gridctx->i_end[0], gridctx->N[0], gridctx->hi[0]);

  DMDAVecRestoreArrayDOF(gridctx->da_x,v_src_local,&array_src); 
  DMDAVecRestoreArray(gridctx->da_x,v_dst,&array_dst); 

  DMRestoreLocalVector(gridctx->da_x,&v_src_local);

  return 0;
}
