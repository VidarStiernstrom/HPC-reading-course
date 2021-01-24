static char help[] = "Solves advection 1D problem u_t + u_x = 0.\n";

#include <petsc.h>
// #include "timestepping.h"
#include "appctx.h"
#include "grids/grid_function.h"
#include "grids/create_layout.h"
#include "IO_utils.h"
#include "scatter_ctx.h"
#include "multigrid.h"
#include "standard.h"
#include "imp_timestepping.h"
#include "adv_1D.h"

extern PetscErrorCode print_vec(GridCtx gridctx, Vec v, PetscInt comp);
extern PetscErrorCode shell2sparse(Mat D_shell, Mat& D_sparse, MatCtx matctx);
extern PetscErrorCode shell2dense(Mat D_shell, Mat& D_sparse, MatCtx matctx);

int main(int argc,char **argv)
{ 
  Vec            v;
  Mat            Dfine;
  PetscInt       stencil_radius, i_xstart, i_xend, Nx, nx, Nt, dofs, tblocks;
  PetscScalar    xl, xr, dx, dxi, dt, dti, t0, Tend, Tpb, tau;
  
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
  tau = 1.0; // SAT parameter

  // Fine space grid
  Nx = 2001;
  dx = (xr - xl)/(Nx-1);
  dxi = 1./dx;

  // Time
  tblocks = 100;
  t0 = 0;
  Tend = 0.5;
  Tpb = Tend/tblocks;
  Nt = 4;
  dt = Tpb/(Nt-1);
  dti = 1./dt;

  nlevels = 4;

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
  F_matctx.timectx.tblocks = tblocks;

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
  MatShellSetContext(Dfine, &F_matctx);
  MatSetUp(Dfine);

  setup_timestepper(F_matctx.timectx, tau);

  DMCreateGlobalVector(F_matctx.gridctx.da_xt,&v);


  mgsolver(Dfine, v, nlevels);
  // standard_solver(Dfine, v);

   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      // Free work space.  All PETSc objects should be destroyed when they
      // are no longer needed.
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  VecDestroy(&v);
  MatDestroy(&Dfine);
  
  ierr = PetscFinalize();
  return ierr;
}



PetscErrorCode print_vec(GridCtx gridctx, Vec v, PetscInt comp)
{
  PetscInt i, rank;
  PetscScalar **v_arr;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 
  printf("Rank %d printing...\n", rank);
  for (i = gridctx.i_start[0]; i < gridctx.i_end[0]; i++) {
    printf("v(%d) = %.16f;\n",i+1,v_arr[i][comp]);
  }
  printf("Rank %d done...\n", rank);
  DMDAVecGetArrayDOF(gridctx.da_xt,v,&v_arr); 

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























