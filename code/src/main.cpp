
static char help[] ="Computes the derivative of a 1D quadratic function.";

/*
   Example program computing the derivative of a 1D quadratic function anc computing the l2-error
   in parallel.
*/

#include "petscsys.h" 
#include <petscdmda.h>
#include "petscvec.h"
#include "sbpops/make_diff_op.h"
#include "diffops/advection.h"

extern PetscErrorCode initial_condition(DM da, Vec v_global);
extern PetscErrorCode analytic_solution(DM da, Vec v_analytic);

int main(int argc,char **argv)
{
  Vec            v_glob, v_loc, v_tmp;
  PetscInt       Nx, nx, i_start;
  PetscInt       order, stencil_radius;
  DM             da;             

  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Nx = 101;
  PetscScalar xl = 0;
  PetscScalar xr = 1;
  PetscPrintf(PETSC_COMM_WORLD,"Differentiating the quadratic function 2x^2 on the domain [%.2f, %.2f] with N = %d grid points, using %d processes.\n",xl,xr,Nx,size);
  PetscScalar hi = (Nx-1)/(xr-xl);
  order = 4;
  stencil_radius = (order+1)/2;
  DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,Nx,1,stencil_radius,0,&da);

  DMSetFromOptions(da);
  DMSetUp(da);
  DMDASetUniformCoordinates(da, xl, xr, 0, 0, 0, 0);
  DMDAGetCorners(da,&i_start,0,0,&nx,0,0);
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Extract global vectors from DMDA; then duplicate for remaining
      vectors that are the same types
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMCreateGlobalVector(da,&v_glob);
  DMCreateLocalVector(da,&v_loc);
  VecDuplicate(v_glob,&v_tmp);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set initial condition on global vector and scatter to local vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  initial_condition(da, v_glob);
  DMGlobalToLocalBegin(da,v_glob,INSERT_VALUES,v_loc);
  DMGlobalToLocalEnd(da,v_glob,INSERT_VALUES,v_loc);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Apply the first derivative D1_central to the initial condition in v_glob,
    storing the result in the vector v_tmp
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  constexpr auto D1 = sbp::make_D1_central_4th_order();
  auto velocity_field = [](const int i){ return 2.; };
  sbp::advection_variable_distributed(D1,velocity_field,da,v_loc,hi,Nx,v_tmp);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute the error
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecCopy(v_tmp,v_glob); 
  analytic_solution(da, v_tmp);
  VecAYPX(v_tmp,-1.,v_glob);
  PetscReal l2_error;
  VecNorm(v_tmp,NORM_2,&l2_error);
  l2_error = sqrt(hi)*l2_error;
  PetscPrintf(PETSC_COMM_WORLD,"The l2-error error is: %g\n",l2_error);



  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecDestroy(&v_glob);
  VecDestroy(&v_loc);
  VecDestroy(&v_tmp);
  DMDestroy(&da);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode initial_condition(DM da, Vec v_global)
{ 
  PetscErrorCode ierr;
  Vec coords;
  ierr = DMGetCoordinates(da,&coords);
  ierr = VecCopy(coords,v_global); //Coords is a borrowed reference. Should not be destroyed.
  ierr = VecPow(v_global,2.);
  CHKERRQ(ierr);
  return 0;
}

PetscErrorCode analytic_solution(DM da, Vec v_analytic)
{ 
  PetscErrorCode ierr;
  Vec coords;
  ierr = DMGetCoordinates(da,&coords);
  ierr = VecCopy(coords,v_analytic); //Coords is a borrowed reference. Should not be destroyed.
  ierr = VecScale(v_analytic,4.);
  CHKERRQ(ierr);
  return 0;
};