
static char help[] ="Computes the derivative of a 1D quadratic function.";

/*
   Example program computing the derivative of a 1D quadratic function anc computing the l2-error
   in parallel.
*/

#include "petscsys.h" 
#include <petscdmda.h>
#include "petscvec.h"
#include "sbpops/D1_central.h"

extern PetscErrorCode initial_condition(DM da, Vec v_global);
extern PetscErrorCode analytic_solution(DM da, Vec v_analytic);

int main(int argc,char **argv)
{
  Vec            v_glob, v_loc, v_tmp;
  PetscScalar    *array_loc, *array_tmp;
  PetscInt       Nx, nx, i_start, i_end;
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
  PetscPrintf(PETSC_COMM_WORLD,"Differentiating the quadratic function x^2 on the domain [%.2f, %.2f] with N = %d grid points, using %d processes.\n",xl,xr,Nx,size);
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

  const int iw = 5;
  const int nc = 4;
  const int cw = 6;
  constexpr sbp::D1_central<iw,nc,cw> d1_4; //4th order central stencil
  DMDAVecGetArray(da,v_loc,&array_loc);
  DMDAVecGetArray(da,v_tmp,&array_tmp);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Apply the first derivative D1_central to the initial condition in v_glob,
    storing the result in the vector v_tmp
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  // Assume that no communication is to be made over the closures, i.e that
  // a processor at least has number of nodes equal to the closure width
  // E.g., on 3 processors then minimal number of global nodes is 6*3 = 18
  // TODO: The index ranges and applies performed by a process is given by
  // the problem size and number of processors, as well as the stencil widths of the derivative
  // We should consider (1) writing an D1_central::apply_distributed(DM, ...) which does this nicely
  // (2) create a new class/struct which bundles together the index ranges and the applies for a processor
  // in a nice way.
  i_end = i_start+nx;
  if (i_start < nc){
    for (PetscInt i = i_start; i < nc; i++) 
    {
      array_tmp[i] = d1_4.apply_left(array_loc,hi,i);
    }
    for (PetscInt i = nc; i < i_end; i++)
    {
      array_tmp[i] = d1_4.apply_interior(array_loc,hi,i);
    }
  }
  if ((nc < i_start) && (i_end < Nx-nc))
  {
    for (PetscInt i = i_start; i < i_end; i++)
    {
      array_tmp[i] = d1_4.apply_interior(array_loc,hi,i);
    }
  }
  if ((i_start < Nx-nc) && (Nx-nc < i_end))
  {
    for (PetscInt i = i_start; i < Nx-nc; i++)
    {
      array_tmp[i] = d1_4.apply_interior(array_loc,hi,i);
    }
    for (PetscInt i = Nx-nc; i < Nx; i++)
    {
      array_tmp[i] = d1_4.apply_right(array_loc,hi,Nx,i);
    }
  }
  DMDAVecRestoreArray(da,v_loc,array_loc);
  DMDAVecRestoreArray(da,v_tmp,array_tmp);
  
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
  ierr = VecScale(v_analytic,2.);
  CHKERRQ(ierr);
  return 0;
};