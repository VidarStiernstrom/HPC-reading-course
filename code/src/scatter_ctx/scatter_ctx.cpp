#include <petsc.h>
#include <petsc/private/dmdaimpl.h> 
#include "scatter_ctx/scatter_ctx.h"

/**
* Build local to local scatter context containing only ghost point communications
* Inputs: da        - DMDA object
*         ltol      - pointer to local to local scatter context
**/
PetscErrorCode build_ltol_1D(DM da, VecScatter *ltol)
{
  PetscInt    stencil_radius, i_xstart, i_xend, ig_xstart, ig_xend, n, i, j, ln, no_com_vals, count, N, dof;
  IS          ix, iy;
  Vec         vglobal, vlocal;
  VecScatter  gtol;

  DMDAGetStencilWidth(da, &stencil_radius);
  DMDAGetCorners(da,&i_xstart,NULL,NULL,&n,NULL,NULL);
  DMDAGetGhostCorners(da,&ig_xstart,NULL,NULL,&ln,NULL,NULL);
  DMDAGetInfo(da, NULL, &N, NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);

  i_xend = i_xstart + n;
  ig_xend = ig_xstart + ln;

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute how many elements to receive
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  no_com_vals = 0;
  if (i_xstart != 0)  // NOT LEFT BOUNDARY, RECEIVE LEFT
  {
    no_com_vals += 1;
  }
  if (i_xend != N) // NOT RIGHT BOUNDARY, RECEIVE RIGHT
  {
    no_com_vals += 1;
  }
  no_com_vals *= stencil_radius*dof;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define communication pattern, from global index ixx[i] to local index iyy[i]
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt ixx[no_com_vals], iyy[no_com_vals];
  count = 0;

  for (i = ig_xstart; i < i_xstart; i++) { // LEFT
    for (j = 0; j < dof; j++) {
      ixx[count] = dof*i + j;
      iyy[count] = dof*(i - ig_xstart) + j;
      count++;
    }
  }

  for (i = i_xend; i < ig_xend; i++) { // RIGHT
    for (j = 0; j < dof; j++) {
      ixx[count] = dof*i + j;
      iyy[count] = dof*(i - ig_xstart) + j;
      count++;
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Build global to local scatter context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,ixx,PETSC_COPY_VALUES,&ix);  
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,iyy,PETSC_COPY_VALUES,&iy);  

  DMGetGlobalVector(da, &vglobal);
  DMGetLocalVector(da, &vlocal);

  VecScatterCreate(vglobal,ix,vlocal,iy, &gtol);  
  VecScatterSetUp(gtol);

  DMRestoreGlobalVector(da, &vglobal);
  DMRestoreLocalVector(da, &vlocal);

  ISDestroy(&ix);
  ISDestroy(&iy);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Map 1D global to local scatter context to local to local (petsc source code)
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecScatterCopy(gtol,ltol);
  VecScatterDestroy(&gtol);

  PetscInt *idx,left;
  DM_DA *dd = (DM_DA*) da->data; 
  left = dd->xs - dd->Xs;
  PetscMalloc1(dd->xe-dd->xs,&idx);
  for (j=0; j<dd->xe-dd->xs; j++) 
  {
    idx[j] = left + j;
  }
  VecScatterRemap(*ltol,idx,NULL);

  return 0;
}

/**
* Build local to local scatter context containing only ghost point communications
* Inputs: da        - DMDA object
*         ltol      - pointer to local to local scatter context
**/
PetscErrorCode build_ltol_2D(DM da, VecScatter *ltol)
{
  AO          ao;
  PetscInt    stencil_radius, i_xstart, i_xend, i_ystart, i_yend, ig_xstart, ig_xend, ig_ystart, ig_yend, nx, ny, i, j, l, lnx, lny, no_com_vals, count, Nx, Ny, dof;
  IS          ix, iy;
  Vec         vglobal, vlocal;
  VecScatter  gtol;

  DMDAGetStencilWidth(da, &stencil_radius);
  DMDAGetCorners(da,&i_xstart,&i_ystart,NULL,&nx,&ny,NULL);
  DMDAGetGhostCorners(da,&ig_xstart,&ig_ystart,NULL,&lnx,&lny,NULL);
  DMDAGetInfo(da, NULL, &Nx, &Ny,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);

  i_xend = i_xstart + nx;
  i_yend = i_ystart + ny;
  ig_xend = ig_xstart + lnx;
  ig_yend = ig_ystart + lny;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute how many elements to receive
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  no_com_vals = 0;
  if (i_ystart != 0)  // NOT BOTTOM, RECEIVE BELOW
  {
    no_com_vals += nx;
  }
  if (i_yend != Ny) // NOT TOP, RECEIVE ABOVE
  {
    no_com_vals += nx;
  }
  if (i_xstart != 0)  // NOT LEFT BOUNDARY, RECEIVE LEFT
  {
    no_com_vals += ny;
  }
  if (i_xend != Nx) // NOT RIGHT BOUNDARY, RECEIVE RIGHT
  {
    no_com_vals += ny;
  }
  no_com_vals *= stencil_radius*dof;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define communication pattern, from global index ixx[i] to local index iyy[i]
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt ixx[no_com_vals], iyy[no_com_vals];
  count = 0;
  for (i = i_xstart; i < i_xend; i++) { // UP
    for (j = i_yend; j < ig_yend; j++) {
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  for (i = i_xstart; i < i_xend; i++) { // DOWN
    for (j = ig_ystart; j < i_ystart; j++) { 
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  for (i = ig_xstart; i < i_xstart; i++) { // LEFT
    for (j = i_ystart; j < i_yend; j++) {
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  for (i = i_xend; i < ig_xend; i++) { // RIGHT
    for (j = i_ystart; j < i_yend; j++) {
      for (l = 0; l < dof; l++) {
        ixx[count] = (i + Nx*j)*dof + l;
        iyy[count] = ((i - ig_xstart) + lnx*(j - ig_ystart))*dof + l;
        count++;
      }
    }
  }

  // Map global indices from natural ordering to petsc application ordering
  DMDAGetAO(da,&ao);
  AOApplicationToPetsc(ao,no_com_vals,ixx);
  AODestroy(&ao);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Build global to local scatter context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,ixx,PETSC_COPY_VALUES,&ix);  
  ISCreateGeneral(PETSC_COMM_SELF,no_com_vals,iyy,PETSC_COPY_VALUES,&iy);  

  DMGetGlobalVector(da, &vglobal);
  DMGetLocalVector(da, &vlocal);

  VecScatterCreate(vglobal,ix,vlocal,iy, &gtol);  
  VecScatterSetUp(gtol);

  DMRestoreGlobalVector(da, &vglobal);
  DMRestoreLocalVector(da, &vlocal);

  ISDestroy(&ix);
  ISDestroy(&iy);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Map 2D global to local scatter context to local to local (petsc source code)
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecScatterCopy(gtol,ltol);
  VecScatterDestroy(&gtol);

  PetscInt *idx,left,up,down;
  DM_DA *dd = (DM_DA*) da->data;
  left  = dd->xs - dd->Xs; down  = dd->ys - dd->Ys; up = down + dd->ye-dd->ys;
  PetscMalloc1((dd->xe-dd->xs)*(up - down),&idx);
  count = 0;
  for (i=down; i<up; i++) {
    for (j=0; j<dd->xe-dd->xs; j++) {
      idx[count++] = left + i*(dd->Xe-dd->Xs) + j;
    }
  }
  VecScatterRemap(*ltol,idx,NULL);

  return 0;
}