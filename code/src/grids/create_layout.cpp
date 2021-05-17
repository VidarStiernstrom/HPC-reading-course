#include "grids/create_layout.h"
#include <cassert>


namespace grid
{    
    partitioned_layout_1d create_layout_1d(const DM& da)
    {   
        PetscInt dim, nx, nlocal, dofs, sw, processor_offset, stencil_offset, g2l_offset;
        DMDAGetInfo(da,&dim,&nx,NULL,NULL,NULL,NULL,NULL,&dofs,&sw,NULL,NULL,NULL,NULL);
        DMDAGetGhostCorners(da,NULL,NULL,NULL,&nlocal,NULL,NULL);
        assert(dim==1);

        // Get the offsets this process has.
        DMDAGetCorners(da,&processor_offset,NULL,NULL,NULL,NULL,NULL);
        stencil_offset = (processor_offset == 0) ? 0 : -sw; // If on left boundary, no offset, otherwise offset by DMDA stencil size (halo size).
        
        // Compute global to local offset. 
        g2l_offset = -(dofs*(processor_offset+stencil_offset));
        return grid::partitioned_layout_1d(grid::extents_1d(nlocal,dofs),g2l_offset,nx);
    }

    partitioned_layout_2d create_layout_2d(const DM& da)
    {   
        PetscInt dim, nx, ny, nxg, nyg, dofs, sw, processor_x_offset, processor_y_offset, stencil_x_offset, stencil_y_offset, g2l_offset, g2l_x_offset, g2l_y_offset;
        DMDAGetInfo(da,&dim,&nx,&ny,NULL,NULL,NULL,NULL,&dofs,&sw,NULL,NULL,NULL,NULL);
        DMDAGetGhostCorners(da,NULL,NULL,NULL,&nxg,&nyg,NULL);
        assert(dim==2);

        // Get the offsets this process has.
        DMDAGetCorners(da,&processor_x_offset,&processor_y_offset,NULL,NULL,NULL,NULL);
        
        stencil_x_offset = (processor_x_offset == 0) ? 0 : -sw; // If on west boundary, no offset, otherwise offset by DMDA stencil width (halo size).
        stencil_y_offset = (processor_y_offset == 0) ? 0 : -sw; // If on south boundary, no offset, otherwise offset by DMDA stencil width (halo size).
        
        // Compute global to local offset. 
        g2l_x_offset = -(processor_x_offset+stencil_x_offset);
        g2l_y_offset = -(processor_y_offset+stencil_y_offset);
        g2l_offset = dofs*(g2l_x_offset + nxg*g2l_y_offset);
        
        return grid::partitioned_layout_2d(grid::extents_2d(nxg,nyg,dofs),g2l_offset,nx,ny);
    }

}
