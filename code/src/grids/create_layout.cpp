#include "grids/create_layout.h"
#include <cassert>


namespace grid
{    
    partitioned_layout_1d create_layout_1d(const DM& da)
    {   
        PetscInt dim, n, dofs, sw, processor_offset, stencil_offset, g2l_offset;
        DMDAGetInfo(da,&dim,NULL,NULL,NULL,&n,NULL,NULL,&dofs,&sw,NULL,NULL,NULL,NULL);
        assert(dim==1);

        // Get the offsets this process has.
        DMDAGetCorners(da,&processor_offset,NULL,NULL,NULL,NULL,NULL);
        stencil_offset = -sw;
        if (processor_offset == 0) stencil_offset = 0; // Left boundary, no ghost points.
        
        // Compute global to local offset. 
        g2l_offset = -(dofs*(processor_offset+stencil_offset));
        return grid::partitioned_layout_1d(grid::extents_1d(n,dofs),g2l_offset);
    }

}
