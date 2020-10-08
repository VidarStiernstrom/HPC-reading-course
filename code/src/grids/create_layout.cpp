#include "grids/create_layout.h"
#include <cassert>


namespace grid
{    
    partitioned_layout_1d create_layout_1d(const DM& da, const PetscInt rank, const PetscInt size)
    {   
        PetscInt dim, N, n, dofs, s, id, processor_offset, stencil_offset, g2l_offset;
        DMBoundaryType b;
        DMDAGetInfo(da,&dim,&N,NULL,NULL,&n,NULL,NULL,&dofs,&s,&b,NULL,NULL,NULL);
        assert(dim==1);
        // Get the offset this process has, based on the number of
        // grid points stored by previous processes. 
        // Assumes that if N/size is not even, then the first k
        // processes store an additional grid point (this seems to be what PETSc does)
        // TODO: Alternative is to send grid point data from previous processes to current
        id = 0;
        processor_offset = 0;
        for (id = 0; id < rank; id++)
        {
        processor_offset += (N / size) + (id < (N % size));
        }
        // Get the ghost region stencil offset. If no ghost regions is used at the boundary
        // the first process should not be offset.
        stencil_offset = (b == DM_BOUNDARY_NONE)? -s*(rank!=0) : -s;
        
        // Compute global to local offset. 
        g2l_offset = -(dofs*(processor_offset+stencil_offset));
        return grid::partitioned_layout_1d(grid::extents_1d(n,dofs),g2l_offset);
    }

}
