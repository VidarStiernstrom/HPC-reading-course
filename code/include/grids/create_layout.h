#pragma once
#include <petscdmda.h>
#include "grids/layout.h"


namespace grid
{
    partitioned_layout_1d create_layout_1d(const DM& da);

}
