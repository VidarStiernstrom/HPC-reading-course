#pragma once 
#include <experimental/mdspan>
#include "grids/layout.h"

namespace stdex = std::experimental;
namespace grid
{
    template <typename T>
    using grid_function_1d = stdex::basic_mdspan<T, extents_1d, PartitionedLayout1D>;

    template <typename T>
    using grid_function_2d = stdex::basic_mdspan<T, extents_2d, PartitionedLayout2D>;
    
    template <typename T>
    using grid_function_3d = stdex::mdspan<T, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>;
}