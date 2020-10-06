#include </Users/vidar/dev/git/mdspan/include/experimental/mdspan>
#pragma once 

namespace stdex = std::experimental;
// Simple tiled layout.
// Hard-coded for 2D, column-major across tiles
// and row-major within each tile
struct PartitionedLayout1D {
  template <class Extents>
  struct mapping {

    // for simplicity
    static_assert(Extents::rank() == 2, "PartitionedLayout1D is hard-coded for 1D layout with Dofs");

    // for convenience
    using index_type = typename Extents::index_type;

    // constructor
    mapping(Extents const& exts, index_type stencil_size, index_type n_procs, index_type proc_id) noexcept
      : extents(exts),
        s_sz(stencil_size),
        n(n_procs),
        id(proc_id)
    {
      assert(exts.extent(0) > 0);
      assert(exts.extent(1) > 0);
      assert(stencil_size > 0);
      assert(n_procs > 0);
      assert(proc_id >= 0);
    }

    mapping() noexcept = default;
    mapping(mapping const&) noexcept = default;
    mapping(mapping&&) noexcept = default;
    mapping& operator=(mapping const&) noexcept = default;
    mapping& operator=(mapping&&) noexcept = default;
    ~mapping() noexcept = default;

    //------------------------------------------------------------
    // Helper members (not part of the layout concept)

    constexpr index_type
    proc_offset() const noexcept {
      return id*(extents.extent(0) / n) + index_type((extents.extent(0) % n) != 0);
    }

    //------------------------------------------------------------
    // Required members

    constexpr index_type
    operator()(index_type i, index_type comp) const noexcept {
      return extents.extent(1)*(i+s_sz - proc_offset())+comp;
    }

    constexpr index_type
    required_span_size() const noexcept {
      return extents.extent(1)*extents.extent(0);
    }

    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_always_strided() noexcept { return true; }
    static constexpr bool is_always_contiguous() noexcept { return true; }
    static constexpr bool is_unique() noexcept { return true; }
    static constexpr bool is_contiguous() noexcept { return true; }
    static constexpr bool is_strided() noexcept { return true; }

   private:

    Extents extents;
    index_type s_sz;
    index_type n;
    index_type id;

  };
};

using extents_type_1d = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
using partitioned_layout_type_1d = typename PartitionedLayout1D::template mapping<extents_type_1d>;

namespace sbp{
    template <typename T>
    using grid_function_1d = stdex::basic_mdspan<T, extents_type_1d, PartitionedLayout1D>;

    template <typename T>
    using grid_function_2d = stdex::mdspan<T, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>;
    template <typename T>
    using grid_function_3d = stdex::mdspan<T, stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>;
}