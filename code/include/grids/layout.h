#pragma once
#include </Users/vidar/dev/git/mdspan/include/experimental/mdspan>
#include <cassert>

namespace stdex = std::experimental;

namespace grid{

  struct PartitionedLayout1D {
    template <class Extents>
    struct mapping {

      // for simplicity
      static_assert(Extents::rank() == 2, "PartitionedLayout1D is hard-coded for 1D layout with Dofs");

      // for convenience
      using index_type = typename Extents::index_type;

      // constructor
      mapping(Extents const& exts, index_type offset) noexcept
        : extents(exts),
          g2l_offset(offset)
      {
        assert(exts.extent(0) > 0);
        assert(exts.extent(1) > 0);
      }

      mapping() noexcept = default;
      mapping(mapping const&) noexcept = default;
      mapping(mapping&&) noexcept = default;
      mapping& operator=(mapping const&) noexcept = default;
      mapping& operator=(mapping&&) noexcept = default;
      ~mapping() noexcept = default;

      //------------------------------------------------------------
      // Helper members (not part of the layout concept)

       // Computes the offset going from global to local indexing
      constexpr index_type
      global_to_local_offset() const noexcept {
        return g2l_offset;
      }

      // Flattens a 2D index (i,comp) to a 1D index.
      constexpr index_type
      flatten(index_type i, index_type comp) const noexcept {
        return extents.extent(1)*i+comp;
      }

      //------------------------------------------------------------
      // Required memberst.
      constexpr index_type
      operator()(index_type i, index_type comp) const noexcept {
        return flatten(i, comp) + global_to_local_offset();
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
      index_type g2l_offset;
    };
  };

  //Alias definitions
  using extents_1d = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  using partitioned_layout_1d = typename PartitionedLayout1D::template mapping<extents_1d>;

}
