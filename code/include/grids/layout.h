#pragma once
#include <experimental/mdspan>
#include <cassert>

namespace stdex = std::experimental;

namespace grid{

  struct PartitionedLayout1D {
    template <class Extents>
    struct mapping {

      // for simplicity
      static_assert(Extents::rank() == 2, "PartitionedLayout1D is hard-coded for 1D layout with Dofs");

      // for convenience
      using index_t = typename Extents::index_type;

      // constructor
      mapping(Extents const& exts, index_t offset, index_t nx) noexcept
        : _extents(exts),
          _g2l_offset(offset),
          _nx(nx)
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
      constexpr index_t
      nx() const noexcept {
        return _nx;
      }

      //------------------------------------------------------------
      // Required memberst.
      constexpr index_t
      operator()(index_t i, index_t comp) const noexcept {
        return _flatten(i, comp) + _g2l_offset;
      }

      constexpr index_t
      required_span_size() const noexcept {
        return _extents.extent(1)*_extents.extent(0);
      }

      static constexpr bool is_always_unique() noexcept { return true; }
      static constexpr bool is_always_strided() noexcept { return true; }
      static constexpr bool is_always_contiguous() noexcept { return true; }
      static constexpr bool is_unique() noexcept { return true; }
      static constexpr bool is_contiguous() noexcept { return true; }
      static constexpr bool is_strided() noexcept { return true; }

     private:

      // Flattens a 2D index (i,comp) to a 1D index.
      constexpr index_t
      _flatten(index_t i, index_t comp) const noexcept {
        return _extents.extent(1)*i+comp;
      }

      Extents _extents;
      index_t _g2l_offset;
      index_t _nx;      
    };
  };

  struct PartitionedLayout2D {
    template <class Extents>
    struct mapping {

      // for simplicity
      static_assert(Extents::rank() == 3, "PartitionedLayout2D is hard-coded for 2D layout with Dofs");

      // for convenience
      using index_t = typename Extents::index_type;

      // constructor
      mapping(Extents const& exts, index_t offset, index_t nx, index_t ny) noexcept
        : _extents(exts),
          _g2l_offset(offset),
          _nx(nx),
          _ny(ny)
      {
        assert(exts.extent(0) > 0);
        assert(exts.extent(1) > 0);
        assert(exts.extent(2) > 0);
      }

      mapping() noexcept = default;
      mapping(mapping const&) noexcept = default;
      mapping(mapping&&) noexcept = default;
      mapping& operator=(mapping const&) noexcept = default;
      mapping& operator=(mapping&&) noexcept = default;
      ~mapping() noexcept = default;

      //------------------------------------------------------------
      // Helper members (not part of the layout concept)
      constexpr index_t
      nx() const noexcept {
        return _nx;
      }

      constexpr index_t
      ny() const noexcept {
        return _nx;
      }

      // Returns the offset going from global to local indexing
      constexpr index_t
      global_to_local_offset() const noexcept {
        return _g2l_offset;
      }

      // Flattens a 3D index (j,i,comp) to a 1D index.
      constexpr index_t
      flatten(index_t j, index_t i, index_t comp) const noexcept {
        return _extents.extent(2)*(i + _extents.extent(0)*j) + comp;
      }

      //------------------------------------------------------------
      // Required members.
      constexpr index_t
      operator()(index_t j, index_t i, index_t comp) const noexcept {
        return flatten(j,i,comp) + global_to_local_offset();
      }

      constexpr index_t
      required_span_size() const noexcept {
        return _extents.extent(0)*_extents.extent(1)*_extents.extent(2);
      }

      static constexpr bool is_always_unique() noexcept { return true; }
      static constexpr bool is_always_strided() noexcept { return true; }
      static constexpr bool is_always_contiguous() noexcept { return true; }
      static constexpr bool is_unique() noexcept { return true; }
      static constexpr bool is_contiguous() noexcept { return true; }
      static constexpr bool is_strided() noexcept { return true; }

     private:

      Extents _extents;
      index_t _g2l_offset;
      index_t _nx,_ny;
    };
  };

  //Alias definitions
  using extents_1d = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  using partitioned_layout_1d = typename PartitionedLayout1D::template mapping<extents_1d>;

  using extents_2d = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent, stdex::dynamic_extent>;
  using partitioned_layout_2d = typename PartitionedLayout2D::template mapping<extents_2d>;
}
