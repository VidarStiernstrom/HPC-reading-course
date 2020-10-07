#pragma once
#include </Users/vidar/dev/git/mdspan/include/experimental/mdspan>

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
      mapping(Extents const& exts, index_type s_sz, index_type com_sz, index_type id) noexcept
        : extents(exts),
          stencil_size(s_sz),
          communicator_size(com_sz),
          process_id(id)
      {
        assert(exts.extent(0) > 0);
        assert(exts.extent(1) > 0);
        assert(s_sz > 0);
        assert(com_sz > 0);
        assert(id >= 0);
      }

      mapping() noexcept = default;
      mapping(mapping const&) noexcept = default;
      mapping(mapping&&) noexcept = default;
      mapping& operator=(mapping const&) noexcept = default;
      mapping& operator=(mapping&&) noexcept = default;
      ~mapping() noexcept = default;

      //------------------------------------------------------------
      // Helper members (not part of the layout concept)

      // All processes should offset by the ghost region stencil size, except from
      // the first process.
      // Note! The ghost region stencil size is not equivalent the stencil size of the
      // finite difference stencils (although they are related).
      constexpr index_type
      stencil_offset() const noexcept {
        return -stencil_size * index_type(process_id != 0);
      }

      // The offset by the number of grid points in preceeding processes.
      // TODO: Assumes that the first n processes store the additional grid points,
      // in case the number of grid points are not even divided among the processes.
      // I'm fairly sure that this is the partitioning that PETSc also uses, but an
      // alternativ is to compute the offset externally by e.g the number of points 
      // owned by each process prior to process_id.
      constexpr index_type
      proc_offset() const noexcept {
        int id = 0;
        int offset = 0;
        for (id = 0; id < process_id; id++)
        {
          offset += (extents.extent(0) / communicator_size) + index_type(id < (extents.extent(0) % communicator_size));
        }
        return offset;
      }

       // Computes the offset going from global to local indexing
      constexpr index_type
      global_to_local_offset() const noexcept {
        return -(extents.extent(1)*(proc_offset()+stencil_offset()));
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
      index_type stencil_size;
      index_type communicator_size;
      index_type process_id;

    };
  };

  //Alias definitions
  using extents_1d = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  using partitioned_layout_1d = typename PartitionedLayout1D::template mapping<extents_1d>;

}
