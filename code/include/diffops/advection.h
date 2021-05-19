#pragma once

#include <petscsystypes.h>
#include <array>
#include "grids/grid_function.h"
#include "partitioned_rhs/rhs.h"
#include "partitioned_rhs/boundary_conditions.h"


namespace sbp{

//=============================================================================
// 1D functions
//=============================================================================
template <class SbpDerivative, typename VelocityFunction>
inline void advection_l(grid::grid_function_1d<PetscScalar> dst,
                        const grid::grid_function_1d<PetscScalar> src, 
                        const PetscInt cls_sz,
                        const SbpDerivative& D1, 
                        const PetscScalar hi,
                        VelocityFunction&& a)
{
 
  for (PetscInt i = 0; i < cls_sz; i++) { 
    dst(i,0) = -std::forward<VelocityFunction>(a)(i)*D1.apply_left(src,hi,i,0);
  }   
}

template <class SbpDerivative, typename VelocityFunction>
inline void advection_r(grid::grid_function_1d<PetscScalar> dst,
                        const grid::grid_function_1d<PetscScalar> src, 
                        const PetscInt cls_sz,
                        const SbpDerivative& D1, 
                        const PetscScalar hi,
                        VelocityFunction&& a)
{
  const PetscInt nx = src.mapping().nx();
  for (PetscInt i = nx-cls_sz; i < nx; i++) {
      dst(i,0) = -std::forward<VelocityFunction>(a)(i)*D1.apply_right(src,hi,i,0);
  }
}

template <class SbpDerivative, typename VelocityFunction>
inline void advection_i(grid::grid_function_1d<PetscScalar> dst,
                                        const grid::grid_function_1d<PetscScalar> src, 
                                        const std::array<PetscInt,2> ind_i,
                                        const SbpDerivative& D1, 
                                        const PetscScalar hi,
                                        VelocityFunction&& a)
{
  for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) {
    dst(i,0) = -std::forward<VelocityFunction>(a)(i)*D1.apply_interior(src,hi,i,0);
  }
}

template <class SbpDerivative, typename VelocityFunction>
inline void advection_local(grid::grid_function_1d<PetscScalar> dst,
                            const grid::grid_function_1d<PetscScalar> src,
                            const std::array<PetscInt,2> ind_i, 
                            const PetscInt halo_sz, 
                            const SbpDerivative& D1, 
                            const PetscScalar hi,
                            VelocityFunction&& a)
{
  const PetscInt cls_sz = D1.closure_size();
  rhs_local(advection_l<decltype(D1),decltype(a)>,
            advection_i<decltype(D1),decltype(a)>,
            advection_r<decltype(D1),decltype(a)>,
            dst, src, ind_i, cls_sz, halo_sz, D1, hi, a);
};

template <class SbpDerivative, typename VelocityFunction>
inline void advection_overlap(grid::grid_function_1d<PetscScalar> dst,
                               const grid::grid_function_1d<PetscScalar> src,
                               const std::array<PetscInt,2>& ind_i,
                               const PetscInt halo_sz,
                               const SbpDerivative& D1,
                               const PetscScalar hi,
                               VelocityFunction&& a)
{
  rhs_overlap(advection_i<decltype(D1),decltype(a)>, dst, src, ind_i, halo_sz, D1, hi, a);
};
  

template <class SbpDerivative, typename VelocityFunction>
inline void advection_serial(grid::grid_function_1d<PetscScalar> dst,
                              const grid::grid_function_1d<PetscScalar> src,
                              const SbpDerivative& D1,
                              const PetscScalar hi,
                              VelocityFunction&& a)
{
  const PetscInt cls_sz = D1.closure_size();
  rhs_serial(advection_l<decltype(D1),decltype(a)>, advection_i<decltype(D1),decltype(a)>, advection_r<decltype(D1),decltype(a)>,
            dst, src, cls_sz, D1, hi, a);
};

template <class SbpInvQuad, typename VelocityFunction>
inline void SAT_bc_l(grid::grid_function_1d<PetscScalar> dst,
                     const grid::grid_function_1d<PetscScalar> src, 
                     const SbpInvQuad& HI, 
                     const PetscScalar hi,
                     VelocityFunction&& a)
{
  const PetscInt i = 0;
  const PetscScalar a_l = std::forward<VelocityFunction>(a)(i);
  const PetscScalar tau = -0.5*(a_l+std::abs(a_l));
  dst(i,0) += 0.5*tau*HI.apply_left(src, hi, i, 0);
}

template <class SbpInvQuad, typename VelocityFunction>
inline void SAT_bc_r(grid::grid_function_1d<PetscScalar> dst,
                     const grid::grid_function_1d<PetscScalar> src, 
                     const SbpInvQuad& HI, 
                     const PetscScalar hi,
                     VelocityFunction&& a)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt i = nx-1;
  const PetscScalar a_r = std::forward<VelocityFunction>(a)(i);
  const PetscScalar tau = 0.5*(a_r-std::abs(a_r));
  dst(i,0) += 0.5*tau*HI.apply_left(src, hi, i, 0);
}

template <class SbpInvQuad, typename VelocityFunction>
inline void advection_bc(grid::grid_function_1d<PetscScalar> dst,
                          const grid::grid_function_1d<PetscScalar> src,
                          const std::array<PetscInt,2>& ind_i,
                          const SbpInvQuad& HI, 
                          const PetscScalar hi,
                          VelocityFunction&& a)
{
  bc(SAT_bc_l<decltype(HI),decltype(a)>,SAT_bc_r<decltype(HI),decltype(a)>,dst,src,ind_i,HI,hi,a);
};

template <class SbpInvQuad, typename VelocityFunction>
inline void advection_bc_serial(grid::grid_function_1d<PetscScalar> dst,
                                const grid::grid_function_1d<PetscScalar> src,
                                const SbpInvQuad& HI, 
                                const PetscScalar hi,
                                VelocityFunction&& a)
{
  bc_serial(SAT_bc_l<decltype(HI),decltype(a)>,SAT_bc_r<decltype(HI),decltype(a)>,dst,src,HI,hi,a);
};

//=============================================================================
// 2D functions
//=============================================================================

 /**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * il *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_ll(grid::grid_function_2d<PetscScalar> dst,
                  const grid::grid_function_2d<PetscScalar> src,
                  const PetscInt cl_sz,
                  const SbpDerivative& D1,
                  const std::array<PetscScalar,2>& hi,
                  VelocityFunction&& a_x,
                  VelocityFunction&& a_y)
{
  for (PetscInt j = 0; j < cl_sz; j++) { 
    for (PetscInt i = 0; i < cl_sz; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_left(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_left(src,hi[1],i,j,0));
    }
  }
}


 /**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * il *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_il(grid::grid_function_2d<PetscScalar> dst,
                  const grid::grid_function_2d<PetscScalar> src,
                  const std::array<PetscInt,2> ind_i,
                  const PetscInt cl_sz,
                  const SbpDerivative& D1,
                  const std::array<PetscScalar,2>& hi,
                  VelocityFunction&& a_x,
                  VelocityFunction&& a_y)
{
  for (PetscInt j = 0; j < cl_sz; j++) { 
    for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) {  
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_interior(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_left(src,hi[1],i,j,0));
    }
  }
}

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * rl *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_rl(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
  const PetscInt nx = src.mapping().nx();
  for (PetscInt j = 0; j < cl_sz; j++) { 
    for (PetscInt i = nx-cl_sz; i < nx; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_right(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_left(src,hi[1],i,j,0));
    }
  }
}

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   * li *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_li(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const std::array<PetscInt,2> ind_j,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
 
 for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    for (PetscInt i = 0; i < cl_sz; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_left(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_interior(src,hi[1],i,j,0));
    }
  }
}

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    * ii *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_ii(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const std::array<PetscInt,2> ind_i,
                    const std::array<PetscInt,2> ind_j,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{

  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_interior(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_interior(src,hi[1],i,j,0));
    }
  }
}

/**
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    * ri *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_ri(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const std::array<PetscInt,2> ind_j,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
  const PetscInt nx = src.mapping().nx();
  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    for (PetscInt i = nx-cl_sz; i < nx; i++) {
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_right(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_interior(src,hi[1],i,j,0));
    }
  }
}


/**
  *   ****************
  *   * lr *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_lr(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
  const PetscInt ny = src.mapping().ny();  
  for (PetscInt j = ny-cl_sz; j < ny; j++) { 
    for (PetscInt i = 0; i < cl_sz; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_left(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_right(src,hi[1],i,j,0));
    }
  }
}

/**
  *   ****************
  *   *    * ir  *   *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_ir(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const std::array<PetscInt,2> ind_i,
                    const PetscInt cl_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
  const PetscInt ny = src.mapping().ny();
  for (PetscInt j = ny-cl_sz; j < ny; j++) { 
    for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_interior(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_right(src,hi[1],i,j,0));
    }
  }
}

/**
  *   ****************
  *   *    *    * rr *
  *   ****************
  *   *    *    *    *
  *   ****************
  *   *    *    *    *
  *   ****************
  **/
template <class SbpDerivative, typename VelocityFunction>
void advection_rr(grid::grid_function_2d<PetscScalar> dst,
                  const grid::grid_function_2d<PetscScalar> src,
                  const PetscInt cl_sz,
                  const SbpDerivative& D1,
                  const std::array<PetscScalar,2>& hi,
                  VelocityFunction&& a_x,
                  VelocityFunction&& a_y)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt ny = src.mapping().ny();
  for (PetscInt j = ny-cl_sz; j < ny; j++) { 
    for (PetscInt i = nx-cl_sz; i < nx; i++) { 
      dst(j,i,0) = -(std::forward<VelocityFunction>(a_x)(i,j)*D1.apply_x_right(src,hi[0],i,j,0) +
                     std::forward<VelocityFunction>(a_y)(i,j)*D1.apply_y_right(src,hi[1],i,j,0));
    }
  }
}

template <class SbpDerivative, typename VelocityFunction>
void advection_local(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const std::array<PetscInt,2>& ind_i,
                    const std::array<PetscInt,2>& ind_j,
                    const PetscInt halo_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
  const PetscInt cl_sz = D1.closure_size();
  rhs_local(advection_ll<decltype(D1),decltype(a_x)>,
            advection_li<decltype(D1),decltype(a_x)>,
            advection_lr<decltype(D1),decltype(a_x)>,
            advection_il<decltype(D1),decltype(a_x)>,
            advection_ii<decltype(D1),decltype(a_x)>,
            advection_ir<decltype(D1),decltype(a_x)>,
            advection_rl<decltype(D1),decltype(a_x)>,
            advection_ri<decltype(D1),decltype(a_x)>,
            advection_rr<decltype(D1),decltype(a_x)>,
            dst,src,ind_i,ind_j,cl_sz,halo_sz,D1,hi,a_x,a_y);
}

template <class SbpDerivative, typename VelocityFunction>
void advection_overlap(grid::grid_function_2d<PetscScalar> dst,
                    const grid::grid_function_2d<PetscScalar> src,
                    const std::array<PetscInt,2>& ind_i,
                    const std::array<PetscInt,2>& ind_j,
                    const PetscInt halo_sz,
                    const SbpDerivative& D1,
                    const std::array<PetscScalar,2>& hi,
                    VelocityFunction&& a_x,
                    VelocityFunction&& a_y)
{
  const PetscInt cl_sz = D1.closure_size();
  rhs_overlap(advection_li<decltype(D1),decltype(a_x)>,
              advection_il<decltype(D1),decltype(a_x)>,
              advection_ii<decltype(D1),decltype(a_x)>,
              advection_ir<decltype(D1),decltype(a_x)>,
              advection_ri<decltype(D1),decltype(a_x)>,
              dst,src,ind_i,ind_j,cl_sz,halo_sz,D1,hi,a_x,a_y);
}

template <class SbpDerivative, typename VelocityFunction>
void advection_serial(grid::grid_function_2d<PetscScalar> dst,
                     const grid::grid_function_2d<PetscScalar> src,
                     const SbpDerivative& D1,
                     const std::array<PetscScalar,2>& hi,
                     VelocityFunction&& a_x,
                     VelocityFunction&& a_y)
{
  const PetscInt cl_sz = D1.closure_size();
  rhs_serial(advection_ll<decltype(D1),decltype(a_x)>,
             advection_li<decltype(D1),decltype(a_x)>,
             advection_lr<decltype(D1),decltype(a_x)>,
             advection_il<decltype(D1),decltype(a_x)>,
             advection_ii<decltype(D1),decltype(a_x)>,
             advection_ir<decltype(D1),decltype(a_x)>,
             advection_rl<decltype(D1),decltype(a_x)>,
             advection_ri<decltype(D1),decltype(a_x)>,
             advection_rr<decltype(D1),decltype(a_x)>,
             dst,src,cl_sz,D1,hi,a_x,a_y);
}

template <class SbpInvQuad, typename VelocityFunction>
void SAT_bc_west(grid::grid_function_2d<PetscScalar> dst,
                           const grid::grid_function_2d<PetscScalar> src,
                           const std::array<PetscInt,2>& ind_j,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi,
                           VelocityFunction&& a_x,
                           VelocityFunction&& a_y)
{
  const PetscInt i = 0;
  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) {
    PetscScalar a_w = std::forward<VelocityFunction>(a_x)(i,j);
    PetscScalar tau_w = -0.5*(a_w+std::abs(a_w));
    dst(j, i, 0) += 0.5*tau_w*HI.apply_x_left(src, hi[0], i, j, 0);
  }
};

template <class SbpInvQuad, typename VelocityFunction>
void SAT_bc_south(grid::grid_function_2d<PetscScalar> dst,
                           const grid::grid_function_2d<PetscScalar> src,
                           const std::array<PetscInt,2>& ind_i,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi,
                           VelocityFunction&& a_x,
                           VelocityFunction&& a_y)
{
  const PetscInt j = 0;
  for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) { 
    PetscScalar a_s = std::forward<VelocityFunction>(a_y)(i,j);
    PetscScalar tau_s = -0.5*(a_s+std::abs(a_s));
    dst(j, i, 0) += 0.5*tau_s*HI.apply_x_left(src, hi[0], i, j, 0);
  }
};

template <class SbpInvQuad, typename VelocityFunction>
void SAT_bc_east(grid::grid_function_2d<PetscScalar> dst,
                           const grid::grid_function_2d<PetscScalar> src,
                           const std::array<PetscInt,2>& ind_j,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi,
                           VelocityFunction&& a_x,
                           VelocityFunction&& a_y)
{
  const PetscInt nx = src.mapping().nx();
  const PetscInt i = nx-1;
  for (PetscInt j = ind_j[0]; j < ind_j[1]; j++) { 
    PetscScalar a_e = std::forward<VelocityFunction>(a_x)(i,j);
    PetscScalar tau_e = 0.5*(a_e-std::abs(a_e));
    dst(j, i, 0) += 0.5*tau_e*HI.apply_x_right(src, hi[0], nx, i, j, 0);
  }
};

template <class SbpInvQuad, typename VelocityFunction>
void SAT_bc_north(grid::grid_function_2d<PetscScalar> dst,
                           const grid::grid_function_2d<PetscScalar> src,
                           const std::array<PetscInt,2>& ind_i,
                           const SbpInvQuad& HI,
                           const std::array<PetscScalar,2>& hi,
                           VelocityFunction&& a_x,
                           VelocityFunction&& a_y)
{
  const PetscInt ny = src.mapping().ny();
  const PetscInt j = ny-1;
  for (PetscInt i = ind_i[0]; i < ind_i[1]; i++) {
    PetscScalar a_n = std::forward<VelocityFunction>(a_y)(i,j);
    PetscScalar tau_n = 0.5*(a_n-std::abs(a_n));
    dst(j, i, 0) += 0.5*tau_n*HI.apply_y_right(src, hi[1], ny, i, j, 0);
  }
};

template <class SbpInvQuad, typename VelocityFunction>
void advection_bc_serial(grid::grid_function_2d<PetscScalar> dst,
                         const grid::grid_function_2d<PetscScalar> src,
                         const SbpInvQuad& HI,
                         const std::array<PetscScalar,2>& hi,
                         VelocityFunction&& a_x,
                         VelocityFunction&& a_y)
{
  bc_serial(SAT_bc_west<decltype(HI),decltype(a_x)>,
            SAT_bc_south<decltype(HI),decltype(a_x)>,
            SAT_bc_east<decltype(HI),decltype(a_y)>,
            SAT_bc_north<decltype(HI),decltype(a_y)>,dst,src,HI,hi,a_x,a_y);
};

template <class SbpInvQuad, typename VelocityFunction>
void advection_bc(grid::grid_function_2d<PetscScalar> dst,
                   const grid::grid_function_2d<PetscScalar> src,
                   const std::array<PetscInt,2>& ind_i,
                   const std::array<PetscInt,2>& ind_j,
                   const SbpInvQuad& HI,
                   const std::array<PetscScalar,2>& hi,
                   VelocityFunction&& a_x,
                   VelocityFunction&& a_y)
{
  bc(SAT_bc_west<decltype(HI),decltype(a_x)>,
     SAT_bc_south<decltype(HI),decltype(a_x)>,
     SAT_bc_east<decltype(HI),decltype(a_y)>,
     SAT_bc_north<decltype(HI),decltype(a_y)>,dst,src,ind_i,ind_j,HI,hi,a_x,a_y);
};

} //namespace sbp