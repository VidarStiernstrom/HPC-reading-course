// #if !defined APPCTX_INCLUDED
// 	#define APPCTX_INCLUDED
#pragma once
#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"
#include "sbpops/ICF_central.h"
#include <array>

#define PROBLEM_TYPE_1D_O2

#if defined PROBLEM_TYPE_1D_O2
	struct GridCtx{
		std::array<PetscInt,1> N, n, i_start, i_end;
		std::array<PetscScalar,1> hi, h, xl, xr;
		PetscInt dofs;
		PetscScalar sw;
		std::function<double(int)> a;
		DM da_xt, da_x;
		KSP ksp, ksp_smo;
		Mat D, D_presmo;
		const sbp::D1_central<sbp::Stencils_2nd,3,1,2> D1;
		const sbp::Int_central<sbp::Interp_2nd,3,1,2,2,1,1,1> ICF;
		const sbp::H_central<sbp::Quadrature_2nd,1> H;
		const sbp::HI_central<sbp::InverseQuadrature_2nd,1> HI;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
	};
#elif defined PROBLEM_TYPE_1D_O4
	struct GridCtx{
		std::array<PetscInt,1> N, n, i_start, i_end;
		std::array<PetscScalar,1> hi, h, xl, xr;
		PetscInt dofs;
		PetscScalar sw;
		std::function<double(int)> a;
		DM da_xt, da_x;
		KSP ksp, ksp_smo;
		Vec b, v_curr;
		Mat D;
		const sbp::D1_central<sbp::Stencils_4th,5,4,6> D1;
		const sbp::Int_central<sbp::Interp_2nd,3,1,2,2,1,1,1> ICF;
		const sbp::H_central<sbp::Quadrature_4th,4> H;
		const sbp::HI_central<sbp::InverseQuadrature_2nd,1> HI;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
	};
#elif defined PROBLEM_TYPE_1D_O6
	struct GridCtx{
		std::array<PetscInt,1> N, n, i_start, i_end;
		std::array<PetscScalar,1> hi, h, xl, xr;
		PetscInt dofs;
		PetscScalar sw;
		std::function<double(int)> a;
		DM da_xt, da_x;
		KSP ksp, ksp_smo;
		Vec b, v_curr;
		Mat D;
		const sbp::D1_central<sbp::Stencils_6th,7,6,9> D1;
		const sbp::Int_central<sbp::Interp_2nd,3,1,2,2,1,1,1> ICF;
		const sbp::H_central<sbp::Quadrature_6th,6> H;
		const sbp::HI_central<sbp::InverseQuadrature_2nd,1> HI;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
	};
#endif	

struct TimeCtx {
  PetscInt N, tblocks;
  PetscScalar Tend, Tpb;
  PetscScalar er[4], HI_el[4], D[4][4];
};

struct MatCtx {
  GridCtx gridctx;
  TimeCtx timectx;
};