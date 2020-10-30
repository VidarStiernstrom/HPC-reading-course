#if !defined APPCTX_INCLUDED
	#define APPCTX_INCLUDED

	#if defined PROBLEM_TYPE_1D_O2
		struct AppCtx{
		std::array<PetscInt,1> N, i_start, i_end;
		std::array<PetscScalar,1> hi, h;
		PetscInt dofs;
		PetscScalar xl, sw;
		std::function<double(int)> a;
		const sbp::D1_central<sbp::Stencils_2nd,3,1,2> D1;
		const sbp::H_central<sbp::Quadrature_2nd,1> H;
		const sbp::HI_central<sbp::InverseQuadrature_2nd,1> HI;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
		};
	#elif defined PROBLEM_TYPE_1D_O4
		struct AppCtx{
		std::array<PetscInt,1> N, i_start, i_end;
		std::array<PetscScalar,1> hi, h;
		PetscInt dofs;
		PetscScalar xl, sw;
		std::function<double(int)> a;
		const sbp::D1_central<sbp::Stencils_4th,5,4,6> D1;
		const sbp::H_central<sbp::Quadrature_4th,4> H;
		const sbp::HI_central<sbp::InverseQuadrature_4th,4> HI;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
		};
	#elif defined PROBLEM_TYPE_1D_O6
		struct AppCtx{
		std::array<PetscInt,1> N, i_start, i_end;
		std::array<PetscScalar,1> hi, h;
		PetscInt dofs;
		PetscScalar xl, sw;
		std::function<double(int)> a;
		const sbp::D1_central<sbp::Stencils_6th,7,6,9> D1;
		const sbp::H_central<sbp::Quadrature_6th,6> H;
		const sbp::HI_central<sbp::InverseQuadrature_6th,6> HI;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
		};
	#elif defined PROBLEM_TYPE_2D_O2
		struct AppCtx{
		std::array<PetscInt,2> N, i_start, i_end;
		std::array<PetscScalar,2> hi, h, xl;
		PetscScalar sw;
		std::function<double(int, int)> a;
		std::function<double(int, int)> b;
		const sbp::D1_central<sbp::Stencils_2nd,3,1,2> D1;
		const sbp::H_central<sbp::Quadrature_6th,6> H;
		const sbp::HI_central<sbp::InverseQuadrature_6th,6> HI;
		VecScatter scatctx;
		grid::partitioned_layout_2d layout;
		};
	#elif defined PROBLEM_TYPE_2D_O4
		struct AppCtx{
		std::array<PetscInt,2> N, i_start, i_end;
		std::array<PetscScalar,2> hi, h, xl;
		PetscScalar sw;
		std::function<double(int, int)> a;
		std::function<double(int, int)> b;
		const sbp::D1_central<sbp::Stencils_4th,5,4,6> D1;
		const sbp::H_central<sbp::Quadrature_6th,6> H;
		const sbp::HI_central<sbp::InverseQuadrature_6th,6> HI;
		VecScatter scatctx;
		grid::partitioned_layout_2d layout;
		};
	#elif defined PROBLEM_TYPE_2D_O6
		struct AppCtx{
		std::array<PetscInt,2> N, i_start, i_end;
		std::array<PetscScalar,2> hi, h, xl;
		PetscScalar sw;
		std::function<double(int, int)> a;
		std::function<double(int, int)> b;
		const sbp::D1_central<sbp::Stencils_6th,7,6,9> D1;
		const sbp::H_central<sbp::Quadrature_6th,6> H;
		const sbp::HI_central<sbp::InverseQuadrature_6th,6> HI;
		VecScatter scatctx;
		grid::partitioned_layout_2d layout;
		};
	#endif	

#endif