#if !defined APPCTX_INCLUDED
	#define APPCTX_INCLUDED

	#if defined PROBLEM_TYPE_1D_O2
		struct AppCtx{
		std::array<PetscInt,1> N, i_start, i_end;
		std::array<PetscScalar,1> hi;
		PetscScalar xl, sw;
		std::function<double(int)> a;
		const sbp::D1_central<sbp::Stencils_2nd,3,1,2> D1;
		VecScatter scatctx;
		};
	#elif defined PROBLEM_TYPE_1D_O4
		struct AppCtx{
		std::array<PetscInt,1> N, i_start, i_end;
		std::array<PetscScalar,1> hi;
		PetscScalar xl, sw;
		std::function<double(int)> a;
		const sbp::D1_central<sbp::Stencils_4th,5,4,6> D1;
		VecScatter scatctx;
		};
	#elif defined PROBLEM_TYPE_1D_O6
		struct AppCtx{
		std::array<PetscInt,1> N, i_start, i_end;
		std::array<PetscScalar,1> hi;
		PetscScalar xl, sw;
		std::function<double(int)> a;
		const sbp::D1_central<sbp::Stencils_6th,7,6,9> D1;
		VecScatter scatctx;
		grid::partitioned_layout_1d layout;
		};
	#elif defined PROBLEM_TYPE_2D_O2
		struct AppCtx{
		std::array<PetscInt,2> N, i_start, i_end;
		std::array<PetscScalar,2> hi;
		PetscScalar xl, yl, sw;
		std::function<double(int, int)> a;
		std::function<double(int, int)> b;
		const sbp::D1_central<sbp::Stencils_2nd,3,1,2> D1;
		VecScatter scatctx;
		};
	#elif defined PROBLEM_TYPE_2D_O4
		struct AppCtx{
		std::array<PetscInt,2> N, i_start, i_end;
		std::array<PetscScalar,2> hi;
		PetscScalar xl, yl, sw;
		std::function<double(int, int)> a;
		std::function<double(int, int)> b;
		const sbp::D1_central<sbp::Stencils_4th,5,4,6> D1;
		VecScatter scatctx;
		};
	#elif defined PROBLEM_TYPE_2D_O6
		struct AppCtx{
		std::array<PetscInt,2> N, i_start, i_end;
		std::array<PetscScalar,2> hi;
		PetscScalar xl, yl, sw;
		std::function<double(int, int)> a;
		std::function<double(int, int)> b;
		const sbp::D1_central<sbp::Stencils_6th,7,6,9> D1;
		VecScatter scatctx;
		};
	#endif	

#endif