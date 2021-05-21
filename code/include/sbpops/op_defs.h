#pragma once

#include "sbpops/D1_central.h"
#include "sbpops/H_central.h"
#include "sbpops/HI_central.h"

#ifndef SBP_OPERATOR_ORDER
#error "SBP_OPERATOR_ORDER not defined (must be one of 2,4,6)"
#endif

// TODO: In the future we can add a preprocessor flag for different
// operator types. e.g #ifdef SBP_OPERATOR_TYPE_CENTRAL
// #ifndef SBP_OPERATOR_TYPE_CENTRAL
// #define SBP_OPERATOR_TYPE_CENTRAL
// #endif

// #ifdef SBP_OPERATOR_TYPE_CENTRAL
#if SBP_OPERATOR_ORDER == 2
	typedef sbp::D1_central<sbp::Stencils_2nd,3,1,2> FirstDerivativeOp;
	typedef sbp::H_central<sbp::Quadrature_2nd,1> NormOp;
	typedef sbp::HI_central<sbp::InverseQuadrature_2nd,1> InverseNormOp;
#elif SBP_OPERATOR_ORDER == 4
	typedef sbp::D1_central<sbp::Stencils_4th,5,4,6> FirstDerivativeOp;
	typedef sbp::H_central<sbp::Quadrature_4th,4> NormOp;
	typedef sbp::HI_central<sbp::InverseQuadrature_4th,4> InverseNormOp;
#elif SBP_OPERATOR_ORDER == 6
	typedef sbp::D1_central<sbp::Stencils_6th,7,6,9> FirstDerivativeOp;
	typedef sbp::H_central<sbp::Quadrature_6th,6> NormOp;
	typedef sbp::HI_central<sbp::InverseQuadrature_6th,6> InverseNormOp;
#endif //SBP_OPERATOR_ORDER
// #endif //SBP_OPERATOR_TYPE_CENTRAL