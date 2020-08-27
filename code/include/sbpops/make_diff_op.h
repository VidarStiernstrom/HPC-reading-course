#include "sbpops/D1_central.h"

namespace sbp {
  // Central first derivative operators. The actual stencils are found in
  // the constructor impl. in sbpops/D1_central.h.
  // TODO: Would be more clear to pass the actual stencils
  // on construction. See comment in D1_central.h.
  constexpr D1_central<3,1,2> make_D1_central_2nd_order(){
    constexpr  D1_central<3,1,2>  D1;
    return D1;
  }
  constexpr D1_central<5,4,6> make_D1_central_4th_order(){
    constexpr  D1_central<5,4,6>  D1;
    return D1;
  }
  constexpr D1_central<7,6,9> make_D1_central_6th_order(){
    constexpr  D1_central<7,6,9> D1;
    return D1;
  }
}

