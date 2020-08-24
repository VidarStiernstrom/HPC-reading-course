#include "sbpops/D1_central.h"
#include <math.h>
#include <iostream>

int main(int argc,char **argv)
{
  // Quadratic function x^2 on the domain [0,14] with h = 1
  double v[13] = {0,1,4,9,16,25,36,49,64,81,100,121,144};
  const double u_x[13] = {0,2,4,6,8,10,12,14,16,18,20,22,24}; //Exact solution
  double hi = 1.;
  constexpr D1_central<5,4,6> d1_4;
  double v_x[13];
  d1_4.apply(v,hi,13,v_x);
  double sum = 0;
  for (int i = 0; i < 13; i++) {
    sum += hi*pow(u_x[i]-v_x[i],2);
  }
  std::cout << "l2 error for quadratic function is " << sqrt(sum) <<std::endl;
  return 0;
}


