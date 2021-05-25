#pragma once
#ifndef INDEX
#define N_DOF 3
#define INDEX(Nx,j,i,comp) (N_DOF*((i) + (Nx)*(j)) + (comp))
#endif