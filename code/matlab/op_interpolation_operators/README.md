# Order-preserving interpolation operators for non-conforming interfaces
This repository contains MATLAB code for interpolation operators corresponding to summation-by-parts finite difference operators and a grid interface with a 2:1 grid size ratio.

The operators denoted by MC are due to Mattsson and Carpenter:

+ Ken Mattsson and Mark H. Carpenter. Stable and accurate interpolation operators for high-order multiblock finite difference schemes. *SIAM J. Sci. Comput.*, 32(4):2298–2320, 2010, [https://doi.org/10.1137/090750068](https://doi.org/10.1137/090750068)

The **order-preserving** (OP) operators can be used to obtain one order higher global convergence rate for equations with second derivatives in space.

IC2F denotes an interpolation operator that takes from the coarse grid to the fine. IF2C takes from the fine grid to the coarse. For each order of accuracy, the MC operators are the adjoints of one another:

+ IC2F is the adjoint of IF2C.

The OP operators come in two adjoint pairs:

+ IC2F_g is the adjoint of IF2C_b,
+ IC2F_b is the adjoint of IF2C_g.

The operators with subscript 'g' are one order more accurate than those with subscript 'b'.

# How do I use the code?
* Clone the repository or simply download the files
* Start MATLAB and navigate into the repository directories.
* The call `[IC2F, IF2C] = MC_orders6to6(N_coarse)` creates MC operators corresponding to 6th order SBP operators, with N_coarse grid points on the coarse side of the interface.
* The call `[IC2F_g, IF2C_b] = OP_orders4to4(N_coarse, 'C2F')` creates a pair of OP operators, corresponding to 4th order SBP operators, where the operator from coarse to fine is one order more accurate than the one from fine to coarse.
* The call `[IC2F_b, IF2C_g] = OP_orders4to4(N_coarse, 'F2C')` creates the other pair of OP operators, where the operator from fine to coarse is one order more accurate than the one from coarse to fine.
* The other kinds of operators are created analogously.