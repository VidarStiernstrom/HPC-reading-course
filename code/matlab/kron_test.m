clear
% close all

m = 101;
xl = -1;
xr = 1;
T = 0.01;

xvec = linspace(xl,xr,m);
hx = xvec(2) - xvec(1);

addpath('/usr/local/Cellar/petsc/3.14.0//share/petsc/matlab/')

vfinalpetsc = PetscBinaryRead('../data/ref_1D/vout');
