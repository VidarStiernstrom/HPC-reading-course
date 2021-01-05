clear

addpath('/usr/local/Cellar/petsc/3.14.0//share/petsc/matlab/')


v = PetscBinaryRead('../data/adv_1D_try/vres');

% mx = 21;
% mt = 11;

% reshape(v,[mt,mx])'