clear
close all

addpath("op_interpolation_operators/");

mf = 11;
xl = -1; yl = -1;
xr = 1; yr = 1;
mc = 0.5*(mf+1);


[IC2F,IF2C] = MC_orders2to2(mc);

vf = sym('v',[mf,1]);