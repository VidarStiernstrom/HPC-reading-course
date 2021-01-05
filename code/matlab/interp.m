clear

addpath("op_interpolation_operators/");

mf = 21;
xl = -1;
xr = 1;
mc = 0.5*(mf+1);

xvec = linspace(xl,xr,mf)';



[IC2F,IF2C] = MC_orders2to2(mc);
% [IC2F,IF2C,Hc,Hf] = OP_orders2to2(mc,'F2C');


% w = sym('w',[mf,1]);
% v = sym('v',[mc,1]);
% IF2C*w
% IC2F*v
yvec = xvec + 1;
IC2F*IF2C*yvec