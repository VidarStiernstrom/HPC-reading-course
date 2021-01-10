clear

addpath("op_interpolation_operators/");

mf = 21;
xl = -1;
xr = 1;
mc = 0.5*(mf+1);

xvec = linspace(xl,xr,mf)';



[IC2F,IF2C] = MC_orders2to2(mc);
% [IC2F,IF2C,Hc,Hf] = OP_orders2to2(mc,'F2C');

v = zeros(mc,1);
v(1) = -0.0000000000000000;
v(2) = -0.0000000000000000;
v(3) = -0.0000000000000015;
v(4) = -0.0000007370713254;
v(5) = -0.1199618898218033;
v(6) = -6.5496972587503199;
v(7) = -0.1199618898218023;
v(8) = -0.0000007370713254;
v(9) = -0.0000000000000015;
v(10) = -0.0000000000000000;
v(11) = -0.0000000000000000;



% w = sym('w',[mf,1]);
% v = sym('v',[mc,1]);
% IF2C*w
% IC2F*v
% yvec = xvec + 1;
% IC2F*IF2C*yvec