% function [D,b,E_R] = build_system_ref(T,m,hx,v0)
clear
close all

Tend = 1.8;
tblocks = 1;
Tpb = Tend/tblocks;
m = 101;
xvec = linspace(-1,1,m)';
hx = xvec(2) - xvec(1);

theta1 = @(x,t) exp(-(x-t).^2/0.1^2);
theta2 = @(x,t) -theta1(x,t);

W = 2;
v0_1 = theta2(xvec,0) - theta1(xvec,0);
v0_2 = theta2(xvec,0) + theta1(xvec,0);
v0 = [v0_1;v0_2];

I2 = speye(2);
Im = speye(m);
Imt = speye(4);
Im2 = speye(2*m);


[Dt,Ht,tvec,ht,e_l,e_r] = d1_gauss_4(Tpb);
Dt = sparse(Dt); Ht = sparse(Ht); e_l = sparse(e_l); e_r = sparse(e_r);
HIt = inv(Ht);

eLt = kron(e_l',Im);
% 
[H,HI,Dx,e_1,e_m,S_1,S_m] = SBP2_BV3(m,hx);
Dx = sparse(Dx); e_1 = sparse(e_1); HI = sparse(HI);
HHI = kr(HI,Imt,I2);
% HHI = kr(I2,HI);
% DDx = -kron([0,1;1,0],Dx);
e1 = [1;0]';
e2 = [0;1]';
et{1} = kron(I2,kron(Im,[1;0;0;0]'));
et{2} = kron(I2,kron(Im,[0;1;0;0]'));
et{3} = kron(I2,kron(Im,[0;0;1;0]'));
et{4} = kron(I2,kron(Im,[0;0;0;1]'));
% 
% g = [0;0;0;0];
% L = [kr(e1,Imt,e_1');
%      kr(e1,Imt,e_m')];

L = [kr(e_1',Imt,e1);
     kr(e_m',Imt,e1)];
P = speye(2*4*m) - HHI*L'*pinv(full(L*HHI*L'))*L;
% L = [kr(e1,e_1');
%      kr(e1,e_m')];
% P = speye(2*m) - HHI*L'*pinv(full(L*HHI*L'))*L;
% 
tau = 1;
% D = kron(Im2,Dt + tau*HIt*e_l*e_l') + kron(P*DDx*P,Imt);
% D = kr(I2, Dt + tau*HIt*e_l*e_l', Im) + P*kr([0,1;1,0], Imt, -Dx)*P;

% D = kr(I2, Imt, Dx);
% D = kr(Imt, I2, Dx);
% D = kr(Imt, Dx, I2);
% D = kr(Dx, Imt, [0,1;1,0]);
% D = kr(Dx, I2, Imt);

D = kr(Im, Dt + tau*HIt*e_l*e_l', I2) + P*kr(-Dx, Imt, [0,1;1,0])*P;
% D = P*kr(-Dx, Imt, [0,1;1,0])*P;


% % v = ones(2*4*m,1);
% btmp = kron(Im2,tau*HIt*e_l);
% E_R = kron(Im2,e_r');
% 
% vcurr = v0;
% V = zeros(2*m,tblocks+1);
% for tblock = 1:tblocks
%     V(:,tblock) = vcurr;
%     b = btmp*vcurr;
%     v = D\b;
%     vcurr = E_R*v;
% end
% V(:,tblock+1) = vcurr;
% 
% van = [theta1(xvec, W - Tend) - theta2(xvec, -W + Tend);
%       theta1(xvec, W - Tend) + theta2(xvec, -W + Tend)];
% err_diff = vcurr - van;
%                 
% err = sqrt(hx)*norm(err_diff)

% figure
% subplot(2,1,1)
% plt1 = plot(xvec,V(1:m,1));
% axis([-1,1,-2,2])
% subplot(2,1,2)
% plt2 = plot(xvec,V(m+1:end,1));
% axis([-1,1,-2,2])
% 
% pause
% for tidx = 2:tblocks+1
%     plt1.YData = V(1:m,tidx);
%     plt2.YData = V(m+1:end,tidx);
%     
%     pause(0.01)
% end
% 
% 
% subplot(2,1,1)
% hold on
% plot(xvec, van(1:m))
% subplot(2,1,2)
% hold on
% plot(xvec, van(m+1:end))

v = zeros(size(D,1),1);

addpath('/usr/local/Cellar/petsc/3.14.0//share/petsc/matlab/')

for idx = 0:m-1
    x = xvec(idx+1);
    v(8*idx+1) = 0*x;
    v(8*idx+2) = 1*x;
    v(8*idx+3) = 2*x;
    v(8*idx+4) = 3*x;
    v(8*idx+5) = 4*x;
    v(8*idx+6) = 5*x;
    v(8*idx+7) = 6*x;
    v(8*idx+8) = 7*x;
end

Dvp = PetscBinaryRead('../data/ref_1D/Dv');
vp = PetscBinaryRead('../data/ref_1D/v');

D*v - Dvp
max(abs(D*v - Dvp))












