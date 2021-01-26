clear
close all

Tend = 0.1;
tblocks = 1;
Tpb = Tend/tblocks;
m = 21;
m2 = m*m;
xvec = linspace(-1,1,m)';
yvec = linspace(-1,1,m)';
[X,Y] = meshgrid(xvec,yvec);
hx = xvec(2) - xvec(1);
hy = yvec(2) - yvec(1);
k = 3;

% rho = @(x,y) 1./(2 + x.*y);
% K = @(x,y) 0*x + 1;

nn = 3;
mm = 4;

uan = @(x,y,t) -nn*cos(nn*pi*x).*sin(mm*pi*y).*sin(pi*sqrt(nn*nn + mm*mm)*t)/sqrt(nn*nn + mm*mm);
van = @(x,y,t) -mm*sin(nn*pi*x).*cos(mm*pi*y).*sin(pi*sqrt(nn*nn + mm*mm)*t)/sqrt(nn*nn + mm*mm);
pan = @(x,y,t) sin(pi*nn*x).*sin(pi*mm*y).*cos(pi*sqrt(nn*nn + mm*mm)*t);

v0_1 = reshape(uan(X,Y,0),[m*m,1]);
v0_2 = reshape(van(X,Y,0),[m*m,1]);
v0_3 = reshape(pan(X,Y,0),[m*m,1]);
v0 = [v0_1,v0_2,v0_3];
v0 = reshape(v0', [k*m*m,1]);

Ik = speye(k);
Imx = speye(m);
Imy = speye(m);
Imt = speye(4);

[Dt,Ht,tvec,ht,e_l,e_r] = d1_gauss_4(Tpb);
Dt = sparse(Dt); Ht = sparse(Ht); e_l = sparse(e_l); e_r = sparse(e_r);
HIt = inv(Ht);

%
[H,HI,Dxy,e_1,e_m,S_1,S_m] = SBP2_BV3(m,hx);
Dxy = sparse(Dxy); e_1 = sparse(e_1); HI = sparse(HI);
HHI = kr(HI,HI,Imt,Ik);
e1 = [1,0,0];
e2 = [0,1,0];
e3 = [0,0,1];
e4_4 = [0,0,0,1];

E1 = kr(Imx,Imy,e4_4,e1);
E2 = kr(Imx,Imy,e4_4,e2);
E3 = kr(Imx,Imy,e4_4,e3);

tauw = [-1;0;0];
taue = [1;0;0];
taus = [0;-1;0];
taun = [0;1;0];

SATw = kr(Imy,HI,Imt,tauw)*kr(Imx,e_1,Imt)*kr(Imx,e_1',Imt,e3);
SATe = kr(Imx,HI,Imt,taue)*kr(Imx,e_m,Imt)*kr(Imx,e_m',Imt,e3);
SATs = kr(HI,Imx,Imt,taus)*kr(e_1,Imy,Imt)*kr(e_1',Imy,Imt,e3);
SATn = kr(HI,Imx,Imt,taun)*kr(e_m,Imy,Imt)*kr(e_m',Imy,Imt,e3);
SAT = SATw + SATe + SATs + SATn;

% L = [kr(e_1',Imt,e1);
%      kr(e_m',Imt,e1)];

% L = [kr(e_1',Imy,Imt,e3);
%     kr(e_m',Imy,Imt,e3);
%     kr(Imx,e_1',Imt,e3);
%     kr(Imx,e_m',Imt,e3)];

% L = [kr(e3,Imt,SBPx.e_1',Imy); % pw
%      kr(e3,Imt,SBPx.e_m',Imy); % pe
%      kr(e3,Imt,Imx,SBPy.e_1'); % ps
%      kr(e3,Imt,Imx,SBPy.e_m')] % pn
% P = speye(k*4*m*m) - sparse(HHI*L'*sparse(pinv(full(L*HHI*L')))*L);
%
tau = 1;

A = [0,0,-1;
    0,0,0;
    -1,0,0];

B = [0,0,0;
    0,0,-1;
    0,-1,0];
%
% w_t + A*w_x + B*w_y = F
%
%
D = kr(Imx,Imy,Dt + tau*HIt*e_l*e_l',Ik) - kr(Imy,Dxy,Imt,A) - kr(Dxy,Imx,Imt,B) - SAT;
% D = - kr(Imy,Dxy,Imt,A) - kr(Dxy,Imx,Imt,B) - SAT;

v = zeros(size(D,1),1);
addpath('/usr/local/Cellar/petsc/3.14.0//share/petsc/matlab/')
%
for idxx = 0:m-1
    for idxy = 0:m-1
        x = xvec(idxx+1);
        y = yvec(idxy+1);
        v(12*(idxy*m + idxx) + 1) = 0*x - 0*y;
        v(12*(idxy*m + idxx) + 2) = 1*x - 1*y;
        v(12*(idxy*m + idxx) + 3) = 2*x - 2*y;
        v(12*(idxy*m + idxx) + 4) = 3*x - 3*y;
        v(12*(idxy*m + idxx) + 5) = 4*x - 4*y;
        v(12*(idxy*m + idxx) + 6) = 5*x - 5*y;
        v(12*(idxy*m + idxx) + 7) = 6*x - 6*y;
        v(12*(idxy*m + idxx) + 8) = 7*x - 7*y;
        v(12*(idxy*m + idxx) + 9) = 8*x - 8*y;
        v(12*(idxy*m + idxx) + 10) = 9*x - 9*y;
        v(12*(idxy*m + idxx) + 11) = 10*x - 10*y;
        v(12*(idxy*m + idxx) + 12) = 11*x - 11*y;
    end
end

Dvp = PetscBinaryRead('../data/aco2D/Dv');
vp = PetscBinaryRead('../data/aco2D/v');

vp = reshape(vp,[3,4,m,m]);
Dvp = reshape(Dvp,[3,4,m,m]);
% Up = squeeze(vp(1,end,:,:));
% Vp = squeeze(vp(2,end,:,:));
% Pp = squeeze(vp(3,end,:,:));


vp = permute(vp,[1,2,4,3]);
Up = squeeze(vp(1,end,:,:));
Vp = squeeze(vp(2,end,:,:));
Pp = squeeze(vp(3,end,:,:));

Dvp = permute(Dvp,[1,2,4,3]);
DUp = squeeze(Dvp(1,end,:,:));
DVp = squeeze(Dvp(2,end,:,:));
DPp = squeeze(Dvp(3,end,:,:));

U = reshape(E1*v,[m,m])';
V = reshape(E2*v,[m,m])';
P = reshape(E3*v,[m,m])';

DU = reshape(E1*D*v,[m,m])';
DV = reshape(E2*D*v,[m,m])';
DP = reshape(E3*D*v,[m,m])';

errU = DU - DUp;
errV = DV - DVp;
errP = DP - DPp;

figure('pos',[0,0,1600,1400]/1.5)

subplot(3,3,1)
surf(X,Y,errU)
title('error')
colorbar
view(2)
subplot(3,3,4)
surf(X,Y,errV)
colorbar
view(2)
subplot(3,3,7)
surf(X,Y,errP)
colorbar
view(2)

subplot(3,3,2)
surf(X,Y,DUp)
title('petsc')
colorbar
view(2)
subplot(3,3,5)
surf(X,Y,DVp)
colorbar
view(2)
subplot(3,3,8)
surf(X,Y,DPp)
colorbar
view(2)

subplot(3,3,3)
surf(X,Y,DU)
title('matlab')
colorbar
view(2)
subplot(3,3,6)
surf(X,Y,DV)
colorbar
view(2)
subplot(3,3,9)
surf(X,Y,DP)
colorbar
view(2)

