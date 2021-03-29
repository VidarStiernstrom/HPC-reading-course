clear
close all

Tend = 0.1;
tblocks = 1;
Tpb = Tend/tblocks;
m = 11;
m2 = m*m;
xvec = linspace(-1,1,m)';
yvec = linspace(-1,1,m)';
[X,Y] = meshgrid(xvec,yvec);
hx = xvec(2) - xvec(1);
hy = yvec(2) - yvec(1);
k = 3;
      
nn = 3;
mm = 4;

uan = @(x,y,t) -nn*cos(nn*pi*x).*sin(mm*pi*y).*sin(pi*sqrt(nn*nn + mm*mm)*t)/sqrt(nn*nn + mm*mm);
van = @(x,y,t) -mm*sin(nn*pi*x).*cos(mm*pi*y).*sin(pi*sqrt(nn*nn + mm*mm)*t)/sqrt(nn*nn + mm*mm);
pan = @(x,y,t) sin(pi*nn*x).*sin(pi*mm*y).*cos(pi*sqrt(nn*nn + mm*mm)*t);

v0_1 = reshape(uan(X,Y,0),[m*m,1]);
v0_2 = reshape(van(X,Y,0),[m*m,1]);
v0_3 = reshape(pan(X,Y,0),[m*m,1]);
v0 = [v0_1,v0_2,v0_3]';
v0 = reshape(v0, [k*m*m,1]);

Ik = speye(k);
Imx = speye(m);
Imy = speye(m);
Imt = speye(4);

[Dt,Ht,tvec,ht,e_l,e_r] = d1_gauss_4(Tpb);
Dt = sparse(Dt); Ht = sparse(Ht); e_l = sparse(e_l); e_r = sparse(e_r);
HIt = inv(Ht);


[H,HI,Dxy,e_1,e_m,S_1,S_m] = SBP6_BV3(m,hx);
Dxy = sparse(Dxy); e_1 = sparse(e_1); HI = sparse(HI);
HIx = kr(HI,Imy);
HIy = kr(Imx,HI);
HHI = kr(HI,HI,Imt,Ik);
e1 = [1,0,0];
e2 = [0,1,0];
e3 = [0,0,1];

E1 = kr(Imx,Imy,e1);
E2 = kr(Imx,Imy,e2);
E3 = kr(Imx,Imy,e3);

tauw = [1;0;0];
taue = [-1;0;0];
taus = [0;1;0];
taun = [0;-1;0];

% y -> x -> t -> k
SATw = kr(HI,Imy,Imt,tauw)*kr(e_1,Imy,Imt)*kr(e_1',Imx,Imt,e3);
SATe = kr(HI,Imy,Imt,taue)*kr(e_m,Imy,Imt)*kr(e_m',Imx,Imt,e3);
SATs = kr(Imx,HI,Imt,taus)*kr(Imx,e_1,Imt)*kr(Imy,e_1',Imt,e3);
SATn = kr(Imx,HI,Imt,taun)*kr(Imx,e_m,Imt)*kr(Imy,e_m',Imt,e3);
SAT = SATw + SATe + SATs + SATn;

tau = 1;

A = [0,0,1;
    0,0,0;
    1,0,0];

B = [0,0,0;
    0,0,1;
    0,1,0];


D = kr(Imx,Imy,Dt + tau*HIt*e_l*e_l',Ik) + kr(Dxy,Imy,Imt,A) + kr(Imx,Dxy,Imt,B) + SAT;

eigD = eig(full(D));

figure('pos',[288         820        1246         516])
fs = 20;

subplot(1,2,1)
spy(D)
ax1 = get(gca);
title('Sparsity pattern of A')
set(gca,'Fontsize',fs)
box on

subplot(1,2,2)
scatter(real(eigD),imag(eigD))
ax2 = get(gca);
xlabel('Re(\lambda)')
ylabel('Im(\lambda)')
grid on
box on
title('Spectrum of A')
set(gca,'Fontsize',fs)

% background color
set(gcf,'color','white')

% save figure using export_fig, https://github.com/altmany/export_fig
addpath('/Users/guser801/MATLAB/export_fig')
export_fig 'sparsspec.pdf'