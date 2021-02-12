% function [D,b,E_R] = build_system_ref(T,m,hx,v0)
clear
close all

Tend = 0.01;
tblocks = 1;
Tpb = Tend/tblocks;
m = 21
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


[H,HI,Dxy,e_1,e_m,S_1,S_m] = SBP2_BV3(m,hx);
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


% D = kr(Imx,Imy,1,Ik) + kr(Dxy,Imy,Imt,A) + kr(Imx,Dxy,Imt,B) + SAT;
% eigD = eig(full(D));
% scatter(real(eigD),imag(eigD))

btmp = kr(Imy,Imx,tau*HIt*e_l,Ik);
E_R = kr(Imy,Imx,e_r',Ik);

DDD = diag(diag(D));
[L,U] = ilu(D);
M1 = @(x) DDD\x;
M2 = @(x) U\(L\x);
M3 = @(x) x;

vcurr = v0;
V = zeros(k*m*m,tblocks+1);
for tblock = 1:tblocks
    V(:,tblock) = vcurr;
    b = btmp*vcurr;
    
%     v = D\b;
    RESTART = 10;
    TOL = 1e-12;
    MAXIT = 100;
    
    [v,~,RELRES1,ITER1,RESVEC1] = gmres(D,b,RESTART,TOL,MAXIT,M1);
    [v,~,RELRES2,ITER2,RESVEC2] = gmres(D,b,RESTART,TOL,MAXIT,M2);
    [v,~,RELRES3,ITER3,RESVEC3] = gmres(D,b,RESTART,TOL,MAXIT,M3);
    
    vcurr = E_R*v;
end
V(:,tblock+1) = vcurr;


Van_1 = reshape(uan(X,Y,Tend),[m*m,1]);
Van_2 = reshape(van(X,Y,Tend),[m*m,1]);
Van_3 = reshape(pan(X,Y,Tend),[m*m,1]);
Van = [Van_1;Van_2;Van_3];

Vapprox1 = E1*vcurr;
err_diff1 = Vapprox1 - Van_1;
err1 = sqrt(hx*hy)*norm(err_diff1)

Vapprox2 = E2*vcurr;
err_diff2 = Vapprox2 - Van_2;
err2 = sqrt(hx*hy)*norm(err_diff2)

Vapprox3 = E3*vcurr;
err_diff3 = Vapprox3 - Van_3;
err3 = sqrt(hx*hy)*norm(err_diff3)

figure
hold on
semilogy(RESVEC1)
semilogy(RESVEC2)
semilogy(RESVEC3)
set(gca,'YScale','log')
legend('diag','lu','none')




