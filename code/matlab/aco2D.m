% function [D,b,E_R] = build_system_ref(T,m,hx,v0)
clear
close all

Tend = 0.242;
tblocks = 100;
Tpb = Tend/tblocks;
m = 41;
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

tauw = [-1;0;0];
taue = [1;0;0];
taus = [0;-1;0];
taun = [0;1;0];

% SATw = kr(tauw,HI,Imy)*kr(e_1,Imy)*kr(e3,e_1',Imy);
SATw = kr(HI,Imy,Imt,tauw)*kr(e_1,Imy,Imt)*kr(e_1',Imy,Imt,e3);
SATe = kr(HI,Imy,Imt,taue)*kr(e_m,Imy,Imt)*kr(e_m',Imy,Imt,e3);
SATs = kr(Imx,HI,Imt,taus)*kr(Imx,e_1,Imt)*kr(Imx,e_1',Imt,e3);
SATn = kr(Imx,HI,Imt,taun)*kr(Imx,e_m,Imt)*kr(Imx,e_m',Imt,e3);
SAT = SATw + SATe + SATs + SATn;
% SATe = kron(taue,HI,Imy)*ee*kron(e3,ee'), 
% SATs = kron(taus,Imx,HI)*es*kron(e3,es'), 
% SATn = kron(taun,Imx,HI)*en*kron(e3,en'), 
% 
% L = [kr(e_1',Imy,Imt,e3);
%      kr(e_m',Imy,Imt,e3);
%      kr(Imx,e_1',Imt,e3);
%      kr(Imx,e_m',Imt,e3)];
%  
% 
% P = speye(k*4*m*m) - sparse(HHI*L'*sparse(pinv(full(L*HHI*L')))*L);
% % 
tau = 1;
% 
A = [0,0,-1;
     0,0,0;
     -1,0,0];
 
B = [0,0,0;
    0,0,-1;
    0,-1,0];
% 
% % w_t + A*w_x + B*w_y = F
% % 
% % 
D = kr(Imx,Imy,Dt + tau*HIt*e_l*e_l',Ik) - kr(Dxy,Imy,Imt,A) - kr(Imx,Dxy,Imt,B) - SAT;
% 
% % FF = @(xvec,yvec,tvec) -3*pi*(rho-1)/rho*kr(cos(3*pi*xvec),sin(4*pi*yvec),cos(5*pi*tvec),e3) ...
% %                        -4*pi*(rho-1
% eigD = eig(full(D));
% figure
% scatter(real(eigD),imag(eigD))
% 
btmp = kr(Imx,Imy,tau*HIt*e_l,Ik);
E_R = kr(Imx,Imy,e_r',Ik);

vcurr = v0;
V = zeros(k*m*m,tblocks+1);
for tblock = 1:tblocks
    V(:,tblock) = vcurr;
    b = btmp*vcurr;
    v = D\b;
    vcurr = E_R*v;
end
V(:,tblock+1) = vcurr;

Van_1 = reshape(uan(X,Y,Tend),[m*m,1]);
Van_2 = reshape(van(X,Y,Tend),[m*m,1]);
Van_3 = reshape(pan(X,Y,Tend),[m*m,1]);
Van = [Van_1;Van_2;Van_3];
% Van = reshape(Van, [k*m*m,1]);

Vapprox = [E1*vcurr;E2*vcurr;E3*vcurr];

err_diff = Vapprox - Van;

err = sqrt(hx*hy)*norm(err_diff)

% figure
% subplot(3,1,1)
% srf1 = surf(X,Y,reshape(E1*vcurr,[m,m]));
% view(3)
% 
% subplot(3,1,2)
% srf2 = surf(X,Y,reshape(Van_1,[m,m]));
% view(3)
% 
% subplot(3,1,3)
% srf3 = surf(X,Y,reshape(Van_1 - E1*vcurr,[m,m]));
% view(3)
% 
% % % % % 
% 
% figure
% subplot(3,1,1)
% srf1 = surf(X,Y,reshape(E2*vcurr,[m,m]));
% view(3)
% 
% subplot(3,1,2)
% srf2 = surf(X,Y,reshape(Van_2,[m,m]));
% view(3)
% 
% subplot(3,1,3)
% srf3 = surf(X,Y,reshape(Van_2 - E2*vcurr,[m,m]));
% view(3)
% 
% % % % % 
% 
% figure
% subplot(3,1,1)
% srf1 = surf(X,Y,reshape(E3*vcurr,[m,m]));
% view(3)
% 
% subplot(3,1,2)
% srf2 = surf(X,Y,reshape(Van_3,[m,m]));
% view(3)
% 
% subplot(3,1,3)
% srf3 = surf(X,Y,reshape(Van_3 - E3*vcurr,[m,m]));
% view(3)
%                 
% figure('pos',[0,0,1000,1000])
% subplot(3,1,1)
% srf1 = surf(X,Y,reshape(E1*v0,[m,m]));
% axis([-1,1,-1,1,-2,2])
% 
% subplot(3,1,2)
% srf2 = surf(X,Y,reshape(E2*v0,[m,m]));
% axis([-1,1,-1,1,-2,2])
% 
% subplot(3,1,3)
% srf3 = surf(X,Y,reshape(E3*v0,[m,m]));
% axis([-1,1,-1,1,-2,2])
% 
% 
% pause
% for tidx = 2:tblocks+1
%     srf1.ZData = reshape(E1*V(:,tidx),[m,m]);
%     srf2.ZData = reshape(E2*V(:,tidx),[m,m]);
%     srf3.ZData = reshape(E3*V(:,tidx),[m,m]);
%     
%     pause(0.01)
% end


% 
% 
% 
% 
% 
% 
% 
% 
% 
