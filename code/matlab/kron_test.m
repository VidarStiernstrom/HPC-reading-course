clear
% close all

addpath("op_interpolation_operators/");

m = 101;
xl = -1;
xr = 1;
T = 0.01;

xvec = linspace(xl,xr,m);
hx = xvec(2) - xvec(1);

Imx = eye(m);
Imt = eye(4);

xvec = linspace(xl,xr,m)';

[Dt,Ht,tvec,ht,e_l,e_r] = d1_gauss_4(T);
HIt = inv(Ht);
% D = full(D);
% Dt = kron(D,Imx);

eLt = kron(e_l',Imx);


[H,HI,Dx,e_1,e_m,S_1,S_m] = SBP6_BV3(m,hx);

v0 = exp(-xvec.^2/0.1^2);
g = [0;0;0;0];
L = [e_1'];
P = eye(m) - L'*pinv(full(L*L'))*L;

tau = 1;
D = kron(Dt + tau*HIt*e_l*e_l', Imx) + kron(Imt, P*Dx*P);
b = kron(tau*HIt*e_l,Imx)*v0;

v = D\b;
vfinal = kron(e_r',Imx)*v;

b = kron(tau*HIt*e_l,Imx)*vfinal;

v = D\b;
vfinal = kron(e_r',Imx)*v;

v = reshape(v,[m,4]);

van = exp(-(xvec-2*T).^2/0.1^2);

err_diff = van - vfinal;
% err = sqrt(err_diff'*H*err_diff)
err = sqrt(hx)*norm(err_diff)

% figure
% plt = plot(xvec,v(:,1));
% axis([xl,xr,-1,1])
% pause
% plt.YData = v(:,2);
% pause
% plt.YData = v(:,3);
% pause
% plt.YData = v(:,4);

% eigD = eig(D);
% figure
% scatter(real(eigD),imag(eigD))

% w = kron(ones(4,1),xvec.^4) + kron(ht*[0:3]',ones(m,1));

addpath('/usr/local/Cellar/petsc/3.14.0//share/petsc/matlab/')

vfinalpetsc = PetscBinaryRead('../data/adv_1D_try/v_final');
% bpetsc = PetscBinaryRead('../data/adv_1D_try/b');

max(abs(vfinal - vfinalpetsc))
% max(abs(reshape(reshape(b,[m,4])',[4*m,1]) - bpetsc))

% 
% w = (-0.040000:0.0080:0.040000);
% figure
% subplot(1,2,1)
% spy(Dt)
% subplot(1,2,2)
% spy(Dx)