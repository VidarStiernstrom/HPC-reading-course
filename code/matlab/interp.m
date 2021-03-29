clear
close all

addpath("op_interpolation_operators/");

mf = 9;
xl = -1; yl = -1;
xr = 1; yr = 1;
mc = 0.5*(mf+1);

xvecf = linspace(xl,xr,mf)';
yvecf = linspace(yl,yr,mf)';
[Xf,Yf] = meshgrid(xvecf,yvecf);

xvecc = linspace(xl,xr,mc)';
yvecc = linspace(yl,yr,mc)';
[Xc,Yc] = meshgrid(xvecc,yvecc);

Imf = eye(mf);
Imc = eye(mc);

e1 = [1,0,0];
e2 = [0,1,0];
e3 = [0,0,1];
e4_1 = [1,0,0,0];

Eff = kr(Imf,Imf);

Ecc = kr(Imc,Imc);

Efc = kr(Imf,Imc);


[IC2F,IF2C] = MC_orders2to2(mc);

R = kr(IF2C,IF2C,eye(12));
% Rx = kr(Imf,IF2C);

vf = zeros(mf*mf*12,1);

for idxx = 0:mf-1
    for idxy = 0:mf-1
        x = xvecf(idxx+1);
        y = yvecf(idxy+1);
        val = x*x*y + y*x + 1;
        for dof = 1:12
            vf(12*(idxy*mf + idxx) + dof) = val + dof - 1;
            vf_mat(idxx+1,idxy+1,dof) = val + dof - 1;
        end
        
        %         vf(12*(idxy*mf + idxx) + 1) = 0*x - 0*y;
        %         vf(12*(idxy*mf + idxx) + 2) = 1*x - 1*y;
        %         vf(12*(idxy*mf + idxx) + 3) = 2*x - 2*y;
        %         vf(12*(idxy*mf + idxx) + 4) = 3*x - 3*y;
        %         vf(12*(idxy*mf + idxx) + 5) = 4*x - 4*y;
        %         vf(12*(idxy*mf + idxx) + 6) = 5*x - 5*y;
        %         vf(12*(idxy*mf + idxx) + 7) = 6*x - 6*y;
        %         vf(12*(idxy*mf + idxx) + 8) = 7*x - 7*y;
        %         vf(12*(idxy*mf + idxx) + 9) = 8*x - 8*y;
        %         vf(12*(idxy*mf + idxx) + 10) = 9*x - 9*y;
        %         vf(12*(idxy*mf + idxx) + 11) = 10*x - 10*y;
        %         vf(12*(idxy*mf + idxx) + 12) = 11*x - 11*y;
        
    end
end

% for idxx = 1:mf
%     for idxy = 1:mf
%         vf
%     end
% end
% vf = ones(mf*mf,1);
% vc = R*vf;

addpath('/usr/local/Cellar/petsc/3.14.0//share/petsc/matlab/')
Rvp = PetscBinaryRead('../data/aco2D/Rv');
vp = PetscBinaryRead('../data/aco2D/v');

vp = reshape(vp,[3,4,mf,mf]);
% vp = permute(vp,[2,1,3,4]);
vp = reshape(vp,[12*mf*mf,1]);

Rvp = reshape(Rvp,[3,4,mc,mc]);
% Rvp = permute(Rvp,[2,1,3,4]);
Rvp = reshape(Rvp,[12*mc*mc,1]);

max(max(abs(vp - vf)))
max(max(abs(Rvp - R*vf)))

% vp = reshape(vp,[3,4,mf,mf]);
% vp = permute(vp,[1,2,4,3]);

% Rv = reshape(R*vf,[3,4,mc,mc]);
% U = reshape(vf,[mf,mf]);
%
% diff = RU - RUp;
% surf(Xc,Yc,diff*100)
% view(2)
% colorbar

% max(abs((RU - RUp
