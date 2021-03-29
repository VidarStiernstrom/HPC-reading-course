clear
close all

addpath('/usr/local/Cellar/petsc/3.14.3/share/petsc/matlab/')
addpath('/Users/guser801/MATLAB/ColorBlindSets_v1_0/ColorBlindSets_v1_0')

% data_dir = "../data/aco2D/";
data_dir = "../rack_data/new/";

Cmat = getColorSet(20);
Typelist = ['o','+','*','d','x'];

fs = 32;
ms = 400;
lw = 1.7;

method = "MGV";
krylovs_outmg = ["pipefgmres","fgmres"];
krylovs_inmg = ["gmres", "pgmres"];


Nx = 2304;
Ny = 2304;
sizes = [1,4,16,36,64];
nlevels = 4;

figure('pos',[0 0 1970         802])
subplot(1,2,1)
hold on
subplot(1,2,2)
hold on

leg = [];


fprintf("-------- Multigrid preconditioner --------\n")
for methodoutidx = 1:numel(krylovs_outmg)
    for methodinidx = 1:numel(krylovs_inmg)
        krylovout = krylovs_outmg(methodoutidx);
        krylovin = krylovs_inmg(methodinidx);
        if ~isequal(krylovout + "_" + krylovin,"pipefgmres_pgmres")
            for sizeidx = 1:numel(sizes)
                size = sizes(sizeidx);
                
                filename = char(data_dir + method + "_" + krylovout + "_" + krylovin + "_Nx" + num2str(Nx) +...
                    "_Ny" + num2str(Ny) + "_size" + num2str(size) + "_nlevels" + num2str(nlevels));
                
                
                if exist(filename, 'file')
                    data = PetscBinaryRead(filename);
                    nits = data(end-1);
                    elaptime = data(end);
                    fprintf("Method: %s, size: %d, nits: %d, elapsed time: %f\n",krylovout + "_" + krylovin,size,nits,elaptime)
                    reshist = data(1:nits);
                    
                    
                    timings(sizeidx) = elaptime;
                else
                    fprintf("Missing file: %s\n",filename);
                    timings(sizeidx) = NaN;
                end
                
            end
            leg = [leg,krylovout + "-" + krylovin];
            S = timings(1)./timings;
            subplot(1,2,1)
            scatter(sizes,S,ms,Typelist(numel(leg)),'MarkerEdgeColor',Cmat(numel(leg),:),'Linewidth',lw)
            
            subplot(1,2,2)
            scatter(sizes,100*S./sizes,ms,Typelist(numel(leg)),'MarkerEdgeColor',Cmat(numel(leg),:),'Linewidth',lw)
        end
    end
end

method = "stand";
krylovs_stand = ["gmres", "pgmres"];
% krylovs_stand = ["pipefgmres"];

fprintf("-------- No preconditioner --------\n")
for methodidx = 1:numel(krylovs_stand)
    for sizeidx = 1:numel(sizes)
        size = sizes(sizeidx);
        krylov = krylovs_stand(methodidx);
        filename = char(data_dir + method + "_" + krylov + "_Nx" + num2str(Nx) +...
            "_Ny" + num2str(Ny) + "_size" + num2str(size));
        
        if exist(filename, 'file')
            data = PetscBinaryRead(filename);
            nits = data(end-1);
            elaptime = data(end);
            fprintf("Method: %s, size: %d, nits: %d, elapsed time: %f\n",krylov,size,nits,elaptime)
            reshist = data(1:nits);
            
            
            timings(sizeidx) = elaptime;
        else
            fprintf("Missing file: %s\n",filename);
            timings(sizeidx) = NaN;
        end
    end
    leg = [leg,"std-" + krylov];
    
    S = timings(1)./timings;
    subplot(1,2,1)
    scatter(sizes,S,ms,Typelist(numel(leg)),'MarkerEdgeColor',Cmat(numel(leg),:),'Linewidth',lw)
    
    subplot(1,2,2)
    scatter(sizes,100*S./sizes,ms,Typelist(numel(leg)),'MarkerEdgeColor',Cmat(numel(leg),:),'Linewidth',lw)
    
end
% leg = ["Linear",leg];
%
subplot(1,2,1)
plot(sizes,sizes,'k--')
title('Fixed size speedup')
axis([1,80,0,80])
set(gca,'Fontsize',fs)
grid on
box on
[h,icons,plots,legend_text] = legend("PIPEFGMRES-GMRES","FGMRES-GMRES","FGMRES-PGMRES","GMRES","PGMRES","Linear",'Location','Northwest');
for idx = 7:11
    icons(idx).Children.MarkerSize = ms/19;
end

xlabel('Number of cores')
ylabel('Speedup')

subplot(1,2,2)
axis([1,80,0,120])
title('Fixed size efficiency')
grid on
hold on
box on
set(gca,'Fontsize',fs)
xlabel('Number of cores')
ylabel('Efficiency %')

set(gcf,'color','white')

addpath('/Users/guser801/MATLAB/export_fig')
export_fig 'speedupeff.pdf'

