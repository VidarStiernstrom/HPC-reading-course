clear
close all

addpath('/usr/local/Cellar/petsc/3.14.3/share/petsc/matlab/')
addpath('/Users/guser801/MATLAB/ColorBlindSets_v1_0/ColorBlindSets_v1_0')

% data_dir = "../data/aco2D/";
data_dir = "../rack_data/new/";

Cmat = getColorSet(20);
Typelist = ['o','+','*','d','x','.','s','v'];
linestyle = {'--','-','--','-','--','-','--','-'};

fs = 28;
lw = 3;


krylovs_outmg = ["pipefgmres","fgmres","fbcgsr"];
krylovs_inmg = ["gmres", "pgmres"];
% krylovs_outmg = ["fgmres"];
% krylovs_inmg = ["gmres"];

Nx = 2304;
Ny = 2304;
nlevels = 4;


figure('pos',[0 0 1471        1320])


sizes = [1,4,16,36];

% size = 4;
for sizeidx = 1:numel(sizes)
    size = sizes(sizeidx);
    
    subplot(2,2,sizeidx)
    hold on
    leg = [];
    
    method = "MGV";
    
    fprintf("-------- Multigrid preconditioner --------\n")
    for methodoutidx = 1:numel(krylovs_outmg)
        for methodinidx = 1:numel(krylovs_inmg)
            krylovout = krylovs_outmg(methodoutidx);
            krylovin = krylovs_inmg(methodinidx);
            filename = char(data_dir + method + "_" + krylovout + "_" + krylovin + "_Nx" + num2str(Nx) +...
                "_Ny" + num2str(Ny) + "_size" + num2str(size) + "_nlevels" + num2str(nlevels));
            
            if ~isequal(krylovout + "_" + krylovin,"pipefgmres_pgmres")
                
                if exist(filename, 'file')
                    data = PetscBinaryRead(filename);
                    
                    nits = data(end-1);
                    elaptime = data(end);
                    fprintf("Method: %s, elapsed time: %f\n",krylovout + "_" + krylovin,elaptime)
                    reshist = data(1:nits);
                    leg = [leg,krylovout + "-" + krylovin];
                    plot(reshist/reshist(1),'LineStyle',linestyle{numel(leg)},'Linewidth',lw,'Color',Cmat(numel(leg),:))
%                     plot(reshist/reshist(1),'Marker',Typelist(numel(leg)),'Linewidth',lw)
%                     plot(reshist/reshist(1),'--')
%                     linestyle(numel(leg))
                else
                    fprintf("Missing file: %s\n",filename);
                end
            end
        end
        
    end
    
    set(gca,'YScale','log')
    box on
    grid on
    
    method = "stand";
    krylovs_stand = ["gmres", "pgmres","bcgs"];
    % krylovs_stand = ["bcgs"];
    
    fprintf("-------- No preconditioner --------\n")
    for methodidx = 1:numel(krylovs_stand)
        krylov = krylovs_stand(methodidx);
        filename = char(data_dir + method + "_" + krylov + "_Nx" + num2str(Nx) +...
            "_Ny" + num2str(Ny) + "_size" + num2str(size));
        data = PetscBinaryRead(filename);
        
        nits = data(end-1);
        elaptime = data(end);
        fprintf("Method: %s, elapsed time: %f\n",krylov,elaptime)
        reshist = data(1:nits);
        
        leg = [leg,"std-" + krylov];
%         plot(reshist/reshist(1),Typelist(numel(leg)),'Linewidth',lw,'Color',Cmat(numel(leg),:))
        plot(reshist/reshist(1),'LineStyle',linestyle{numel(leg)},'Linewidth',lw,'Color',Cmat(numel(leg),:))
        set(gca,'YScale','log','XScale','log')
%         set(gca,'YScale','log')
        box on
        grid on
    end
    
    if size == 4
        lgd = legend("PIPEFGMRES-GMRES","FGMRES-GMRES","FGMRES-PGMRES","FBCGS-GMRES","FBCGS-PGMRES","GMRES","PGMRES","BCGS",'Location','Northeast');
    end
    
    % legend('MG - pipefgmres','MG - fgmres',...
    %     'No precond. - gmres','No precond. - pgmres')
    set(gca,'Fontsize',fs);
    if size == 1
        title(num2str(size) + " core")
    else
        title(num2str(size) + " cores")
    end
    axis([0,10^4,10^-13,10^2])
    set(gca,'Fontsize',fs)
    xlabel('Iteration')
    ylabel('Relative residual')
end

lgd.FontSize = 16;
set(gcf,'color','white')

addpath('/Users/guser801/MATLAB/export_fig')
export_fig 'reshist.pdf'