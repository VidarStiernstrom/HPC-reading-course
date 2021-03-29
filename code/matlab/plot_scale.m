clear
close all

% data_dir = "../data/aco2D/";
data_dir = "../rack_data/aco2D/Nx2240_Ny2240/";

method = "MG";
krylovs_mg = ["pipefgmres","fgmres","fbcgsr"];


Nx = 2240;
Ny = 2240;
sizes = [1,4,8,20,40];
nlevels = 4;

figure
hold on

fprintf("-------- Multigrid preconditioner --------\n")
for methodidx = 1:numel(krylovs_mg)
    for sizeidx = 1:numel(sizes)
        size = sizes(sizeidx);
        krylov = krylovs_mg(methodidx);
        filename = char(data_dir + method + "_" + krylov + "_Nx" + num2str(Nx) +...
            "_Ny" + num2str(Ny) + "_size" + num2str(size) + "_nlevels" + num2str(nlevels));
        
        
        if exist(filename, 'file')
            data = PetscBinaryRead(filename);
            nits = data(end-1);
            elaptime = data(end);
            fprintf("Method: %s, elapsed time: %f\n",krylov,elaptime)
            reshist = data(1:nits);
            
            timings(sizeidx) = elaptime;
        else
            timings(sizeidx) = NaN;
        end
       
        
        
    end
    scatter(sizes,timings,'filled');
    box on
    grid on
    
    S = timings(1)./timings;
end

legend(krylovs_mg)

method = "stand";
krylovs_stand = ["gmres", "pgmres","bcgs"];

% Nx = 1440;
% Ny = 1440;


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
            fprintf("Method: %s, elapsed time: %f\n",krylov,elaptime)
            reshist = data(1:nits);
            
            timings(sizeidx) = elaptime;
        end
        
        
    end
    scatter(sizes,timings,'filled');
    box on
    grid on
end
%
set(gca,'YScale','log')
legend([krylovs_mg,krylovs_stand])
% 

% figure
