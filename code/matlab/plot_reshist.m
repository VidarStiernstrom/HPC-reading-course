clear
close all

data_dir = "../data/aco2D/";

method = "MG";
krylovs_mg = ["pipefgmres","fgmres","fbcgsr"];


Nx = 2*384;
Ny = 2*384;
size = 4;
nlevels = 4;

figure 
hold on

fprintf("-------- Multigrid preconditioner --------\n")
for methodidx = 1:numel(krylovs_mg)
    krylov = krylovs_mg(methodidx);
    filename = char(data_dir + method + "_" + krylov + "_Nx" + num2str(Nx) +...
        "_Ny" + num2str(Ny) + "_size" + num2str(size) + "_nlevels" + num2str(nlevels));
    data = PetscBinaryRead(filename);

    nits = data(end-1);
    elaptime = data(end);
    fprintf("Method: %s, elapsed time: %f\n",krylov,elaptime)
    reshist = data(1:nits);

    plot(reshist,'Linewidth',2)
    set(gca,'YScale','log')
    box on
    grid on
end



method = "stand";
krylovs_stand = ["fgmres", "pipefgmres", "gmres", "bcgsl", "lgmres", "dgmres", "pgmres"];

Nx = 2*384;
Ny = 2*384;
size = 4;

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

    plot(reshist)
    set(gca,'YScale','log')
    box on
    grid on
end

legend([krylovs_mg,krylovs_stand])