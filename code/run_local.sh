
make clean
make init

export PETSC_ARCH=

make bin/main_aco

mpirun -n 4 bin/main_aco 1 -ksp_type pipefgmres -ksp_monitor_true_residual 	-ksp_converged_reason -ksp_view_final_residual -citations out$count.bib
