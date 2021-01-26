
make clean
make init

make bin/main_aco

mpirun -n 8 bin/main_aco -ksp_monitor_true_residual 	-ksp_converged_reason 	-ksp_view_final_residual 
