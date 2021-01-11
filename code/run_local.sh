
make clean
make init

make bin/adv_1D

# mpirun -n 2 bin/adv_1D

# mpirun -n 3 bin/adv_1D

# mpirun -n 4 bin/adv_1D

# mpirun -n 5 bin/adv_1D

# mpirun -n 6 bin/adv_1D

# mpirun -n 7 bin/adv_1D

# mpirun -n 2 bin/adv_1D -ksp_monitor
mpirun -n 2 bin/adv_1D -ksp_monitor_true_residual	-ksp_converged_reason	-pc_sor_backward	-pc_sor_omega 1



# mpirun -n 4 bin/first_order_ode
# 
# mpirun -n 8 bin/first_order_ode