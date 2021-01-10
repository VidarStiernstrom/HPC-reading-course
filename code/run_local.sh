
make clean
make init

make bin/first_order_ode

# mpirun -n 2 bin/adv_1D

# mpirun -n 3 bin/adv_1D

# mpirun -n 4 bin/adv_1D

# mpirun -n 5 bin/adv_1D

# mpirun -n 6 bin/adv_1D

# mpirun -n 7 bin/adv_1D

# mpirun -n 2 bin/adv_1D -ksp_monitor
mpirun -n 2 bin/first_order_ode	-ksp_converged_reason -ksp_monitor



# mpirun -n 4 bin/first_order_ode
# 
# mpirun -n 8 bin/first_order_ode