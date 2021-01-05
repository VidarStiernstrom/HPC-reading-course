
make clean
make init

make bin/adv_1D

mpirun -n 2 bin/adv_1D



# mpirun -n 4 bin/first_order_ode
# 
# mpirun -n 8 bin/first_order_ode