
make clean
make init

make bin/first_order_ode

mpirun -n 8 bin/first_order_ode -ksp_monitor -malloc_dump

# mpirun -n 4 bin/first_order_ode
# 
# mpirun -n 8 bin/first_order_ode