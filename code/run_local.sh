
make clean
make init

make bin/main_ref

mpirun -n 2 bin/main_ref  -ksp_converged_reason 	-ksp_view_final_residual
