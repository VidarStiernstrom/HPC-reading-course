
make clean
make init

export PETSC_ARCH=

make bin/main_aco

for method in pipefgmres fgmres fbcgsr
do
	mpirun -n 4 bin/main_aco 1 -ksp_type $method -ksp_monitor 	-ksp_converged_reason -ksp_view_final_residual
done

for method in fgmres pipefgmres gmres bcgs bcgsl lgmres dgmres pgmres
do
	mpirun -n 4 bin/main_aco 0 -ksp_type $method -ksp_monitor 	-ksp_converged_reason -ksp_view_final_residual
done