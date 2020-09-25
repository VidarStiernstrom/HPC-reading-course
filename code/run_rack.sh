module add ABINIT/8.10.3 PETSc/3.12.4-intel-2019b

export PETSC_ARCH=

make clean
make bin/sim_adv_ts

for ncores in 4 8 20
do
	echo $cores
	for (( i=1; i<2; i++ ))
	do
		sbatch -p node -n $ncores batch.sh $ncores
	done
done