#!/bin/bash -l
#SBATCH -A snic2020-5-358
#SBATCH -t 0-00:05:00
#SBATCH -J p_ts/

module load ABINIT/8.10.3 PETSc/3.12.4-intel-2019b
${PETSC_DIR}/lib/petsc/bin/petscmpiexec -n $1 bin/sim_adv_ts