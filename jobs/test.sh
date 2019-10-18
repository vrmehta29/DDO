#!/bin/bash
#
SBATCH --job-name=testDDO
#SBATCH --array=0-12
SBATCH --time=2:00:00
#SBATCH --ntasks=1
SBATCH --cpus-per-task=4
SBATCH --mem=4G
SBATCH --gres gpu:1
#SBATCH --gres-flags=enforce-binding

#LISTOFMOVS=/fastscratch/bgeuther/listofmovs.txt
#readarray -t ALLMOVIES < ${LISTOFMOVS}
#MOVIE=${ALLMOVIES[$SLURM_ARRAY_TASK_ID]}

module load singularity

cd /home/c-mehtav/DDO/

singularity run --nv /gpfs/ctgs0/fastscratch/bgeuther/tf19.01.simg python myexperiment1.py #--input_movie ${MOVIE}

