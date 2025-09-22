#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --error=/gpfs/home5/sfris1/two-towers-confounding-project/results/test_varying/.submitit/%j/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=varying
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home5/sfris1/two-towers-confounding-project/results/test_varying/.submitit/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=3600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /gpfs/home5/sfris1/two-towers-confounding-project/results/test_varying/.submitit/%j/%j_%t_log.out --error /gpfs/home5/sfris1/two-towers-confounding-project/results/test_varying/.submitit/%j/%j_%t_log.err /gpfs/home5/sfris1/two-towers-confounding-project/.venv/bin/python -u -m submitit.core._submit /gpfs/home5/sfris1/two-towers-confounding-project/results/test_varying/.submitit/%j
