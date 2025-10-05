import random
import numpy as np
import itertools

hyperparameter_file = 'scripts/hparams_varying_single_experiment.txt'
hyperparameter_file_main = 'scripts/hparams_varying_single_experiment_main.txt'



parameters = {
    'experiment': ['test'],
    'data': ['Custom_dataset'],
    'relevance': ['linear'],
    'logging_policy_ranker': ['linear'],
    'relevance_tower': ['linear'],
    'policy_strength': [1],
    'policy_temperature': [0.0],
    'random_state': [2021],
    'param_shift': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    'freeze_bias_tower': [True],
    'single_param': [True],
    'param_idx': [0],
    'logging_policy_sampler': ['e_greedy'],
    'save_test_datasets': [False],
    'load_test_datasets': [False],
}

# Helper function to format a line nicely
def format_line(params: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())

# Main experiment combinations
main_keys = [
    "experiment", "data", "relevance", "logging_policy_ranker",
    "relevance_tower", "policy_strength", "policy_temperature",
    "random_state", "logging_policy_sampler",
    "save_test_datasets", "load_test_datasets"
]

# Param shift combinations
shift_keys = main_keys + [
    "param_shift", "param_idx", "single_param", "freeze_bias_tower"
]

# Write main experiments
with open(hyperparameter_file_main, "w") as f_main:
    for combo in itertools.product(*(parameters[k] for k in main_keys)):
        params = dict(zip(main_keys, combo))
        f_main.write(format_line(params) + "\n")
num_jobs_main = sum(1 for _ in itertools.product(*(parameters[k] for k in main_keys)))

# Write parameter-shift experiments
with open(hyperparameter_file, "w") as f:
    for combo in itertools.product(*(parameters[k] for k in shift_keys)):
        params = dict(zip(shift_keys, combo))
        f.write(format_line(params) + "\n")
num_jobs = sum(1 for _ in itertools.product(*(parameters[k] for k in shift_keys)))


job_script_main = f"""#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Test-Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00

#SBATCH --array=1-{num_jobs_main}
#SBATCH --output=slurm/slurm_array_testing_%A_%a.out

module purge
module load 2023
module load CUDA/12.4.0
source .venv/bin/activate

HPARAMS_FILE="scripts/hparams_varying_single_experiment_main.txt"

srun python main.py -m $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
"""

with open("scripts/test_varying_array_main_single_param.job", "w") as f:
    f.write(job_script_main)

job_script = f"""#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Test-Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00

#SBATCH --array=1-{num_jobs}
#SBATCH --output=slurm/slurm_array_testing_%A_%a.out

module purge
module load 2023
module load CUDA/12.4.0
source .venv/bin/activate

HPARAMS_FILE="scripts/hparams_varying_single_experiment.txt"

srun python varying.py -m $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
"""

with open("scripts/test_varying_array_single_param.job", "w") as f:
    f.write(job_script)