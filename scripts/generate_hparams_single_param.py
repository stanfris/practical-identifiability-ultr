import random
import numpy as np
import itertools

hyperparameter_file = 'scripts/hparams_varying_single_experiment.txt'
hyperparameter_file_main = 'scripts/hparams_varying_single_experiment_main.txt'

parameters = {
    'experiment': ['test_baidu'],
    'data': ['Custom_dataset_deep'],
    'relevance': ['deep'],
    'logging_policy_ranker': ['ordered'],
    'relevance_tower': ['deeper'],
    'policy_strength': [1],
    'policy_temperature': [0.0],
    'random_state': [2021],
    'param_shift': [-3.0, -1.5, 0.0, 1.5, 3.0],
    'freeze_bias_tower': [True],
    'single_param': [True],
    'param_idx': [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
       86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    'logging_policy_sampler': ['e_greedy'],
    'use_baidu': [True],
    'baidu_subset': ['train_Baidu_ULTRA_part1_media_type_position.npz'],
}

# Helper function to format a line nicely
def format_line(params: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())

# Main experiment combinations
main_keys = [
    "experiment", "data", "relevance", "logging_policy_ranker",
    "relevance_tower", "policy_strength", "policy_temperature",
    "random_state", "logging_policy_sampler", "use_baidu", "baidu_subset"
]

# Param shift combinations
shift_keys = main_keys + [
    "param_shift", "param_idx", "single_param", "freeze_bias_tower"
]

# Write main experiments
with open(hyperparameter_file_main, "w") as f_main:
    for combo in itertools.product(*(parameters[k] for k in main_keys)):
        params = dict(zip(main_keys, combo))
        # construct the dataset name
        params['test_dataset_name'] = f"test_dataset_" + "_".join([
            f"policy_temperature{params.get('policy_temperature')}",
            f"num_queries{params.get('num_queries')}",
            f"D{params.get('D')}",
            f"s_doc{params.get('s_doc')}",
            ".pkl"
        ])
        params['test_click_dataset_name'] = params['test_dataset_name'].replace("dataset", "click_dataset")
        f_main.write(format_line(params) + "\n")
num_jobs_main = sum(1 for _ in itertools.product(*(parameters[k] for k in main_keys)))

# Write parameter-shift experiments
with open(hyperparameter_file, "w") as f:
    for combo in itertools.product(*(parameters[k] for k in shift_keys)):
        params = dict(zip(shift_keys, combo))
        params['test_dataset_name'] = f"test_dataset_" + "_".join([
            f"policy_temperature{params.get('policy_temperature')}",
            f"num_queries{params.get('num_queries')}",
            f"D{params.get('D')}",
            f"s_doc{params.get('s_doc')}",
            ".pkl",
        ])
        params['test_click_dataset_name'] = params['test_dataset_name'].replace("dataset", "click_dataset")
        f.write(format_line(params) + "\n")
num_jobs = sum(1 for _ in itertools.product(*(parameters[k] for k in shift_keys)))

a_100 = False
if a_100:

    job_script_main = f"""#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Test-Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00

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
#SBATCH --time=00:30:00

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

else:
    job_script_main = f"""#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=Test-Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

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

#SBATCH --partition=rome
#SBATCH --job-name=Test-Run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

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
