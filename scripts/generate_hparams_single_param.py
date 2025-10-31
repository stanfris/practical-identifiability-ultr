import random
import numpy as np
import itertools

hyperparameter_file = 'scripts/hparams_varying_single_experiment.txt'
hyperparameter_file_main = 'scripts/hparams_varying_single_experiment_main.txt'

parameters = {
    'experiment': ['deep_target_label_ranked'],
    'data': ['Custom_dataset_deep'],
    'relevance': ['deep'],
    'logging_policy_ranker': ['ordered'],
    'relevance_tower': ['deep'],
    'policy_strength': [1],
    'policy_temperature': [0, 0.333, 0.667],
    'random_state': [2021],
    'param_shift': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    'freeze_bias_tower': [True],
    'single_param': [True],
    'param_idx': [0],
    'logging_policy_sampler': ['e_greedy'],
    'save_test_datasets': [True],
    'load_test_datasets': [True],
    'num_queries': [1, 10, 20],
    'docs_per_group': [10],
    'D': [2],
    'label_type': ['deep_overlap'],
    's_doc' : [-0.2, 0.3]
}

# Helper function to format a line nicely
def format_line(params: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())

# Main experiment combinations
main_keys = [
    "experiment", "data", "relevance", "logging_policy_ranker",
    "relevance_tower", "policy_strength", "policy_temperature",
    "random_state", "logging_policy_sampler",
    "save_test_datasets", "load_test_datasets",
    "num_queries", "docs_per_group", "D", "label_type", "s_doc"
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