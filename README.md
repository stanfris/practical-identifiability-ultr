# Diagnosing Identifiability in Two-Tower Models for Unbiased Learning to Rank

Repository for the paper `Diagnosing Identifiability in Two-Tower Models for Unbiased
Learning to Rank`, in preparation for SIGIR 2026.

## Setup:
This project uses [Mamba](https://mamba.readthedocs.io/en/latest/index.html) for environment management. To set up a Python environment, run:
```bash
mamba env create -f environment.yml
mamba activate two-tower-confounding
```

## Data
The project uses synthetic datasets, which are generated while running experiments, as well as Real-Wolrd Click datsets. We make use of the BAIDU ULTR dataset, which can be downloaded from [HuggingFace]https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr. By default, our code expects raw .zip files under `~/ltr_datasets/download/`. But you can change the directory to your preference under: `config/config.yaml`. 

## Experiments
We manage our experiments with Hydra, with all code configuations under `config/`. We also provide scripts for each experiment under `scripts/`. To begin, make sure all bash scripts are executable:
```bash
chmod +x scripts/*.sh  
```

To run, e.g., a well-specified linear two-tower model trained on users following a linear relevance behavior, you can use:
```bash
./scripts/2_model_fit_linear.sh
```
Optionally, you can launch the job on a SLURM cluster to distribute training jobs:
```bash
./scripts/2_model_fit_linear.sh +launcher=slurm
```
You can edit the launch parameters for SLURM under: `config/launcher/slurm.yaml`.

For running identifiability analysis, experiments are split into main and varying jobs. This allows the computation of identifiability to be parallalized across compute nodes. To do so, consider the hyperparameter generation file in `generate_hparams_single_param.py`.     

```
    'policy_temperature': [0.0],
    'param_shift': [-3.0, -1.5, 0.0, 1.5, 3.0],
    'freeze_bias_tower': [True],
    'single_param': [True],
    'param_idx': [0],
    'logging_policy_sampler': ['e_greedy'],
    'baidu_subset': ['train_Baidu_ULTRA_part1_media_type_position.npz'],
    'bias_type': ["media_type"]
```

Here, users can change what types of parameters they wish to and vary. The most important parameters to consider here are `param_shift`, `param_idx` and `bias_type`. These indicate which parameters should be shifted and what amount. 

## Results
We publish all simulation results under `results/`, orgainzed by the experimental script that created the results. All code for our visualizations is under `notebooks/`. The primary notebooks to consider here are under a1, a2 and a3. These currently run all visualizations present in the paper, but can be easily adapted to show results for new datasets. 

## Thanks
This repository has been adapted from the repository developed by Hager et al. (2025).

### Reference
```
Hager, P., Zoeter, O., & de Rijke, M. (2025, July). Unidentified and Confounded? Understanding Two-Tower Models for Unbiased Learning to Rank. In Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR) (pp. 347-357).
```