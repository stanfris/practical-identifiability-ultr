# Diagnosing Identifiability in Two-Tower Models for Unbiased Learning to Rank

Repository for the paper `Diagnosing Identifiability in Two-Tower Models for Unbiased
Learning to Rank`.

## Setup:
This project uses [Mamba](https://mamba.readthedocs.io/en/latest/index.html) for environment management. To set up a Python environment, run:
```bash
mamba env create -f environment.yml
mamba activate two-tower-confounding
```

## Data
The project uses synthetic datasets, which are generated while running experiments, as well as Real-Wolrd Click datsets. We make use of the BAIDU ULTR dataset, which can be downloaded from [HuggingFace](https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr). By default, our code expects raw .zip files under `~/ltr_datasets/download/`. But you can change the directory to your preference under: `config/config.yaml`. 

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

The contents of our paper consist of 3 primary components, the deterministic setting, a synthetic feature setting and evaluation on Baidu_ULTR. 

For running identifiability analysis, experiments are split into main and varying jobs. This allows the computation of identifiability to be parallalized across compute nodes. We include the hyperparameter configurations (and ability to generate those) under  `generate_hparams_deterministic_custom_data.py`, `generate_hparams_synthetic_feature_separation.py` and `generate_hparams_baidu_ULTR_pos_bias.py`. An example of the configuration is shown below:

```python
    'policy_temperature': [0.0],
    'param_shift': [-3.0, -1.5, 0.0, 1.5, 3.0],
    'freeze_bias_tower': [True],
    'single_param': [True],
    'param_idx': [0],
    'logging_policy_sampler': ['e_greedy'],
    'baidu_subset': ['train_Baidu_ULTRA_part1_media_type_position.npz'],
    'bias_type': ["media_type"]
```

Here, users can change what types of parameters they wish to and vary. The most important parameters to consider here are `param_shift`, `param_idx` and `bias_type`. These indicate which parameters should be shifted and what amount. After selecting appropriate parameters, `run_wrapper_single_param.sh` can be used to generate full outputs, submitting all slurm jobs with appropriate dependencies. 

## Results
We publish all simulation results under `results/`, organized by the experimental script that created the results. All code for our visualizations is under `notebooks/`. The primary notebooks to consider here are under RQ1, RQ2 and RQ3. These currently run all visualizations present in the paper for each of our (without requiring users to run any additional code), but can also be easily adapted to show results of new experiments. 

## Credit
This repository has been adapted from the repository developed by Hager et al. (2025).

### References
Hager, P., Zoeter, O., & de Rijke, M. (2025, July). Unidentified and Confounded? Understanding Two-Tower Models for Unbiased Learning to Rank. In Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR) (pp. 347-357).
