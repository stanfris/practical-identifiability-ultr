# Understanding Two-Tower Models for Unbiased Learning to Rank

Repository for the paper `Unidentified and Confounded? Understanding Two-Tower Models for Unbiased Learning to Rank` accepted at ICTIR 2025.

## Setup:
This project uses [Mamba](https://mamba.readthedocs.io/en/latest/index.html) for environment management. To set up a Python environment, run:
```bash
mamba env create -f environment.yml
mamba activate two-tower-confounding
```

## Data
The project uses classic LTR datasets as the foundation for its click simulation. The code supports: [MSLR30K](https://www.microsoft.com/en-us/research/project/mslr/), and [Yahoo! Webscope](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&guccounter=1), which have to be manually downloaded as of 2025. By default, our code expects raw .zip files under `~/ltr_datasets/download/`. But you can change the directory to your preference under: `config/config.yaml`. 

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

## Results
We publish all simulation results under `results/`, orgainzed by the experimental script that created the results. All code for our visualizations is under `notebooks/`.

### Reference
```
@inproceedings{Hager2025TwoTowers,
  author = {Philipp Hager and Onno Zoeter and Maarten de Rijke},
  title = {Unidentified and Confounded? Understanding Two-Tower Models for Unbiased Learning to Rank},
  booktitle = {Proceedings of the 11th ACM SIGIR / 15th International Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR`25)},
  organization = {ACM},
  year = {2025},
}
```

### License
This repository uses the [MIT License](https://github.com/philipphager/two-tower-confounding/blob/main/LICENSE).
