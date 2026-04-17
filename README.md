# Diagnosing Identifiability in Two-Tower Models for Unbiased Learning to Rank

Repository for the paper `Diagnosing Identifiability in Two-Tower Models for Unbiased
Learning to Rank`.

This repository extends and is based on the implementation from “Understanding Two-Tower Models for Unbiased Learning to Rank” (Hager et al., 2025), found [Here](https://github.com/philipphager/two-tower-confounding).
## Setup:
This project uses [Mamba](https://mamba.readthedocs.io/en/latest/index.html) for environment management. To set up a Python environment, run:
```bash
mamba env create -f environment.yml
mamba activate two-tower-confounding
```

## Data
The project uses synthetic datasets, which are generated while running experiments, as well as Real-Wolrd Click datsets. We make use of the BAIDU ULTR dataset, which can be downloaded from [HuggingFace](https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr). By default, our code expects raw .zip files under `~/ltr_datasets/download/`. But you can change the directory to your preference under: `config/config.yaml`. 

After files for BAIDU ULTR are included, run `notebooks/parse_Baidu_ULTR.ipynb` to parse dataset files into a useable format, and include them in the `~/ltr_datasets` folder.


## Experiments
We manage our experiments with Hydra, with all code configuations under `config/`. We also provide scripts for each experiment under `scripts/`. To begin, make sure all bash scripts are executable:
```bash
chmod +x scripts/*.sh  
```

To run, e.g., a well-specified linear two-tower model trained on users following a deep relevance behavior with different policy temperatures, you can use:
```bash
./scripts/1_example.sh
```
Optionally, you can launch the job on a SLURM cluster to distribute training jobs:
```bash
./scripts/1_example.sh +launcher=slurm
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

To reproduce the results we generated, please use the python files to generate the hyperparameter configurations, and run each by running the respective 'run_wrapper' bash file. 

## Results
We publish all simulation results under `results/`, organized by the experimental script that created the results. All code for our visualizations is under `notebooks/`. The primary notebooks to consider here are under RQ1, RQ2 and RQ3. These currently run all visualizations present in the paper for each of our (without requiring users to run any additional code), but can also be easily adapted to show results of new experiments. 

### Running An Identifiability Test
For Baidu-ULTR position-bias identifiability experiments, the main configuration point is [scripts/generate_hparams_baidu_ULTR_pos_bias.py](/Users/stanf/Documents/Uni/Project_AI/two-tower-confounding/scripts/generate_hparams_baidu_ULTR_pos_bias.py). This script writes the Hydra sweep files used by the baseline run in `main.py` and the shifted runs in `varying.py`.

To test a different model or dataset, edit the `parameters` dictionary in that script. The fields that usually matter are:
- `relevance_tower`: chooses the relevance model architecture. The repo currently provides `linear`, `deep`, and `deeper` under [config/relevance_tower](/Users/stanf/Documents/Uni/Project_AI/two-tower-confounding/config/relevance_tower).
- `baidu_subset`: chooses which parsed Baidu dataset file to use, for example `train_Baidu_ULTRA_part1.npz` or `train_Baidu_ULTRA_very_short.npz`. These files are expected under `~/ltr_datasets` by default. For different dataset families, other configs/loaders can be implemented.
- `param_idx`: chooses which positions are perturbed in the identifiability sweep. For position bias, each `param_idx` corresponds to one position parameter in the bias tower. Set this list to the positions you want to test, for example `range(20)` for 20 positions or a shorter list if dataset only covers fewer positions.
- `param_shift`: chooses the magnitudes of the perturbations applied at each tested position. The default script uses `[-3.0, -1.5, 0.0, 1.5, 3.0]`.

After editing the generator, launch the jobs with:

```bash
python scripts/generate_hparams_baidu_ULTR_pos_bias.py
./scripts/run_wrapper_baidu_ULTR_pos_bias.sh
```

The wrapper first submits the baseline model job and then submits the parameter-shift sweep with a dependency on the baseline. The outputs are written under `results/Baidu_ULTR_position/...`, with one result folder per model/dataset configuration.

Once those jobs are finished, run the CLI on the produced result folder and the matching parsed dataset file:

```bash
python scripts/identifiability_cli.py
```

The script will ask for:
- the results folder path
- the parsed Baidu `.npz` data path
- the plot output path
- the summary CSV path

For the output paths, pressing Enter accepts the default path under `tmp/`, and generates results for the Baidu-ULTR dataset.
- results folder: `results/Baidu_ULTR_position/baidu_subset=train_Baidu_ULTRA_part1.npz,data=Custom_dataset_deep,experiment=Baidu_ULTR_position,logging_policy_ranker=ordered,logging_policy_sampler=e_greedy,policy_temperature=0.0,relevance=deep,relevance_tower=deeper`
- data path: `../ltr_datasets/train_Baidu_ULTRA_part1.npz`

You can still skip the prompts by passing flags directly:

```bash
python scripts/identifiability_cli.py \
  --folder-path "results/Baidu_ULTR_position/baidu_subset=train_Baidu_ULTRA_part1.npz,data=Custom_dataset_deep,experiment=Baidu_ULTR_position,logging_policy_ranker=ordered,logging_policy_sampler=e_greedy,policy_temperature=0.0,relevance=deep,relevance_tower=deeper" \
  --data-path "../ltr_datasets/train_Baidu_ULTRA_part1.npz" \
  --output-path "tmp/identifiability_train_Baidu_ULTRA_part1.pdf" \
  --summary-csv "tmp/identifiability_train_Baidu_ULTRA_part1.csv"
```

The CLI reads the sweep CSVs in the chosen results folder, attaches sample counts from the selected `.npz` dataset, prints the per-position identifiability table, and saves a summary CSV plus plot for that exact model/dataset combination.


### Reference
```
@inproceedings{Fris2026DiagnosingIdentifiability,
  author = {Stan Fris and Philipp Hager},
  title = {Diagnosing Identifiability in Two-Tower Models for Unbiased Learning to Rank},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`26)},
  organization = {ACM},
  year = {2026},
}
```

### License
This repository uses the [MIT License](https://github.com/stanfris/practical-identifiability-ultr/blob/main/LICENSE).
