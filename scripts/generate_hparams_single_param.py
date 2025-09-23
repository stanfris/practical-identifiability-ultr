import random
import numpy as np
hyperparameter_file = 'scripts/hparams_varying_single_experiment.txt'
hyperparameter_file_main = 'scripts/hparams_varying_single_experiment_main.txt'


parameters = {
    'experiment': ['test_single_varying'],
    'datasets': ['Custom_dataset'],
    'relevance': ['linear'],
    'logging_policy_ranker': ['linear'],
    'relevance_tower': ['linear'],
    'policy_strength': [1],
    'policy_temperature': [0.333, 0.667, 1.0],
    'random_state': [2021],
    'param_shift': [2.0],
    'freeze_bias_tower': [True],
    'single_param' : [True],
    'param_idx' : np.arange(0,10,1).tolist(), # only if single_param is True
}

with open(hyperparameter_file, 'w') as f:
    with open (hyperparameter_file_main, 'w') as f2:
        for experiment in parameters['experiment']:
            for dataset in parameters['datasets']:
                for relevance in parameters['relevance']:
                    for logging_policy_ranker in parameters['logging_policy_ranker']:
                        for relevance_tower in parameters['relevance_tower']:
                            for policy_strength in parameters['policy_strength']:
                                for policy_temperature in parameters['policy_temperature']:
                                    for random_state in parameters['random_state']:
                                        line = (
                                            f"experiment={experiment} "
                                            f"data={dataset} "
                                            f"relevance={relevance} "
                                            f"logging_policy_ranker={logging_policy_ranker} "
                                            f"relevance_tower={relevance_tower} "
                                            f"policy_strength={policy_strength} "
                                            f"policy_temperature={policy_temperature} "
                                            f"random_state={random_state} "
                                            )
                                        f2.write(line + "\n")
                                        for freeze_bias_tower in parameters['freeze_bias_tower']:
                                            for param_shift in parameters['param_shift']:
                                                for param_idx in parameters['param_idx']:
                                                    for single_param in parameters['single_param']:
                                                        line = (
                                                            f"experiment={experiment} "
                                                            f"data={dataset} "
                                                            f"relevance={relevance} "
                                                            f"logging_policy_ranker={logging_policy_ranker} "
                                                            f"relevance_tower={relevance_tower} "
                                                            f"policy_strength={policy_strength} "
                                                            f"policy_temperature={policy_temperature} "
                                                            f"random_state={random_state} "
                                                            f"param_shift={param_shift} "
                                                            f"param_idx={param_idx} "
                                                            f"single_param={str(single_param)} "
                                                            f"freeze_bias_tower={str(freeze_bias_tower)} "
                                                        )
                                                        f.write(line + "\n")