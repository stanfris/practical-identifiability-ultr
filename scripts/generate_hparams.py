import random
import numpy as np
hyperparameter_file = 'scripts/hparams_varying_experiment.txt'
hyperparameter_file_main = 'scripts/hparams_varying_experiment_main.txt'


parameters = {
    'experiment': ['test_varying'],
    'datasets': ['Custom_dataset'],
    'relevance': ['linear'],
    'logging_policy_ranker': ['linear'],
    'relevance_tower': ['linear'],
    'policy_strength': [1],
    'policy_temperature': [0, 0.333, 0.667, 1.0],
    'random_state': [2021],
    'param_shift': np.arange(-5.0, 6.0, 1.0).tolist(),
    'freeze_bias_tower': [True],
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
                                                line = (
                                                    f"experiment={experiment} "
                                                    f"data={dataset} "
                                                    f"relevance={relevance} "
                                                    f"logging_policy_ranker={logging_policy_ranker} "
                                                    f"relevance_tower={relevance_tower} "
                                                    f"policy_strength={policy_strength} "
                                                    f"policy_temperature={policy_temperature} "
                                                    f"random_state={random_state} "
                                                    f"param_shift={param_shift}"
                                                    f" freeze_bias_tower={str(freeze_bias_tower)}"
                                                )
                                                f.write(line + "\n")


                                