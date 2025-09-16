import random

import hydra
import numpy as np
import optax
import pandas as pd
import torch
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from two_tower_confounding.metrics import NDCG, MRR, NegativeLogLikelihood
from two_tower_confounding.models.two_tower import TwoTowerModel
from two_tower_confounding.simulation.simulator import Simulator
from two_tower_confounding.trainer import Trainer
from two_tower_confounding.utils import np_collate
import wandb


def train_val_test_datasets(config: DictConfig):
    """
    Simulate clicks on the original train, val, and test datasets of a LTR dataset.
    Note that this should not be used when using embedding towers, as test query-doc pairs
    do not appear during training. To test embedding towers, use the cross-validation method.
    """

    #### LTR Datasets ####
    dataset = instantiate(config.data.dataset)
    preprocessor = instantiate(config.data.preprocessor)

    train_dataset = preprocessor(dataset.load("train"))
    val_dataset = preprocessor(dataset.load("val"))
    test_dataset = preprocessor(dataset.load("test"))

    #### Simulate user clicks ####
    logging_policy_ranker = instantiate(config.logging_policy_ranker)
    logging_policy_ranker.fit(train_dataset)
    logging_policy_sampler = instantiate(config.logging_policy_sampler)

    simulator = Simulator(
        logging_policy_ranker=logging_policy_ranker,
        logging_policy_sampler=logging_policy_sampler,
        bias_strength=config.bias_strength,
        random_state=config.random_state,
    )

    train_click_dataset = simulator(train_dataset, config.train_clicks)
    val_click_dataset = simulator(val_dataset, config.val_clicks)
    test_click_dataset = simulator(test_dataset, config.test_clicks)

    return train_click_dataset, val_click_dataset, test_click_dataset, test_dataset


def cross_val_datasets(config: DictConfig):
    """
    Simulate train, val, test clicks on the train partition of a LTR dataset.
    This is used to simulate clicks in a cross-validation setting for embedding towers.
    """

    #### LTR Datasets ####
    dataset = instantiate(config.data.dataset)
    preprocessor = instantiate(config.data.preprocessor)

    rating_dataset = preprocessor(dataset.load("train"))

    #### Simulate user clicks ####
    logging_policy_ranker = instantiate(config.logging_policy_ranker)
    logging_policy_ranker.fit(rating_dataset)
    logging_policy_sampler = instantiate(config.logging_policy_sampler)

    simulator = Simulator(
        logging_policy_ranker=logging_policy_ranker,
        logging_policy_sampler=logging_policy_sampler,
        bias_strength=config.bias_strength,
        random_state=config.random_state,
    )

    total_clicks = config.train_clicks + config.val_clicks + config.test_clicks
    click_dataset = simulator(rating_dataset, total_clicks)

    train_click_dataset, val_click_dataset = torch.utils.data.random_split(
        click_dataset,
        lengths=[config.train_clicks, config.val_clicks + config.test_clicks],
    )
    val_click_dataset, test_click_dataset = torch.utils.data.random_split(
        val_click_dataset,
        lengths=[config.val_clicks, config.test_clicks],
    )

    return train_click_dataset, val_click_dataset, test_click_dataset, rating_dataset


@hydra.main(version_base="1.3", config_path="config/", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="stanfris2-0-university-of-amsterdam",
            # Set the wandb project where this run will be logged.
            project="project_AI",
            name=f"Two-Tower_bs{config.bias_strength}_pw{config.use_propensity_weighting}_rs{config.random_state}_cv{config.use_cross_validation}",
            # Track hyperparameters and run metadata.
            config={
                "architecture": "Two-Tower",
                "dataset": "Custom_dataset",
                "bias_strength": config.random_state,
                "use_propensity_weighting": config.use_propensity_weighting,
                "random_state": config.random_state,
                "use_cross_validation": config.use_cross_validation,
                "train_clicks": config.train_clicks,
                "val_clicks": config.val_clicks,
                "test_clicks": config.test_clicks,
            },
        )

    if config.use_cross_validation:
        train_click_dataset, val_click_dataset, test_click_dataset, test_dataset = (
            cross_val_datasets(config)
        )
    else:
        train_click_dataset, val_click_dataset, test_click_dataset, test_dataset = (
            train_val_test_datasets(config)
        )

    train_click_loader = DataLoader(
        train_click_dataset,
        batch_size=512,
        collate_fn=np_collate,
        shuffle=True,
    )
    val_click_loader = DataLoader(
        val_click_dataset,
        batch_size=512,
        collate_fn=np_collate,
    )
    test_click_loader = DataLoader(
        test_click_dataset,
        batch_size=512,
        collate_fn=np_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        collate_fn=np_collate,
    )

    #### Two-tower model ####
    # Use hydra partial instantiation to pass rngs and dataset properties:


    trainer = Trainer(
        optimizer=optax.adamw(learning_rate=0.001),
        metrics={
            "ndcg": NDCG(),
            "ndcg@3": NDCG(top_k=3),
            "ndcg@5": NDCG(top_k=5),
            "ndcg@10": NDCG(top_k=10),
            "mrr@10": MRR(top_k=10),
        },
        click_metrics={
            "nll": NegativeLogLikelihood(),
        },
        epochs=50,
        run=run,
    )

    # load model
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    val_df = trainer.test_clicks(model, val_click_loader)

    test_click_df = trainer.test_clicks(model, test_click_loader)
    test_rel_df = trainer.test_relevance(model, test_loader)
    test_lp_df = trainer.test_logging_policy(test_click_loader)
    test_df = pd.concat([test_click_df, test_rel_df], axis=1)

    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    test_lp_df.to_csv("test_logging_policy.csv", index=False)

    bias_df = trainer.get_position_bias(model, test_dataset.n_positions)
    bias_df.to_csv("bias.csv", index=False)
    relevance_df = trainer.get_relevance_scores(model, test_dataset.n_features)
    relevance_df.to_csv("relevance.csv", index=False)

if __name__ == "__main__":
    main()
