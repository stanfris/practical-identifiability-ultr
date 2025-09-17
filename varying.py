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

from two_tower_confounding.models.towers import *
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
    def load_two_tower(config, dataset, bias_path="bias.csv", relevance_path="relevance.csv", param_shift=0.0) -> TwoTowerModel:
        """Rebuild TwoTowerModel and load parameters from CSV files."""

        # 1. Rebuild fresh towers with Hydra
        bias_tower = instantiate(
            config.bias_tower,
            positions=dataset.n_positions,
        )
        relevance_tower = instantiate(
            config.relevance_tower,
            query_doc_features=dataset.n_features,
            query_doc_pairs=dataset.n_documents,
        )

        rngs = nnx.Rngs(config.random_state)
        model = TwoTowerModel(
            bias_tower=bias_tower(rngs=rngs),
            relevance_tower=relevance_tower(rngs=rngs),
            use_propensity_weighting=config.use_propensity_weighting,
        )

        # 2. Load saved CSVs
        bias_df = pd.read_csv(bias_path)
        relevance_df = pd.read_csv(relevance_path)

        bias_values = bias_df["examination"].to_numpy()
        if param_shift != 0.0:
            print(bias_values)
            bias_values += param_shift
            print(bias_values)
            print(f"⚠️  Shift bias tower parameters by {param_shift:.4f} ⚠️")

        relevance_values = relevance_df["relevance"].to_numpy()

        # 3. Inject parameters depending on tower type
        # --- Bias ---
        if isinstance(model.bias_tower, EmbeddingBiasTower):
            model.bias_tower.embedding.embedding.value = bias_values.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported bias tower type: {type(model.bias_tower)}")

        # --- Relevance ---
        if isinstance(model.relevance_tower, EmbeddingRelevanceTower):
            model.relevance_tower.embeddings.embedding.value = relevance_values.reshape(-1, 1)
        elif isinstance(model.relevance_tower, LinearRelevanceTower):
            model.relevance_tower.layer.kernel.value = relevance_values.reshape(-1, 1)
        elif isinstance(model.relevance_tower, DeepRelevanceTower):
            model.relevance_tower.output.kernel.value = relevance_values.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported relevance tower type: {type(model.relevance_tower)}")
    
        print("Inside tower after shift:", model.bias_tower.embedding.embedding.value[:10])
        print(f"✅ Loaded parameters from {bias_path} and {relevance_path}")
        return model

    model = load_two_tower(config, test_dataset, bias_path="bias.csv", relevance_path="relevance.csv", param_shift=3.0)

    print("bias before retraining: ", trainer.get_position_bias(model, test_dataset.n_positions))
    print("relevance before retraining: ", trainer.get_relevance_scores(model, test_dataset.n_features))

    trainer = Trainer(
        optimizer=base_optimizer,
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
        freeze_bias_tower=config.freeze_bias_tower,
    )
    
    trainer.train(model, train_click_loader, val_click_loader)
    val_df = trainer.test_clicks(model, val_click_loader)

    test_click_df = trainer.test_clicks(model, test_click_loader)
    test_rel_df = trainer.test_relevance(model, test_loader)
    test_lp_df = trainer.test_logging_policy(test_click_loader)

    print("bias after retraining: ", trainer.get_position_bias(model, test_dataset.n_positions))
    print("relevance after retraining: ", trainer.get_relevance_scores(model, test_dataset.n_features))

if __name__ == "__main__":
    main()
