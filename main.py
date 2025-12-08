from importlib.resources import path
import random

import hydra
import numpy as np
import optax
import pandas as pd
import torch
from flax import config, nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from two_tower_confounding.models.towers import *
from two_tower_confounding.metrics import NDCG, MRR, NegativeLogLikelihood
from two_tower_confounding.models.two_tower import TwoTowerModel
from two_tower_confounding.trainer import Trainer
from two_tower_confounding.utils import np_collate, train_val_test_datasets, load_custom_click_dataset
import wandb
import os
import orbax.checkpoint as ocp
import flax.serialization as serialization
import jax
from pathlib import Path
import pickle as pkl



@hydra.main(version_base="1.3", config_path="config/", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    if not config.use_baidu:
        train_click_dataset, val_click_dataset, test_click_dataset, test_dataset = (
            train_val_test_datasets(config, varying=False)
        )
        print(test_click_dataset)

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
    else:
        _, train_click_dataset, unique_list = load_custom_click_dataset(config.baidu_subset, config)
        # _, val_click_dataset, _ = load_custom_click_dataset(config.baidu_subset, config)
        test_dataset, test_click_dataset, _ = load_custom_click_dataset(config.baidu_subset, config)

        train_click_loader = DataLoader(
            train_click_dataset,
            batch_size=512,
            collate_fn=np_collate,
            shuffle=True,
        )

        val_click_loader = DataLoader(
            train_click_dataset,
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

    # print entry of train dataloader
    batch = next(iter(train_click_loader))
    print("Sample batch keys:", batch.keys())
    print("Sample batch['lp_query_doc_features'] shape:", batch["lp_query_doc_features"].shape)
    print("Sample batch['lp_query_doc_features']" , batch["lp_query_doc_features"][:2, :2, :])
    #### Two-tower model ####
    # Use hydra partial instantiation to pass rngs and dataset properties:
    bias_tower = instantiate(
        config.bias_tower,
        positions=test_dataset.n_positions,
        feature_sizes=unique_list
    )
    relevance_tower = instantiate(
        config.relevance_tower,
        query_doc_features=test_dataset.n_features,
        query_doc_pairs=test_dataset.n_documents,
    )

    rngs = nnx.Rngs(config.random_state)
    model = TwoTowerModel(
        bias_tower=bias_tower(rngs=rngs),
        relevance_tower=relevance_tower(rngs=rngs),
        use_propensity_weighting=config.use_propensity_weighting,
    )

    base_optimizer = optax.adamw(learning_rate=0.01)

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
        n_features=test_dataset.n_features,
    )

    trainer.train(model, train_click_loader, val_click_loader)

    # test_click_df, _ = trainer.test_clicks(model, test_click_loader, save_outputs=True, output_path="test_click_outputs.csv")
    test_rel_df = trainer.test_relevance(model, test_loader)
    test_lp_df = trainer.test_logging_policy(test_click_loader)
    # test_df = pd.concat([test_click_df, test_rel_df], axis=1)

    test_rel_df.to_csv("test.csv", index=False)
    test_lp_df.to_csv("test_logging_policy.csv", index=False)

    trainer.get_position_bias(model, test_dataset.n_positions, unique_list)
    relevance_df = trainer.get_relevance_scores(model, test_dataset.n_features)
    relevance_df.to_csv("relevance.csv", index=False)
    if config.relevance == "deep":
        trainer.save_model_params(model, ckpt_dir="checkpoint")

    # predicted_relevance_df = trainer.get_predicted_relevance(model, test_loader, examination_0)
    
    # predicted_relevance_df.to_csv("predicted_relevance.csv", index=False)

if __name__ == "__main__":
    main()
