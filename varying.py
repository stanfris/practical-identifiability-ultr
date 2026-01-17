import random

import hydra
import jax
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
from two_tower_confounding.trainer import Trainer
from two_tower_confounding.utils import np_collate, train_val_test_datasets, load_custom_click_dataset

import os
import orbax.checkpoint as ocp
from pathlib import Path
import pickle as pkl


def load_model_params(model, ckpt_dir="checkpoint", rng_seed=0):
    """
    Load model parameters into an existing model, keeping RNG separate.
    """
    ckptr = ocp.StandardCheckpointer()
    
    # Split the new model into RNG and other state
    _, _, other_state = nnx.split(model, nnx.RngState, ...)
    
    # Restore saved parameters (other_state)
    ckpt_path = Path(ckpt_dir).resolve()
    restored_other_state = ckptr.restore(ckpt_path, other_state)

    # Merge restored state into the live model
    nnx.update(model, restored_other_state)
    
    print("Relevance tower parameters successfully restored!")
    return model


def load_two_tower_incremental(config, dataset, relevance_path="relevance.csv", param_shift=0.0, param_idx=0, unique_list=[], test_dataset=None) -> TwoTowerModel:
    """Rebuild TwoTowerModel and load parameters from CSV files."""

    if unique_list != []:
        bias_tower = instantiate(
            config.bias_tower,
            positions=test_dataset.n_positions,
            unique_list=unique_list,
        )
    else:
        bias_tower = instantiate(
            config.bias_tower,
            positions=test_dataset.n_positions,
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


    # 3. Inject parameters depending on tower type
    model = load_model_params(model, ckpt_dir="checkpoint")

    bias_path = "bias.csv"
    bias_df = pd.read_csv(bias_path)
    bias_values = bias_df["examination"].to_numpy()

    if config.use_baidu:
        bias_types = ["position", "media_type", "displayed_time", "serp_height", "slipoff_count_after_click"]

    else:
        if param_shift != 0.0:
            print("bias_values before shift:", model.bias_tower.get_position_bias())
            model.bias_tower.frozen_param_idx = param_idx
            model.bias_tower.frozen_param_val = float(model.bias_tower.get_position_bias()[param_idx] + param_shift)
            print("bias_values after shift:", model.bias_tower.get_position_bias())
            print(model.bias_tower.frozen_param_val)
            print(f"Shift bias tower {param_idx} parameters by {param_shift:.4f}")

    print(f"✅ Loaded parameters from {bias_path} and {relevance_path}")
    return model
    
@hydra.main(version_base="1.3", config_path="config/", config_name="config")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    if not config.use_baidu:
        train_click_dataset, val_click_dataset, test_click_dataset, test_dataset = (
            train_val_test_datasets(config, varying=True)
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

    if config.use_baidu:
        model = load_two_tower_incremental(config, test_dataset, relevance_path="relevance.csv", param_shift=config.param_shift, param_idx=config.param_idx, unique_list=unique_list, test_dataset=test_dataset)
    else:
        model = load_two_tower_incremental(config, test_dataset, relevance_path="relevance.csv", param_shift=config.param_shift, param_idx=config.param_idx, test_dataset=test_dataset)
    print("completed loading of model")

    trainer = Trainer(
        optimizer=optax.adamw(learning_rate=0.006),
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
        freeze_bias_tower=config.freeze_bias_tower,
    )
    
    trainer.train(model, train_click_loader, val_click_loader)

    test_click_df = trainer.test_clicks(model, test_click_loader)
    test_rel_df = trainer.test_relevance(model, test_loader)
    test_lp_df = trainer.test_logging_policy(test_click_loader)

    test_click_df.to_csv(f"test_clicks_param_shift_{config.param_shift}_idx{config.param_idx}_bias_type{config.bias_type}.csv", index=False)
    print(f"✅ Saved test clicks results to test_clicks_param_shift_{config.param_shift}_idx{config.param_idx}.csv")
    test_rel_df.to_csv(f"test_relevance_param_shift_{config.param_shift}_idx{config.param_idx}.csv", index=False)
    print(f"✅ Saved test relevance results to test_relevance_param_shift_{config.param_shift}_idx{config.param_idx}.csv")
    test_lp_df.to_csv(f"test_logging_policy_param_shift_{config.param_shift}_idx{config.param_idx}.csv", index=False)
    print(f"✅ Saved test logging policy results to test_logging_policy_param_shift_{config.param_shift}_idx{config.param_idx}.csv")

    print(f"✅ Saved relevance parameters to relevance_param_shift_{config.param_shift}_idx{config.param_idx}_bias_type{config.bias_type}.csv")
    if config.use_baidu:
        trainer.get_position_bias(model, test_dataset.n_positions, unique_list, bias_csv_name=f"bias_param_shift_{config.param_shift}_idx{config.param_idx}_bias_type{config.bias_type}")
    else:
        trainer.get_position_bias(model, test_dataset.n_positions, bias_csv_name=f"bias_param_shift_{config.param_shift}_idx{config.param_idx}")


    print("Finished")

if __name__ == "__main__":
    main()
