from copy import deepcopy
from functools import partial
import shutil
from typing import Dict
from xml.parsers.expat import model
import json

import jax
import jax.numpy as jnp
import pandas as pd
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import optax
from flax import serialization
import orbax.checkpoint as ocp
import os
from pathlib import Path

from two_tower_confounding.metrics import Metric, Average


class Trainer:
    def __init__(
        self,
        optimizer: GradientTransformation,
        metrics: Dict[str, Metric] = None,
        click_metrics: Dict[str, Metric] = None,
        epochs: int = 50,
        patience: int = 2,  # Note that NNX patience is off by 1, so 2 means 3 epochs.
        run = wandb.run,
        n_features: int = 100,
        freeze_bias_tower: bool = False,
    ):
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else {}
        self.click_metrics = click_metrics if click_metrics is not None else {}
        self.epochs = epochs
        self.patience = patience
        self.click_metrics["loss"] = Average("loss")
        self.run = run
        self.n_features = n_features
        self.freeze_bias_tower = freeze_bias_tower

    def train(
        self,
        model: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):

        if self.freeze_bias_tower:
            print("Freezing bias tower during training.")

            mask = self.freeze_bias_tower_mask(model)

            optim = optax.multi_transform(
                {
                    "train": self.optimizer,
                    "frozen": optax.set_to_zero(),
                },
                mask
            )
        else:   
            optim = self.optimizer

        optimizer = nnx.Optimizer(model, optim)

        click_metrics = nnx.MultiMetric(**deepcopy(self.click_metrics))
        early_stopping = EarlyStopping(patience=self.patience, min_delta=0.00001)
        best_state = nnx.state(model)

        for epoch in range(self.epochs):
            # Enable non-deterministic operations:
            model.train()

            for batch in tqdm(train_loader, desc=f"Train - Epoch: {epoch}"):
                self._train_step(model, optimizer, click_metrics, batch)

            train_metrics = click_metrics.compute()
            click_metrics.reset()

            # Disable random operations, such as dropout:
            model.eval()

            for batch in tqdm(val_loader, desc=f"Val - Epoch: {epoch}"):
                self._test_click_step(model, click_metrics, batch)

            val_metrics = click_metrics.compute()
            early_stopping = early_stopping.update(val_metrics["loss"])
            click_metrics.reset()

            print(
                f"Epoch {epoch} - "
                f"Train loss: {train_metrics['loss']:.8f}, "
                f"Val loss: {val_metrics['loss']:.8f}, "
                f"has improved: {early_stopping.has_improved}\n"
            )

            if self.run is not None:
                # convert to floats before logging
                train_metrics_float = jax.tree.map(float, train_metrics)
                val_metrics_float = jax.tree.map(float, val_metrics)
                features = jnp.eye(self.n_features)
                relevance = model.relevance_tower({"query_doc_features": features}).squeeze()
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics_float["loss"],
                    "val_loss": val_metrics_float["loss"],
                    **{f"train/{k}": v for k, v in train_metrics_float.items()},
                    **{f"val/{k}": v for k, v in val_metrics_float.items()},
                    **{f"relevance_{i}": float(relevance[i]) for i in range(self.n_features)},
                })

            if early_stopping.has_improved:
                best_state = nnx.state(model)

            if early_stopping.should_stop:
                print("Stopping early, loading best model state")
                nnx.update(model, best_state)
                break


    def test_clicks(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
        save_outputs: bool = False,
        output_path: str | None = None,
    ):
        click_metrics = nnx.MultiMetric(**deepcopy(self.click_metrics))
        model.eval()
        all_outputs = [] 
        
        for batch in tqdm(test_loader, desc="Test"):
            outputs = self._test_click_step(model, click_metrics, batch)
            if save_outputs:
                outputs_dict = {
                    "click": jnp.array(outputs.click).tolist(),
                    "relevance": jnp.array(outputs.relevance).tolist(),
                    "examination": jnp.array(outputs.examination).tolist(),
                }
                all_outputs.append(outputs_dict)

        test_metrics = click_metrics.compute()
        click_metrics.reset()
        print(f"Test: {jax.tree.map(float, test_metrics)}")

        if self.run is not None:
            test_metrics_float = jax.tree.map(float, test_metrics)
            wandb.log({f"test/{k}": v for k, v in test_metrics_float.items()})

        if save_outputs:
            if output_path is not None:
                with open(output_path, "w") as f:
                    json.dump(all_outputs, f)
                print(f"Saved outputs to {output_path}")
            else:
                print("Outputs were collected but not saved (no path specified).")

            return pd.DataFrame(test_metrics, index=[0]), all_outputs

        return pd.DataFrame(test_metrics, index=[0])

    def test_relevance(
        self,
        model: nnx.Module,
        test_loader: DataLoader,
    ):
        metrics = nnx.MultiMetric(**deepcopy(self.metrics))
        model.eval()

        for batch in tqdm(test_loader, desc="Test"):
            self._test_relevance_step(model, metrics, batch)

        test_metrics = metrics.compute()
        metrics.reset()
        return pd.DataFrame(test_metrics, index=[0])

    def get_position_bias(self, model, positions: int):
        if hasattr(model, "bias_tower"):
            positions = jnp.arange(positions)
            examination = model.bias_tower({"positions": positions}).squeeze()
            return pd.DataFrame(
                {
                    "position": positions,
                    "examination": examination - examination[0],
                }
            )
        else:
            return pd.DataFrame({})
        
    def get_relevance_scores(self, model, features: int):
        if hasattr(model, "relevance_tower"):
            feature_vectors = jnp.eye(features)

            relevance = model.relevance_tower({"query_doc_features": feature_vectors}).squeeze()

            return pd.DataFrame(
                {
                    "feature": jnp.arange(features),
                    "relevance": relevance,
                }
            )
        else:
            return pd.DataFrame({})
        
    def get_predicted_relevance(        
        self,
        model: nnx.Module,
        test_loader: DataLoader,
    ):
        model.eval()
        relevance_list = []
        for batch in tqdm(test_loader, desc="Test"):
                relevance = model.predict_relevance(batch)
                relevance = jnp.broadcast_to(relevance, batch["labels"].shape)
                relevance_list.append((relevance, batch["labels"]))
        return relevance_list

    def save_model_params(self, model, ckpt_dir="checkpoint"):
        """
        Save model parameters, excluding RNG state.
        """
        # Split model into RNG state and other parameters
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        
        # Ensure checkpoint directory exists
        ckpt_path = Path(ckpt_dir).resolve()
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Save using Orbax
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(ckpt_path, other_state, force=True)
        ckptr.wait_until_finished()

        print(f"Model parameters saved to {ckpt_path}")

    def test_logging_policy(self, test_loader: DataLoader):
        metrics = nnx.MultiMetric(**deepcopy(self.metrics))

        @nnx.jit()
        def metric_fn(metrics):
            metrics.update(
                relevance=1 / (1 + batch["positions"]),
                relevance_labels=batch["labels"],
                mask=batch["mask"],
            )

        for batch in tqdm(test_loader, desc="Test logging policy"):
            metric_fn(metrics)
        test_metrics = metrics.compute()
        print(f"Test logging policy: {jax.tree.map(float, test_metrics)}")
        return pd.DataFrame(test_metrics, index=[0])

    @partial(nnx.jit, static_argnums=(0))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
    ):

        def loss_fn(model, batch):
            output = model(batch)
            loss = model.compute_loss(output, batch).mean()
            return loss, output

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, output), grads = grad_fn(model, batch)

        metrics.update(
            loss=loss,
            click=output.click,
            click_labels=batch["clicks"],
            mask=batch["mask"],
        )
        optimizer.update(grads)

    @partial(nnx.jit, static_argnums=(0))
    def _test_click_step(
        self,
        model: nnx.Module,
        click_metrics: nnx.MultiMetric,
        batch,
    ):
        output = model(batch)
        loss = model.compute_loss(output, batch)
        click_metrics.update(
            loss=loss,
            click=output.click,
            click_labels=batch["clicks"],
            mask=batch["mask"],
        )
        return output

    @partial(nnx.jit, static_argnums=(0))
    def _test_relevance_step(
        self,
        model: nnx.Module,
        metrics: nnx.MultiMetric,
        batch,
    ):
        relevance = model.predict_relevance(batch)
        # cast to same shape as batch["labels"]
        relevance = jnp.broadcast_to(relevance, batch["labels"].shape)
        metrics.update(
            relevance=relevance,
            relevance_labels=batch["labels"],
            mask=batch["mask"],
        )


    def freeze_bias_tower_mask(self, model: nnx.Module):
        """
        Returns a pytree of labels ('train' or 'frozen') for optax.multi_transform.
        """
        # Get a structured dict of model state (params + other metadata)
        graphdef, params, rng_state = nnx.split(model, nnx.Param, ...)

        def label_fn(path, _):
            # path is a tuple of keys, e.g. ("bias_tower", "embedding", "embedding")
            if any("bias_tower" in str(k) for k in path):
                print("multi_transform sees frozen path:", path)
                return "frozen"
            else:
                print("multi_transform sees train path:", path)
                return "train"

        return jax.tree_util.tree_map_with_path(label_fn, params)