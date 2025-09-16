from functools import partial
from typing import Dict

import jax.random
import numpy as np
import optax
import rax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from optax._src.base import GradientTransformation
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_tower_confounding.data.base import RatingDataset
from two_tower_confounding.models.towers import LinearRelevanceTower, DeepRelevanceTower, EmbeddingRelevanceTower


class ExpertRanker:
    def __init__(
        self,
        max_label: int = 4,
        *,
        policy_strength: float,
        random_state: int,
    ):
        self.policy_strength = policy_strength
        self.max_label = max_label
        self.rng = np.random.default_rng(random_state)

    def fit(self, dataset: RatingDataset):
        pass

    def __call__(
        self,
        *,
        labels: np.ndarray,
        where: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        labels = labels.copy()

        # Generate uniform random scores
        random_scores = self.rng.uniform(0, self.max_label, size=labels.shape)

        # Compute scores as an interpolation between random and label-based scores
        abs_strength = abs(self.policy_strength)
        ordered_scores = np.sign(self.policy_strength) * labels
        scores = abs_strength * ordered_scores + (1 - abs_strength) * random_scores

        return np.where(where, scores, -np.inf)


class NeuralRanker:
    def __init__(
        self,
        max_epochs: int = 250,
        max_label: int = 4,
        batch_size: int = 512,
        patience: int = 1,
        optimizer: GradientTransformation = optax.adamw(learning_rate=0.001),
        *,
        model_type: str,
        policy_strength: float,
        random_state: int,
    ):
        self.model_type = model_type
        self.max_epochs = max_epochs
        self.max_label = max_label
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer = optimizer
        self.policy_strength = policy_strength
        self.rngs = nnx.Rngs(random_state)

    def fit(self, dataset: RatingDataset):
        loader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=dataset.collate_fn
        )

        if self.model_type == "linear":
            self.model = LinearRelevanceTower(
                query_doc_features=dataset.n_logging_policy_features,
                rngs=self.rngs,
                use_bias=True,
            )
        elif self.model_type == "deep":
            self.model = DeepRelevanceTower(
                query_doc_features=dataset.n_logging_policy_features,
                layers=2,
                hidden_units=32,
                dropout=0.1,
                rngs=self.rngs,
            )
        # elif self.model_type == "embedding":
        #     self.model = EmbeddingRelevanceTower(
        #         query_doc_pairs=dataset.n_documents,
        #         rngs=self.rngs,
        #         embedding_dims=dataset.n_logging_policy_features,  # optional: tie embedding dim to feature count
        #     )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if self.policy_strength == 0:
            print(
                "Policy strength is 0, skipping training as model scores won't be used"
            )
            return

        optimizer = nnx.Optimizer(self.model, self.optimizer)
        early_stopping = EarlyStopping(min_delta=0.0005, patience=self.patience)
        best_state = None

        for epoch in tqdm(
            range(self.max_epochs),
            desc=f"Training {self.model_type} logging policy on all "
            f"{dataset.n_logging_policy_features} available features",
        ):
            epoch_loss = 0

            for batch in loader:
                epoch_loss += self._train_step(self.model, optimizer, batch)

            epoch_loss = epoch_loss / len(loader)
            early_stopping = early_stopping.update(epoch_loss)

            if early_stopping.has_improved:
                best_state = nnx.state(self.model)

            if early_stopping.should_stop:
                nnx.update(self.model, best_state)
                break

    def __call__(
        self,
        *,
        lp_query_doc_features: np.ndarray,
        where: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # Obtain model scores:
        self.model.eval()
        batch = {"query_doc_features": lp_query_doc_features}
        labels = self.model(batch)

        # Generate uniform random scores [0, max_label]
        random_scores = jax.random.uniform(
            self.rngs.params(),
            shape=labels.shape,
            maxval=self.max_label,
        )

        # Compute scores as an interpolation between random and label-based scores
        abs_strength = abs(self.policy_strength)
        ordered_scores = np.sign(self.policy_strength) * labels
        scores = abs_strength * ordered_scores + (1 - abs_strength) * random_scores

        return np.where(where, scores, -np.inf)

    @partial(nnx.jit, static_argnums=(0))
    def _train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: Dict,
    ):
        def loss_fn(model, batch):
            y_predict = model({"query_doc_features": batch["lp_query_doc_features"]})
            y = batch["labels"]
            return rax.pointwise_mse_loss(y_predict, y, where=batch["mask"])

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch)
        optimizer.update(grads)

        return loss
