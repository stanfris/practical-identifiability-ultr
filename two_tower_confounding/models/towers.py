from typing import Dict, List, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

class LinearRelevanceTower(nnx.Module):
    """
    Relevance tower using a linear model \gamma_{q,d} = w^T x_{q,d}.
    By default does not use bias.
    """

    def __init__(
        self,
        query_doc_features: int,
        *,
        rngs: nnx.Rngs,
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.layer = nnx.Linear(
            in_features=query_doc_features,
            out_features=1,
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(self, batch: Dict) -> Array:
        x = batch["query_doc_features"]
        return self.layer(x).squeeze()


class DeepRelevanceTower(nnx.Module):
    """
    Relevance tower using a feed forward network with elu activations.
    """

    def __init__(
        self,
        query_doc_features: int,
        layers: int,
        hidden_units: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()
        self.modules = get_sequential(
            features=query_doc_features,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            in_features=hidden_units if layers > 0 else query_doc_features,
            out_features=1,
            rngs=rngs,
        )

    def __call__(self, batch: Dict) -> Array:
        x = batch["query_doc_features"]

        for module in self.modules:
            x = module(x)

        return self.output(x).squeeze()


class EmbeddingBiasTower(nnx.Module):
    """
    Bias tower allocating a separate parameter per position \theta_{k}.
    Uses a single embedding dimension by default.
    """

    def __init__(
        self,
        positions: int,
        embedding_dims: int = 1,
        *,
        rngs: nnx.Rngs,
        frozen_param_idx: int = -1,
        frozen_param_val: float = None,
        **kwargs,
    ):
        super().__init__()
        self.positions = positions
        self.embedding = nnx.Embed(
            num_embeddings=positions,
            features=embedding_dims,
            rngs=rngs,
        )
        self.frozen_param_idx = frozen_param_idx
        self.frozen_param_val = frozen_param_val

    def __call__(self, batch: Dict) -> Array:
        x = batch["positions"]
        if self.frozen_param_val is not None:
            mask = x == self.frozen_param_idx
            embedding = self.embedding(x).squeeze()
            embedding = jnp.where(mask, self.frozen_param_val, embedding)
        else:
            embedding = self.embedding(x).squeeze()
        embedding = jnp.atleast_2d(embedding)
        return embedding

    def get_position_bias(self) -> Array:
        positions = jnp.arange(self.positions)
        return self({"positions": positions}).squeeze()

class ByPassEmbeddingBiasTower(nnx.Module):
    """
    Bias tower allocating a separate parameter per position \theta_{k}.
    Bypasses the embedding for the selected dimension.
    """

    def __init__(
        self,
        positions: int,
        embedding_dims: int = 1,
        *,
        rngs: nnx.Rngs,
        frozen_param_idx: int = -1,
        frozen_param_val: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nnx.Embed(
            num_embeddings=positions,
            features=embedding_dims,
            rngs=rngs,
        )
        self.positions = positions
        self.frozen_param_idx = frozen_param_idx
        self.frozen_param_val = frozen_param_val

    def __call__(self, batch: Dict) -> Array:
        x = batch["positions"]
        mask = x == self.frozen_param_idx
        embedding = self.embedding(x).squeeze()
        embedding = jnp.where(mask, self.frozen_param_val, embedding)
        embedding = jnp.atleast_2d(embedding)
        return embedding
    
    def get_position_bias(self) -> Array:
        positions = jnp.arange(self.positions)
        return self({"positions": positions}).squeeze()
    

class MultiEmbeddingBiasTower(nnx.Module):
    """
    Bias tower using multiple embeddings for categorical features followed by an MLP.
    Expects bias features in batch["lp_query_doc_features"] of shape [B, N, F] (N = number of positions, F = number of categorical features).
    """
    def __init__(
        self,
        feature_sizes: List[int],
        embedding_dims: int = 1,
        hidden_dims: int = 32,
        layers: int = 2,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()

        F = len(feature_sizes)
        concat_dim = F * embedding_dims

        # --- Embeddings ---
        self.embeddings = [
            nnx.Embed(
                num_embeddings=size,
                features=embedding_dims,
                rngs=rngs
            )
            for size in feature_sizes
        ]

        # --- MLP ---
        mlp_modules = get_sequential(
            features=concat_dim,
            hidden_units=hidden_dims,
            layers=layers,
            dropout=dropout,
            rngs=rngs,
        )

        # Final linear → scalar relevance
        mlp_modules.append(
            nnx.Linear(
                in_features=hidden_dims if layers > 0 else concat_dim,
                out_features=1,
                rngs=rngs
            )
        )

        self.mlp = nnx.Sequential(*mlp_modules)


    def __call__(self, batch: Dict) -> Array:
        """
        batch["lp_query_doc_features"]: int array of shape [B, N, F] (N = number of positions, F = number of categorical features).
        Returns: Array of shape [B, N] with bias scores.
        """
        # read bias features (one column per feature)
        if "lp_query_doc_features" not in batch:
            raise KeyError(
                "MultiEmbeddingBiasTower expects batch['lp_query_doc_features'] with shape [B, N, F]."
            )

        bias_feats = batch["lp_query_doc_features"] 

        _, _, F = bias_feats.shape

        # embed each feature separately
        embedded_feats = []
        for f in range(F):
            feat_f = bias_feats[:, :, f]  # shape [B, N]
            embedded_f = self.embeddings[f](feat_f)  # shape [B, N, embedding_dims]
            embedded_feats.append(embedded_f)

        x = jnp.concatenate(embedded_feats, axis=-1)  # shape [B, N, F * embedding_dims]

        x = self.mlp(x) # shape [B, N, 1]

        x = x.squeeze(-1)  # shape [B, N]

        return x


def get_sequential(
    features: int,
    hidden_units: int,
    layers: int,
    dropout: float,
    *,
    rngs: nnx.Rngs,
) -> List[Callable]:
    modules = []

    for _ in range(layers):
        modules.extend(
            [
                nnx.Linear(in_features=features, out_features=hidden_units, rngs=rngs),
                nnx.elu,
                nnx.Dropout(rate=dropout, rngs=rngs),
            ]
        )
        features = hidden_units

    return modules
