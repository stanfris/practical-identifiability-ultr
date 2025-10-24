from typing import Dict, List, Callable

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
        **kwargs,
    ):
        super().__init__()
        self.embedding = nnx.Embed(
            num_embeddings=positions,
            features=embedding_dims,
            rngs=rngs,
        )

    def __call__(self, batch: Dict) -> Array:
        x = batch["positions"]
        x = self.embedding(x).squeeze()
        x = jnp.atleast_2d(x)
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
