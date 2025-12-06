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
    
class MultiEmbeddingBiasTower(nnx.Module):
    """
    Bias tower:
      - One embedding per feature (in a fixed order)
      - Concatenate embeddings
      - Pass through an MLP
    """

    def __init__(
        self,
        feature_sizes: List[int],     # list of vocabulary sizes (ordered)
        embedding_dims: int = 8,
        hidden_dims: int = 32,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__()

        # Create embeddings in a fixed order
        self.embeddings = [
            nnx.Embed(num_embeddings=size, features=embedding_dims, rngs=rngs)
            for size in feature_sizes
        ]

        # Small MLP to combine embeddings
        self.mlp = nnx.Sequential(
            [
                nnx.Dense(hidden_dims, rngs=rngs),
                nnx.elu,
                nnx.Dense(1, rngs=rngs),
            ]
        )

    def __call__(self, batch: Dict) -> Array:
        """
        batch["lp_query_doc_features"]: [B, T, F] integer IDs
        F must match len(feature_sizes)
        """
        x = batch["lp_query_doc_features"]

        # Embed each feature column using the corresponding embedding module
        embedded_cols = [
            emb(x[:, :, i])       # → [B, T, embedding_dims]
            for i, emb in enumerate(self.embeddings)
        ]

        # Concatenate all feature embeddings
        concat = jnp.concatenate(embedded_cols, axis=-1)    # [B, T, F*embedding_dims]

        # Flatten for MLP
        flat = concat.reshape(-1, concat.shape[-1])         # [B*T, dim]
        out = self.mlp(flat)                                # [B*T, 1]

        return out.reshape(concat.shape[:2]).squeeze()




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
