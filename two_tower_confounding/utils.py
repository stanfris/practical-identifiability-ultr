from pathlib import Path
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.training import checkpoints
from jax import Array


def np_collate(batch):
    """
    Collates a list of dictionaries into a single dict.
    The method assumes that all dicts in the list have the same keys.

    E.g.: batch = [{"query_doc_features": [...]}, {"query_doc_features": [...]}]
    -> {"query_doc_features": [...]}
    """
    keys = batch[0].keys()
    return {key: np.stack([sample[key] for sample in batch]) for key in keys}


def reduce_per_query(loss: Array, where: Array) -> Array:
    loss = loss.reshape(len(loss), -1)
    where = where.reshape(len(where), -1)

    # Adopt Rax safe_reduce as jnp.mean can return NaN if all inputs are 0,
    # which happens easily for pairwise loss functions without any valid pair.
    # Replace NaNs with 0 after reduce, but propagate if the loss already contains NaNs:
    is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(loss)))
    output = jnp.mean(loss, where=where, axis=1)
    output = jnp.where(jnp.isnan(output) & is_input_valid, 0.0, output)

    return output


def collect_metrics(results: List[Dict[str, Array]]) -> pd.DataFrame:
    """
    Collects batches of metrics into a single pandas DataFrame:
    [
        {"ndcg": [0.8, 0.3], "MRR": [0.9, 0.2]},
        {"ndcg": [0.2, 0.1], "MRR": [0.1, 0.02]},
        ...
    ]
    """
    # Convert Jax Arrays to numpy:
    np_results = [dict_to_numpy(r) for r in results]
    # Unroll values in batches into individual rows:
    df = pd.DataFrame(np_results)
    return df.explode(column=list(df.columns)).reset_index(drop=True)


def aggregate_metrics(df: pd.DataFrame, ignore_columns=["query"]) -> Dict[str, float]:
    df = df.drop(columns=ignore_columns, errors="ignore")
    return df.mean(axis=0).to_dict()


def dict_to_numpy(_dict: Dict[str, Array]) -> Dict[str, np.ndarray]:
    return {k: jax.device_get(v) for k, v in _dict.items()}


def save_state(state, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / name).absolute()
    checkpoints.save_checkpoint(path, state, step=0, overwrite=True)
