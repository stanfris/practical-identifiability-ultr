from pathlib import Path
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.training import checkpoints
from jax import Array
from omegaconf import DictConfig, OmegaConf
from two_tower_confounding.simulation.simulator import Simulator
import os
import pickle as pkl
from hydra.utils import instantiate
from two_tower_confounding.data.base import RatingDataset
from two_tower_confounding.simulation.datasets import ClickDataset

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


def train_val_test_datasets(config: DictConfig, varying: bool = False):
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
    if varying and config.load_test_datasets:
        print("Loading pre-saved test datasets", config.test_dataset_name, config.test_click_dataset_name)
        with open(f"../test_datasets/{config.test_dataset_name}", "rb") as f:
            test_dataset = pkl.load(f)
        with open(f"../test_datasets/{config.test_click_dataset_name}", "rb") as f:
            test_click_dataset = pkl.load(f)
    else:
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
    if not varying or not config.load_test_datasets:
        test_click_dataset = simulator(test_dataset, config.test_clicks)

    if not varying and config.save_test_datasets:
        print("Saving test datasets", config.test_dataset_name, config.test_click_dataset_name)
        os.makedirs("../test_datasets", exist_ok=True)
        with open(f"../test_datasets/{config.test_dataset_name}", "wb") as f:
            pkl.dump(test_dataset, f)
        with open(f"../test_datasets/{config.test_click_dataset_name}", "wb") as f:
            pkl.dump(test_click_dataset, f)

    return train_click_dataset, val_click_dataset, test_click_dataset, test_dataset



def load_custom_click_dataset(path: str, config: DictConfig) -> ClickDataset:
    dataset_dir = Path(config.dataset_dir).expanduser()
    file_path = dataset_dir / path
    data = np.load(file_path, allow_pickle=True)
    padded_positions = data["padded_positions"]
    mask = data["mask"]
    padded_clicks = data["padded_clicks"]
    sessions_per_query = data["sessions_per_query"]
    sessions_per_doc_pos = data["sessions_per_doc_pos"]
    query_doc_features = data["query_doc_features"]
    lp_query_doc_features = data["lp_query_doc_features"]
    query_doc_ids = data["query_doc_ids"]
    n = data["n"]
    queries = data["queries"]

    rating_dataset = RatingDataset(
        query = queries,
        query_doc_ids=query_doc_ids,
        query_doc_features=query_doc_features,
        lp_query_doc_features=lp_query_doc_features,
        labels=padded_clicks,
        mask=mask,
        n=n,
    )

    # -----------------------------
    # Construct ClickDataset
    # -----------------------------
    sessions = np.arange(len(rating_dataset))  # each session corresponds to a row in RatingDataset

    click_dataset = ClickDataset(
        rating_dataset=rating_dataset,
        sessions=sessions,
        clicks=padded_clicks,
        positions=padded_positions,
        sessions_per_query=sessions_per_query,
        sessions_per_doc_pos=sessions_per_doc_pos,
    )

    print("RatingDataset.query.shape:", rating_dataset.query.shape)
    print("RatingDataset.query_doc_features.shape:", rating_dataset.query_doc_features.shape)
    print("RatingDataset.lp_query_doc_features.shape:", rating_dataset.lp_query_doc_features.shape)
    print("ClickDataset.clicks.shape:", click_dataset.clicks.shape)
    print("ClickDataset.positions.shape:", click_dataset.positions.shape)
    print("ClickDataset.sessions_per_query.shape:", click_dataset.sessions_per_query.shape)
    print("ClickDataset.sessions_per_doc_pos.shape:", click_dataset.sessions_per_doc_pos.shape)
    return rating_dataset, click_dataset
