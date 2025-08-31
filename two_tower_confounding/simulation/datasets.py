from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from two_tower_confounding.data.base import RatingDataset


class ClickDataset(Dataset):

    def __init__(
        self,
        rating_dataset: RatingDataset,
        sessions: np.ndarray,
        clicks: np.ndarray,
        positions: np.ndarray,
        sessions_per_query: np.ndarray,
        sessions_per_doc_pos: np.ndarray,
    ):
        self.sessions = sessions
        self.clicks = clicks
        self.positions = positions
        self.query = rating_dataset.query
        self.query_doc_features = rating_dataset.query_doc_features
        self.query_doc_ids = rating_dataset.query_doc_ids
        self.labels = rating_dataset.labels
        self.mask = rating_dataset.mask
        self.n = rating_dataset.n
        self.sessions_per_query = sessions_per_query
        self.sessions_per_doc_pos = sessions_per_doc_pos

    def __getitem__(self, idx):
        session_idx = self.sessions[idx]
        position_idx = self.positions[idx]
        doc_idx = np.arange(len(position_idx))

        sessions_per_doc_pos = self.sessions_per_doc_pos[
            session_idx, doc_idx, position_idx
        ]
        sessions_per_query = self.sessions_per_query[session_idx]
        propensities = sessions_per_doc_pos / sessions_per_query

        return {
            "query": self.query[session_idx],
            "query_doc_features": self.query_doc_features[session_idx][position_idx],
            "query_doc_ids": self.query_doc_ids[session_idx][position_idx],
            "labels": self.labels[session_idx][position_idx],
            "propensities": propensities,
            "clicks": self.clicks[idx],
            "positions": np.arange(len(position_idx)),
            "mask": self.mask[session_idx][position_idx],
            "n": self.n[session_idx],
        }

    def __len__(self):
        return len(self.sessions)

    def sample_features(
        self, n_samples: int, random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=random_state)
        sample_idx = rng.choice(len(self), size=n_samples)
        session_idx = self.sessions[sample_idx]
        position_idx = self.positions[sample_idx]

        x = self.query_doc_features[session_idx]
        mask = self.mask[session_idx]

        x = np.take_along_axis(x, position_idx[:, :, None], axis=1)
        mask = np.take_along_axis(mask, position_idx, axis=1)

        return x, mask

    @staticmethod
    def collate_fn(batch):
        """
        Collates a list of dictionaries into a single dict.
        The method assumes that all dicts in the list have the same keys.

        E.g.: batch = [{"query_doc_features": [...]}, {"query_doc_features": [...]}]
        -> {"query_doc_features": [...]}
        """
        keys = batch[0].keys()
        return {key: np.stack([sample[key] for sample in batch]) for key in keys}
