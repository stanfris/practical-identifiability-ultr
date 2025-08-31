from typing import Callable

import numpy as np
from scipy.special import expit
from tqdm import tqdm

from two_tower_confounding.data.base import RatingDataset
from two_tower_confounding.simulation.datasets import ClickDataset


class Simulator:
    def __init__(
        self,
        logging_policy_ranker: Callable,
        logging_policy_sampler: Callable,
        bias_strength: float,
        max_label: int = 4,
        *,
        random_state: int,
    ):
        self.logging_policy_ranker = logging_policy_ranker
        self.logging_policy_sampler = logging_policy_sampler
        self.bias_strength = bias_strength
        self.rng = np.random.RandomState(random_state)
        self.max_label = max_label
        self.random_state = random_state

    def __call__(
        self,
        rating_dataset: RatingDataset,
        n_sessions: int,
    ) -> ClickDataset:
        # Uniform query distribution:
        sessions = self.rng.randint(0, rating_dataset.n_queries, size=(n_sessions,))
        sampled_clicks = []
        sampled_positions = []

        sessions_per_query = np.zeros(rating_dataset.n_queries)
        sessions_per_doc_pos = np.zeros(
            (
                rating_dataset.n_queries,
                rating_dataset.n_positions,
                rating_dataset.n_positions,
            )
        )

        scores = self.logging_policy_ranker(
            lp_query_doc_features=rating_dataset.lp_query_doc_features,
            labels=rating_dataset.labels,
            where=rating_dataset.mask,
        )

        for session_idx in tqdm(sessions, desc="Simulating clicks.."):
            sample = rating_dataset[session_idx]

            positions = self.logging_policy_sampler(
                scores=scores[session_idx],
                where=sample["mask"],
            )
            clicks = self.sample_clicks(
                labels=sample["labels"],
                positions=positions,
                where=sample["mask"],
            )

            # Keep track of the number of sessions per query and per document position:
            doc_idx = np.arange(len(positions))
            sessions_per_query[session_idx] += 1
            sessions_per_doc_pos[session_idx, doc_idx, positions] += 1

            sampled_positions.append(positions)
            sampled_clicks.append(clicks)

        return ClickDataset(
            rating_dataset=rating_dataset,
            sessions=sessions,
            clicks=np.stack(sampled_clicks),
            positions=np.stack(sampled_positions),
            sessions_per_query=sessions_per_query,
            sessions_per_doc_pos=sessions_per_doc_pos,
        )

    def sample_clicks(
        self, labels: np.ndarray, positions: np.ndarray, where: np.ndarray
    ):
        bias = get_position_bias(len(labels), strength=self.bias_strength)
        relevance = get_relevance(labels, max_label=self.max_label)

        # Rank documents according to their position and apply position bias:
        click_logits = bias + relevance[positions]
        assert np.isfinite(click_logits).all()
        click_prob = expit(click_logits)
        # Ensure masked items cannot be clicked:
        click_prob = np.where(where, click_prob, 0)

        return self.rng.binomial(n=1, p=click_prob)


def get_position_bias(n: int, strength: float = 1):
    return -strength * np.log(np.arange(n) + 1)


def get_relevance(labels: np.ndarray, max_label: int):
    # Normalize labels to logit range of: [-max_label/2, max_label/2]:
    return labels - (max_label // 2)
