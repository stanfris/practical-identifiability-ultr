import numpy as np


class EGreedySampler:
    """
    Displays the original ranking in 1 - epsilon of the cases,
    and in epsilon of the cases displays a uniform random ranking.
    """

    def __init__(
        self,
        random_state: int,
        policy_temperature: float,
    ):
        self.rng = np.random.RandomState(random_state)
        self.policy_temperature = policy_temperature

    def __call__(
        self,
        *,
        scores: np.ndarray,
        where: np.ndarray,
    ) -> np.ndarray:
        scores = scores.copy()

        if self.rng.rand() < self.policy_temperature:
            scores = self.rng.random(scores.shape)

        scores = np.where(where, scores, -np.inf)
        # Rank in descending order:
        return np.argsort(-scores)


class GumbelMaxSampler:
    """
    Sampling permutations, 4.1 of:
    https://dl.acm.org/doi/pdf/10.1145/3336191.3371844
    This is the non-differentiable version of the Gumbel Softmax Trick
    """

    def __init__(
        self,
        max_label: int = 5,
        *,
        random_state: int,
        policy_temperature: float,
    ):
        self.rng = np.random.RandomState(random_state)
        self.max_label = max_label
        self.temperature = policy_temperature

    def __call__(
        self,
        *,
        scores: np.ndarray,
        where: np.ndarray,
        eps: float = 0.1e-6,
    ) -> np.ndarray:
        scores = scores.copy()

        # Mask padding docs:
        scores = np.where(where, scores, -np.inf)

        # Sample Gumbel noise:
        noise = self.rng.rand(*scores.shape)
        gumbel_noise = -np.log(-np.log(noise + eps) + eps)

        # Add noise and scale by temperature:
        perturbed_scores = scores + self.temperature * gumbel_noise

        # Rank in descending order:
        return np.argsort(-perturbed_scores)


class PivotRankSampler:
    """
    Selects a random document to swap with the document at a fixed pivot rank.
    Ranks are in range of: 0, ..., n-1.
    """

    def __init__(
        self,
        random_state: int,
        pivot_rank: int,
    ):
        self.rng = np.random.RandomState(random_state)
        self.pivot_rank = pivot_rank

    def __call__(
        self,
        *,
        scores: np.ndarray,
        where: np.ndarray,
    ) -> np.ndarray:
        scores = scores.copy()
        scores = np.where(where, scores, -np.inf)

        valid_doc_idx = np.where(where)[0]
        swap_idx = np.random.choice(valid_doc_idx)

        temp_score = scores[self.pivot_rank]
        scores[self.pivot_rank] = scores[swap_idx]
        scores[swap_idx] = temp_score

        # Rank in descending order:
        return np.argsort(-scores)
