from enum import Enum
from typing import Optional, Union

import jax
import numpy as np
import pandas as pd
from flax import nnx

from two_tower_confounding.data.base import RatingDataset
from two_tower_confounding.data.utils.features import parse_feature_selection
from two_tower_confounding.data.utils.tensor import log1p, pad


class Relevance(Enum):
    ORIGINAL = "original"
    LINEAR = "linear"
    DEEP = "deep"


class Preprocessor:
    def __init__(
        self,
        normalize_features: bool,
        generate_query_document_ids: bool,
        sample_queries: Optional[int] = None,
        top_documents_per_query: Optional[int] = None,
        min_relevance_per_query: Optional[int] = None,
        *,
        random_state: int,
        features: str,
        relevance: Union[Relevance, str],
        relevance_noise: float,
        relevance_quantization: bool,
    ):
        self.normalize_features = normalize_features
        self.generate_query_document_ids = generate_query_document_ids
        self.sample_queries = sample_queries
        self.top_documents_per_query = top_documents_per_query
        self.min_relevance_per_query = min_relevance_per_query
        self.random_state = random_state
        self.features = features
        self.relevance = (
            Relevance(relevance) if isinstance(relevance, str) else relevance
        )
        self.relevance_noise = relevance_noise
        self.relevance_quantization = relevance_quantization

        if self.relevance == Relevance.LINEAR:
            self.relevance_fn = LinearRelevance(
                random_state=random_state,
                noise=relevance_noise,
            )
        elif self.relevance == Relevance.DEEP:
            self.relevance_fn = DeepRelevance(
                random_state=random_state,
                noise=relevance_noise,
            )

    def __call__(self, df: pd.DataFrame) -> RatingDataset:
        if self.top_documents_per_query is not None:
            print(
                f"Truncate queries to max. {self.top_documents_per_query} documents, "
                f"keeping the most relevant query-document pairs"
            )
            df = df.sort_values(by=["query", "label"], ascending=[True, False])
            df = df.groupby(["query"]).head(self.top_documents_per_query)

        if self.min_relevance_per_query is not None:
            query_df = (
                df.groupby(["query"]).agg(max_label=("label", "max")).reset_index()
            )
            keep_df = query_df[query_df.max_label >= self.min_relevance_per_query]
            df = df[df["query"].isin(keep_df["query"])]
            dropped_queries = len(query_df) - len(keep_df)
            print(
                f"Dropped: {dropped_queries}/{len(query_df)} queries "
                f"without any relevant document"
            )

        if self.generate_query_document_ids:
            print("Generate unique query-document ids starting from 1, 2, ...")
            df["query_doc_id"] = np.arange(1, len(df) + 1)

        if self.normalize_features:
            print("Log-transform query-document features")
            df["query_doc_features"] = df["query_doc_features"].map(lambda x: log1p(x))

        # Converting long (query-doc per row) to wide (query with all docs/row) format:
        df = (
            df.groupby(["query"])
            .agg(
                labels=("label", list),
                query_doc_ids=("query_doc_id", list),
                query_doc_features=("query_doc_features", list),
            )
            .reset_index()
        )

        if self.sample_queries is not None:
            print(f"Sampling {self.sample_queries}/{len(df)} queries")
            df = df.sample(n=self.sample_queries, random_state=self.random_state)

        # Pad all queries to the same number of documents and mask padded documents:
        df["n"] = df["query_doc_ids"].map(len)
        max_n = (
            self.top_documents_per_query
            if self.top_documents_per_query is not None
            else df.n.max()
        )
        print(f"Pad all queries to {max_n} docs")
        df["query_doc_ids"] = df["query_doc_ids"].map(lambda x: pad(x, max_n))
        df["query_doc_features"] = df["query_doc_features"].map(lambda x: pad(x, max_n))
        df["labels"] = df["labels"].map(lambda x: pad(x, max_n))
        df["mask"] = df["n"].map(np.ones).map(lambda x: pad(x, max_n).astype(bool))
        df["n"] = df["n"].map(lambda x: min(x, max_n))

        # Optionally generate new relevance labels:
        df = self.generate_labels(df)
        # Optionally select a subset of features:
        df = self.select_features(df)
        # Convert to PyTorch dataset:
        return RatingDataset(
            query=df["query"].values,
            query_doc_ids=np.stack(df["query_doc_ids"]),
            query_doc_features=np.stack(df["query_doc_features"]),
            lp_query_doc_features=np.stack(df["lp_query_doc_features"]),
            labels=np.stack(df["labels"]),
            mask=np.stack(df["mask"]),
            n=df["n"].values,
        )

    def generate_labels(self, df: pd.DataFrame):
        query_document_features = np.stack(df["query_doc_features"])
        labels = None

        if self.relevance == Relevance.LINEAR:
            print(
                f"Generating linear relevance labels with {self.relevance_noise} noise"
            )
            labels = self.relevance_fn(query_document_features)
            labels = scale_relevance(labels)
        elif self.relevance == Relevance.DEEP:
            print(
                f"Generating non-linear relevance labels with {self.relevance_noise} noise"
            )
            labels = self.relevance_fn(query_document_features)
            labels = scale_relevance(labels)
        elif self.relevance == Relevance.ORIGINAL:
            print(
                f"Using relevance labels originally provided in the dataset, no noise applied"
            )
            labels = np.stack(df["labels"])

        if self.relevance_quantization:
            print(f"Rounding to labels nearest integer for quantized relevance")
            labels = np.round(labels)

        df["labels"] = list(labels)
        return df

    def select_features(self, df):
        query_document_features = np.stack(df["query_doc_features"])
        total_features = query_document_features.shape[2]
        features = parse_feature_selection(self.features, total_features)
        print(
            f"Select query-document features for two-tower model: {self.features}, "
            f"{len(features)}/{total_features} available features"
        )

        # Keep all features for logging policy training:
        df["lp_query_doc_features"] = df["query_doc_features"]

        # Select subset of features for all downstream models:
        df["query_doc_features"] = df["query_doc_features"].map(
            lambda x: x[:, features]
        )

        return df


def scale_relevance(x, max_label: float = 4):
    lower, upper = np.percentile(x, [5, 95])
    x = np.clip(x, lower, upper)
    return (x - lower) / (upper - lower) * max_label


class LinearRelevance:
    def __init__(self, *, random_state: int, noise: float):
        self.noise = noise
        self.weights = None
        self.rngs = np.random.default_rng(random_state)

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        queries, documents, features = query_document_features.shape

        if self.weights is None:
            # Ensure subsequent calls to this function use the same weights:
            self.weights = self.rngs.standard_normal(features)

        scores = query_document_features.dot(self.weights)
        noise = self.noise * self.rngs.standard_normal(scores.shape)

        return scores + noise


class DeepRelevance:
    def __init__(self, hidden_units=16, *, random_state: int, noise: float):
        self.noise = noise
        self.hidden_units = hidden_units
        self.rngs = np.random.default_rng(random_state)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        queries, documents, features = query_document_features.shape

        if self.W1 is None:
            # Ensure subsequent calls to this function use the same weights:
            self.W1 = self.rngs.standard_normal((features, self.hidden_units))
            self.b1 = self.rngs.standard_normal(self.hidden_units)
            self.W2 = self.rngs.standard_normal(self.hidden_units)
            self.b2 = self.rngs.standard_normal()

        hidden = np.tanh(query_document_features.dot(self.W1) + self.b1)
        scores = hidden.dot(self.W2) + self.b2
        noise = self.noise * self.rngs.standard_normal(scores.shape)

        return scores + noise
