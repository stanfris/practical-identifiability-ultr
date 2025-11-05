# from two_tower_confounding.models.towers import *
# from two_tower_confounding.metrics import NDCG, MRR, NegativeLogLikelihood
# from two_tower_confounding.models.two_tower import TwoTowerModel
# from two_tower_confounding.simulation.simulator import Simulator
# from two_tower_confounding.trainer import Trainer
# from two_tower_confounding.utils import np_collate

# import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import two_tower_confounding as ttc


# -------------------------------
# DeepRelevance Model Definition
# -------------------------------
class DeepRelevance:
    def __init__(self, hidden_units=[32, 32, 32], *, random_state: int, noise: float = 0.0):
        self.hidden_units = hidden_units
        self.noise = noise
        self.rng = np.random.default_rng(random_state)
        self.layers = []

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        n_docs, n_features = query_document_features.shape
        if not self.layers:
            input_size = n_features
            for units in self.hidden_units:
                W = self.rng.standard_normal((input_size, units))
                b = self.rng.standard_normal(units)
                self.layers.append((W, b))
                input_size = units
            W_out = self.rng.standard_normal(input_size)
            b_out = self.rng.standard_normal()
            self.output_layer = (W_out, b_out)

        hidden = query_document_features
        for (W, b) in self.layers:
            hidden = np.tanh(hidden.dot(W) + b)
        scores = hidden.dot(self.output_layer[0]) + self.output_layer[1]
        noise = self.noise * self.rng.standard_normal(scores.shape)
        return scores + noise


# ---------------------------------------------
# Data generation
# ---------------------------------------------
def generate_deep_score_and_features_overlap(
    num_queries, num_groups, docs_per_group, D, s_group, s_doc, rng, deep_model
):
    all_scores = []
    all_data = []
    boundaries = []

    for qid in range(num_queries):
        begin = 0.0
        end = 1.0
        q_boundaries = []

        for grp_idx in range(num_groups):
            for doc_idx in range(docs_per_group):
                a = 0.0
                if doc_idx == 0:
                    b = rng.uniform(begin, end + s_doc)
                    q_boundaries.append((begin, end + s_doc, doc_idx))
                elif doc_idx == docs_per_group - 1:
                    b = rng.uniform(begin - s_doc, end)
                    q_boundaries.append((begin - s_doc, end, doc_idx))
                else:
                    b = rng.uniform(begin - s_doc, end + s_doc)
                    q_boundaries.append((begin - s_doc, end + s_doc, doc_idx))

                features = np.array([[a, b]])
                score = deep_model(features)[0]
                all_scores.append(score)
                all_data.append((qid, grp_idx, doc_idx, [a, b]))

                begin += 1
                end += 1

        boundaries.append(q_boundaries)

    return np.array(all_scores), np.array(all_data, dtype=object), boundaries


# ---------------------------------------------
# Visualization function with baseline lines and vertical caps
# ---------------------------------------------
def visualize_deep_relevance_with_caps(
    num_queries=2,
    docs_per_group=10,
    s_doc=0.0,
    hidden_units=[32, 32, 32],
    noise=0.0,
    random_state=41,
    show_boundaries=True,
    show_fill=True,
    fill_cmap="tab10",
    point_cmap="tab10",
    seed=42,
    jitter=0.0,
    baseline_gap=0.1,  # vertical gap between baseline lines
    cap_height=0.03    # height of the vertical caps
):
    rng = np.random.default_rng(seed)
    deep_relevance = DeepRelevance(hidden_units=hidden_units, random_state=random_state, noise=noise)

    scores, data, boundaries = generate_deep_score_and_features_overlap(
        num_queries=num_queries,
        num_groups=1,
        docs_per_group=docs_per_group,
        D=2,
        s_group=0.0,
        s_doc=s_doc,
        rng=rng,
        deep_model=deep_relevance,
    )

    doc_ids = np.array([d[2] for d in data])
    b_values = np.array([d[3][1] for d in data])

    # Smooth model curve
    a = np.zeros_like(np.linspace(0, 10, 500))
    b = np.linspace(0, 10, 500)
    X_plot = np.column_stack((a, b))
    y = np.ravel(deep_relevance(X_plot))

    plt.figure(figsize=(10, 6))
    plt.plot(b, y, color='black', lw=2, label='Deep relevance curve')

    # Colormaps
    fill_colors = plt.get_cmap(fill_cmap).colors[:docs_per_group]
    point_colors = plt.get_cmap(point_cmap).colors[:docs_per_group]

    # Compute mean boundaries across queries
    mean_bounds = []
    for doc_idx in range(docs_per_group):
        lefts = [q_bounds[doc_idx][0] for q_bounds in boundaries]
        rights = [q_bounds[doc_idx][1] for q_bounds in boundaries]
        mean_bounds.append((np.mean(lefts), np.mean(rights)))

    # --- Fill and boundary lines ---
    if show_fill:
        for i, (left, right) in enumerate(mean_bounds):
            color = fill_colors[i % len(fill_colors)]
            plt.axvspan(left, right, color=color, alpha=0.25)
            # Small horizontal line with vertical caps
            baseline_y = -baseline_gap * i - 1
            shift = 0.05
            plt.hlines(y=baseline_y, xmin=left+shift, xmax=right-shift, color=color, linewidth=3, alpha=0.9)
            # vertical caps
            plt.vlines([left+shift, right-shift], baseline_y - cap_height, baseline_y + cap_height, color=color, linewidth=2, alpha=0.9)

    if show_boundaries:
        for i, (left, right) in enumerate(mean_bounds):
            color = fill_colors[i % len(fill_colors)]
            plt.axvline(x=left, color=color, linestyle='--', linewidth=1.3, alpha=0.9)
            plt.axvline(x=right, color=color, linestyle='--', linewidth=1.3, alpha=0.9)

    # --- Plot sampled points ---
    for doc_idx in np.unique(doc_ids):
        mask = doc_ids == doc_idx
        color = point_colors[doc_idx % len(point_colors)]
        y_vals = scores[mask] + rng.normal(0, jitter, size=np.sum(mask))
        plt.scatter(
            b_values[mask],
            y_vals,
            color=color,
            s=90,
            edgecolor='black',
            linewidth=0.8,
            alpha=0.95,
            label=f"Rank {doc_idx}",
        )

    plt.xlabel('Feature b (second input dimension)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(
        f'Deep Relevance Visualization\n'
        f'{num_queries} queries × {docs_per_group} docs | Separation={s_doc}',
        fontsize=14,
    )
    plt.xlim(0, 10)
    plt.ylim(min(-baseline_gap*docs_per_group, min(y)-0.2)-1, max(y)+0.2)
    plt.grid(alpha=0.25)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.savefig('deep_relevance_visualization_with_caps.png', dpi=300)


# ---------------------------------------------
# Example Usage
# ---------------------------------------------
# visualize_deep_relevance_with_caps(
#     num_queries=3,
#     docs_per_group=10,
#     s_doc=0.1,
#     hidden_units=[32, 32, 32],
#     noise=0.0,
#     random_state=41,
#     seed=42,
#     fill_cmap="tab10",
#     point_cmap="tab10",
#     show_fill=True,
#     show_boundaries=True,
#     jitter=0.02,
#     baseline_gap=0.15,
#     cap_height=0.15
# )






def generate_deep_score_and_features_overlap(num_queries, num_groups, docs_per_group, D, s_group, s_doc, rng):
    """
    Generate features and scores using a deep learning model with hierarchical Gaussian noise.
    """
    all_scores = []
    all_data = []

    deep_model = DeepRelevance(hidden_units=[32, 32, 32], random_state=rng, noise=0.0)
    if num_queries != 10000:
        for qid in range(num_queries):
            for _ in range(num_groups):
                for doc_idx in range(docs_per_group):
                    a = 0
                    if doc_idx == 0:
                        b = rng.uniform(doc_idx, min(doc_idx + 1 + s_doc, 10))
                    elif doc_idx == docs_per_group - 1:
                        b = rng.uniform(max(0, doc_idx - s_doc), min(10, doc_idx + 1))
                    else:
                        b = rng.uniform(max(0, doc_idx - s_doc), min(10, doc_idx + 1 + s_doc))

                    # Document-level features
                    features = np.array([[a, b]])
                    score = deep_model(features)[0]
                    all_scores.append(score)
                    all_data.append((qid, [a, b]))  # qid starts from 0
    else:
        print("writing custom test dataset")
        b = 0
        for qid in range(num_queries):
            for _ in range(num_groups):
                for doc_idx in range(docs_per_group):
                    a = 0
                    b += 10/(num_queries)
                    # Document-level features
                    features = np.array([[a, b]])
                    score = deep_model(features)[0]
                    all_scores.append(score)
                    all_data.append((qid, [a, b]))  # qid starts from 0       

    return all_scores, all_data


class DeepRelevance:
    def __init__(self, hidden_units=[16, 8], *, random_state: int, noise: float):
        """
        Parameters
        ----------
        hidden_units : list[int]
            A list specifying the number of units in each hidden layer.
            Example: [32, 16, 8] creates 3 hidden layers.
        random_state : int
            Seed for reproducibility.
        noise : float
            Standard deviation of Gaussian noise added to output.
        """
        self.hidden_units = hidden_units
        self.noise = noise
        self.rng = np.random.default_rng(random_state)
        self.layers = []  # Will hold (W, b) tuples

    def __call__(self, query_document_features: np.ndarray) -> np.ndarray:
        n_docs, n_features = query_document_features.shape

        # Initialize weights only once
        if not self.layers:
            input_size = n_features
            for units in self.hidden_units:
                W = self.rng.standard_normal((input_size, units))
                b = self.rng.standard_normal(units)
                self.layers.append((W, b))
                input_size = units

            # Output layer
            W_out = self.rng.standard_normal(input_size)
            b_out = self.rng.standard_normal()
            self.output_layer = (W_out, b_out)

        # Forward pass
        hidden = query_document_features
        for (W, b) in self.layers:
            hidden = np.tanh(hidden.dot(W) + b)

        scores = hidden.dot(self.output_layer[0]) + self.output_layer[1]

        # Add noise
        noise = self.noise * self.rng.standard_normal(scores.shape)
        return scores + noise



# all_scores, all_data = generate_deep_score_and_features_overlap(
#     num_queries=num_queries,
#     num_groups=num_groups,
#     docs_per_group=docs_per_group,
#     D=D,
#     s_group=s_group,
#     s_doc=s_doc,
#     rng=rng
# )


# rng = np.random.default_rng(42)
# all_scores, all_data = generate_deep_score_and_features_overlap(
#     num_queries=100, num_groups=1, docs_per_group=10,
#     D=2, s_group=0, s_doc=0, rng=rng
# )

# scores = np.array(all_scores)
# b_vals = np.array([x[1][1] for x in all_data])  # second element in [a, b]

# # --- Plot ---
# plt.figure(figsize=(8,5))
# plt.plot(b_vals, scores, 'o-', alpha=0.7)
# plt.xlabel("Feature b")
# plt.ylabel("DeepRelevance Score")
# plt.title("Model Output vs Feature b")
# plt.grid(True)
# plt.show()
# plt.savefig('deep_relevance_scores_vs_b.png', dpi=300)

scores = []
b_vals = []



with open("../ltr_datasets/dataset/Custom_dataset_deep-num_groups1_docs10_D2_sgroup0.0_sdoc0.3_seed2021_num_queries20_labeltypedeep/Fold1/test.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        score = float(parts[0])
        # Find the part starting with '1:' and extract its numeric value
        b_part = [p for p in parts if p.startswith("1:")][0]
        b_val = float(b_part.split(":")[1])

        scores.append(score)
        b_vals.append(b_val)

# Plot score vs feature b
plt.figure(figsize=(8,5))
plt.scatter(b_vals, scores, c=scores, cmap='viridis', s=60, edgecolor='k', alpha=0.8)
plt.xlabel("Feature b")
plt.ylabel("Score")
plt.title("Score vs Feature b")
plt.grid(True)
plt.show()
plt.savefig('deep_relevance_scores_vs_b_from_file2.png', dpi=300)