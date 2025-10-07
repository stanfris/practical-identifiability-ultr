
from pathlib import Path

from two_tower_confounding.data.datasets.svmlight import SVMLightDataSet
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import pickle
from pathlib import Path
from typing import Dict

from two_tower_confounding.data.utils.file import verify_file, unarchive

import os
import random
import shutil
import numpy as np

class CustomDataset(SVMLightDataSet):
    name = "custom_dataset"
    zip_file = "Custom_dataset.zip"
    file = "Custom_dataset"
    checksum = "08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae"
    fold_split_map = {
        1: {
            "train": "Fold1/train.txt",
            "val": "Fold1/vali.txt",
            "test": "Fold1/test.txt",
        },
    }

    def __init__(self, base_dir: Path):
        super().__init__(
            name=self.name,
            zip_file=self.zip_file,
            file=self.file,
            checksum=self.checksum,
            fold_split_map=self.fold_split_map,
            base_dir=base_dir,
        )


def create_custom_dataset(initial_path, filename, 
                          num_groups=1, docs_per_group=10, 
                          D=100, s_group=0.5, s_doc=0.5, 
                          random_seed=42, num_queries=10):
    """
    Generate a synthetic LTR-style dataset using a hierarchical Gaussian model
    and balanced quantile-based relevance labels (1–5).
    """
    rng = np.random.default_rng(random_seed)
    os.makedirs(initial_path, exist_ok=True)

    path = os.path.join(initial_path, filename)

    # global feature vector and weight vector
    global_feature_vector = rng.normal(0, 1, D)
    global_weight_vector = rng.normal(0, 1, D)

    all_scores, all_data = [], []

    for query in range(num_queries):
        for i in range(num_groups):
            group_sampled_features = rng.normal(0, 1, D)
            group_feature_vector = (1 - s_group) * global_feature_vector + s_group * group_sampled_features

            for j in range(docs_per_group):
                individual_sample = rng.normal(0, 1, D)
                doc_feature_vector = (1 - s_doc) * group_feature_vector + s_doc * individual_sample
                score = float(np.dot(global_weight_vector, doc_feature_vector))
                all_scores.append(score)
                all_data.append((query, doc_feature_vector))

    # Compute 5-level relevance bins using quantiles
    thresholds = np.percentile(all_scores, [20, 40, 60, 80])

    def score_to_label(s):
        if s <= thresholds[0]: return 1
        elif s <= thresholds[1]: return 2
        elif s <= thresholds[2]: return 3
        elif s <= thresholds[3]: return 4
        else: return 5

    # Write RankLib/LibSVM format
    with open(path, 'w') as f:
        for (score, (qid, features)) in zip(all_scores, all_data):
            label = score_to_label(score)
            feature_str = ' '.join(f"{k+1}:{features[k]:.4f}" for k in range(D))
            f.write(f"{label} qid:{qid} {feature_str}\n")

    print(f"✅ Dataset written: {path}")


def write_custom_dataset(initial_path, file, data, zip_path,
                         num_groups=1, docs_per_group=10, 
                         D=100, s_group=0.5, s_doc=0.5, 
                         random_seed=42, num_queries=10):
    """
    Generate train/val/test splits and zip them into a single dataset archive.
    """
    # Create Folds directory
    fold_dir = Path(initial_path) / "Fold1"
    os.makedirs(fold_dir, exist_ok=True)

    print(f"⚙️ Creating dataset splits in: {fold_dir}")

    for split_name in ["train.txt", "vali.txt", "test.txt"]:
        create_custom_dataset(
            initial_path=fold_dir,
            filename=split_name,
            num_groups=num_groups,
            docs_per_group=docs_per_group,
            D=D,
            s_group=s_group,
            s_doc=s_doc,
            random_seed=random_seed,
            num_queries=num_queries
        )

    # Zip everything
    zip_path = Path(zip_path)
    os.makedirs(zip_path.parent, exist_ok=True)

    output_filename = zip_path.stem
    print(f"📦 Zipping dataset from {initial_path} → {zip_path}")
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=initial_path)

    print(f"✅ Dataset zipped successfully at: {zip_path}")


# ==========================================================
# === Dataset Parser / Loader ===
# ==========================================================
class CustomDatasetDeep_Parser:
    def __init__(
        self,
        name: str,
        zip_file: str,
        file: str,
        checksum: str,
        fold_split_map: Dict[int, Dict[str, str]],
        base_dir: Path,
        dataset_params: Dict = None,
    ):
        self.base_dir = Path(base_dir).expanduser()
        self.dataset_params = dataset_params or {}

        # --- Generate parameter-specific suffix ---
        param_suffix = self._param_suffix(self.dataset_params)

        # --- Dynamic naming ---
        self.name = f"{name}-{param_suffix}" if param_suffix else name
        self.file = f"{file}-{param_suffix}" if param_suffix else file
        self.zip_file = f"{Path(zip_file).stem}-{param_suffix}.zip" if param_suffix else zip_file
        self.checksum = checksum
        self.fold_split_map = fold_split_map

        print(f"📁 Initialized dataset: {self.name}")

    # --- Helper for consistent param-based naming ---
    def _param_suffix(self, params: Dict) -> str:
        if not params:
            return ""
        parts = [
            f"num_groups{params.get('num_groups', 1)}",
            f"docs{params.get('docs_per_group', 10)}",
            f"D{params.get('D', 100)}",
            f"sgroup{params.get('s_group', 0.5)}",
            f"sdoc{params.get('s_doc', 0.5)}",
            f"seed{params.get('random_seed', 42)}",
            f"num_queries{params.get('num_queries', 10)}",
        ]
        return "_".join(parts)

    # --- Directory helpers ---
    @property
    def dataset_directory(self):
        path = self.base_dir / "dataset" / self.file
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_directory(self):
        path = self.base_dir / "download"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_directory(self):
        # parameter-specific cache folder
        path = self.base_dir / "cache" / self.file
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- Generation ---
    def _generate_dataset_if_needed(self):
        zip_path = self.download_directory / self.zip_file
        if zip_path.exists():
            print(f"📦 Found existing dataset zip: {zip_path}")
            return

        print("⚙️ Generating custom dataset...")
        write_custom_dataset(
            initial_path=self.dataset_directory,
            file=self.file,
            data=None,
            zip_path=zip_path,
            **self.dataset_params,
        )
        print(f"✅ Generated dataset: {zip_path}")

    # --- Loading ---
    def load(self, split: str, fold: int = 1) -> pd.DataFrame:
        cache_path = self.cache_directory / f"{self.name}-{fold}-{split}.pckl"
        print(f"\nLoading dataset: {self.name}, fold: {fold}, split: {split}")

        if not cache_path.exists():
            self._generate_dataset_if_needed()

            zip_path = self.download_directory / self.zip_file
            archive_dir = self.dataset_directory
            shutil.unpack_archive(str(zip_path), str(archive_dir))

            file_path = archive_dir / self.fold_split_map[fold][split]
            df = self._parse_svmlight(file_path)
            pickle.dump(df, open(cache_path, "wb"))

        return pickle.load(open(cache_path, "rb"))

    def _parse_svmlight(self, path: Path) -> pd.DataFrame:
        print(f"Parsing svmlight file: {path}")
        features, label, query = load_svmlight_file(str(path), query_id=True)
        features = np.asarray(features.todense())
        df = pd.DataFrame({"query_doc_features": list(features)})
        df["label"] = label
        df["query"] = query
        return df
    

class CustomDatasetDeep(CustomDatasetDeep_Parser):
    name = "custom_dataset_deep"
    zip_file = "Custom_dataset_deep.zip"
    file = "Custom_dataset_deep"
    checksum = "08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae"
    fold_split_map = {
        1: {
            "train": "Fold1/train.txt",
            "val": "Fold1/vali.txt",
            "test": "Fold1/test.txt",
        },
    }

    def __init__(self, base_dir: Path, **dataset_params):
        super().__init__(
            name=self.name,
            zip_file=self.zip_file,
            file=self.file,
            checksum=self.checksum,
            fold_split_map=self.fold_split_map,
            base_dir=base_dir,
            dataset_params=dataset_params,  
        )
