import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from two_tower_confounding.data.utils.file import verify_file, unarchive


class SVMLightDataSet:
    def __init__(
        self,
        name: str,
        zip_file: str,
        file: str,
        checksum: str,
        fold_split_map: Dict[int, Dict[str, str]],
        base_dir: Path,
    ):
        self.name = name
        self.zip_file = zip_file
        self.file = file
        self.checksum = checksum
        self.fold_split_map = fold_split_map
        self.base_dir = Path(base_dir).expanduser()

    @property
    def dataset_directory(self):
        path = self.base_dir / "dataset"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_directory(self):
        path = self.base_dir / "download"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_directory(self):
        path = self.base_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, split: str, fold: int = 1) -> pd.DataFrame:
        """
        Parses and caches a LTR dataset in svmlight format to a pandas DataFrame
        in long format with one row query query-document pair.

        Please place any dataset in its original .ZIP in the following directory:
        ~/my/base/directory/download/
        """
        cache_path = self.cache_directory / f"{self.name}-{fold}-{split}.pckl"
        print(f"\nLoading: {self.name}, fold: {1}, split: {split}")

        if not cache_path.exists():
            zip_path = self.download_directory / self.zip_file
            verify_file(zip_path, self.checksum)
            archive_path = unarchive(zip_path, self.dataset_directory / self.file)

            file_path = archive_path / self.fold_split_map[fold][split]
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
