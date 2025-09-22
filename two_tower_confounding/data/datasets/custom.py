
from pathlib import Path

from two_tower_confounding.data.datasets.svmlight import SVMLightDataSet


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