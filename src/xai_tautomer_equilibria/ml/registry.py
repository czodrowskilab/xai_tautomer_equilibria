from pathlib import Path
import json
import numpy as np


class DatasetRegistry:

    def __init__(self, root: Path):
        self.root = Path(root)

    def list_datasets(self):
        datasets = []
        for subset in self.root.iterdir():
            if subset.is_dir():
                for cfg in subset.iterdir():
                    if (cfg / "X.npy").exists():
                        datasets.append(cfg)
        return datasets

    def load(self, dataset_path: Path):
        X = np.load(dataset_path / "X.npy")
        y = np.load(dataset_path / "y.npy")

        with open(dataset_path / "config.json") as f:
            config = json.load(f)

        with open(dataset_path / "features.json") as f:
            features = json.load(f)

        return X, y, config, features