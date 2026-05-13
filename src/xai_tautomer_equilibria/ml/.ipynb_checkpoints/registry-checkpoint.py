from pathlib import Path
import json
import numpy as np


class DatasetRegistry:

    def __init__(self, root: Path):
        self.root = Path(root)





    def list_datasets(self):
        datasets = []
        for subset in self.root.iterdir():
            if not subset.is_dir():
                continue
    
            for dataset in subset.iterdir():
                if (dataset / "config.json").exists():
                    datasets.append(dataset)
    
        return datasets

    def load(self, dataset_path: Path):
        X = np.load(dataset_path / "X.npy")
        y = np.load(dataset_path / "y.npy")

        with open(dataset_path / "config.json") as f:
            config = json.load(f)

        with open(dataset_path / "features.json") as f:
            features = json.load(f)

        return X, y, config, features