from pathlib import Path
import json
import numpy as np


class Dataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.X = None
        self.y = None
        self.features = None
        self.config = None

    def load(self):
        self.X = np.load(self.path / "X.npy")
        self.y = np.load(self.path / "y.npy")

        with open(self.path / "features.json") as f:
            self.features = json.load(f)

        with open(self.path / "config.json") as f:
            self.config = json.load(f)

        return self