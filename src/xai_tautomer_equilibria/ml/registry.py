# src/xai_tautomer_equilibria/ml/registry.py

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

    def get_by_id(self, dataset_id: str) -> Path:
        """Return the directory for `dataset_id`, or raise KeyError."""
        for subset in self.root.iterdir():
            if not subset.is_dir():
                continue
            candidate = subset / dataset_id
            if (candidate / "config.json").exists():
                return candidate
        raise KeyError(f"dataset_id not found: {dataset_id}")

    def load(self, dataset_path: Path):
        X = np.load(dataset_path / "X.npy")
        y = np.load(dataset_path / "y.npy")

        with open(dataset_path / "config.json") as f:
            config = json.load(f)

        with open(dataset_path / "features.json") as f:
            features = json.load(f)

        return X, y, config, features


# =========================================================
# MODEL REGISTRY
# =========================================================

class ModelEntry:
    """One row from models/registry.json."""


    __slots__ = (
        "model_id", "path", "target", "variant", "subset",
        "radius", "n_bits", "model_family", "seed",
        "dataset_id", "framework", "version",
    )
    
    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))
        self.path = Path(self.path)

    def __repr__(self):
        return (
            f"ModelEntry(model_id={self.model_id!r}, "
            f"target={self.target!r}, seed={self.seed})"
        )

    def load(self):
        """Load the pickled model from disk."""
        import pickle

        pkl = self.path / f"{self.model_id}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(pkl)
        with open(pkl, "rb") as f:
            return pickle.load(f)


class ModelRegistry:
    """
    Loads models/registry.json and exposes ModelEntry lookups,
    mirroring the API of DatasetRegistry.
    """

    def __init__(self, root: Path):
        """
        root: directory containing registry.json AND processed files.
        """
        self.root = Path(root)
        self._entries: list[ModelEntry] = []

        registry_path = self.root / "registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"registry.json not found at {registry_path}")

        raw = json.loads(registry_path.read_text())
        for row in raw:
            self._entries.append(ModelEntry(**row))

    def list_models(self) -> list[ModelEntry]:
        return list(self._entries)

    def get(self, model_id: str) -> ModelEntry:
        for e in self._entries:
            if e.model_id == model_id:
                return e
        raise KeyError(f"model_id not found: {model_id}")

    def by_target(self, target: str) -> list[ModelEntry]:
        return [e for e in self._entries if e.target == target]

    def by_model_family(self, family: str) -> list[ModelEntry]:
        return [e for e in self._entries if e.model_family == family]

    def by_dataset(self, dataset_id: str) -> list[ModelEntry]:
        return [e for e in self._entries if e.dataset_id == dataset_id]