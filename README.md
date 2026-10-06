# xai_tautomer_equilibria



## Introduction

Tautomerism is a fundamental problem in drug discovery: Tautomers of the same molecule can exhibit different physicochemical properties, like solubility, lipophilicity, hydrogen-bonding capacity and protein-binding affinity. Computational methods exist for predicting tautomer equilibria, but the role of aromaticity impacting this equilibria is often overlooked or treated implicitly.

This project explores the relationship between tautomer equilibria and aromaticity using a machine learning approach. We build a curated dataset of tautomer pairs from Tautobase, featurize them with Morgan fingerprints and use aromaticity descriptors (HOMA) and train a range of linear and tree-based regression models for the prediction of tautomer stability.

To understand what the importance of aromaticity in model predictions, we employ two different explainability approaches:

- **Intrinsic feature importances** — the model's own learned attributions.
- **Extrinsic feature importances** — SHAP values, which attribute predictions to features in a model-agnostic way.





## Usage

# Build datasets
python -m xai_tautomer_equilibria.pipeline.build_datasets

# Tune and train
python scripts/tune_cluster_linux.py
python scripts/run_train_linux.py

# Explainability for one seed
python all_ml_models_intrinsic.py 
python all_ml_models_shap.py      

# Aggregate Importances
python bar_plot_generator.py --kind intrinsic
python bar_plot_generator.py --kind shap      


## Installation

```bash

.........