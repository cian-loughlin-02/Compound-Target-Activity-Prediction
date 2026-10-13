# Compound–Target Activity Prediction

Machine-learning virtual screening that predicts whether a small molecule is **active** against a given protein target, directly from its SMILES string. Built as my undergraduate thesis (BSc Chemistry with AI, DCU).

## Overview

Early-stage drug discovery needs to narrow enormous compound libraries down to a testable shortlist. This project trains per-target classifiers on the [DUD-E](http://dude.docking.org/) benchmark (102 targets), then uses the trained ensemble to screen unseen compounds and flag likely actives.

The pipeline has two phases:

1. **Training & validation:**  for each target, convert SMILES to molecular fingerprints, train and tune four model families, and evaluate with cross-validation.
2. **Screening:**  apply the trained models to a set of unseen SMILES and take a majority vote to label each compound active/inactive.

## Approach

- **Representation:** SMILES → Morgan (circular) fingerprints via RDKit.
- **Models:** Random Forest, Support Vector Machine, k-Nearest Neighbours, and a Multi-Layer Perceptron.
- **Tuning:** hyperparameter optimisation with Optuna.
- **Evaluation:** 3- and 5-fold cross-validation, scored with ROC-AUC plus enrichment factor (EF 0.5% / 1%),  which matter more than raw accuracy in screening, where actives are rare.

## Results

- Best model (**SVM**) reached a mean ROC-AUC of **~0.99** across the 102 DUD-E targets, with EF 0.5% / 1% around 60–63.
- In the screening phase, an ensemble of **408 trained models** labelled **348 of 458** unseen compounds as active by majority vote.

> **Note:** DUD-E is known to carry decoy bias. Its actives and decoys are often separable on simple chemical properties, so the near-perfect AUCs partly reflect the benchmark rather than true protein–ligand discrimination. A more rigorous follow-up would use scaffold or time-based splits and prospective validation. This limitation is discussed in the report.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── Final_Code.ipynb        # main pipeline: preprocessing → training → screening
├── final_code.py           # script version of the pipeline
│
├── DUD-E/                       # DUD-E dataset (downloaded separately)
├── pre-processed/               # fingerprints / features for the DUD-E targets
├── New_Data_Pre_Processed/      # features for the unseen compounds to screen
├── experimental-data/           # experimental / validation compounds
│
├── Optimised-Parameters/        # Optuna-tuned hyperparameters
├── models_3fold/                # models trained under 3-fold cross-validation
├── models_5fold/                # models trained under 5-fold cross-validation
│
├── scoring/                     # phase-2 screening outputs (majority vote)
└── results/                     # metrics and figures
```


## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `rdkit`, `scikit-learn`, `torch`, `optuna`, `numpy`, `pandas`, `matplotlib`.

## Usage

The full pipeline: preprocessing, per-target training with Optuna tuning, and ensemble screening — runs from the main notebook, with a script version for headless runs:

```bash
jupyter notebook "Final_Code.ipynb"
# or
python final_code.py
```

Models were trained under both **3-fold and 5-fold cross-validation** (`models_3fold/`, `models_5fold/`); tuned hyperparameters are stored in `Optimised-Parameters/`, and screening results land in `scoring/` and `results/`.

## Data

DUD-E is publicly available at <http://dude.docking.org/>. The dataset and the preprocessed feature / model folders are not committed to this repo — download or regenerate them and place them in the folders above.


## Acknowledgements

Undergraduate thesis, Dublin City University. Supervised by Dr Darren Fayne (Molecular Design Group).
