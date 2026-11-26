# AAI-501 Final Team Project: Low-Cost Human Activity Recognition

How cheaply and reliably can a smartphone recognize what its user is doing?
We compare classifiers, feature selection, sensor subsets, clustering, and a
1D CNN on the UCI Human Activity Recognition dataset. See
[PROJECT_PLAN.md](PROJECT_PLAN.md) for the research questions and task split.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch (in `requirements.txt`) is needed only for the CNN notebook.

### Get the data

The dataset is not committed to the repository.

1. Download the dataset from
   [UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
   Use the UCI version, not the Kaggle copy, because only the UCI version
   includes the raw `Inertial Signals`.
2. Unzip it, then unzip the inner `UCI HAR Dataset.zip`. Put the resulting
   `UCI HAR Dataset` folder in either of these places (the loader checks
   both):
   - `data/UCI HAR Dataset/features.txt`
   - `human+activity+recognition+using+smartphones/UCI HAR Dataset/features.txt`
     (the folder name the UCI download unzips to)
3. Check that it loads: `python -m src.har_data`

## Using the data

```python
from src.har_data import load_features, load_raw_signals, sensor_masks

data = load_features()              # 561 features, official subject split
masks = sensor_masks(data.feature_names)
X_acc = data.X_train.loc[:, masks["accelerometer"]]
X_raw, y_raw = load_raw_signals("train")   # shape (7352, 128, 9)
```

## Notebooks

Run the notebooks from the `notebooks/` folder, in this order. Each one
saves its tables to `results/` and its plots to `figures/`, using its
letter as a prefix (`A_`, `B_`, `C_`).

| Notebook | What it does | Run time |
|---|---|---|
| [A_classifiers.ipynb](notebooks/A_classifiers.ipynb) | Exploratory data analysis, 8 baseline classifiers, tuning, final test comparison, sitting vs. standing error analysis | a few minutes |
| [B_deep_learning_week1.ipynb](notebooks/B_deep_learning_week1.ipynb) | Raw-signal plots and the proposed CNN design | seconds |
| [B_feature_selection.ipynb](notebooks/B_feature_selection.ipynb) | Feature rankings (random forest, mutual information, L1), accuracy vs. number of features, RFECV, sensor comparison, PCA vs. selection | about 7 minutes |
| [C_clustering_generalization.ipynb](notebooks/C_clustering_generalization.ipynb) | k-means, Gaussian mixture and agglomerative clustering, accuracy for each test person, leave-one-person-out CV, leakage experiment | about 2.5 minutes |
| [C_cnn_raw_signals.ipynb](notebooks/C_cnn_raw_signals.ipynb) | 1D CNN on the raw signals vs. engineered features | about 5 minutes (CPU) |

`B_feature_selection.ipynb` and `C_cnn_raw_signals.ipynb` read the saved
results of earlier notebooks, so run A before B, and B before the CNN
notebook.

## Evaluation protocol

- The official split is used throughout: 21 people for training, 9
  different people for testing.
- Model choices are made with 5-fold cross-validation on the training set,
  grouped by person (`GroupKFold`), so no person is in both training and
  validation folds. The test set is used only for final scores.
- Main metric: macro F1. Accuracy is also reported.
- `random_state=42` everywhere.

## Main results

All scores are macro F1 on the 9 test people unless stated otherwise.

| Question | Result |
|---|---|
| Best classifier (A) | Linear SVM (`C = 0.1`) on all 561 features: **0.963**. Hardest pair: sitting vs. standing. |
| How few features? (B) | 50 features chosen by L1 logistic regression: **0.942** (98% of the full score). RFECV's best subset: 121 features (CV 0.944). |
| Which sensors? (B) | Accelerometer only: **0.912**. Gyroscope only: 0.831. Accelerometer only with 30 features: 0.889. |
| Structure without labels (C) | k = 2 clusters match moving vs. still almost perfectly (ARI 0.9995). k = 6 matches the six activities only partly (ARI 0.474). |
| New people (C) | Leave-one-person-out accuracy ranges from 68.7% to 99.7% (mean 94.0%). |
| Leakage (C) | A random split inflates macro F1 by 4.8 points (linear SVM) and 9.7 points (5-NN) compared with a person-based split. |
| Raw signals vs. features (C) | 1D CNN on raw signals: **0.902** (9 channels) and 0.903 (accelerometer only), below the SVM on engineered features. |

## Repository layout

| Path | Contents |
|---|---|
| `src/har_data.py` | Shared data loading (features, raw signals, sensor masks) |
| `src/evaluation.py` | Shared grouped cross-validation, test scoring, and saving helpers |
| `notebooks/` | One notebook per team member (`A_`, `B_`, `C_`) |
| `results/` | Saved metrics as CSV |
| `figures/` | Plots used in the report and slides |
| `reports/` | Proposal, final paper, slides |

## Team

| Member | Area |
|---|---|
| Person A | Classifier comparison, tuning, error analysis |
| Person B | Feature selection, sensor comparison |
| Person C | Clustering, generalization to new people, CNN |

## Citation

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
*Human activity recognition using smartphones* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C54S4K
