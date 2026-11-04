# AAI-501 Final Team Project: Low-Cost Human Activity Recognition

How cheaply and reliably can a smartphone recognize what its user is doing?
We compare classifiers, feature selection, sensor subsets, and clustering on
the UCI Human Activity Recognition dataset. See
[PROJECT_PLAN.md](PROJECT_PLAN.md) for the research questions and task split.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get the data

The dataset is not committed to the repository.

1. Download the dataset from
   [UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
   Use the UCI version, not the Kaggle copy, because only the UCI version
   includes the raw `Inertial Signals`.
2. Unzip it, then unzip the inner `UCI HAR Dataset.zip`, so that this path
   exists: `data/UCI HAR Dataset/features.txt`.
3. Check that it loads: `python -m src.har_data`

## Using the data

```python
from src.har_data import load_features, load_raw_signals, sensor_masks

data = load_features()              # 561 features, official subject split
masks = sensor_masks(data.feature_names)
X_acc = data.X_train.loc[:, masks["accelerometer"]]
X_raw, y_raw = load_raw_signals("train")   # shape (7352, 128, 9)
```

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Shared, reusable code (data loading) |
| `notebooks/` | One notebook per team member (`A_`, `B_`, `C_`) |
| `results/` | Saved metrics as CSV |
| `figures/` | Plots used in the report and slides |
| `reports/` | Proposal, final paper, slides |

## Team

| Member | Area |
|---|---|
| Person A | Classifier comparison, tuning, error analysis |
| Person B | Feature selection, sensor comparison |
| Person C | Clustering, generalization to new people, optional CNN |

## Citation

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
*Human activity recognition using smartphones* [Dataset]. UCI Machine

