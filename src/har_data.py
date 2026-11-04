"""Shared data loading for the UCI HAR (Human Activity Recognition) dataset.

Every team member should load data through this module so that all
experiments use the same train/test split, labels, and feature masks.

Example
-------
>>> from src.har_data import load_features, sensor_masks
>>> data = load_features()
>>> data.X_train.shape
(7352, 561)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Places the unzipped "UCI HAR Dataset" folder may live, checked in order.
_CANDIDATE_DIRS = [
    PROJECT_ROOT / "data" / "UCI HAR Dataset",
    PROJECT_ROOT / "human+activity+recognition+using+smartphones"
    / "UCI HAR Dataset",
]

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

RAW_SIGNALS = [
    f"{signal}_{axis}"
    for signal in ("body_acc", "body_gyro", "total_acc")
    for axis in ("x", "y", "z")
]


@dataclass
class HARData:
    """Container for the engineered-feature version of the dataset."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    subject_train: np.ndarray
    subject_test: np.ndarray
    feature_names: list


def find_data_dir():
    """Return the path to the unzipped 'UCI HAR Dataset' folder."""
    for path in _CANDIDATE_DIRS:
        if (path / "features.txt").exists():
            return path
    searched = "\n  ".join(str(p) for p in _CANDIDATE_DIRS)
    raise FileNotFoundError(
        "UCI HAR Dataset not found. Unzip it to one of:\n  " + searched
    )


def _feature_names(data_dir):
    """Read feature names and make duplicates unique (the raw file has some)."""
    names = pd.read_csv(
        data_dir / "features.txt", sep=r"\s+", header=None
    )[1].tolist()
    seen = {}
    unique = []
    for name in names:
        count = seen.get(name, 0)
        unique.append(name if count == 0 else f"{name}__{count}")
        seen[name] = count + 1
    return unique


def _read_matrix(path):
    return np.loadtxt(path)


def load_features(data_dir=None):
    """Load the 561 engineered features with labels and subject IDs.

    Uses the official subject-wise split (21 train / 9 test subjects),
    so the test set measures performance on people the model never saw.
    """
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    names = _feature_names(data_dir)

    def split(part):
        X = pd.DataFrame(
            _read_matrix(data_dir / part / f"X_{part}.txt"), columns=names
        )
        y = _read_matrix(data_dir / part / f"y_{part}.txt").astype(int)
        subj = _read_matrix(
            data_dir / part / f"subject_{part}.txt"
        ).astype(int)
        return X, y, subj

    X_train, y_train, s_train = split("train")
    X_test, y_test, s_test = split("test")
    return HARData(X_train, X_test, y_train, y_test, s_train, s_test, names)


def load_raw_signals(part, data_dir=None):
    """Load raw inertial windows for deep learning.

    Returns
    -------
    X : ndarray of shape (n_windows, 128, 9)
        128 time steps (2.56 s at 50 Hz) x 9 channels (see RAW_SIGNALS).
    y : ndarray of shape (n_windows,)
    """
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    folder = data_dir / part / "Inertial Signals"
    channels = [
        _read_matrix(folder / f"{name}_{part}.txt") for name in RAW_SIGNALS
    ]
    X = np.stack(channels, axis=-1)
    y = _read_matrix(data_dir / part / f"y_{part}.txt").astype(int)
    return X, y


def sensor_masks(feature_names):
    """Boolean masks splitting features by sensor.

    'Gyro' features come from the gyroscope; everything else is derived
    from the accelerometer (including the gravity-based angle features).
    """
    names = pd.Series(feature_names)
    gyro = names.str.contains("Gyro").to_numpy()
    return {"accelerometer": ~gyro, "gyroscope": gyro}


if __name__ == "__main__":
    d = load_features()
    print("train", d.X_train.shape, "test", d.X_test.shape)
    print("train subjects:", sorted(set(d.subject_train)))
    print("test subjects:", sorted(set(d.subject_test)))
    for k, v in sensor_masks(d.feature_names).items():
        print(f"{k}: {v.sum()} features")
    X_raw, y_raw = load_raw_signals("test")
    print("raw test windows:", X_raw.shape)
