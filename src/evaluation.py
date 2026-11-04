"""Shared evaluation helpers so every team member reports results the same way.

All cross-validation is grouped by subject: one person's windows never
appear in both the training and validation folds.
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
N_SPLITS = 5


def group_cv():
    """Return the subject-grouped cross-validation splitter used project-wide."""
    return GroupKFold(n_splits=N_SPLITS)


def cv_score(model, X, y, groups):
    """Cross-validate a model with subject-grouped folds.

    Returns
    -------
    dict
        Mean and standard deviation of accuracy and macro F1, plus mean fit
        time in seconds.
    """
    scores = cross_validate(
        model, X, y, groups=groups, cv=group_cv(),
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    return {
        "cv_accuracy": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "cv_fit_time_s": scores["fit_time"].mean(),
    }


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    """Fit on the full training set and score once on the held-out test set.

    Returns
    -------
    metrics : dict
        Test accuracy, macro F1, fit time, and prediction time in seconds.
    y_pred : ndarray
        Test-set predictions, for confusion matrices and error analysis.
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    return metrics, y_pred


def save_figure(fig, name):
    """Save a matplotlib figure to figures/<name>.png at report quality."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def save_results(df, name):
    """Save a results table to results/<name>.csv."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(df).to_csv(RESULTS_DIR / f"{name}.csv", index=True)
