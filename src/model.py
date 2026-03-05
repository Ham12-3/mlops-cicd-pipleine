"""
Model definition and persistence.
"""

import json
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators: int = 50, random_state: int = 42) -> BaseEstimator:
    """Create the default classifier."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def save_model(model: BaseEstimator, path: str | Path) -> None:
    """Save model to disk with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> BaseEstimator:
    """Load model from disk."""
    return joblib.load(path)


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Write metrics dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
