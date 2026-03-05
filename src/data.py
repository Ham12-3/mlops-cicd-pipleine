"""
Data loading and feature engineering for the ML pipeline.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_synthetic_data(n_samples: int = 1000, n_features: int = 20, random_state: int = 42):
    """
    Load or generate synthetic classification data.
    Returns (X_train, X_test, y_train, y_test).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(10, n_features),
        n_redundant=2,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def validate_feature_matrix(X: np.ndarray, min_samples: int = 1) -> None:
    """
    Validate feature matrix: no NaN/Inf, sufficient samples.
    Raises ValueError if validation fails.
    """
    if X.size == 0:
        raise ValueError("Feature matrix is empty")
    if X.shape[0] < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {X.shape[0]}")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Feature matrix contains NaN or Inf")


def validate_labels(y: np.ndarray, min_classes: int = 2) -> None:
    """
    Validate labels: at least min_classes present.
    Raises ValueError if validation fails.
    """
    if y.size == 0:
        raise ValueError("Labels are empty")
    n_classes = len(np.unique(y))
    if n_classes < min_classes:
        raise ValueError(f"Need at least {min_classes} classes, got {n_classes}")
