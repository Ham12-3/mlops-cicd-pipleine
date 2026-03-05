"""
Data processing and feature engineering tests (unit tests for data validation).
"""

import numpy as np
import pytest

from src.data import (
    load_synthetic_data,
    validate_feature_matrix,
    validate_labels,
)


def test_load_synthetic_data_returns_four_arrays() -> None:
    """Data loading should return train/test splits."""
    X_train, X_test, y_train, y_test = load_synthetic_data(n_samples=200, random_state=42)
    assert X_train.shape[0] == 160  # 80% of 200
    assert X_test.shape[0] == 40
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]


def test_validate_feature_matrix_accepts_valid_matrix() -> None:
    """Valid feature matrix (no NaN/Inf, enough samples) should not raise."""
    X = np.random.randn(10, 5)
    validate_feature_matrix(X, min_samples=5)


def test_validate_feature_matrix_rejects_empty() -> None:
    """Empty feature matrix should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validate_feature_matrix(np.array([]).reshape(0, 5), min_samples=1)


def test_validate_feature_matrix_rejects_nan() -> None:
    """Feature matrix with NaN should raise ValueError."""
    X = np.random.randn(10, 5)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_feature_matrix(X)


def test_validate_feature_matrix_rejects_insufficient_samples() -> None:
    """Too few samples should raise ValueError."""
    X = np.random.randn(3, 5)
    with pytest.raises(ValueError, match="at least 10"):
        validate_feature_matrix(X, min_samples=10)


def test_validate_labels_accepts_binary() -> None:
    """Binary labels should pass validation."""
    validate_labels(np.array([0, 1, 0, 1]), min_classes=2)


def test_validate_labels_rejects_single_class() -> None:
    """Single-class labels should raise when min_classes=2."""
    with pytest.raises(ValueError, match="at least 2 classes"):
        validate_labels(np.array([0, 0, 0]), min_classes=2)
