"""
Unit tests for model creation and persistence.
"""

import tempfile
from pathlib import Path

from src.model import create_model, load_model, save_metrics, save_model


def test_create_model() -> None:
    model = create_model(n_estimators=5, random_state=42)
    assert model.n_estimators == 5
    assert model.random_state == 42


def test_save_load_model_roundtrip() -> None:
    model = create_model(n_estimators=5, random_state=42)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.joblib"
        save_model(model, path)
        assert path.exists()
        loaded = load_model(path)
        assert loaded.n_estimators == model.n_estimators


def test_save_metrics_creates_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub" / "metrics.json"
        save_metrics({"accuracy": 0.9, "f1": 0.88}, out)
        assert out.exists()
        assert out.read_text().find("0.9") != -1
