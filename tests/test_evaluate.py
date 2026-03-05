"""
Unit tests for evaluate module (lightweight: no full training).
"""

import argparse
import json
import tempfile
from pathlib import Path

from src.data import load_synthetic_data
from src.evaluate import main as eval_main
from src.model import create_model, save_model


def test_evaluate_produces_metrics_file() -> None:
    """Evaluate script should write metrics.json with accuracy and f1."""
    with tempfile.TemporaryDirectory() as d:
        model_path = Path(d) / "model.joblib"
        out_path = Path(d) / "metrics.json"
        # Fit model on same data regime evaluate uses (n_samples=200, random_state=42).
        x_train, _, y_train, _ = load_synthetic_data(n_samples=200, random_state=42)
        model = create_model(n_estimators=5, random_state=42)
        model.fit(x_train, y_train)
        save_model(model, model_path)
        args = argparse.Namespace(
            model_path=str(model_path), out=str(out_path), n_samples=200, random_state=42
        )
        eval_main(args)
        assert out_path.exists()
        m = json.loads(out_path.read_text())
        assert "accuracy" in m and "f1" in m
        assert 0 <= m["accuracy"] <= 1 and 0 <= m["f1"] <= 1
