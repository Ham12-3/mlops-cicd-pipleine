"""
Evaluation entrypoint. Run with: python -m src.evaluate --model-path <path> --out metrics.json
"""

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.data import load_synthetic_data, validate_feature_matrix, validate_labels
from src.model import load_model, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    parser.add_argument(
        "--out", type=str, default="metrics.json", help="Output path for metrics JSON"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Total samples (same as train for same holdout)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Must match train for comparable metrics"
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)
    # Use same n_samples and random_state as train so test set is the same holdout (20% of data).
    _, X_test, _, y_test = load_synthetic_data(
        n_samples=args.n_samples, random_state=args.random_state
    )
    validate_feature_matrix(X_test, min_samples=1)
    validate_labels(y_test, min_classes=2)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    metrics = {"accuracy": accuracy, "f1": f1}
    save_metrics(metrics, args.out)
    print(f"Metrics saved to {args.out}: accuracy={accuracy:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()
