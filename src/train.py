"""
Training entrypoint. Run with: python -m src.train
"""

import argparse
from pathlib import Path

from src.data import load_synthetic_data, validate_feature_matrix, validate_labels
from src.model import create_model, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--out-dir", type=str, default="artifacts", help="Directory for model output"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_synthetic_data(
        n_samples=args.n_samples, random_state=args.random_state
    )
    validate_feature_matrix(X_train, min_samples=10)
    validate_labels(y_train, min_classes=2)

    model = create_model(random_state=args.random_state)
    model.fit(X_train, y_train)

    model_path = out_dir / "model.joblib"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
