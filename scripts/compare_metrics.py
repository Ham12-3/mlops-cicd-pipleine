#!/usr/bin/env python3
"""
Compare new metrics to baseline and exit with non-zero if key metrics drop beyond threshold.
Usage: python scripts/compare_metrics.py --new <path> --baseline <path> [--accuracy-threshold 0.005] [--f1-threshold 0.005]
"""
import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    """Load metrics JSON; exit with clear error if missing or invalid."""
    if not path.exists():
        print(f"ERROR: Metrics file not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(2)


def compare(
    new: dict,
    baseline: dict,
    accuracy_threshold: float = 0.005,
    f1_threshold: float = 0.005,
) -> tuple[bool, list[str]]:
    """
    Compare new metrics to baseline. Returns (passed, list of failure messages).
    Fails if accuracy or f1 drops by more than the given fraction (e.g. 0.005 = 0.5%).
    """
    failures: list[str] = []
    for key, threshold in [("accuracy", accuracy_threshold), ("f1", f1_threshold)]:
        b_val = baseline.get(key)
        n_val = new.get(key)
        if b_val is None:
            failures.append(f"Baseline missing key: {key}")
            continue
        if n_val is None:
            failures.append(f"New metrics missing key: {key}")
            continue
        drop = float(b_val) - float(n_val)
        if drop > threshold:
            failures.append(
                f"{key}: drop {drop:.4f} exceeds threshold {threshold} (baseline={b_val:.4f}, new={n_val:.4f})"
            )
    return (len(failures) == 0, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare new metrics to baseline for gating")
    parser.add_argument("--new", type=str, required=True, help="Path to new metrics.json")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline metrics.json")
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.005,
        help="Max allowed accuracy drop (default 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.005,
        help="Max allowed F1 drop (default 0.005 = 0.5%%)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new = load_metrics(Path(args.new))
    baseline = load_metrics(Path(args.baseline))
    passed, failures = compare(
        new,
        baseline,
        accuracy_threshold=args.accuracy_threshold,
        f1_threshold=args.f1_threshold,
    )
    if passed:
        print("Metrics gate passed: no significant regression.")
        sys.exit(0)
    for msg in failures:
        print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
