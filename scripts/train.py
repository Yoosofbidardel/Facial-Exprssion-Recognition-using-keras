"""Convenience script for launching model training from a configuration file."""

import argparse
from pathlib import Path

from facial_expression_recognition.config import ExperimentConfig
from facial_expression_recognition.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the facial expression recognition model.")
    parser.add_argument("--config", type=Path, default=Path("configs/example_experiment.yaml"), help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    history = train(config)
    print("Training finished. Final epoch metrics:")
    for metric, values in history.items():
        print(f"  {metric}: {values[-1]:.4f}")


if __name__ == "__main__":
    main()

