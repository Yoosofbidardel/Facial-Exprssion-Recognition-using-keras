"""Command-line interface for training and inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from facial_expression_recognition.config import ExperimentConfig
from facial_expression_recognition.inference import predict_image, run_webcam
from facial_expression_recognition.training import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Facial expression recognition utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new model from a YAML config.")
    train_parser.add_argument("--config", type=Path, required=True, help="Path to the experiment configuration YAML.")

    predict_parser = subparsers.add_parser("predict-image", help="Predict facial expression for a still image.")
    predict_parser.add_argument("--model", type=Path, required=True, help="Path to a trained model (.h5).")
    predict_parser.add_argument("--labels", type=str, nargs="+", required=True, help="Class labels matching the training order.")
    predict_parser.add_argument("--image", type=Path, required=True, help="Path to the image to evaluate.")

    webcam_parser = subparsers.add_parser("predict-webcam", help="Run real-time prediction from a webcam feed.")
    webcam_parser.add_argument("--model", type=Path, required=True, help="Path to a trained model (.h5).")
    webcam_parser.add_argument("--labels", type=str, nargs="+", required=True, help="Class labels matching the training order.")
    webcam_parser.add_argument("--camera-index", type=int, default=0, help="Camera index for OpenCV VideoCapture.")
    webcam_parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum confidence to render labels.")

    return parser


def _handle_train(config_path: Path) -> None:
    config = ExperimentConfig.load(config_path)
    history = train(config)
    print("Training complete. Final metrics:")
    for metric, values in history.items():
        print(f"{metric}: {values[-1]:.4f}")


def _handle_predict_image(model_path: Path, labels: List[str], image_path: Path) -> None:
    label, scores = predict_image(model_path, image_path, labels)
    print(f"Prediction: {label}")
    print("Class probabilities:")
    for cls, score in zip(labels, scores):
        print(f"  {cls}: {score:.4f}")


def _handle_predict_webcam(model_path: Path, labels: List[str], camera_index: int, confidence_threshold: float) -> None:
    run_webcam(model_path, labels, camera_index=camera_index, confidence_threshold=confidence_threshold)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _handle_train(args.config)
    elif args.command == "predict-image":
        _handle_predict_image(args.model, args.labels, args.image)
    elif args.command == "predict-webcam":
        _handle_predict_webcam(args.model, args.labels, args.camera_index, args.confidence_threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

