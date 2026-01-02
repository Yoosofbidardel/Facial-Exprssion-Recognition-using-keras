"""Run a webcam demo using a trained model."""

import argparse
from pathlib import Path

from facial_expression_recognition.inference import run_webcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam inference.")
    parser.add_argument("--model", type=Path, default=Path("outputs/model.h5"), help="Path to the trained Keras model.")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        help="Class labels in training order.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum confidence to render a label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_webcam(args.model, args.labels, camera_index=args.camera_index, confidence_threshold=args.confidence_threshold)


if __name__ == "__main__":
    main()

