"""Inference helpers for images and webcam streams."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_trained_model(model_path: Path):
    """Load a serialized Keras model."""

    return load_model(model_path)


def preprocess_face(frame, gray: bool = True, target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """Prepare a face image for model prediction."""

    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    expanded = np.expand_dims(normalized, axis=(0, -1)) if gray else np.expand_dims(normalized, axis=0)
    return expanded


def predict_image(model_path: Path, image_path: Path, labels: List[str]) -> Tuple[str, np.ndarray]:
    """Predict the dominant emotion for a single image."""

    model = load_trained_model(model_path)
    img = image.load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    scores = model.predict(img_array)[0]
    return labels[int(np.argmax(scores))], scores


def run_webcam(model_path: Path, labels: List[str], camera_index: int = 0, confidence_threshold: float = 0.5) -> None:
    """Start a webcam session for real-time emotion detection."""

    model = load_trained_model(model_path)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_color = frame[y : y + h, x : x + w]
                roi_processed = preprocess_face(roi_color, gray=True)
                scores = model.predict(roi_processed)[0]
                label_idx = int(np.argmax(scores))
                confidence = scores[label_idx]
                label = labels[label_idx] if label_idx < len(labels) else f"class_{label_idx}"

                if confidence >= confidence_threshold:
                    cv2.putText(
                        frame,
                        f"{label}: {confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Emotion Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
