"""Dataset preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from facial_expression_recognition.config import DatasetConfig


def _build_datagen(config: DatasetConfig, training: bool) -> ImageDataGenerator:
    """Create an image data generator with optional augmentation."""

    if training and config.augment:
        return ImageDataGenerator(horizontal_flip=True, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
    return ImageDataGenerator()


def build_generators(config: DatasetConfig) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """Construct training and validation generators."""

    train_gen = _build_datagen(config, training=True).flow_from_directory(
        Path(config.train_dir),
        target_size=(config.image_size, config.image_size),
        color_mode=config.color_mode,
        batch_size=config.batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = _build_datagen(config, training=False).flow_from_directory(
        Path(config.val_dir),
        target_size=(config.image_size, config.image_size),
        color_mode=config.color_mode,
        batch_size=config.batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    return train_gen, val_gen


def resolve_class_names(generator, explicit_names: List[str] | None) -> List[str]:
    """Resolve class names using configured values or generator metadata."""

    if explicit_names:
        return explicit_names
    return list(generator.class_indices.keys())

