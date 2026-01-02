"""Configuration utilities for facial expression recognition experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Paths and settings for training and validation data."""

    train_dir: Path
    val_dir: Path
    image_size: int = 48
    batch_size: int = 64
    color_mode: str = "grayscale"
    class_names: Optional[List[str]] = None
    augment: bool = True


@dataclass
class ModelConfig:
    """Hyperparameters for building the convolutional network."""

    base_filters: int = 64
    dense_units: List[int] = field(default_factory=lambda: [256, 512])
    dropout_rate: float = 0.25
    learning_rate: float = 5e-4
    num_classes: int = 7


@dataclass
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 30
    monitor_metric: str = "val_accuracy"
    output_dir: Path = Path("outputs")
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.1


@dataclass
class ExperimentConfig:
    """Top-level configuration container."""

    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration values from a YAML file."""

        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        num_classes_override = raw.get("model", {}).get("num_classes")
        dataset_cfg = DatasetConfig(
            train_dir=Path(raw["dataset"]["train_dir"]),
            val_dir=Path(raw["dataset"]["val_dir"]),
            image_size=raw["dataset"].get("image_size", 48),
            batch_size=raw["dataset"].get("batch_size", 64),
            color_mode=raw["dataset"].get("color_mode", "grayscale"),
            class_names=raw["dataset"].get("class_names"),
            augment=raw["dataset"].get("augment", True),
        )

        inferred_classes = len(dataset_cfg.class_names) if dataset_cfg.class_names else 7
        model_cfg = ModelConfig(
            base_filters=raw.get("model", {}).get("base_filters", 64),
            dense_units=raw.get("model", {}).get("dense_units", [256, 512]),
            dropout_rate=raw.get("model", {}).get("dropout_rate", 0.25),
            learning_rate=raw.get("model", {}).get("learning_rate", 5e-4),
            num_classes=num_classes_override or inferred_classes,
        )

        training_cfg = TrainingConfig(
            epochs=raw.get("training", {}).get("epochs", 30),
            monitor_metric=raw.get("training", {}).get("monitor_metric", "val_accuracy"),
            output_dir=Path(raw.get("training", {}).get("output_dir", "outputs")),
            early_stopping_patience=raw.get("training", {}).get("early_stopping_patience", 5),
            reduce_lr_patience=raw.get("training", {}).get("reduce_lr_patience", 2),
            reduce_lr_factor=raw.get("training", {}).get("reduce_lr_factor", 0.1),
        )

        return cls(dataset=dataset_cfg, model=model_cfg, training=training_cfg)
