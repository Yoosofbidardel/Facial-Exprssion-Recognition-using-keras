"""High-level training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from facial_expression_recognition.config import ExperimentConfig
from facial_expression_recognition.data import build_generators, resolve_class_names
from facial_expression_recognition.model import build_cnn


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_callbacks(config: ExperimentConfig) -> Tuple[EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]:
    training_cfg = config.training
    _ensure_output_dir(training_cfg.output_dir)

    checkpoint_path = training_cfg.output_dir / "model_weights.h5"
    early_stopping = EarlyStopping(monitor=training_cfg.monitor_metric, patience=training_cfg.early_stopping_patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=training_cfg.reduce_lr_factor, patience=training_cfg.reduce_lr_patience, min_lr=1e-5)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=training_cfg.monitor_metric,
        save_weights_only=True,
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    return early_stopping, reduce_lr, checkpoint


def train(config: ExperimentConfig) -> Dict[str, list]:
    """Run the training loop with the provided configuration."""

    train_gen, val_gen = build_generators(config.dataset)
    class_names = resolve_class_names(train_gen, config.dataset.class_names)

    model_cfg = config.model
    model_cfg.num_classes = len(class_names)

    model = build_cnn(model_cfg, input_shape=(config.dataset.image_size, config.dataset.image_size, 1 if config.dataset.color_mode == "grayscale" else 3))

    callbacks = _build_callbacks(config)

    history = model.fit(
        x=train_gen,
        steps_per_epoch=train_gen.n // train_gen.batch_size,
        epochs=config.training.epochs,
        validation_data=val_gen,
        validation_steps=val_gen.n // val_gen.batch_size,
        callbacks=list(callbacks),
    )

    model.save(config.training.output_dir / "model.h5")
    return history.history

