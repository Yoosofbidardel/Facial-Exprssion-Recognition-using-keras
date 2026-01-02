"""Model creation utilities."""

from __future__ import annotations

from typing import Tuple

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from facial_expression_recognition.config import ModelConfig


def _conv_block(filters: int, kernel_size: Tuple[int, int] = (3, 3)):
    """Define a convolutional block with BN, ReLU, pooling, and dropout."""

    return [
        Conv2D(filters, kernel_size, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
    ]


def build_cnn(config: ModelConfig, input_shape=(48, 48, 1)) -> Model:
    """Create the CNN architecture inspired by the original project."""

    model = Sequential()
    model.add(Conv2D(config.base_filters, (3, 3), padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(config.dropout_rate))

    for filters in (config.base_filters * 2, config.base_filters * 8, config.base_filters * 8):
        for layer in _conv_block(filters):
            model.add(layer)
        model.add(Dropout(config.dropout_rate))

    model.add(Flatten())

    for units in config.dense_units:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(config.dropout_rate))

    model.add(Dense(config.num_classes, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

