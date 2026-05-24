# src/model_architecture.py
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import os
def build_cnn_model(spec_shape, meta_shape, num_classes, config):
    # Input spektrum (1D CNN)
    spec_input = Input(shape=spec_shape, name="spectrum_input")
    x = layers.Conv1D(32, 5, activation="relu", padding="same")(spec_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)

    # Input metadata (one-hot)
    meta_input = Input(shape=meta_shape, name="metadata_input")
    m = layers.Dense(64, activation="relu")(meta_input)

    # Gabungkan keduanya
    combined = layers.concatenate([x, m])
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)

    # Output multilabel
    output = layers.Dense(num_classes, activation="sigmoid")(combined)

    model = Model(inputs=[spec_input, meta_input], outputs=output)
    return model
