import numpy as np, pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


def timing_features(rec, pre_rr, post_rr):
    """Beat intervals relative to each patient's own median interval."""
    base = pd.Series(pre_rr).groupby(rec).transform("median").to_numpy()
    t = np.stack([pre_rr / base, post_rr / base], axis=1)
    return np.clip(t, 0, 3).astype("float32")


def build():
    sig = keras.Input((256, 1), name="signal")
    x = sig
    for filters, k in [(32, 7), (64, 5), (128, 3)]:
        x = layers.Conv1D(filters, k, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if filters < 128:
            x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    t = keras.Input((2,), name="timing")
    x = layers.Concatenate()([x, layers.Dense(16, activation="relu")(t)])
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(4, activation="softmax")(x)
    model = keras.Model([sig, t], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model