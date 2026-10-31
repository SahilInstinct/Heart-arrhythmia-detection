import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from dataset import load, CLASSES

keras.utils.set_random_seed(42)
d = load()
m = d["split"] == "DS1"                      # DS2 stays untouched
X, y, rec = d["X"][m][..., None], d["y"][m], d["rec"][m]

baseline = pd.Series(d["pre_rr"][m]).groupby(rec).transform("median").to_numpy()
T = np.stack([d["pre_rr"][m] / baseline, d["post_rr"][m] / baseline], axis=1)
T = np.clip(T, 0, 3).astype("float32")


def build(use_timing):
    sig = keras.Input((256, 1))
    x = sig
    for filters, k in [(32, 7), (64, 5), (128, 3)]:
        x = layers.Conv1D(filters, k, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if filters < 128:
            x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    inputs = sig
    if use_timing:
        t = keras.Input((2,))
        x = layers.Concatenate()([x, layers.Dense(16, activation="relu")(t)])
        inputs = [sig, t]
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def run_cv(use_timing, epochs=15):
    oof = np.zeros(len(y), dtype=int)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(folds.split(X, y, groups=rec)):
        w = np.sqrt(compute_class_weight("balanced", classes=np.arange(4), y=y[tr]))
        pick = lambda idx: [X[idx], T[idx]] if use_timing else X[idx]
        model = build(use_timing)
        model.fit(pick(tr), y[tr], epochs=epochs, batch_size=256,
                  class_weight=dict(enumerate(w)), verbose=0)
        oof[va] = model.predict(pick(va), verbose=0).argmax(1)
        print(f"  fold {k}: records {sorted(set(rec[va]))}\n"
              f"          beats per class {np.bincount(y[va], minlength=4)}, "
              f"macro F1 {f1_score(y[va], oof[va], average='macro', zero_division=0):.3f}")
    return oof


print("Baseline (always N) macro F1:",
      round(f1_score(y, np.zeros_like(y), average="macro", zero_division=0), 3))
for name, use_timing in [("MORPHOLOGY ONLY", False), ("MORPHOLOGY + TIMING", True)]:
    print(f"\n=== {name} ===")
    oof = run_cv(use_timing)
    print(classification_report(y, oof, target_names=CLASSES, digits=3, zero_division=0))