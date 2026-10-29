import os, json
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from dataset import load, CLASSES

keras.utils.set_random_seed(42)
os.makedirs("reports", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---- data: DS1 only; DS2 is not touched here ----
d = load()
m = d["split"] == "DS1"
X, y, rec = d["X"][m], d["y"][m], d["rec"][m]

# validation = whole records held out
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr, va = next(splitter.split(X, y, groups=rec))
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
print("validation records:", sorted(set(rec[va])))
print("train counts [N S V F]:", np.bincount(ytr, minlength=4))
print("val counts   [N S V F]:", np.bincount(yva, minlength=4))

# ---- baselines ----
mf1 = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
print("\nBaseline A (always N)      macro F1:", round(mf1(yva, np.zeros_like(yva)), 3))
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42).fit(Xtr, ytr)
print("Baseline B (grad. boosting) macro F1:", round(mf1(yva, hgb.predict(Xva)), 3))

# ---- CNN ----
Xtr3, Xva3 = Xtr[..., None], Xva[..., None]
w = compute_class_weight("balanced", classes=np.arange(4), y=ytr)
class_weight = dict(enumerate(np.sqrt(w)))

model = keras.Sequential([
    keras.layers.Input(shape=(256, 1)),
    keras.layers.Conv1D(32, 7, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy")


class BestMacroF1(keras.callbacks.Callback):
    """Track validation macro F1, stop when it stalls, restore the best weights."""
    def __init__(self, X, y, patience=6):
        super().__init__()
        self.X, self.y, self.patience = X, y, patience

    def on_train_begin(self, logs=None):
        self.best, self.wait, self.best_w = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        f1 = mf1(self.y, self.model.predict(self.X, verbose=0).argmax(1))
        print(f"  val macro F1 = {f1:.3f}")
        if f1 > self.best:
            self.best, self.wait, self.best_w = f1, 0, self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_w)


model.fit(Xtr3, ytr, epochs=40, batch_size=256, class_weight=class_weight,
          callbacks=[BestMacroF1(Xva3, yva)], verbose=2)

# ---- validation report ----
pred = model.predict(Xva3, verbose=0).argmax(1)
print("\nCNN validation report")
print(classification_report(yva, pred, target_names=CLASSES, digits=3, zero_division=0))

cm = confusion_matrix(yva, pred, normalize="true")
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig("reports/val_confusion_matrix.png", dpi=150, bbox_inches="tight")

json.dump(classification_report(yva, pred, target_names=CLASSES, output_dict=True,
                                zero_division=0),
          open("reports/val_metrics.json", "w"), indent=2)

model.save("model/arrhythmia_model.keras")
print("model size (KB):", round(os.path.getsize("model/arrhythmia_model.keras") / 1e3))