import json, os
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from dataset import load, CLASSES
from model_def import build, timing_features

keras.utils.set_random_seed(42)
os.makedirs("reports", exist_ok=True)
os.makedirs("model", exist_ok=True)

d = load()
tr, te = d["split"] == "DS1", d["split"] == "DS2"


def inputs(m):
    return [d["X"][m][..., None],
            timing_features(d["rec"][m], d["pre_rr"][m], d["post_rr"][m])]


ytr, yte = d["y"][tr], d["y"][te]
w = np.sqrt(compute_class_weight("balanced", classes=np.arange(4), y=ytr))

model = build()
model.fit(inputs(tr), ytr, epochs=15, batch_size=256,
          class_weight=dict(enumerate(w)), verbose=2)

# ---- the one and only DS2 evaluation ----
pred = model.predict(inputs(te), verbose=0).argmax(1)
print(classification_report(yte, pred, target_names=CLASSES, digits=3, zero_division=0))
json.dump(classification_report(yte, pred, target_names=CLASSES, output_dict=True,
                                zero_division=0),
          open("reports/test_metrics.json", "w"), indent=2)

cm = confusion_matrix(yte, pred, normalize="true")
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig("reports/test_confusion_matrix.png", dpi=150, bbox_inches="tight")

# how much do S and F results depend on single patients?
rec_te = d["rec"][te]
for c, name in [(1, "S"), (3, "F")]:
    sel = yte == c
    per = (np.stack([rec_te[sel], pred[sel] == c], axis=1))
    print(f"\n{name} recall by record (largest first):")
    for r in sorted(set(per[:, 0]), key=lambda r: -(per[:, 0] == r).sum())[:3]:
        g = per[per[:, 0] == r, 1]
        print(f"  record {r}: {len(g)} beats, recall {g.mean():.2f}")

model.save("model/arrhythmia_model.keras")
print("model size (KB):", round(os.path.getsize("model/arrhythmia_model.keras") / 1e3))