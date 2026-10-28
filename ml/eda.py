import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from dataset import load, CLASSES

os.makedirs("reports", exist_ok=True)
d = load()
X, y, rec, split = d["X"], d["y"], d["rec"], d["split"]
names = np.array(CLASSES)[y]

# 1. Sanity: no NaNs, per-beat z-score really applied
print("NaNs:", np.isnan(X).sum(),
      "| mean of beat means:", X.mean(1).mean().round(3),
      "| mean of beat stds:", X.std(1).mean().round(3))

# 2. Class balance per split (%)
print((pd.crosstab(names, split, normalize="columns") * 100).round(2).reindex(CLASSES))

# 3. Mean waveform per class over 30 random beats
t = (np.arange(X.shape[1]) - X.shape[1] // 2) / 360 * 1000      # ms from the beat
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
for k, ax in enumerate(axes):
    Xk = X[y == k]
    for b in Xk[rng.choice(len(Xk), 30, replace=False)]:
        ax.plot(t, b, color="grey", alpha=0.2)
    ax.plot(t, Xk.mean(0), color="red")
    ax.set_title(f"{CLASSES[k]} (n={len(Xk)})"); ax.set_xlabel("ms from beat")
fig.savefig("reports/class_waveforms.png", dpi=150, bbox_inches="tight")

# 4. Beats per patient record and class (log colour, counts annotated)
counts = pd.crosstab(rec, names)[CLASSES]
rec_split = pd.Series(split, index=rec).groupby(level=0).first()
counts.index = [f"{r} {rec_split[r]}" for r in counts.index]
plt.figure(figsize=(5, 12))
sns.heatmap(np.log1p(counts), annot=counts, fmt="d", cmap="Blues", cbar=False)
plt.savefig("reports/beats_per_record.png", dpi=150, bbox_inches="tight")

# 5. Timing: seconds since the previous beat, by class
rr = pd.DataFrame({"class": names, "pre_rr": d["pre_rr"].clip(max=2.0)})
plt.figure(figsize=(6, 4))
sns.boxplot(data=rr, x="class", y="pre_rr", order=CLASSES)
plt.ylabel("seconds since previous beat")
plt.savefig("reports/rr_by_class.png", dpi=150, bbox_inches="tight")