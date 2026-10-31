import sys
sys.path.insert(0, "ml")
import numpy as np, pandas as pd
from dataset import load
from model_def import timing_features

d = load()
te = d["split"] == "DS2"
X, y, rec = d["X"][te], d["y"][te], d["rec"][te]
T = timing_features(rec, d["pre_rr"][te], d["post_rr"][te])

rng = np.random.default_rng(7)
idx = np.concatenate([rng.choice(np.where(y == c)[0], 6, replace=False) for c in range(4)])

df = pd.DataFrame(X[idx], columns=[f"s{i}" for i in range(256)])
df["pre_rr_ratio"], df["post_rr_ratio"] = T[idx, 0], T[idx, 1]
df["label"], df["record"] = y[idx], rec[idx]
df.sample(frac=1, random_state=0).to_csv("ui/samples.csv", index=False)   # shuffled
print("saved", len(df), "beats")