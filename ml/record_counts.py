import numpy as np, pandas as pd
from dataset import load, CLASSES

d = load()
df = pd.DataFrame({"rec": d["rec"], "cls": np.array(CLASSES)[d["y"]], "split": d["split"]})
t = pd.crosstab([df.split, df.rec], df.cls)[CLASSES]

for c in ["S", "F"]:
    for s in ["DS1", "DS2"]:
        col = t.loc[s][c].sort_values(ascending=False)
        print(f"{c} in {s}: total {col.sum()}, top records {col.head(4).to_dict()}, "
              f"top-1 share {col.iloc[0] / col.sum():.0%}")