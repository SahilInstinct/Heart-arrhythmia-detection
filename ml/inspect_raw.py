from pathlib import Path
import pandas as pd
        
d = Path("data/mitdb")
records = sorted(p.stem for p in d.glob("*.csv"))
print(len(records), "signal files:", records)
print(len(list(d.glob("*annotations.txt"))), "annotation files")

sig = pd.read_csv(d / "100.csv")
print("\n100.csv shape:", sig.shape)
print("columns:", sig.columns.tolist())
print(sig.head(3))

print("\n100annotations.txt, first 8 lines (raw):")
with open(d / "100annotations.txt") as f:
    for _ in range(8):
        print(repr(f.readline()))