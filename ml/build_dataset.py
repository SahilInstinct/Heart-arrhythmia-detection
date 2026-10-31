from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/mitdb")
OUT = Path("data/processed")
FS = 360          # MIT-BIH sampling rate (Hz)
HALF = 128        # samples either side of the beat -> 256-sample window (~0.71 s)

# Inter-patient split (de Chazal et al.): no patient appears in both sets.
# Paced records 102, 104, 107, 217 are excluded, as AAMI recommends.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]                 # train
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]                 # test

# AAMI classes: 0=N, 1=S, 2=V, 3=F, 4=Q
AAMI = {}
for cls, symbols in enumerate(["NLRej", "AaJS", "VE", "F", "/fQ"]):
    for s in symbols:
        AAMI[s] = cls
CLASS_NAMES = "NSVFQ"

def load_lead(rec):
    df = pd.read_csv(RAW / f"{rec}.csv")
    df.columns = [c.strip("'") for c in df.columns]
    if "MLII" not in df.columns:
        return None
    return df["MLII"].to_numpy(dtype=np.float32)



def load_beats(rec):
    positions, labels = [], []
    with open(RAW / f"{rec}annotations.txt") as f:
        next(f)                                   # header line
        for line in f:
            parts = line.split()
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            if parts[2] in AAMI:                  # skips '+', '~', '|', etc.
                positions.append(int(parts[1]))
                labels.append(AAMI[parts[2]])
    return np.array(positions), np.array(labels)


def main():
    X, y, rec_ids, pre_rr_all, post_rr_all = [], [], [], [], []

    for rec in sorted(DS1 + DS2):
        lead = load_lead(rec)
        if lead is None:
            print(f"record {rec}: no MLII lead, skipped")
            continue
        pos, lab = load_beats(rec)
        rr = np.diff(pos) / FS                    # seconds between consecutive beats
        pre_rr = np.r_[rr[0], rr]                 # edge beats reuse their neighbour
        post_rr = np.r_[rr, rr[-1]]

        for p, l, a, b in zip(pos, lab, pre_rr, post_rr):
            if p - HALF < 0 or p + HALF > len(lead):
                continue
            beat = lead[p - HALF:p + HALF]
            beat = (beat - beat.mean()) / (beat.std() + 1e-6)   # per-beat z-score
            X.append(beat); y.append(l); rec_ids.append(rec)
            pre_rr_all.append(a); post_rr_all.append(b)

    X = np.stack(X).astype("float32")
    y = np.array(y, dtype="int8")
    rec_ids = np.array(rec_ids, dtype="int16")
    split = np.where(np.isin(rec_ids, DS1), "DS1", "DS2")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "beats.npz", X=X, y=y, rec=rec_ids, split=split,
                        pre_rr=np.array(pre_rr_all, dtype="float32"),
                        post_rr=np.array(post_rr_all, dtype="float32"))

    print("X shape:", X.shape)
    names = pd.Series([CLASS_NAMES[k] for k in y], name="class")
    print(pd.crosstab(names, pd.Series(split, name="split"), margins=True))


if __name__ == "__main__":
    main()