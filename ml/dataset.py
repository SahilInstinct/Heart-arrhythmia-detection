import numpy as np

CLASSES = ["N", "S", "V", "F"]

def load(path="data/processed/beats.npz"):
    d = np.load(path)
    y = d["y"].astype(int)
    keep = y < 4                                   # drops Q
    return {
        "X": d["X"][keep], "y": y[keep], "rec": d["rec"][keep],
        "split": d["split"][keep],
        "pre_rr": d["pre_rr"][keep], "post_rr": d["post_rr"][keep],
    }
    
    