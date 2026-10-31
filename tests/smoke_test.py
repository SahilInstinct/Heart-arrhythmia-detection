import sys
sys.path.insert(0, "ml")
import numpy as np, requests
from dataset import load, CLASSES
from model_def import timing_features

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
d = load()
te = d["split"] == "DS2"
X, y = d["X"][te], d["y"][te]
T = timing_features(d["rec"][te], d["pre_rr"][te], d["post_rr"][te])

print(requests.get(f"{URL}/health").json())
rng = np.random.default_rng(1)
for c in range(4):
    i = rng.choice(np.where(y == c)[0])
    r = requests.post(f"{URL}/predict", json={
        "signal": X[i].tolist(),
        "pre_rr_ratio": float(T[i, 0]), "post_rr_ratio": float(T[i, 1])}).json()
    print(f"true {CLASSES[c]} -> predicted {r['class']} ({r['confidence']:.2f}), {r['latency_ms']} ms")