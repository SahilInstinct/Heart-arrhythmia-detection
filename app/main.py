import time, logging
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

CLASSES = ["N", "S", "V", "F"]
MODEL_PATH = "model/arrhythmia_model.keras"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = keras.models.load_model(MODEL_PATH)
    model([np.zeros((1, 256, 1), "float32"), np.ones((1, 2), "float32")], training=False)  # warm-up
    state["model"] = model
    log.info("model loaded")
    yield
    state.clear()


app = FastAPI(title="Arrhythmia API", version="1.0", lifespan=lifespan)


class Beat(BaseModel):
    signal: list[float] = Field(min_length=256, max_length=256)
    pre_rr_ratio: float = Field(ge=0)
    post_rr_ratio: float = Field(ge=0)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/predict")
def predict(beat: Beat):
    t0 = time.perf_counter()
    x = np.asarray(beat.signal, dtype="float32")
    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="signal contains NaN or inf")
    x = ((x - x.mean()) / (x.std() + 1e-6)).reshape(1, 256, 1)
    t = np.clip([[beat.pre_rr_ratio, beat.post_rr_ratio]], 0, 3).astype("float32")

    probs = state["model"]([x, t], training=False).numpy()[0]
    i = int(probs.argmax())
    ms = (time.perf_counter() - t0) * 1000
    log.info("predict class=%s confidence=%.3f latency_ms=%.1f", CLASSES[i], probs[i], ms)
    return {"class": CLASSES[i], "confidence": float(probs[i]),
            "probabilities": dict(zip(CLASSES, map(float, probs))),
            "latency_ms": round(ms, 1)}