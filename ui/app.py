import os
from pathlib import Path
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
CODES = ["N", "S", "V", "F"]
NAMES = {"N": "Normal", "S": "Supraventricular ectopic",
         "V": "Ventricular ectopic", "F": "Fusion beat"}
SIG_COLS = [f"s{i}" for i in range(256)]

st.set_page_config(page_title="Arrhythmia classifier")
st.title("Heartbeat arrhythmia classifier")
st.caption("Research demo trained on the MIT-BIH database. Not a medical device.")


@st.cache_data
def load_samples():
    return pd.read_csv(Path(__file__).parent / "samples.csv")


df = load_samples()
source = st.radio("Beat source", ["Sample beat", "Upload CSV"], horizontal=True)
signal, truth, record = None, None, None
pre, post = 1.0, 1.0

if source == "Sample beat":
    st.caption("Beats from patients the model never saw during training.")
    k = st.selectbox("Choose a beat", range(len(df)), format_func=lambda i: f"Sample beat #{i + 1}")
    row = df.iloc[k]
    signal = row[SIG_COLS].to_numpy(dtype=float)
    pre, post = float(row.pre_rr_ratio), float(row.post_rr_ratio)
    truth, record = CODES[int(row.label)], int(row.record)
else:
    f = st.file_uploader("CSV with one beat: 256 values in a single row", type="csv")
    help_txt = "Gap compared with the patient's usual gap. 1.0 = on schedule, 0.7 = 30% early."
    pre = st.number_input("Interval before this beat (relative)", 0.0, 3.0, 1.0, 0.05, help=help_txt)
    post = st.number_input("Interval after this beat (relative)", 0.0, 3.0, 1.0, 0.05, help=help_txt)
    if f:
        vals = pd.read_csv(f, header=None).to_numpy().flatten()
        if vals.size == 256:
            signal = vals.astype(float)
        else:
            st.error(f"Expected 256 values, got {vals.size}.")

if signal is not None:
    t_ms = (np.arange(256) - 128) / 360 * 1000
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot(t_ms, signal)
    ax.axvline(0, color="red", alpha=0.3, linestyle="--")
    ax.set_xlabel("Milliseconds from annotated beat")
    ax.set_ylabel("Normalised amplitude")
    st.pyplot(fig)
    st.caption(f"Timing context: {pre:.2f}x usual interval before, {post:.2f}x after.")

    if st.button("Classify beat", type="primary"):
        payload = {"signal": [float(v) for v in signal],
                   "pre_rr_ratio": pre, "post_rr_ratio": post}
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            r.raise_for_status()
            res = r.json()
        except requests.RequestException as e:
            st.error(f"Prediction service unavailable: {e}")
        else:
            c = res["class"]
            st.metric("Prediction", f"{c}: {NAMES[c]}",
                      f"model score {res['confidence']:.0%}", delta_color="off")
            st.bar_chart(pd.Series(res["probabilities"]))
            st.caption(f"Server latency: {res['latency_ms']} ms. The score is the model's own "
                       "certainty, not a calibrated probability.")
            if c in ("S", "F"):
                st.warning("S and F predictions are unreliable: on unseen patients this model "
                           "found few of these beats and produced many false alarms.")
            if truth is not None:
                msg = f"Cardiologist label: {truth} (record {record})"
                (st.success if truth == c else st.error)(msg)