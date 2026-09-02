"""Egreen Quanta — working local digital-signature telemetry prototype."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from torch import nn
from torch.nn import functional as F


SEED = 42
FEATURES = ["Nonce entropy", "Verification latency", "Failed handshakes", "Payload size", "Side-channel variance"]
CLASSES = ["NORMAL", "WEAK NONCE", "SIDE CHANNEL"]
SCENARIOS = {"Normal baseline": 0, "Weak nonce anomaly": 1, "Side-channel anomaly": 2}
THRESHOLD = 0.50


def bootstrap_telemetry(label: int, rng: np.random.Generator) -> np.ndarray:
    """Starter data for the neural workflow; it is not live telemetry."""
    values = np.array([rng.normal(7.96, .04), rng.normal(.20, .06), rng.poisson(.05), rng.normal(1.0, .25), rng.normal(.001, .001)], dtype=np.float32)
    if label == 1:
        values += np.array([-rng.uniform(1.0, 4.5), rng.uniform(.15, 1.0), rng.integers(1, 5), 0, 0])
    elif label == 2:
        values += np.array([0, rng.uniform(.8, 4), rng.integers(0, 3), 0, rng.uniform(.03, .25)])
    return np.clip(values, [0, 0, 0, .001, 0], [8, 50, 100, 10000, 5]).astype(np.float32)


def make_dataset(n_per_class: int = 650):
    rng = np.random.default_rng(SEED)
    y = np.repeat(np.arange(3), n_per_class)
    x = np.vstack([bootstrap_telemetry(label, rng) for label in y])
    order = rng.permutation(len(y))
    return x[order], y[order]


def graph_batch(x: torch.Tensor) -> torch.Tensor:
    wallet = x
    gateway = x * torch.tensor([1, 1.03, 1, 1, 1.05]) + torch.tensor([0, 0, .15, 0, 0])
    validator = x * torch.tensor([1, 1.06, 1, 1, 1.02]) + torch.tensor([0, 0, .25, 0, 0])
    return torch.stack((wallet, gateway, validator), dim=1)


ADJ = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
ADJ = torch.diag(torch.pow(ADJ.sum(1), -.5)) @ ADJ @ torch.diag(torch.pow(ADJ.sum(1), -.5))


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = nn.Linear(5, 16), nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.first(torch.matmul(ADJ, x)))
        return F.relu(self.second(torch.matmul(ADJ, x))).mean(1)


class HybridDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GraphEncoder()
        self.quantum_projection = nn.Linear(4, 4)
        self.classifier = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(.12), nn.Linear(32, 3))

    def forward(self, telemetry):
        theta = np.pi * torch.sigmoid(telemetry[:, :2])
        amp_a = torch.stack((torch.cos(theta[:, 0]), torch.sin(theta[:, 0])), 1)
        amp_b = torch.stack((torch.cos(theta[:, 1]), torch.sin(theta[:, 1])), 1)
        quantum = self.quantum_projection(torch.einsum("bi,bj->bij", amp_a, amp_b).flatten(1))
        return self.classifier(torch.cat((self.gcn(graph_batch(telemetry)), quantum), 1))


@dataclass
class TrainingResult:
    model: HybridDetector
    mean: np.ndarray
    std: np.ndarray
    accuracy: float
    loss: list[float]
    confusion: np.ndarray


@st.cache_resource(show_spinner="Training hybrid neural detector…")
def train_detector() -> TrainingResult:
    torch.manual_seed(SEED)
    x, y = make_dataset()
    split = int(.8 * len(y))
    x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]
    mean, std = x_train.mean(0), x_train.std(0).clip(1e-6)
    train_x = torch.tensor((x_train - mean) / std)
    test_x = torch.tensor((x_test - mean) / std)
    train_y, test_y = torch.tensor(y_train), torch.tensor(y_test)
    model, optimizer = HybridDetector(), torch.optim.AdamW(HybridDetector().parameters(), lr=.003)
    # Optimizer must own this model's parameters (kept explicit to prevent a silent training bug).
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    losses = []
    for _ in range(42):
        model.train(); optimizer.zero_grad()
        value = F.cross_entropy(model(train_x), train_y)
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        losses.append(float(value.detach()))
    model.eval()
    with torch.no_grad():
        predicted = model(test_x).argmax(1).numpy()
    cm = np.zeros((3, 3), dtype=int)
    for actual, pred in zip(y_test, predicted): cm[actual, pred] += 1
    return TrainingResult(model, mean, std, float((predicted == y_test).mean()), losses, cm)


@st.cache_resource
def signing_key():
    return ec.generate_private_key(ec.SECP256R1())


def shannon_entropy(data: bytes) -> float:
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counts[counts > 0] / len(data)
    return float(-(probabilities * np.log2(probabilities)).sum())


def live_telemetry(message: bytes, samples: int, test_invalid_signature: bool):
    """Perform local ECDSA operations and collect actual timing telemetry."""
    key = signing_key(); timings, signatures = [], []
    for _ in range(samples):
        signature = key.sign(message, ec.ECDSA(hashes.SHA256()))
        started = time.perf_counter()
        key.public_key().verify(signature, message, ec.ECDSA(hashes.SHA256()))
        timings.append((time.perf_counter() - started) * 1000)
        signatures.append(signature)
    failed = 0
    if test_invalid_signature:
        bad = bytearray(signatures[-1]); bad[-1] ^= 1
        try:
            key.public_key().verify(bytes(bad), message, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            failed = 1
    features = np.array([shannon_entropy(signatures[-1]), np.mean(timings), failed, len(message) / 1024, np.var(timings)], dtype=np.float32)
    return features, True, len(signatures[-1]), float(np.mean(timings))


def infer(training: TrainingResult, raw: np.ndarray):
    normalized = torch.tensor(((raw - training.mean) / training.std)[None, :])
    training.model.eval()
    with torch.no_grad(): probabilities = F.softmax(training.model(normalized), 1)[0].numpy()
    anomaly = float(1 - probabilities[0]); index = int(probabilities.argmax())
    label = "NORMAL" if anomaly < THRESHOLD else CLASSES[index]
    return probabilities, anomaly, label, float(probabilities[index])


def probability_chart(probabilities):
    colors = ["#35d39e", "#ffb547", "#ff5c77"]
    fig = go.Figure(go.Bar(x=CLASSES, y=probabilities, marker_color=colors, text=[f"{p:.1%}" for p in probabilities], textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=8, r=8, t=28, b=8), yaxis=dict(range=[0, 1], tickformat=".0%"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def network_chart(danger: bool):
    edge = "#ff5c77" if danger else "#35d39e"; nodes = ["Signer", "API gateway", "Validator"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1, None, 1, 2], y=[0, .6, None, .6, 0], mode="lines", line=dict(color=edge, width=5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 1, 2], y=[0, .6, 0], mode="markers+text", text=nodes, textposition="top center", marker=dict(size=38, color="#14233d", line=dict(color=edge, width=2))))
    fig.update_layout(height=280, margin=dict(l=8, r=8, t=28, b=8), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


st.set_page_config(page_title="Egreen Quanta", page_icon="🛡️", layout="wide")
st.markdown("""<style>
 .stApp {background: radial-gradient(circle at 80% 0%, #18345c 0, #0b1220 38%, #080d16 100%); color:#eff6ff}
 [data-testid="stMetric"] {background:#111d31; border:1px solid #243958; border-radius:14px; padding:14px}
 .hero {padding:10px 0 18px}.eyebrow{color:#35d39e;letter-spacing:.14em;font-size:.78rem;font-weight:700}.muted{color:#9eb0ca}
 </style>""", unsafe_allow_html=True)
st.markdown("<div class='hero'><div class='eyebrow'>LOCAL CRYPTOGRAPHY PROTOTYPE</div><h1>🛡️ Egreen Quanta</h1><p class='muted'>Real ECDSA execution with hybrid neural telemetry triage</p></div>", unsafe_allow_html=True)

training = train_detector()
if "history" not in st.session_state: st.session_state.history = []

with st.sidebar:
    st.header("Live scan control")
    payload = st.text_area("Message to sign", value="EgreenQuanta transaction payload 88492", max_chars=10000)
    samples = st.slider("Verification samples", 5, 100, 20)
    test_invalid = st.checkbox("Perform one invalid-signature check", value=True)
    scan = st.button("Sign, verify & scan", type="primary", use_container_width=True)
    st.divider()
    st.caption("The scan executes local ECDSA P-256. It does not observe remote traffic or recover ECDSA nonces.")

if scan or "latest" not in st.session_state:
    raw, verified, sig_size, latency = live_telemetry(payload.encode("utf-8"), samples, test_invalid)
    probs, anomaly, detected, confidence = infer(training, raw)
    st.session_state.latest = dict(raw=raw, probabilities=probs, anomaly=anomaly, detected=detected, confidence=confidence, verified=verified, sig_size=sig_size, latency=latency, scenario="Live local ECDSA scan")
    st.session_state.history.insert(0, {"Scenario": "Live local ECDSA scan", "Detected": detected, "Threat score": round(anomaly, 4), "Confidence": round(confidence, 4)})

r = st.session_state.latest
status = "ALERT" if r["anomaly"] >= THRESHOLD else "NOMINAL"
st.markdown(f"<div class='eyebrow'>{status} · {r['detected']} · {r['confidence']:.0%} MODEL CONFIDENCE</div>", unsafe_allow_html=True)
metrics = st.columns(5)
for col, name, value in zip(metrics, FEATURES, r["raw"]):
    display = f"{value:.2f}" if name != "Failed handshakes" else str(int(value))
    col.metric(name, display)

left, right = st.columns(2)
with left:
    st.subheader("Threat classification")
    (st.error if status == "ALERT" else st.success)(f"{status}: {r['detected']}")
    st.metric("Anomaly score", f"{r['anomaly']:.1%}", help="One minus the normal-class probability.")
    st.plotly_chart(probability_chart(r["probabilities"]), use_container_width=True)
with right:
    st.subheader("Transaction path")
    st.plotly_chart(network_chart(status == "ALERT"), use_container_width=True)
    st.caption("Graph roles model a transaction path; this prototype does not discover network topology automatically.")

st.divider()
crypto, defense, evaluation = st.columns(3)
with crypto:
    st.subheader("Cryptographic validation")
    st.metric("ECDSA verification", "PASSED" if r["verified"] else "FAILED")
    st.caption(f"P-256 signature: {r['sig_size']} bytes · sign + verify: {r['latency']:.2f} ms")
with defense:
    st.subheader("Migration posture")
    if status == "ALERT": st.warning("Review transaction and initiate your approved PQC migration procedure.")
    else: st.info("Classical path retained; threat score is below the review threshold.")
    st.caption("ML-DSA migration requires separately managed keys and compatibility testing; it is not automatically substituted for ECDSA here.")
with evaluation:
    st.subheader("Model health")
    st.metric("Bootstrap hold-out accuracy", f"{training.accuracy:.1%}")
    with st.expander("Training details"):
        st.line_chart(pd.DataFrame({"cross-entropy loss": training.loss}))
        st.dataframe(pd.DataFrame(training.confusion, index=CLASSES, columns=CLASSES), use_container_width=True)

with st.expander("Recent scans", expanded=False):
    st.dataframe(pd.DataFrame(st.session_state.history[:12]), use_container_width=True, hide_index=True)

st.caption("Live measurements are actual local cryptographic measurements. Train the default neural model on labelled operational telemetry before relying on its alerts.")
