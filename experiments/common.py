"""Shared evaluation / model code for the leakage-controlled experiments.

Protocol used throughout (identical for every dataset and regime):
  * GroupShuffleSplit by student id -> no student appears in both train and test.
  * All fitted preprocessing (one-hot, scaling, class weights) is learned on
    the training split only and applied unchanged to validation / test.
  * Torch models use the validation split for early stopping; the test split
    is scored exactly once.
  * Confidence intervals come from a student-level bootstrap (resampling
    *students*, not observations, then pooling all their observations).
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, mean_absolute_error)

SEED = 42
LABELS = [0, 1, 2, 3]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


# ── metrics ──────────────────────────────────────────────────────────────────
def ordinal_metrics(y_true, y_pred):
    """Point estimates for every metric reported in the paper."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = np.abs(y_pred - y_true)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS)),
        "one_off_accuracy": float((err <= 1).mean()),
        "severe_underestimation": float((y_pred <= y_true - 2).mean()),
    }


def per_class_f1(y_true, y_pred):
    f1s = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    return {f"level_{k}": float(v) for k, v in zip(LABELS, f1s)}


def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return cm.astype(int).tolist()


# ── student-level bootstrap ──────────────────────────────────────────────────
def student_bootstrap_ci(pred_df, n_boot=1000, seed=SEED, metrics=("accuracy", "macro_f1", "mae", "qwk")):
    """95% CI for each metric.

    pred_df: DataFrame with columns [student, y_true, y_pred]. Resampling unit
    is the *student*; every observation belonging to a drawn student enters
    the resample. Returns {metric: (lo, hi)} plus point estimates.
    """
    rng = np.random.default_rng(seed)
    by_student = {s: (g.y_true.to_numpy(), g.y_pred.to_numpy())
                  for s, g in pred_df.groupby("student")}
    students = list(by_student)
    point = ordinal_metrics(pred_df.y_true, pred_df.y_pred)
    draws = {m: np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        chosen = rng.choice(students, size=len(students), replace=True)
        yt = np.concatenate([by_student[s][0] for s in chosen])
        yp = np.concatenate([by_student[s][1] for s in chosen])
        m = ordinal_metrics(yt, yp)
        for name in metrics:
            draws[name][b] = m[name]
    out = {}
    for name in metrics:
        lo, hi = np.percentile(draws[name], [2.5, 97.5])
        out[name] = {"point": point[name], "lo": float(lo), "hi": float(hi)}
    return out


# ── flat models ──────────────────────────────────────────────────────────────
def make_flat_models(n_classes=4):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=15,
                                               n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                       random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 tree_method="hist", n_jobs=-1,
                                 random_state=SEED, eval_metric="mlogloss"),
    }


# ── sequence models (PyTorch, MPS when available) ────────────────────────────
def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _AdditiveAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = torch.nn.Linear(dim, dim // 2)
        self.v = torch.nn.Linear(dim // 2, 1, bias=False)

    def forward(self, h):                       # h: (B, T, D)
        e = self.v(torch.tanh(self.W(h)))       # (B, T, 1)
        a = torch.softmax(e, dim=1)
        return (a * h).sum(dim=1), a.squeeze(-1)


class SequenceClassifier(torch.nn.Module):
    """LSTM or BiLSTM-Attention over short token sequences."""

    def __init__(self, input_dim, hidden=64, n_classes=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden, batch_first=True,
                                 bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.attn = _AdditiveAttention(d) if bidirectional else None
        if self.attn is None:                   # vanilla LSTM: take last state
            self.head_in = d
        else:
            self.head_in = d
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.head_in, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, n_classes),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.attn is not None:
            ctx, alpha = self.attn(out)
            return self.head(ctx), alpha
        return self.head(out[:, -1, :]), None


def fit_sequence_model(name, X_tr, y_tr, X_val, y_val, n_classes=4, max_epochs=60, patience=8):
    """Train SequenceClassifier with class weights; early stop on val macro-F1."""
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything()
    device = _device()
    bidirectional = "BiLSTM" in name
    model = SequenceClassifier(X_tr.shape[2], n_classes=n_classes,
                               bidirectional=bidirectional).to(device)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.long)),
                        batch_size=256, shuffle=True)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_f1, best_state, wait = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)[0].argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val, val_pred, average="macro", labels=LABELS, zero_division=0)
        if f1 > best_f1:
            best_f1, wait = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_sequence(model, X):
    device = _device()
    model.eval()
    probs, alphas = [], []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.tensor(X[i:i + 1024], dtype=torch.float32, device=device)
            logits, alpha = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            if alpha is not None:
                alphas.append(alpha.cpu().numpy())
    p = np.vstack(probs)
    alpha = np.vstack(alphas) if alphas else None
    return p.argmax(axis=1), p, alpha


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ── result persistence ───────────────────────────────────────────────────────
def save_results(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"saved -> {path}")


def evaluate_run(name, y_true, y_pred, students, results_dir, dataset, regime):
    """Metrics + student-level bootstrap CI + per-student predictions for one model."""
    pred_df = pd.DataFrame({"student": students, "y_true": y_true, "y_pred": y_pred})
    ci = student_bootstrap_ci(pred_df)
    payload = {
        "dataset": dataset, "regime": regime, "model": name,
        "metrics": ordinal_metrics(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
        "bootstrap_ci": ci,
    }
    tag = f"{dataset}_{regime}_{name}".replace(" ", "")
    pred_df.to_csv(os.path.join(results_dir, f"preds_{tag}.csv"), index=False)
    return payload
