"""EdNet-KT1 contamination dose-response.

The prospective label (second-half mean correctness) is FIXED; the feature
window grows from the mid-sequence cut (p=0.5) into the labelled period
(p=1.0 = the full sequence, whose correctness determines the label). The
curve therefore shows the graded approach to the reconstruction ceiling on a
cohort that is byte-identical to the main experiment (pinned user list).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (evaluate_run, fit_sequence_model, ordinal_metrics, predict_sequence,
                    save_results, seed_everything)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

DATA = "/Users/Apple/Desktop/灵感投稿/教育会议/EI/03_Data"
KT1 = f"{DATA}/KT1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N_BINS = 10
CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MODELS = ["XGBoost", "BiLSTM-Attention"]


def label_from_rate(r):
    return 0 if r >= 0.85 else 1 if r >= 0.70 else 2 if r >= 0.50 else 3


def load_cohort():
    """Read exactly the pinned users; return per-user arrays + prospective label."""
    qmap = pd.read_csv(f"{DATA}/contents/questions.csv", low_memory=False)
    qmap = dict(zip(qmap.question_id, qmap.correct_answer))
    uids = pd.read_csv(os.path.join(OUT, "ednet_sample_uids.csv")).uid.astype(str).tolist()
    users = {}
    for i, uid in enumerate(uids):
        path = os.path.join(KT1, uid + ".csv")
        try:
            df = pd.read_csv(path, usecols=["user_answer", "question_id", "elapsed_time", "timestamp"])
        except Exception:
            continue
        df = df.sort_values("timestamp")
        correct = (df.user_answer.to_numpy() == df.question_id.map(qmap).to_numpy())
        elapsed = df.elapsed_time.to_numpy(dtype=float)
        ts = df.timestamp.to_numpy(dtype=float)
        n = len(correct) // 2 * 2
        if n < 20:
            continue
        users[uid] = (correct[:n].astype(float), elapsed[:n], ts[:n])
        if (i + 1) % 5000 == 0:
            print(f"  read {i+1:,} pinned users", flush=True)
    return users


def features_at(c, e, t, p):
    k = max(int(p * len(c)), 1)
    cs, es, tsn = c[:k], e[:k], t[:k]
    roll = pd.Series(cs).rolling(20, min_periods=5).mean().dropna()
    flat = [float(cs.mean()), float(len(cs)), float(np.log1p(es.mean())),
            float(roll.std()) if len(roll) > 1 else 0.0,
            float(np.log1p(tsn[-1] - tsn[0]))]
    tok = np.zeros((N_BINS, 4))
    edges = np.linspace(0, k, N_BINS + 1)
    bins = np.clip(np.digitize(np.arange(k), edges[1:-1]), 0, N_BINS - 1)
    for j in range(N_BINS):
        m = bins == j
        if m.any():
            tok[j] = (cs[m].mean(), float(m.sum()), np.log1p(es[m].mean()),
                      np.log1p(np.diff(tsn[m]).mean() if m.sum() > 1 else 0.0))
    return flat, tok


def main():
    seed_everything()
    print("reading pinned cohort ...", flush=True)
    users = load_cohort()
    uids = sorted(users)
    print(f"cohort: {len(uids):,} users", flush=True)

    # fixed prospective label from the second half
    labels = np.array([label_from_rate(users[u][0][len(users[u][0]) // 2:].mean()) for u in uids])
    toks_u = [users[u] for u in uids]

    raw = {p: [features_at(*u, p) for u in toks_u] for p in CUTOFFS}
    print("features computed for all cut-offs", flush=True)

    pos = np.arange(len(uids))
    p_tr, p_tmp = train_test_split(pos, test_size=0.30, random_state=42, stratify=labels)
    p_va, p_te = train_test_split(p_tmp, test_size=0.50, random_state=42, stratify=labels[p_tmp])

    results = {}
    for p in CUTOFFS:
        print(f"\n=== window p = {p:.1f} ===", flush=True)
        flats = np.array([r[0] for r in raw[p]])
        toks = np.stack([r[1] for r in raw[p]])
        sc = StandardScaler().fit(flats[p_tr])
        F_tr, F_va, F_te = sc.transform(flats[p_tr]), sc.transform(flats[p_va]), sc.transform(flats[p_te])
        y_tr, y_va, y_te = labels[p_tr], labels[p_va], labels[p_te]

        entry = {}
        xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            tree_method="hist", n_jobs=-1, random_state=42,
                            eval_metric="mlogloss")
        xgb.fit(F_tr, y_tr)
        yhat = xgb.predict(F_te)
        entry["XGBoost"] = ordinal_metrics(y_te, yhat)
        m = entry["XGBoost"]
        print(f"  XGBoost            acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} qwk={m['qwk']:.3f}", flush=True)

        def build(Fs, ppos):
            rep = np.repeat(Fs, N_BINS, axis=0).reshape(len(Fs), N_BINS, -1)
            return np.concatenate([toks[ppos], rep], axis=2)
        A_tr, A_va, A_te = build(F_tr, p_tr), build(F_va, p_va), build(F_te, p_te)
        mean = A_tr[:, :, :4].reshape(-1, 4).mean(0)
        std = A_tr[:, :, :4].reshape(-1, 4).std(0)
        std[std == 0] = 1.0
        for A in (A_tr, A_va, A_te):
            A[:, :, :4] = (A[:, :, :4] - mean) / std
        net = fit_sequence_model("BiLSTM-Attention", A_tr, y_tr, A_va, y_va)
        yhat, _, _ = predict_sequence(net, A_te)
        entry["BiLSTM-Attention"] = ordinal_metrics(y_te, yhat)
        m = entry["BiLSTM-Attention"]
        print(f"  BiLSTM-Attention   acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} qwk={m['qwk']:.3f}", flush=True)
        results[f"{p:.1f}"] = entry

    save_results(os.path.join(OUT, "ednet_dose.json"), results)


if __name__ == "__main__":
    main()
