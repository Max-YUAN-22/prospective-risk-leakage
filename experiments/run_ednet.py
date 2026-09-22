"""EdNet-KT1 leakage-controlled experiment.

Each user's interaction sequence is split in half by position:
  prospective     -- features from the FIRST half of the sequence; label from
                     mean correctness of the SECOND half -> X_t -> Y_{t+1}.
  contemporaneous -- features from the FULL sequence; label from mean
                     correctness of the FULL sequence (the label-defining
                     signal is itself in the feature window).

Users are independent (one row per user), so the 70/15/15 stratified split is
student-level by construction. Users with fewer than 20 interactions are
excluded; a deterministic random sample of N_USERS users (seed 42) is used.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (evaluate_run, fit_sequence_model, make_flat_models, param_count,
                    predict_sequence, save_results, seed_everything)

DATA = "/Users/Apple/Desktop/灵感投稿/教育会议/EI/03_Data"
KT1 = f"{DATA}/KT1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N_BINS = 10
N_USERS = 30_000
MIN_INTERACTIONS = 20


def label_from_rate(r):
    return 0 if r >= 0.85 else 1 if r >= 0.70 else 2 if r >= 0.50 else 3


def load_users():
    qmap = pd.read_csv(f"{DATA}/contents/questions.csv", low_memory=False)
    qmap = dict(zip(qmap.question_id, qmap.correct_answer))

    files = sorted(f for f in os.listdir(KT1) if f.endswith(".csv"))
    rng = np.random.default_rng(42)
    pick = rng.choice(len(files), size=min(N_USERS * 2, len(files)), replace=False)
    pick.sort()  # deterministic read order

    rows_flat, rows_seq, labels_pro, labels_con = [], [], [], []
    n_read = 0
    for idx in pick:
        fn = files[idx]
        path = os.path.join(KT1, fn)
        try:
            df = pd.read_csv(path, usecols=["user_answer", "question_id", "elapsed_time", "timestamp"])
        except Exception:
            continue
        n_read += 1
        if len(df) < MIN_INTERACTIONS:
            continue
        uid = fn[:-4]
        df = df.sort_values("timestamp")
        correct = (df.user_answer.to_numpy() == df.question_id.map(qmap).to_numpy())
        elapsed = df.elapsed_time.to_numpy(dtype=float)
        ts = df.timestamp.to_numpy(dtype=float)
        if len(df) % 2 == 1:                      # even halves
            correct, elapsed, ts = correct[:-1], elapsed[:-1], ts[:-1]
        h = len(correct) // 2

        # label sources
        rate_second = correct[h:].mean()
        rate_full = correct.mean()
        labels_pro.append((uid, rate_second))
        labels_con.append((uid, rate_full))

        # sequence tokens over the FIRST half and over the FULL sequence
        def bin_tensor(c, e, t):
            n = len(c)
            tok = np.zeros((N_BINS, 4))
            edges = np.linspace(0, n, N_BINS + 1)
            bins = np.clip(np.digitize(np.arange(n), edges[1:-1]), 0, N_BINS - 1)
            for k in range(N_BINS):
                m = bins == k
                if m.any():
                    tok[k] = (c[m].mean(), float(m.sum()), np.log1p(e[m].mean()),
                              np.log1p(np.diff(t[m]).mean() if m.sum() > 1 else 0.0))
            return tok
        rows_seq.append((uid, bin_tensor(correct[:h], elapsed[:h], ts[:h]),
                         bin_tensor(correct, elapsed, ts)))

        # flat aggregates
        def flat_for(seg_correct, seg_elapsed, seg_ts):
            gaps = np.diff(seg_ts)
            roll = pd.Series(seg_correct.astype(float)).rolling(20, min_periods=5).mean().dropna()
            return [float(seg_correct.mean()), float(len(seg_correct)),
                    float(np.log1p(seg_elapsed.mean())),
                    float(roll.std()) if len(roll) > 1 else 0.0,
                    float(np.log1p(seg_ts[-1] - seg_ts[0]))]
        f_first = flat_for(correct[:h], elapsed[:h], ts[:h])
        f_full = flat_for(correct, elapsed, ts)
        rows_flat.append([uid] + f_first + f_full)
        if n_read % 5000 == 0:
            print(f"  read {n_read:,} files, kept {len(rows_flat):,} users", flush=True)
        if len(rows_flat) >= N_USERS:
            break

    flat = pd.DataFrame(rows_flat, columns=[
        "uid", "acc_first", "n_first", "log_elapsed_first", "roll_sd_first", "log_span_first",
        "acc_full", "n_full", "log_elapsed_full", "roll_sd_full", "log_span_full"])
    seq = pd.DataFrame(rows_seq, columns=["uid", "tok_first", "tok_full"]).set_index("uid")
    pro = pd.DataFrame(labels_pro, columns=["uid", "rate"]).set_index("uid")
    con = pd.DataFrame(labels_con, columns=["uid", "rate"]).set_index("uid")
    return flat, seq, pro, con


def main():
    seed_everything()
    print(f"sampling users from {KT1} ...", flush=True)
    flat, seq, pro, con = load_users()
    flat = flat.set_index("uid")
    flat["label_pro"] = pro.rate.map(label_from_rate)
    flat["label_con"] = con.rate.map(label_from_rate)
    flat = flat.dropna(subset=["label_pro", "label_con"])
    print(f"users kept: {len(flat):,}")
    print("label distribution (prospective):", flat.label_pro.value_counts().sort_index().to_dict())

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    regimes = {
        "prospective": dict(features=["acc_first", "n_first", "log_elapsed_first",
                                      "roll_sd_first", "log_span_first"],
                            label="label_pro"),
        "contemporaneous": dict(features=["acc_full", "n_full", "log_elapsed_full",
                                          "roll_sd_full", "log_span_full",
                                          "acc_first", "n_first"],
                                label="label_con"),
    }
    summary = {"users": int(len(flat)),
               "min_interactions": MIN_INTERACTIONS,
               "label_distribution": {int(k): int(v) for k, v in
                                      flat.label_pro.value_counts().sort_index().items()},
               "n_features": {r: len(cfg["features"]) for r, cfg in regimes.items()}}
    results = []

    Xall = {r: flat[cfg["features"]].to_numpy(dtype=float) for r, cfg in regimes.items()}
    yall = {r: flat[cfg["label"]].astype(int).to_numpy() for r, cfg in regimes.items()}
    uids = flat.index.to_numpy()
    toks_first = np.stack(seq.tok_first.reindex(flat.index).to_numpy())
    toks_full = np.stack(seq.tok_full.reindex(flat.index).to_numpy())

    for regime, cfg in regimes.items():
        print(f"\n=== regime: {regime} ({len(cfg['features'])} features) ===")
        X, y = Xall[regime], yall[regime]
        pos = np.arange(len(flat))
        p_tr, p_tmp = train_test_split(pos, test_size=0.30, random_state=42, stratify=y)
        p_va, p_te = train_test_split(p_tmp, test_size=0.50, random_state=42, stratify=y[p_tmp])
        sc = StandardScaler().fit(X[p_tr])
        X_tr, X_va, X_te = sc.transform(X[p_tr]), sc.transform(X[p_va]), sc.transform(X[p_te])
        y_tr, y_va, y_te = y[p_tr], y[p_va], y[p_te]
        st_te = uids[p_te]

        for name, model in make_flat_models().items():
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_te)
            results.append(evaluate_run(name, y_te, yhat, st_te, OUT, "ednet", regime))
            m = results[-1]["metrics"]
            print(f"  {name:18s} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} qwk={m['qwk']:.3f}")

        # regime-matching tokens + flat features appended to every time step
        toks = toks_first if regime == "prospective" else toks_full

        def build(p, Xs):
            t = toks[p]
            flat_rep = np.repeat(Xs, N_BINS, axis=0).reshape(len(p), N_BINS, -1)
            return np.concatenate([t, flat_rep], axis=2)
        Xtr_s, Xva_s, Xte_s = build(p_tr, X_tr), build(p_va, X_va), build(p_te, X_te)
        mean = Xtr_s[:, :, :4].reshape(-1, 4).mean(0)   # first 4 dims are token dims
        std = Xtr_s[:, :, :4].reshape(-1, 4).std(0)
        std[std == 0] = 1.0
        for A in (Xtr_s, Xva_s, Xte_s):
            A[:, :, :4] = (A[:, :, :4] - mean) / std

        for name in ("LSTM", "BiLSTM-Attention"):
            net = fit_sequence_model(name, Xtr_s, y_tr, Xva_s, y_va)
            yhat, _, alpha = predict_sequence(net, Xte_s)
            results.append(evaluate_run(name, y_te, yhat, st_te, OUT, "ednet", regime))
            res = results[-1]
            res["params"] = int(param_count(net))
            if alpha is not None:
                np.save(os.path.join(OUT, f"attention_ednet_{regime}.npy"), alpha)
                res["attention_mean"] = alpha.mean(axis=0).tolist()
            print(f"  {name:18s} acc={res['metrics']['accuracy']:.3f} "
                  f"f1={res['metrics']['macro_f1']:.3f} qwk={res['metrics']['qwk']:.3f} "
                  f"params={res['params']:,}")

    save_results(os.path.join(OUT, "ednet_results.json"),
                 {"summary": summary, "results": results})


if __name__ == "__main__":
    main()
