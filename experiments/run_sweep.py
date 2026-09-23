"""Neural hyperparameter-sensitivity check (reviewer-robustness analysis).

For each dataset (prospective regime) and each sequence model (LSTM,
BiLSTM-Attention), sweep hidden size x learning rate on a small grid, select
the configuration by VALIDATION macro-F1 only, and report its test metrics
next to the a-priori configuration used in the main tables. The main-table
protocol is unchanged; this analysis only asks whether the claim "temporal
models are competitive but not dominant" survives tuning.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LABELS, SequenceClassifier, ordinal_metrics, param_count,
                    predict_sequence, save_results, seed_everything, _device)
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
GRID = {"hidden": [32, 64, 128], "lr": [3e-4, 1e-3, 3e-3]}
EPOCHS, PATIENCE = 40, 6


def train_once(hidden, lr, X_tr, y_tr, X_va, y_va, bidirectional):
    seed_everything()
    device = _device()
    model = SequenceClassifier(X_tr.shape[2], hidden=hidden, bidirectional=bidirectional).to(device)
    counts = np.bincount(y_tr, minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.long)),
                        batch_size=256, shuffle=True)
    Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
    best_f1, best_state, wait = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb)[0], yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv)[0].argmax(1).cpu().numpy()
        f1 = f1_score(y_va, vp, average="macro", labels=LABELS, zero_division=0)
        if f1 > best_f1:
            best_f1, wait = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def oulad_data():
    import run_oulad as ro
    df = ro.load_enrolments()
    df = ro.student_split(df)
    vle_flat, seq_e, seq_f, seq_ids = ro.vle_features(df)
    asx = ro.assessment_features(df)
    demo = df[["enrol_id"] + ro.CAT_COLS + ro.NUM_COLS]
    Xreg = demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE], on="enrol_id") \
               .merge(asx[["enrol_id"] + ro.EARLY_ASX], on="enrol_id")
    data = df[["enrol_id", "id_student", "label", "split"]].merge(Xreg, on="enrol_id")
    fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
    tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
    X_tr, X_va, X_te, _ = ro.preprocess(tr[fea], va[fea], te[fea])
    dense = lambda m: np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
    seq_lookup = dict(zip(seq_ids, seq_e))
    zero = np.zeros((ro.N_BINS, 2))
    cat = lambda ids, flat: np.concatenate(
        [np.stack([seq_lookup.get(i, zero) for i in ids]),
         np.repeat(flat, ro.N_BINS, 0).reshape(len(ids), ro.N_BINS, -1)], axis=2)
    A_tr, A_va, A_te = (cat(f.enrol_id.to_numpy(), dense(X)) for f, X in
                        ((tr, X_tr), (va, X_va), (te, X_te)))
    mean, std = A_tr[:, :, :2].reshape(-1, 2).mean(0), A_tr[:, :, :2].reshape(-1, 2).std(0)
    std[std == 0] = 1.0
    for A in (A_tr, A_va, A_te):
        A[:, :, 0] = (A[:, :, 0] - mean[0]) / std[0]
        A[:, :, 1] = (A[:, :, 1] - mean[1]) / std[1]
    return (A_tr, A_va, A_te, tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy())


def ednet_data():
    import run_ednet as red
    flat, seq, pro, con = red.load_users()
    flat = flat.set_index("uid")
    flat["label_pro"] = pro.rate.map(red.label_from_rate)
    flat["label_con"] = con.rate.map(red.label_from_rate)
    flat = flat.dropna(subset=["label_pro", "label_con"])
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    cfg = {"features": ["acc_first", "n_first", "log_elapsed_first",
                        "roll_sd_first", "log_span_first"], "label": "label_pro"}
    X = flat[cfg["features"]].to_numpy(float)
    y = flat[cfg["label"]].astype(int).to_numpy()
    pos = np.arange(len(flat))
    p_tr, p_tmp = train_test_split(pos, test_size=0.30, random_state=42, stratify=y)
    p_va, p_te = train_test_split(p_tmp, test_size=0.50, random_state=42, stratify=y[p_tmp])
    sc = StandardScaler().fit(X[p_tr])
    tr_f, va_f, te_f = sc.transform(X[p_tr]), sc.transform(X[p_va]), sc.transform(X[p_te])
    toks = np.stack(seq.tok_first.reindex(flat.index).to_numpy())

    def build(p, Xs):
        rep = np.repeat(Xs, red.N_BINS, 0).reshape(len(p), red.N_BINS, -1)
        return np.concatenate([toks[p], rep], axis=2)
    A_tr, A_va, A_te = build(p_tr, tr_f), build(p_va, va_f), build(p_te, te_f)
    mean, std = A_tr[:, :, :4].reshape(-1, 4).mean(0), A_tr[:, :, :4].reshape(-1, 4).std(0)
    std[std == 0] = 1.0
    for A in (A_tr, A_va, A_te):
        A[:, :, :4] = (A[:, :, :4] - mean) / std
    return (A_tr, A_va, A_te, y[p_tr], y[p_va], y[p_te])


def main():
    seed_everything()
    payload = {}
    for dataset, loader in (("oulad", oulad_data), ("ednet", ednet_data)):
        A_tr, A_va, A_te, y_tr, y_va, y_te = loader()
        payload[dataset] = {}
        for name, bidi in (("LSTM", False), ("BiLSTM-Attention", True)):
            rows = []
            for h, lr in itertools.product(GRID["hidden"], GRID["lr"]):
                model, val_f1 = train_once(h, lr, A_tr, y_tr, A_va, y_va, bidi)
                yhat, _, _ = predict_sequence(model, A_te)
                m = ordinal_metrics(y_te, yhat)
                rows.append({"hidden": h, "lr": lr, "val_macro_f1": round(float(val_f1), 4),
                             "params": int(param_count(model)),
                             **{k: round(m[k], 4) for k in ("accuracy", "macro_f1", "qwk", "mae")}})
                print(f"{dataset} {name} h={h} lr={lr}: valF1={val_f1:.3f} testAcc={m['accuracy']:.3f} "
                      f"testF1={m['macro_f1']:.3f} QWK={m['qwk']:.3f}", flush=True)
            best = max(rows, key=lambda r: r["val_macro_f1"])
            payload[dataset][name] = {"grid": GRID, "rows": rows, "best_by_val": best}
            print(f"  -> best by val: {best}", flush=True)
    save_results(os.path.join(OUT, "hp_sweep.json"), payload)


if __name__ == "__main__":
    main()
