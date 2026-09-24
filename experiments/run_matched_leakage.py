"""Matched-window leakage experiment (OULAD).

Separates the two components of the formulation effect measured in the main
tables. All four configurations share target, student-level split, protocol,
and models; only the feature vector changes:

  clean30   prospective window at p=0.30                          (honest)
  cont30    clean30 + final-exam score          (label-defining injection)
  clean90   prospective window at p=0.90                          (honest)
  cont90    clean90 + final-exam score          (label-defining injection)

  window-extension effect = clean90 - clean30   (legitimate later information)
  leakage step            = cont90 - clean90    (pure label-defining signal
                                                 at a matched window)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (evaluate_run, fit_sequence_model, make_flat_models,
                    predict_sequence, save_results, seed_everything)
import run_oulad as ro

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CONFIGS = [0.30, 0.90]


def features_at(df, demo, p):
    dfp = df.copy()
    dfp["cutoff_day"] = (p * dfp.module_presentation_length).round().astype(int)
    vle_flat, seq, _, seq_ids = ro.vle_features(dfp)
    flat = demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE], on="enrol_id") \
               .merge(ro.assessment_features(dfp)[["enrol_id"] + ro.EARLY_ASX], on="enrol_id")
    return flat, seq, seq_ids


def with_exam(flat, df):
    ex = ro.assessment_features(df)[["enrol_id", "exam_score"]]
    return flat.merge(ex, on="enrol_id")


def main():
    seed_everything()
    base = ro.load_enrolments()
    base = ro.student_split(base)
    demo = base[["enrol_id"] + ro.CAT_COLS + ro.NUM_COLS]

    flat30, seq30, ids30 = features_at(base, demo, 0.30)
    flat90, seq90, ids90 = features_at(base, demo, 0.90)
    sets = {
        "clean30": (flat30, seq30, ids30),
        "cont30": (with_exam(flat30, base), seq30, ids30),
        "clean90": (flat90, seq90, ids90),
        "cont90": (with_exam(flat90, base), seq90, ids90),
    }

    results = {}
    for cname, (Xreg, seq, seq_ids) in sets.items():
        print(f"\n=== {cname} ({Xreg.shape[1]-1} features) ===", flush=True)
        data = base[["enrol_id", "id_student", "label", "split"]].merge(Xreg, on="enrol_id")
        fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
        tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
        X_tr, X_va, X_te, _ = ro.preprocess(tr[fea], va[fea], te[fea])
        y_tr, y_va, y_te = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()

        entry = {}
        for name, model in make_flat_models().items():
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_te)
            r = evaluate_run(name, y_te, yhat, te.id_student.to_numpy(), OUT,
                             "oulad_matched", cname)
            entry[name] = r["metrics"]
            print(f"  {name:18s} acc={entry[name]['accuracy']:.3f} f1={entry[name]['macro_f1']:.3f} "
                  f"qwk={entry[name]['qwk']:.3f}", flush=True)

        lookup = dict(zip(seq_ids, seq))
        zero = np.zeros((ro.N_BINS, 2))
        dense = lambda m: np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
        concat = lambda ids, flatm: np.concatenate(
            [np.stack([lookup.get(i, zero) for i in ids]),
             np.repeat(flatm, ro.N_BINS, 0).reshape(len(ids), ro.N_BINS, -1)], axis=2)
        A_tr = concat(tr.enrol_id.to_numpy(), dense(X_tr))
        A_va = concat(va.enrol_id.to_numpy(), dense(X_va))
        A_te = concat(te.enrol_id.to_numpy(), dense(X_te))
        mean, std = A_tr[:, :, :2].reshape(-1, 2).mean(0), A_tr[:, :, :2].reshape(-1, 2).std(0)
        std[std == 0] = 1.0
        for A in (A_tr, A_va, A_te):
            A[:, :, 0] = (A[:, :, 0] - mean[0]) / std[0]
            A[:, :, 1] = (A[:, :, 1] - mean[1]) / std[1]
        for name in ("LSTM", "BiLSTM-Attention"):
            net = fit_sequence_model(name, A_tr, y_tr, A_va, y_va)
            yhat, _, _ = predict_sequence(net, A_te)
            r = evaluate_run(name, y_te, yhat, te.id_student.to_numpy(), OUT,
                             "oulad_matched", cname)
            entry[name] = r["metrics"]
            print(f"  {name:18s} acc={entry[name]['accuracy']:.3f} f1={entry[name]['macro_f1']:.3f} "
                  f"qwk={entry[name]['qwk']:.3f}", flush=True)
        results[cname] = entry

    # gap decomposition summary (best tree = XGBoost by convention with main text)
    summary = {}
    for model in results["clean30"]:
        c30, k30 = results["clean30"][model], results["cont30"][model]
        c90, k90 = results["clean90"][model], results["cont90"][model]
        summary[model] = {
            "window_extension_acc": round(c90["accuracy"] - c30["accuracy"], 3),
            "leakage_step_acc": round(k90["accuracy"] - c90["accuracy"], 3),
            "window_extension_qwk": round(c90["qwk"] - c30["qwk"], 3),
            "leakage_step_qwk": round(k90["qwk"] - c90["qwk"], 3),
            "total_cont30_to_cont90_acc": round(k90["accuracy"] - c30["accuracy"], 3),
        }
    save_results(os.path.join(OUT, "matched_leakage.json"),
                 {"summary": summary, "results": results})


if __name__ == "__main__":
    main()
