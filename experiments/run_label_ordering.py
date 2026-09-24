"""Label-ordering sensitivity for the OULAD ordinal target (Limitations (1)).

Re-runs the full leakage-controlled experiment under an alternative ordering
of the four native outcomes -- graded failure treated as MOST severe and
withdrawal second (Distinction < Pass < Withdrawn < Fail) -- and reports the
contemporaneous-minus-prospective gaps next to the primary ordering, where
withdrawal is most severe. The formulation effect should be invariant to any
fixed relabelling shared by both regimes; this script verifies that claim.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (evaluate_run, fit_sequence_model, make_flat_models,
                    predict_sequence, save_results, seed_everything)
import run_oulad as ro

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ALT_MAP = {"Distinction": 0, "Pass": 1, "Withdrawn": 2, "Fail": 3}


def main():
    seed_everything()
    df = ro.load_enrolments()
    df["label"] = df.final_result.map(ALT_MAP)
    df = ro.student_split(df)
    print("alternative ordering:", ALT_MAP)
    print("label distribution:", df.label.value_counts().sort_index().to_dict())

    vle_flat, seq_early, seq_full, seq_ids = ro.vle_features(df)
    asx = ro.assessment_features(df)
    demo = df[["enrol_id"] + ro.CAT_COLS + ro.NUM_COLS]
    reg_flat = {
        "prospective": demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE], on="enrol_id")
                           .merge(asx[["enrol_id"] + ro.EARLY_ASX], on="enrol_id"),
        "contemporaneous": demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE + ro.FULL_VLE], on="enrol_id")
                               .merge(asx[["enrol_id"] + ro.EARLY_ASX + ro.FULL_ASX], on="enrol_id"),
    }
    base = df[["enrol_id", "id_student", "label", "split"]]
    results = []
    for regime, Xreg in reg_flat.items():
        data = base.merge(Xreg, on="enrol_id")
        fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
        tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
        X_tr, X_va, X_te, _ = ro.preprocess(tr[fea], va[fea], te[fea])
        y_tr, y_va, y_te = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()

        for name, model in make_flat_models().items():
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_te)
            results.append(evaluate_run(name, y_te, yhat, te.id_student.to_numpy(),
                                        OUT, "oulad_altorder", regime))

        seq_tensor = seq_early if regime == "prospective" else seq_full
        seq_lookup = dict(zip(seq_ids, seq_tensor))
        zero = np.zeros((ro.N_BINS, 2))
        dense = lambda m: np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
        concat = lambda ids, flatm: np.concatenate(
            [np.stack([seq_lookup.get(i, zero) for i in ids]),
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
            results.append(evaluate_run(name, y_te, yhat, te.id_student.to_numpy(),
                                        OUT, "oulad_altorder", regime))
        for r in results[-6:]:
            m = r["metrics"]
            print(f"  {regime:15s} {r['model']:18s} acc={m['accuracy']:.3f} qwk={m['qwk']:.3f}", flush=True)

    # gap summary under the alternative ordering
    gaps = {"accuracy": [], "qwk": []}
    by = {(r["regime"], r["model"]): r["metrics"] for r in results}
    for name in ("LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost",
                 "LSTM", "BiLSTM-Attention"):
        for k in gaps:
            gaps[k].append(by[("contemporaneous", name)][k] - by[("prospective", name)][k])
    summary = {"ordering": ALT_MAP, "gaps": {
        "accuracy_pp": [round(100 * g, 1) for g in gaps["accuracy"]],
        "qwk": [round(g, 3) for g in gaps["qwk"]],
        "accuracy_range": [round(100 * min(gaps["accuracy"]), 1), round(100 * max(gaps["accuracy"]), 1)],
        "qwk_range": [round(min(gaps["qwk"]), 2), round(max(gaps["qwk"]), 2)],
    }}
    save_results(os.path.join(OUT, "ordering_sensitivity.json"), {"summary": summary, "results": results})
    print("gaps under alternative ordering:", summary["gaps"]["accuracy_range"], summary["gaps"]["qwk_range"])


if __name__ == "__main__":
    main()
