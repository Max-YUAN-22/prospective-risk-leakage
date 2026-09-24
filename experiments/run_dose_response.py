"""Dose-response of leakage inflation (OULAD).

Holds target, student-level split, protocol, and models fixed while growing
the feature window from the prospective cut (30% of the course) to the full
course (100%, where the window contains the label-defining signals). Only the
cut-off varies between points, so the curve isolates the effect of
window-label overlap. Uses the identical split across all cut-offs: the split
is stratified by the label, which does not depend on the cut-off.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (evaluate_run, fit_sequence_model, make_flat_models, predict_sequence,
                    save_results, seed_everything)
import run_oulad as ro

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CUTOFFS = [0.30, 0.50, 0.70, 0.90, 1.00]
MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost",
          "LSTM", "BiLSTM-Attention"]


def main():
    seed_everything()
    base = ro.load_enrolments()          # cutoff_day = 0.30 * length
    base = ro.student_split(base)        # identical split reused for every p
    vle_full_frame, _, _, _ = ro.vle_features(base)   # not used; warms nothing, skip
    demo = base[["enrol_id"] + ro.CAT_COLS + ro.NUM_COLS]
    results = {}

    for p in CUTOFFS:
        print(f"\n=== feature window p = {p:.2f} ===", flush=True)
        dfp = base.copy()
        dfp["cutoff_day"] = (p * dfp.module_presentation_length).round().astype(int)
        vle_flat, seq_e, seq_ids = None, None, None
        # vle_features computes "early" at cutoff_day and the early tensor there;
        # we only need the early parts.
        vle_flat_all, seq_all, _, seq_ids = ro.vle_features(dfp)
        vle_flat = vle_flat_all[["enrol_id"] + ro.EARLY_VLE]
        asx = ro.assessment_features(dfp)[["enrol_id"] + ro.EARLY_ASX]
        Xreg = demo.merge(vle_flat, on="enrol_id").merge(asx, on="enrol_id")

        data = base[["enrol_id", "id_student", "label", "split"]].merge(Xreg, on="enrol_id")
        fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
        tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
        from run_oulad import preprocess
        X_tr, X_va, X_te, _ = preprocess(tr[fea], va[fea], te[fea])
        y_tr, y_va, y_te = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()

        entry = {}
        for name, model in make_flat_models().items():
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_te)
            r = evaluate_run(name, y_te, yhat, te.id_student.to_numpy(), OUT,
                             "oulad_dose", f"p{int(p*100):03d}")
            entry[name] = r["metrics"]
            print(f"  {name:18s} acc={entry[name]['accuracy']:.3f} qwk={entry[name]['qwk']:.3f}", flush=True)

        seq_lookup = dict(zip(seq_ids, seq_all))
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
            r = evaluate_run(name, y_te, yhat, te.id_student.to_numpy(), OUT,
                             "oulad_dose", f"p{int(p*100):03d}")
            entry[name] = r["metrics"]
            print(f"  {name:18s} acc={entry[name]['accuracy']:.3f} qwk={entry[name]['qwk']:.3f}", flush=True)
        results[f"{p:.2f}"] = entry

    save_results(os.path.join(OUT, "dose_response.json"), results)


if __name__ == "__main__":
    main()
