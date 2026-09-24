"""Temporal generalization: train on the 2013 cohorts, test on the 2014 cohorts.

Answers the deployment question the random student-level split cannot: does
the formulation effect survive a full cohort shift (new students, one year
later)? Students appearing in the 2014 presentations are removed from the
2013 training pool, so no learner crosses the split. Validation for early
stopping is carved from the 2013 pool only.
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
MODELS = ["XGBoost", "LSTM", "BiLSTM-Attention"]


def main():
    seed_everything()
    df = ro.load_enrolments()
    df["cohort"] = df.code_presentation.str[:4]
    test_ids = set(df[df.cohort == "2014"].id_student)
    train = df[df.cohort == "2013"]
    train = train[~train.id_student.isin(test_ids)].copy()   # no cross-cohort student
    test = df[df.cohort == "2014"].copy()
    print(f"train pool (2013, test-students removed): {len(train):,} enrolments / "
          f"{train.id_student.nunique():,} students")
    print(f"test (2014): {len(test):,} enrolments / {test.id_student.nunique():,} students")

    # validation carve-out from the 2013 pool, by student, stratified by worst label
    from sklearn.model_selection import train_test_split
    worst = train.groupby("id_student").label.max().reset_index()
    tr_s, va_s = train_test_split(worst, test_size=0.15, random_state=42,
                                  stratify=worst.label)
    split_of = {}
    split_of.update(dict.fromkeys(tr_s.id_student, "train"))
    split_of.update(dict.fromkeys(va_s.id_student, "val"))
    train["split"] = train.id_student.map(split_of)
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)

    vle_flat, seq_e, seq_f, seq_ids = ro.vle_features(df)
    asx = ro.assessment_features(df)
    demo = df[["enrol_id"] + ro.CAT_COLS + ro.NUM_COLS]
    reg = {
        "prospective": demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE], on="enrol_id")
                           .merge(asx[["enrol_id"] + ro.EARLY_ASX], on="enrol_id"),
        "contemporaneous": demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE + ro.FULL_VLE], on="enrol_id")
                               .merge(asx[["enrol_id"] + ro.EARLY_ASX + ro.FULL_ASX], on="enrol_id"),
    }
    base = df[["enrol_id", "id_student", "label", "split"]]
    results = {}
    for regime, Xreg in reg.items():
        data = base.merge(Xreg, on="enrol_id")
        fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
        tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
        X_tr, X_va, X_te, _ = ro.preprocess(tr[fea], va[fea], te[fea])
        y_tr, y_va, y_te = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()

        seq_tensor = seq_e if regime == "prospective" else seq_f
        lookup = dict(zip(seq_ids, seq_tensor))
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

        entry = {}
        for name in MODELS:
            if name in make_flat_models():
                model = make_flat_models()[name]
                model.fit(X_tr, y_tr)
                yhat = model.predict(X_te)
            else:
                net = fit_sequence_model(name, A_tr, y_tr, A_va, y_va)
                yhat, _, _ = predict_sequence(net, A_te)
            r = evaluate_run(name, y_te, yhat, te.id_student.to_numpy(), OUT,
                             "oulad_temporal", regime)
            entry[name] = r["metrics"]
            m = entry[name]
            print(f"  {regime:15s} {name:18s} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} "
                  f"qwk={m['qwk']:.3f}", flush=True)
        results[regime] = entry

    summary = {m: {"gap_acc": round(results["contemporaneous"][m]["accuracy"]
                                   - results["prospective"][m]["accuracy"], 3),
                   "gap_qwk": round(results["contemporaneous"][m]["qwk"]
                                    - results["prospective"][m]["qwk"], 3)}
               for m in MODELS}
    save_results(os.path.join(OUT, "temporal_generalization.json"),
                 {"summary": summary, "results": results})
    print("gaps under cohort shift:", summary)


if __name__ == "__main__":
    main()
