"""SHAP attribution for the best tree ensemble (XGBoost) under the OULAD
prospective protocol. Retrains the identical model on the identical split as
run_oulad.py (deterministic seed) and saves mean-absolute-SHAP rankings to
results/shap_oulad.json for the manuscript figure.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_oulad as ro
from common import seed_everything
from xgboost import XGBClassifier
import shap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

PRETTY = {
    "gender": "Gender", "region": "Region", "highest_education": "Prior education",
    "imd_band": "IMD band", "age_band": "Age band", "disability": "Disability",
    "num_of_prev_attempts": "Previous attempts", "studied_credits": "Studied credits",
    "clicks_early": "Clicks (early window)", "days_early": "Active days (early)",
    "meanday_early": "Mean daily clicks (early)", "stdday_early": "Daily-click SD (early)",
    "n_submitted_early": "Assessments submitted (early)",
    "mean_score_early": "Mean assessment score (early)",
    "min_score_early": "Min assessment score (early)",
    "any_missing_early": "No early assessment (flag)",
    "clicks_full": "Clicks (full course)", "days_full": "Active days (full)",
    "meanday_full": "Mean daily clicks (full)", "stdday_full": "Daily-click SD (full)",
    "n_submitted_full": "Assessments submitted (full)",
    "mean_score_full": "Mean assessment score (full)",
    "min_score_full": "Min assessment score (full)",
    "max_score_full": "Max assessment score (full)",
    "exam_score": "Final exam score",
}


def pretty_name(raw):
    name = raw.split("__", 1)[1] if "__" in raw else raw
    for key in sorted(PRETTY, key=len, reverse=True):
        if name == key:
            return PRETTY[key]
        if name.startswith(key + "_"):
            return f"{PRETTY[key]} = {name[len(key)+1:].replace('_', ' ')}"
    return name.replace("_", " ")


def main():
    seed_everything()
    df = ro.load_enrolments()
    df = ro.student_split(df)
    vle_flat, _, _, _ = ro.vle_features(df)
    asx = ro.assessment_features(df)
    demo = df[["enrol_id"] + ro.CAT_COLS + ro.NUM_COLS]
    Xreg = demo.merge(vle_flat[["enrol_id"] + ro.EARLY_VLE], on="enrol_id") \
               .merge(asx[["enrol_id"] + ro.EARLY_ASX], on="enrol_id")
    data = df[["enrol_id", "id_student", "label", "split"]].merge(Xreg, on="enrol_id")
    fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
    tr, te = data[data.split == "train"], data[data.split == "test"]
    X_tr, _, X_te, pre = ro.preprocess(tr[fea], te[fea], te[fea])
    y_tr, y_te = tr.label.to_numpy(), te.label.to_numpy()

    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                          tree_method="hist", n_jobs=-1, random_state=42,
                          eval_metric="mlogloss")
    model.fit(X_tr, y_tr)
    print(f"test acc: {model.score(X_te, y_te):.3f}")

    names = [pretty_name(n) for n in pre.get_feature_names_out()]
    sample = X_te[: min(2000, len(X_te))]
    sv = shap.TreeExplainer(model).shap_values(sample)      # (n, F, K) for multiclass
    sv = np.asarray(sv)
    if sv.ndim == 2:                                        # binary edge case
        sv = sv[:, :, None]
    mean_abs = np.abs(sv).mean(axis=0).mean(axis=1)         # avg over samples and classes
    order = np.argsort(mean_abs)[::-1]
    payload = {
        "dataset": "oulad", "regime": "prospective", "model": "XGBoost",
        "n_background": int(X_tr.shape[0]), "n_explained": int(len(sample)),
        "features": [{"name": names[i], "mean_abs": float(mean_abs[i])} for i in order],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "shap_oulad.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print("top 10:")
    for f in payload["features"][:10]:
        print(f"  {f['mean_abs']:.4f}  {f['name']}")
    print(f"saved -> {OUT}/shap_oulad.json")


if __name__ == "__main__":
    main()
